# Long-running execution slices

`max_iterations` bounds one execution slice. It no longer claims that the
user's task succeeded.

## Status contract

| Status | Meaning | `success` | Resumable |
|---|---|---:|---:|
| `completed` | The model/task produced a real completion verdict | true | no |
| `suspended` | A per-slice guard fired; continuation state is intact | false | yes |
| `blocked` | New input, authority, or budget is required | false | no |
| `failed` | Execution failed | false | no |

`PipelineResult.status`, `termination_reason`, `resumable`, and
`checkpoint_id` are read-only compatibility properties. The existing result
dataclass field order is unchanged. `pipeline.complete` remains the terminal
event for a pipeline invocation, but its payload now contains these fields;
consumers must not interpret the event name alone as task completion.

Iteration, tool-call, and wall-clock slice guards produce `suspended`. Cost
budget exhaustion produces `blocked`. Token pressure does not produce a task
verdict: it requests context compaction and continues when work is pending.

## Continuing a slice

Use `CONTINUE_RUN` so Stage 1 does not append a fake user message:

```python
from xgen_agent_runtime import CONTINUE_RUN

result = await pipeline.run(user_input, state)
while result.resumable:
    result = await pipeline.run(CONTINUE_RUN, result.state)
```

A continuation resets the iteration counter and slice-local events, but keeps
the active turn's messages, token usage, and cost. Session cost accounting
adds only the new slice's delta. This prevents slicing from bypassing a task's
cost budget or double-counting spend.

The synchronous host bridges auto-continue up to 20 additional slices by
default. Configure `max_continuation_slices` to change that bound. If the
bound is reached, streaming emits a `task_suspended` agent event and the
non-streaming bridge returns a `[SUSPENDED]` marker instead of `[ERROR]` or a
false success.

## Checkpoints

When Stage 20 persistence is configured, every suspended boundary forces a
checkpoint even when an `every_n_turns` policy would normally skip it. The
payload includes status, termination reason, resumability, full canonical
messages, and turn cost. `checkpoint_id` is exposed on state/result and is
restored by `state_from_record` / `restore_state_from_checkpoint`.

`FileSessionPersistence` uses tempfile + fsync + atomic replace and accepts
both its v1 and v2 formats. Its v2 format round-trips suspended state.

## Compaction ownership and ordering

There must be one history-compaction owner for a thread:

- SDK providers: the runtime owns canonical `state.messages` compaction.
- Codex/Claude subprocess providers: the native CLI thread owns compaction;
  the host disables runtime Stage 2 compaction for those backends.

Runtime compaction is request-boundary maintenance, not an agent iteration:

1. Stage 2 triggers proactively above 80% of the context window.
2. A Stage 16 token dimension requests synchronous compaction for the next
   pass instead of returning `complete`.
3. The target low watermark is 70%. `context.compaction_target_missed`
   reports compactors that fail to reach it; Stage 4 still performs the hard
   preflight recheck.
4. The SDK internal tool loop runs the same compaction boundary before each
   nested model request, closing the gap where Stage 2 cannot execute.
5. Background results are installed only when every message identity in the
   captured prefix still matches, preventing a stale summary from replacing a
   newer synchronous compaction.

This follows the public Codex harness rule that compaction replaces history
and continues the current turn rather than ending it. OpenAI's Responses
compaction endpoint likewise returns a new compacted context item to use in
subsequent requests; it is not a task-completion response. See the
[Codex harness](https://github.com/openai/codex) and the
[Responses compact API](https://developers.openai.com/api/reference/resources/responses/methods/compact).

For Codex CLI `exec resume`, the runtime sends only the message suffix after
the last assistant turn. It does not resend the system prompt or flattened
full transcript already owned (and possibly compacted) by the native thread.
Raw `text.delta` events retain every model message for audit. At the
user-facing host bridge only, exact consecutive repeats from a
message-granularity CLI stream are coalesced; token-granularity SDK output is
never deduplicated.

## Diagnosing “50 iterations, 40 tool calls”

Iterations, model requests, and tool calls are different counters. One loop
iteration can execute zero, one, or several tools, and the SDK internal loop
can make several model/tool rounds inside one outer iteration. An evaluator or
loop strategy may also have had a smaller `max_turns` than the host's
`max_iterations`. Adaptive evaluation now records
`metadata.evaluation_suggested_max_turns` without overwriting the host's hard
cap. The stopping event includes the exact dimension, effective cap, status,
and reason so callers no longer have to infer the cause from counts.
