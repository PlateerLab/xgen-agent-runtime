# Durable rollout recording

Rollout recording is an opt-in host capability adapted from the Codex harness.
It records the pipeline's existing event stream as ordered JSONL without
changing `Pipeline.run`, `Pipeline.run_stream`, `PipelineState`, `PipelineEvent`,
or the agent node's result shape.

## Enablement

Set the host setting below to a truthy value:

```text
GENY_ROLLOUT_RECORDING_ENABLED=true
```

It is disabled by default because events may contain user prompts, model
responses, and tool inputs/results. A turn without a `workflow_id` is left
unrecorded and emits a warning; it never falls back to a process-temporary path.

For enabled workflow turns the host writes one unique file beneath:

```text
<workspace_storage_root>/executor/rollouts/rollout-*.jsonl
```

Raw interaction IDs are hashed in filenames. The directory retains the newest
100 generated rollout files; unrelated files and symlinks are never pruned.

## Lifecycle and failure behavior

The synchronous host bridge creates the recorder inside its private asyncio
loop, overlays it on the existing free-shape `session_runtime`, and restores the
original runtime after the turn. Pipeline shutdown happens before recorder
shutdown, so cancellation drains every accepted event prefix before the event
loop closes.

Terminal events are flushed and fsynced before they are published to consumers.
If recording fails during a configured run, the failure uses the existing
`pipeline.error` / `[ERROR]` result path instead of producing a plausible but
incomplete audit file. Shutdown and retention errors are logged during teardown
without leaking a background writer task.

## Pre-release verification

The automated suite covers:

- streaming and non-streaming terminal records;
- early stream close and state-runtime restoration;
- persistent disk failure and writer-task cleanup;
- path traversal, symlink safety, uniqueness, and retention;
- the full `AgentTurnExecutor -> Pipeline -> JSONL` path with a network-free LLM.

After a consuming service updates its runtime release, the minimum smoke check
is one enabled non-streaming turn and one cancelled streaming turn. Confirm that
the completed file ends with `pipeline.complete`, the cancelled file is valid
JSONL, and the service has no pending `rollout-recorder:*` task at turn teardown.
