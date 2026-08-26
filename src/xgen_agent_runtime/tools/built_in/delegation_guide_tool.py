"""DelegationGuide — 위임 스킬 게이트웨이 (Guide + 컴팩트 멤버, DocGuide 동형).

위임 표면은 세 층(DelegateTask 단일 동사 / SubAgent* 상주 서브에이전트 /
Task* 백그라운드 태스크 레지스트리)이라, 어느 한 도구의 description 도
"언제 무엇을 쓰는가"의 교차 판단을 소유할 수 없다. 그 결정 지도와 수명주기
지식을 이 게이트웨이가 요청 시에만 공개한다 — 매 턴 컨텍스트를 태우지 않는다.

topic 없음 → 결정 지도. topic → 심층 가이드 (subagents / tasks / patterns).
"""

from __future__ import annotations

from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

_MAP = """\
Delegation skill — three surfaces, pick by shape of the work:

  1. DelegateTask(task)            ONE VERB, the default. Hands the whole task
     to your persistent companion sub-agent; returns immediately and the
     completion arrives automatically as a [SUB_AGENT_RESULT] turn. Never
     wait/poll for it — continue the conversation.
  2. SubAgent* (Spawn/Assign/...)  Standing WORKERS you manage yourself:
     typed sub-agents that stay alive, keep their own context across
     assignments, and notify your inbox. Use for repeated roles
     (researcher, summarizer, critic) or parallel long-running work.
  3. Task* (Create/Get/...)        RAW background task registry (no LLM):
     machine jobs like local_bash with streamed output. Use when the work
     is a process, not an agent.

Rules of thumb:
  - One-shot "do X for me in the background"  -> DelegateTask.
  - A role you will assign to repeatedly       -> SubAgentSpawn + SubAgentAssign.
  - A shell/process you want to run and watch  -> TaskCreate + TaskOutput.
  - Scheduled/recurring work is NOT delegation -> use the Job tools.

Topics (call DelegationGuide(topic=...) for the deep guide):
  - subagents  spawn -> assign -> inbox lifecycle, types, custom instructions
  - tasks      task kinds, statuses, output streaming, cancel
  - patterns   parallel fan-out, standing pipelines, when NOT to delegate
"""

_TOPICS = {
    "subagents": """\
Persistent sub-agents — lifecycle:
  1. SubAgentSpawn(subagent_type, ...)  create a standing worker you own.
     Types: worker (default, general) / researcher / summarizer / critic.
     Custom role: pass instructions (system-prompt override) and optionally a
     model override at spawn.
  2. SubAgentAssign(sub_agent_id, task) hand it a task; it runs autonomously
     in the background. The same sub-agent keeps its context across
     assignments while the pod lives — later tasks can build on earlier ones.
  3. SubAgentInboxRead()                completion/failure notifications land
     in your inbox; reading drains it by default. You are also woken
     automatically on completion — the inbox is for catching up, not polling.
  4. SubAgentList() / SubAgentStop(id)  inventory and teardown (stop cancels
     in-flight work and frees the worker).
Do not spawn a new sub-agent per task — reuse a standing one per role.""",
    "tasks": """\
Background task registry — machine jobs, no LLM:
  - TaskCreate(kind, payload)  submit; returns task_id. Built-in kinds include
    'local_bash' (payload.command) — a process in your execution host.
  - TaskGet(task_id)           one record; status is pending | running | done |
    failed | cancelled (transitions are driven by the runner, not you).
  - TaskOutput(task_id)        accumulated output bytes (UTF-8 decoded,
    truncation-aware) — safe to call while running; call again for more.
  - TaskList(...)              recent tasks, filterable by status/kind.
  - TaskUpdate(task_id, ...)   only 'payload' is mutable, and only pre-run.
  - TaskStop(task_id)          cancel a running task.
Use tasks for processes you want to observe; use DelegateTask when the work
needs judgment (an agent), not just execution.""",
    "patterns": """\
Patterns:
  - Fan-out: spawn 2-3 typed sub-agents, SubAgentAssign each a slice, keep
    answering the user; merge when [SUB_AGENT_RESULT] turns arrive.
  - Standing pipeline: one researcher + one summarizer sub-agent reused all
    conversation — assign, don't respawn.
  - Long shell job: TaskCreate(local_bash) then periodic TaskOutput reads
    woven into your turns; TaskStop if the user changes course.
  - When NOT to delegate: quick lookups or single tool calls — calling the
    tool yourself is faster than any delegation round-trip. And scheduled/
    recurring work belongs to the Job tools, not delegation.""",
}


class DelegationGuideTool(Tool):
    """위임 스킬 게이트웨이 — 계층형 가이드, 점진공개."""

    @property
    def name(self) -> str:
        return "DelegationGuide"

    @property
    def description(self) -> str:
        return (
            "START HERE for delegation/background work — the delegation "
            "skill. No topic: the decision map across DelegateTask / "
            "SubAgent* / Task*. topic: deep guide (subagents, tasks, "
            "patterns). Free, instant."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": sorted(_TOPICS),
                    "description": "Deep-guide topic. Omit for the decision map.",
                },
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        topic = str((input or {}).get("topic") or "").strip()
        if topic and topic in _TOPICS:
            return ToolResult(
                content=_TOPICS[topic],
                metadata={"topic": topic, "topics": sorted(_TOPICS)},
            )
        return ToolResult(content=_MAP, metadata={"topics": sorted(_TOPICS)})
