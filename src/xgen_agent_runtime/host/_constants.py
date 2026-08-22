"""AgentTurnExecutor 가 항상 쓰는 상수/순수 헬퍼 — agent_geny 에서 이전.

서버·커넥터 공용. ``_self_evolution_policy`` 는 관리자 설정 판정을 host.setting
으로 주입받는다(서버=config DB→env, 커넥터=env) — SDK/CLI/웹/커넥터가 **같은
판정**을 써야 하는 보안 계약(deploy/guest 차단)이 갈라지지 않게. 관련:
``xgeny-shared-host-extraction``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

#: LLM 이 시스템 프롬프트를 안 주면 쓰는 기본값.
default_prompt = "You are a helpful AI assistant."

#: 외부 CLI subprocess 가 에이전트 루프를 소유하는 백엔드.
_CLI_BACKENDS = ("claude_code", "codex")

#: 자기진화 시스템 프롬프트 블록 — 등록된 턴에만 붙는다.
SELF_EVOLUTION_PROMPT_BLOCK = """

# Self-evolution (editing your own workflow graph)
You can PERMANENTLY extend your own capabilities by editing your workflow graph
with the WorkflowSelf tool: attach RAG/document search, tool/MCP/API/DB nodes,
memory, routers, and more. When the user asks you to gain an ability you lack
(e.g. "attach RAG to yourself"), use WorkflowSelf — start with
action='guidance' if unsure. Do NOT confuse this with any harness-provided
"Workflow"/orchestration tool: only WorkflowSelf edits the XGEN graph."""


#: 내장 메모리 시스템 프롬프트 블록 — 메모리 provider 가 붙은 턴에만 붙는다.
MEMORY_PROMPT_BLOCK = """

# Agent Memory (persistent)
You have a persistent memory vault that survives across conversations. Retrieved
context (Pinned Facts / Relevant Knowledge) may already be injected above.
Tools:
- memory_categories() — vault overview. Use FIRST when unsure where to look.
- memory_list(category, tag) — list notes in a folder.
- memory_read(filename) — read one note's full body.
- memory_search(query, max_results) — search across notes (keyword + semantic).
- memory_write(title, content, category, tags, importance) — save durable
  knowledge, decisions, insights. Link related notes with [[filename]].
- memory_pin(title, content, tags) — pin a must-always-know fact (user
  preferences, binding decisions). Pinned facts are always injected.
Policy:
- When the user EXPLICITLY asks you to remember something ("기억해",
  "remember this", "저장해"), you MUST call memory_pin (personal facts,
  preferences, how to address them) or memory_write (knowledge/decisions)
  BEFORE answering — acknowledging without saving loses it forever.
- Also proactively memory_write durable facts, decisions and lessons you
  learn mid-conversation; skip transient chatter. Conversations themselves
  are archived automatically — notes are for distilled knowledge.
- Update or extend existing notes instead of duplicating. Write in the
  user's language.
"""


def _coerce_text(value: Any) -> str:
    """STREAM STR | STR | list 포트 값을 단일 문자열로 평탄화."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_coerce_text(v) for v in value)
    if hasattr(value, "__iter__") and not isinstance(value, dict):
        try:
            return "".join(str(chunk) for chunk in value)
        except TypeError:
            return str(value)
    return str(value)


def _self_evolution_policy(
    kwargs: Dict[str, Any], get_setting: Callable[[str], str]
) -> Tuple[bool, str]:
    """(allowed, reason) — WorkflowSelf(자기진화) 배선 정책의 단일 판정.

    ★ 보안: 배포(deploy_)·게스트(guest_) 실행에서는 절대 허용하지 않는다 —
    익명/프롬프트 인젝션이 라이브 그래프를 영구 변조할 수 있다(감사 CRITICAL).
    관리자 설정 판정은 ``get_setting`` (host.setting) 으로 — 서버·커넥터 동일.
    """
    iid = str(kwargs.get("interaction_id") or "")
    if iid.startswith("deploy_") or iid.startswith("guest_"):
        return False, "deploy/guest 컨텍스트 (보안 차단)"
    if not bool(kwargs.get("enable_self_evolution", True)):
        return False, "노드 파라미터 enable_self_evolution=off"
    raw = (get_setting("GENY_TOOLS_WORKFLOW_SELF_ENABLED") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False, "관리자 설정 GENY_TOOLS_WORKFLOW_SELF_ENABLED=off"
    if not str(kwargs.get("workflow_id") or ""):
        return False, "workflow_id 없음 (비정형 실행)"
    return True, ""
