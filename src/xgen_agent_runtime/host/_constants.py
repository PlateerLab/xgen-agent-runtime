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
#:
#: 철학: 시스템 프롬프트는 **일반화된 사실**만 담는다 — 개별 도구 사용법은 그
#: 도구의 description 이 담는다(중복 금지). 여기 남은 두 가지는 도구 설명이 담을
#: 수 없는 것들이다: (a) 능력의 존재 사실(스키마만으로는 모델이 놓친다 — 프로드
#: 실증 "편집 기능이 없다" 확신 답변), (b) 하네스 자체 Workflow 도구와의 교차
#: 구분(어느 한 도구 설명도 소유할 수 없는 cross-tool 사실).
SELF_EVOLUTION_PROMPT_BLOCK = """

# Self-evolution (editing your own workflow graph)
You can PERMANENTLY extend your own capabilities by editing your own XGEN
workflow graph — the WorkflowSelf tool carries the details. Do NOT confuse it
with any harness-provided "Workflow"/orchestration tool: only WorkflowSelf
edits the XGEN graph."""


#: 내장 메모리 시스템 프롬프트 블록 — 메모리 provider 가 붙은 턴에만 붙는다.
#:
#: 철학: **일반화된 지침만** — 도구 목록/시그니처/"MUST call" 드릴은 금지다.
#: 개별 memory_* 도구의 사용법은 각 도구 description 이 이미 담고 있다(중복이
#: 곧 드리프트 위험이다). 여기 남는 것은 어느 한 도구 설명도 소유할 수 없는
#: 볼트의 정보 구조(자동 아카이브 vs 노트 vs 핀 주입)와 가벼운 행동 원칙뿐이다.
MEMORY_PROMPT_BLOCK = """

# Agent Memory (persistent)
You have a persistent memory vault that survives across conversations, with
tools to read and write it. Retrieved context (Pinned Facts / Relevant
Knowledge) may already be injected above. Conversations are archived
automatically — write notes only for distilled, durable knowledge. When the
user asks you to remember something, persist it in memory rather than only
acknowledging it. Prefer updating existing notes over duplicating them, and
write notes in the user's language.
"""


#: CLI 백엔드에서 **도구 브릿지가 없을 때**(데스크톱 사이드카 등) 붙는 메모리 안내 —
#: 자동 계층(Pinned Facts/Relevant Knowledge 주입 + 턴 기록)만 있음을 알리고
#: 도구는 광고하지 않는다. 도구를 약속했는데 CLI 에 보이지 않으면 유령 호출이 된다.
MEMORY_AUTO_PROMPT_BLOCK = """

# Agent Memory (persistent, automatic)
You have a persistent memory vault that survives across conversations. It is
managed automatically on this backend: relevant context (Pinned Facts /
Relevant Knowledge) is injected above when available, and this conversation is
recorded at the end of the turn. There are NO memory tools available here —
do not attempt to call memory_* tools; simply answer using the injected context.
"""

#: host.build_turn_delegation() 반환 dict 에서 **실제 위임 백엔드**를 뜻하는 키 —
#: 하나라도 non-None 이면 위임이 배선된 것이다 (xgen-workflow delegation 모듈 계약:
#: subagent_manager=상주 매니저 프록시, task_runner/task_registry=백그라운드 태스크).
_DELEGATION_BACKEND_KEYS = ("subagent_manager", "task_runner", "task_registry")


def _delegation_wired(extras: Any) -> bool:
    """호스트가 돌려준 위임 extras 가 실제 백엔드를 담고 있는가 (빈 dict/None → False)."""
    if not isinstance(extras, dict) or not extras:
        return False
    return any(extras.get(k) is not None for k in _DELEGATION_BACKEND_KEYS)


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
