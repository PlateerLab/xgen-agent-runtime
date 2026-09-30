"""AgentTurnExecutor 가 항상 쓰는 상수/순수 헬퍼 — agent_geny 에서 이전.

서버·커넥터 공용. ``_self_evolution_policy`` 는 관리자 설정 판정을 host.setting
으로 주입받는다(서버=config DB→env, 커넥터=env) — SDK/CLI/웹/커넥터가 **같은
판정**을 써야 하는 보안 계약(deploy/guest 차단)이 갈라지지 않게. 관련:
``geny-shared-host-extraction``.
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


#: 실행 효율 원칙 — SDK 도구 루프가 있는 모든 턴에 붙는다.
#:
#: 도구 결과 하나를 받을 때마다 모델은 시스템 프롬프트·도구 정의·지금까지의 대화
#: 전체를 다시 읽는다. 비용 ≈ (고정 앞부분 크기) × (모델 왕복 수) 라서 왕복 수가
#: 곧 비용이다. 실측 (2026-09-17 dev, claude-sonnet-4-6, 위배상품 5개 점검):
#: 같은 검색을 품목마다 따로 불러 모델 왕복 26회·입력 30만 토큰이었고, 결국 한
#: 스크립트로 다섯 개를 한 번에 검색하자 한 왕복에 끝났다.
EFFICIENCY_PROMPT_BLOCK = """

# Working efficiently
Every tool call is a full model round trip that re-reads this whole conversation,
so the number of round trips is the cost. Finish the task in as few as possible:
- When the same tool is needed for several items, use ToolBatch (one call, all inputs)
  instead of calling the tool once per item.
- To deliver tabular results as a spreadsheet, use TableExport when it is available
  instead of writing a script — it also gives the user a download button.
- When several different independent lookups or actions are needed, request them
  together in ONE response (parallel tool calls) instead of one per turn.
- For repetitive work over a list of items, write ONE script (e.g. Bash/Python) that
  processes all items and prints a compact summary, instead of calling a tool per item.
- Keep tool output small: print only what you need (counts, matched rows, file paths),
  not whole documents or raw pages.
- Do not re-read files, re-list directories, or re-run commands whose results you
  already have. Use paths and facts already given in the conversation.
- If a tool rejects your input, fix the argument exactly as the error says before
  retrying; never repeat the same failing call."""


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
write notes in the user's language. Your earlier turns in this conversation are
in the message history: treat them as history, not as facts to re-verify, and
act on the current request.
"""


#: 쓰기 도구(memory_write/memory_pin)가 이 턴에 **없을 때**의 메모리 지침 — 호스트 정책(게스트·
#: 동결본)이 쓰기 도구를 뺀 뒤에 고른다. 위 블록은 "기억하라면 저장하라" 고 약속하는데 도구가
#: 없으면 모델이 파일 쓰기로 그 약속을 메운다(2026-09-21 관측). 문구와 표면은 같은 판정에서 나온다.
MEMORY_READONLY_PROMPT_BLOCK = """

# Agent Memory (read-only here)
You have a persistent memory vault with tools to read and search it; retrieved context (Pinned
Facts / Relevant Knowledge) may already be injected above. In this conversation the vault cannot be
changed: there are no memory write tools. If the user asks you to remember something, say that this
conversation cannot store new memories and keep it in mind for the rest of the conversation. Do not
write files or notes to imitate memory. Your earlier turns in this conversation are in the message
history: treat them as history, not as facts to re-verify, and act on the current request.
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

# ── CLI 표면 각주 ────────────────────────────────────────────────────
#
# CLI 백엔드(claude_code·codex)에서도 도구는 **같은 레지스트리의 같은 도구**다(host.tool_surface).
# 다른 것은 이름 하나다 — CLI 는 우리 도구를 MCP 서버 하나로 받으므로 이름 앞에 접두가 붙는다
# (``Bash`` → ``mcp__connector__Bash``). 그 사실은 어느 도구 설명도 소유할 수 없어서 프롬프트가
# 한 번 말한다. 도구별 각주(메모리·자기진화·위임을 따로)는 두지 않는다 — 각주가 도구마다 있으면
# 빠진 도구가 생기고, 빠진 도구는 모델에게 "없는 것"이 된다.


def cli_tool_naming_note(server: str, provider: str) -> str:
    """CLI 백엔드에서 우리 도구가 어떤 이름으로 보이는지 — 모든 도구에 한 번에 적용된다."""
    if provider == "codex":
        return (
            "\n\n# Tool names on this backend\n"
            f"Every tool named in these instructions is served by the MCP server '{server}' — call it"
            " through that server. Those are your only tools."
        )
    return (
        "\n\n# Tool names on this backend\n"
        f"Every tool named in these instructions is served by the MCP server '{server}': a tool named X"
        f" appears as mcp__{server}__X. Those are your only tools."
    )


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
    if kwargs.get("_frozen"):
        # 고정본의 그래프는 불변이다. 도구를 세워 두면 모델은 편집을 시도하고, 커밋 단계에서
        # 거절당한 뒤에야 안다 — 그 사이 "능력을 넓혀 드리겠다" 는 약속을 한다(2026-09-30 실측).
        return False, "고정본(동결) 실행"
    if not bool(kwargs.get("enable_self_evolution", True)):
        return False, "노드 파라미터 enable_self_evolution=off"
    raw = (get_setting("GENY_TOOLS_WORKFLOW_SELF_ENABLED") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False, "관리자 설정 GENY_TOOLS_WORKFLOW_SELF_ENABLED=off"
    if not str(kwargs.get("workflow_id") or ""):
        return False, "workflow_id 없음 (비정형 실행)"
    return True, ""
