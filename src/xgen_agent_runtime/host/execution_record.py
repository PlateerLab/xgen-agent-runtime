"""턴-종료 실행 기록 — Geny ``manager.record_execution`` 이중 기록 미러.

Geny 는 에이전트 실행 1회마다 호스트가 두 곳에 기록한다
(``Geny/backend/service/memory/manager.py``):

  1. **daily 결과 카드** — 실행당 노트 1장. 제목 ``Execution #N — <입력>``,
     성공/실패 태그, 실패는 importance HIGH. (Geny Opsidian 의 DAILY 2124
     가 이것 — "Per-execution result cards (one per agent run)".)
  2. **executions 일자 저널** — ``executions-<YYYY-MM-DD>.md`` 하루 1파일에
     실행 요약 한 줄 append. (Geny EXECUTIONS 12 = 12일치.)

XGEN 은 provider 에 ``record_execution`` 프로토콜 구현만 있고 호출자가
없어 두 카테고리가 통째로 비어 있었다(2026-07-13 검토에서 확정된 격차).
본 모듈이 그 호출자 — runner 의 턴 finally 가 provider 를 닫기 직전에
동기(제한시간 10s)로 부른다. LLM 불필요·증류 여부와 무관하게 매 턴 기록.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

logger = logging.getLogger("editor.geny_bridge.execution_record")

DAILY_CATEGORY = "daily"
EXECUTIONS_CATEGORY = "executions"
_TASK_CLIP = 200
_RESULT_CLIP = 1500


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _next_execution_number(provider: Any) -> int:
    """daily 실행 카드 수 + 1 — DB 가 원장이라 pod 재시작에도 이어진다."""
    try:
        # index().list_notes 는 limit=50 기본이라 포화된다 — notes.list 는
        # 메타 전량 반환 (양 백엔드 공통).
        metas = await provider.notes().list(category=DAILY_CATEGORY, tag="execution")
        return len(metas) + 1
    except Exception:  # noqa: BLE001 — 번호는 표시용, 실패해도 기록은 한다
        return 1


OUTCOME_MARK = {"ok": "✅", "partial": "⚠️", "failed": "❌"}


def classify_outcome(
    success: bool, *, tool_calls: int = 0, tool_failures: int = 0, blocked: int = 0
) -> str:
    """``ok`` / ``partial`` / ``failed``.

    ok      = 끝까지 돌았고 도구 실패·반복 차단이 없다
    partial = 끝까지 돌았지만 도구 실패나 반복 차단이 있었다 — "됐다" 고 적지 않는다
    failed  = 오류·취소·미완료
    """
    if not success:
        return "failed"
    if int(tool_failures or 0) > 0 or int(blocked or 0) > 0:
        return "partial"
    return "ok"


def _build_card_body(
    *,
    number: int,
    input_text: str,
    output_text: str,
    success: bool,
    duration_ms: int,
    session_id: str,
    provider_name: str,
    model: str,
    error: str,
    outcome: str = "",
    tool_calls: int = 0,
    tool_failures: int = 0,
    blocked: int = 0,
) -> str:
    """Geny ``_build_execution_entry`` 형식 미러 (구조 동일, 필드 간소)."""
    outcome = outcome or classify_outcome(
        success, tool_calls=tool_calls, tool_failures=tool_failures, blocked=blocked
    )
    mark = OUTCOME_MARK[outcome]
    secs = duration_ms / 1000.0
    lines = [
        f"### [{mark}] Execution #{number}",
        f"> **Task:** {_clip(input_text, _TASK_CLIP)}",
        f"> **Duration:** {secs:.1f}s · **Session:** {session_id or '-'}",
    ]
    if tool_calls:
        tools_line = f"> **Tools:** {int(tool_calls)} calls · {int(tool_failures)} failed"
        if blocked:
            tools_line += f" · {int(blocked)} blocked (repeated failure)"
        lines.append(tools_line)
    if outcome == "partial":
        lines.append(
            "> **Outcome:** partial — some tool calls failed; do not treat the task as done"
        )
    if provider_name or model:
        lines.append(f"> **Model:** {provider_name}/{model}".rstrip("/"))
    lines.append("")
    if output_text.strip():
        lines.append("**Result:**")
        lines.append(_clip(output_text, _RESULT_CLIP))
    if error:
        lines.append("")
        lines.append(f"**Error:** {_clip(error, 500)}")
    return "\n".join(lines).strip() + "\n"


async def record_turn_execution(
    provider: Any,
    *,
    input_text: str,
    output_text: str,
    success: bool,
    duration_ms: int,
    session_id: str,
    provider_name: str = "",
    model: str = "",
    error: str = "",
    cancelled: bool = False,
    tool_calls: int = 0,
    tool_failures: int = 0,
    blocked: int = 0,
) -> None:
    """실행 1회를 daily 카드 + executions 저널에 기록 (best-effort).

    ``success`` 는 "파이프라인이 끝까지 돌았다" 이지 "일이 됐다" 가 아니다. 도구가 8번 실패하고
    모델이 문장만 쓴 턴을 ✅ 로 남기면, 다음 턴 검색에 "만들었다" 가 올라와 에이전트가 없는 도구를
    있다고 믿는다(2026-09-21 관측). 그래서 도구 실패·반복 차단이 있으면 ⚠️(부분) 로 남기고 카드에
    도구 수를 적는다 — 제목은 요청문이지 완료 주장이 아니다.

    실패는 로그만 — 턴 결과에 절대 영향 없음. 벡터 자동 색인은 notes.write
    훅이 처리한다.
    """
    from xgen_agent_runtime.memory.provider import Importance, NoteDraft, NotePatch

    notes = provider.notes()
    number = await _next_execution_number(provider)
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")

    # ── ① daily 결과 카드 (실행당 1장) ────────────────────────────
    outcome = classify_outcome(
        success, tool_calls=tool_calls, tool_failures=tool_failures, blocked=blocked
    )
    tags = [
        "execution",
        {"ok": "success", "partial": "partial", "failed": "failure"}[outcome],
        "auto",
    ]
    importance = Importance.MEDIUM if outcome == "ok" else Importance.HIGH
    title = f"Execution #{number} — {_clip(input_text, 60)}"
    if cancelled:
        # Geny 의 silent 처리 미러 — 감사용으로 남기되 시각적으로 구분되고
        # 불활성(저중요도)이어야 한다.
        tags.append("cancelled")
        importance = Importance.LOW
        title += " · cancelled"
    body = _build_card_body(
        number=number,
        input_text=input_text,
        output_text=output_text,
        success=success,
        duration_ms=duration_ms,
        session_id=session_id,
        provider_name=provider_name,
        model=model,
        error=error,
        outcome=outcome,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        blocked=blocked,
    )
    try:
        await notes.write(
            NoteDraft(
                title=title,
                body=body,
                category=DAILY_CATEGORY,
                tags=tags,
                importance=importance,
                filename=f"exec-{number:04d}-{uuid.uuid4().hex[:8]}.md",
                frontmatter={
                    "session_id": session_id,
                    "execution_number": number,
                    "success": bool(success),
                    "outcome": outcome,
                    "tool_calls": int(tool_calls),
                    "tool_failures": int(tool_failures),
                    "duration_ms": int(duration_ms),
                },
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("execution card write failed (non-critical)", exc_info=True)

    # ── ② executions 일자 저널 (하루 1파일 append) ────────────────
    journal_filename = f"executions-{day}.md"
    mark = OUTCOME_MARK[outcome]
    line = (
        f"- {now.strftime('%H:%M')} {mark} #{number} "
        f"{_clip(input_text, 60)} ({duration_ms / 1000.0:.1f}s)"
    )
    try:
        existing = await notes.read(journal_filename)
        if existing is None:
            await notes.write(
                NoteDraft(
                    title=f"Executions {day}",
                    body=line + "\n",
                    category=EXECUTIONS_CATEGORY,
                    tags=["execution", "journal"],
                    importance=Importance.MEDIUM,
                    filename=journal_filename,
                    frontmatter={"day": day},
                )
            )
        else:
            await notes.update(journal_filename, NotePatch(append_body=line + "\n"))
    except Exception:  # noqa: BLE001
        logger.debug("execution journal append failed (non-critical)", exc_info=True)
