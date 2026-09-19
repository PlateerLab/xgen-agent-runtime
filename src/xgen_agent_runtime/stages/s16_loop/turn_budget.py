"""턴 입력 토큰 예산 — 한 턴이 모델에 먹인 누적 입력이 상한을 넘으면 마무리시키고 끝낸다.

근거 (dev 실제 사용 28일, 2026-09-19, 벤치 제외 7,159턴 · 입력 2.36억 토큰):
- 턴당 입력 p50 1만 · p95 8.7만 · p99 24.8만. 그런데 **50만을 넘는 턴 33개(0.46%)가 전체 입력의
  30%**, 100만을 넘는 11개(0.15%)가 24% 다. 최대는 3,712만(도구 442회, 옛 20슬라이스 시절).
- 50만 초과 턴의 상태: running(끝내 답 없이 멈춤) 22 · success 10 · cancelled 1. 3분의 2가
  답도 못 받고 버려진 턴이다. 반복 차단(4.26/4.29)은 "같은 것을 되풀이" 하는 낭비를 잡지만,
  매번 다른 일을 하며 끝없이 가는 턴은 슬라이스 상한(10×20회)까지 간다 — 이것이 그 두 번째 안전망.
- 벤치에서도 086 이 82회·330만 토큰까지 가서 외부 20분 제한에 걸렸다(산출물 없음).

방식: 누적 입력(캐시 읽기·생성 포함 — 모델이 실제로 처리한 프롬프트 길이)이
- ``soft`` 를 넘으면 마지막 도구 결과에 한 번 "마무리하라" 를 붙인다(실행은 계속).
- ``hard`` 를 넘으면 "도구를 더 쓰지 말고 지금까지 한 것·남은 것·이어가는 법을 보고하라" 를
  붙이고, **그다음 모델 응답으로 턴을 끝낸다**(도구를 또 부르면 그 한 번만 실행하고 끝). 정상
  완료(complete)로 끝나므로 자동 이어가기는 타지 않고, 사용자는 새 메시지로 이어갈 수 있다.
도메인 규칙 없음 — 토큰 수만 본다. 기본 50만/100만은 위 분포에서 잡았다 (p99 의 2배 / 4배).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from xgen_agent_runtime.core.state import PipelineState

logger = logging.getLogger(__name__)

__all__ = [
    "BUDGET_KEY",
    "DEFAULT_HARD_TOKENS",
    "DEFAULT_SOFT_TOKENS",
    "TurnInputBudget",
    "budget_stopped",
    "turn_input_tokens",
]

#: ``state.shared`` 키 — 턴 단위(연속 슬라이스에 이어지고 새 턴에서 비운다).
BUDGET_KEY = "loop.turn_budget"

DEFAULT_SOFT_TOKENS = 500_000
DEFAULT_HARD_TOKENS = 1_000_000

_SOFT_NOTE = (
    "[Turn budget: {used:,} of {hard:,} input tokens used. Wrap up now — finish the single "
    "most valuable remaining step, then report what is done and what is left. Do not start "
    "new exploration.]"
)
_FINAL_NOTE = (
    "[Turn budget exhausted: {used:,} input tokens (limit {hard:,}). Do not call any more tools. "
    "In this response, report what has been done, what remains, and exactly how to continue "
    "(files, commands, next step). The turn ends after this response.]"
)


def turn_input_tokens(state: PipelineState) -> int:
    """이 턴에서 모델이 처리한 프롬프트 토큰 합 — 캐시 읽기·생성분 포함 (연속 슬라이스 누적)."""
    total = 0
    for u in state.turn_token_usage:
        total += (
            int(getattr(u, "input_tokens", 0) or 0)
            + int(getattr(u, "cache_creation_input_tokens", 0) or 0)
            + int(getattr(u, "cache_read_input_tokens", 0) or 0)
        )
    return total


def _append_note(state: PipelineState, note: str) -> bool:
    """마지막 도구 결과에 안내를 붙인다 — 반복 차단이 쓰는 것과 같은 자리라 어느 클라이언트에서든
    다음 요청에 실린다. 도구 결과가 없고 마지막이 assistant 면 user 메시지로 붙인다. 못 붙이면 False."""
    if not state.messages:
        return False
    last = state.messages[-1]
    if not isinstance(last, dict):
        return False
    content = last.get("content")
    if last.get("role") == "user" and isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str) or inner is None:
                    block["content"] = f"{inner or ''}\n\n{note}"
                elif isinstance(inner, list):
                    inner.append({"type": "text", "text": note})
                else:
                    block["content"] = f"{inner}\n\n{note}"
                return True
        return False
    if last.get("role") == "assistant":
        state.add_message("user", note)
        return True
    return False


@dataclass
class TurnInputBudget:
    """LoopStage 가 컨트롤러 결정 뒤에 부른다. ``apply`` 는 (바뀔 수 있는) 결정을 돌려준다."""

    soft_tokens: int = DEFAULT_SOFT_TOKENS
    hard_tokens: int = DEFAULT_HARD_TOKENS

    def __post_init__(self) -> None:
        self.soft_tokens = int(self.soft_tokens)
        self.hard_tokens = int(self.hard_tokens)
        if self.hard_tokens <= 0:
            raise ValueError("TurnInputBudget.hard_tokens must be > 0")
        if self.soft_tokens <= 0 or self.soft_tokens >= self.hard_tokens:
            raise ValueError("TurnInputBudget.soft_tokens must be in (0, hard_tokens)")

    def _record(self, state: PipelineState) -> Dict[str, Any]:
        rec = state.shared.get(BUDGET_KEY)
        if not isinstance(rec, dict):
            rec = {}
            state.shared[BUDGET_KEY] = rec
        return rec

    def apply(self, state: PipelineState, decision: str) -> str:
        used = turn_input_tokens(state)
        rec = self._record(state)
        calls = len(state.turn_token_usage)

        if rec.get("stopped"):
            return "complete"  # 이미 끝낸 턴 — 어떤 결정도 되살리지 않는다

        # 마무리 응답이 왔다 → 결정이 뭐든 끝낸다 (도구를 또 불렀어도 그 한 번은 이미 실행됐다).
        final_calls = rec.get("final_calls")
        if isinstance(final_calls, int) and calls > final_calls:
            rec["stopped"] = True
            rec["used"] = used
            state.completion_signal = "TURN_INPUT_BUDGET"
            state.completion_detail = (
                f"turn input budget: {used:,} tokens (limit {self.hard_tokens:,})"
            )
            state.add_event(
                "loop.turn_budget",
                {
                    "phase": "stop",
                    "used": used,
                    "soft": self.soft_tokens,
                    "hard": self.hard_tokens,
                    "calls": calls,
                    "iteration": state.iteration,
                },
            )
            return "complete"

        if decision != "continue":
            return decision  # 어차피 끝나는 턴 — 안내를 붙일 곳도, 이유도 없다

        if used >= self.hard_tokens and final_calls is None:
            if _append_note(state, _FINAL_NOTE.format(used=used, hard=self.hard_tokens)):
                rec["final_calls"] = calls
                rec["used"] = used
                state.add_event(
                    "loop.turn_budget",
                    {
                        "phase": "final",
                        "used": used,
                        "soft": self.soft_tokens,
                        "hard": self.hard_tokens,
                        "calls": calls,
                        "iteration": state.iteration,
                    },
                )
            return decision

        if used >= self.soft_tokens and rec.get("soft_calls") is None and final_calls is None:
            if _append_note(state, _SOFT_NOTE.format(used=used, hard=self.hard_tokens)):
                rec["soft_calls"] = calls
                state.add_event(
                    "loop.turn_budget",
                    {
                        "phase": "soft",
                        "used": used,
                        "soft": self.soft_tokens,
                        "hard": self.hard_tokens,
                        "calls": calls,
                        "iteration": state.iteration,
                    },
                )
        return decision


def budget_stopped(state: PipelineState) -> Optional[Dict[str, Any]]:
    """호스트용 — 이 턴이 예산으로 끝났으면 기록을, 아니면 None."""
    rec = state.shared.get(BUDGET_KEY)
    if isinstance(rec, dict) and rec.get("stopped"):
        return rec
    return None
