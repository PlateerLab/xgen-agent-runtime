"""거부된 호출이 쌓이면 턴을 끝낸다 — 건너뛰기만 하는 가드의 다음 단계 (4.45.0).

근거 (Harness-Bench, 4.29.0 가드 배포 뒤, 2026-09-20~21):
4.29.0 같은 호출·같은 결과 가드는 N번째 똑같은 호출부터 **실행하지 않고** "반복하지 마라" 를
돌려준다(N=8, 4.36.0 부터 5 — ``repeat_guard.SAME_RESULT_SKIP_AT``). 그런데 Qwen3.8-27B 는 이 안내를 무시하고 같은 호출을 수십 번 더 했고, 가드는 매번
건너뛰기만 하니 턴 입력 예산 hard(300만)가 올 때까지 끝나지 않았다. 건너뛴 호출은 실행되지 않아
도구 스팬이 남지 않는다 — ``usage.calls`` 가 도구 스팬 수보다 훨씬 크게 드러난다:

    033 지식 QA        LLM 127 / 도구 32 → 건너뜀 95, 입력 3,052,260 (예산 종료)
    087 CLI 버그        LLM 129 / 도구 35 → 건너뜀 94, 입력 3,062,643 (예산 종료)
    086 SQL 마이그레이션 LLM  88 / 도구 56 → 건너뜀 32, 입력 3,111,853 (예산 종료)

세 과제·세 도메인에서 같다. 같은 기간 실사용 턴에는 이 모양이 0건이었다(오탐 위험 낮음).

방식: 이 턴에 하네스가 **실행을 거부한 호출**(같은 호출·같은 결과 건너뛰기, 반복 실패 차단)이
``stop_after`` 번 쌓이면, 턴 예산 마무리(turn_budget.py)와 같은 안내 — 도구 없이 한 것·남은 것·
이어가는 법을 보고하라 — 를 붙이고 **그다음 모델 응답으로 턴을 끝낸다**. 거부가 3번이면 모델은
이미 같은 결과 경고와 거부 안내 3번을 무시한 뒤다(문턱이 5 로 낮아진 4.36.0 이후에도 건너뛰기만으로는
끝나지 않는다 — 문턱은 루프가 *시작되는* 지점만 앞당긴다). 도메인 규칙 없음 — 가드가 센 횟수만 본다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.stages.s16_loop.turn_budget import _append_note

__all__ = [
    "DEFAULT_STOP_AFTER",
    "REFUSED_KEY",
    "REPEAT_STOP_KEY",
    "RepeatStop",
    "note_refused",
    "repeat_stopped",
]

#: ``state.shared`` 키 — 이 턴에 실행을 거부한 호출 수(Stage 10 이 올린다). 턴 단위.
REFUSED_KEY = "tool.refused_calls"
#: ``state.shared`` 키 — 이 모듈의 진행 기록. 턴 단위.
REPEAT_STOP_KEY = "loop.repeat_stop"

DEFAULT_STOP_AFTER = 3

_FINAL_NOTE = (
    "[Stopped: {refused} of your tool calls were refused this turn because they repeated a call "
    "that already returned the same result (or kept failing the same way). Repeating them will not "
    "produce new information. Do not call any more tools. In this response, report what has been "
    "done, what remains, what is blocking you, and exactly how to continue. The turn ends after this "
    "response.]"
)


def note_refused(shared: Dict[str, Any], n: int) -> int:
    """Stage 10 용 — 이번 라운드에 거부한 호출 수를 더하고 누적을 돌려준다."""
    total = int(shared.get(REFUSED_KEY) or 0) + max(0, int(n))
    shared[REFUSED_KEY] = total
    return total


@dataclass
class RepeatStop:
    """LoopStage 가 컨트롤러 결정 뒤에 부른다. ``apply`` 는 (바뀔 수 있는) 결정을 돌려준다."""

    stop_after: int = DEFAULT_STOP_AFTER

    def __post_init__(self) -> None:
        self.stop_after = int(self.stop_after)
        if self.stop_after <= 0:
            raise ValueError("RepeatStop.stop_after must be > 0")

    def apply(self, state: PipelineState, decision: str) -> str:
        rec = state.shared.get(REPEAT_STOP_KEY)
        if not isinstance(rec, dict):
            rec = {}
            state.shared[REPEAT_STOP_KEY] = rec
        if rec.get("stopped"):
            return "complete"

        calls = len(state.turn_token_usage)
        refused = int(state.shared.get(REFUSED_KEY) or 0)
        final_calls = rec.get("final_calls")

        # 마무리 응답이 왔다 → 결정이 뭐든 끝낸다 (도구를 또 불렀어도 그 한 번은 이미 처리됐다).
        if isinstance(final_calls, int) and calls > final_calls:
            rec["stopped"] = True
            rec["refused"] = refused
            state.completion_signal = "REPEAT_STOP"
            state.completion_detail = f"repeated calls refused {refused} times"
            state.add_event(
                "loop.repeat_stop",
                {"phase": "stop", "refused": refused, "calls": calls, "iteration": state.iteration},
            )
            return "complete"

        if decision != "continue" or final_calls is not None:
            return decision
        if refused >= self.stop_after:
            if _append_note(state, _FINAL_NOTE.format(refused=refused)):
                rec["final_calls"] = calls
                rec["refused"] = refused
                state.add_event(
                    "loop.repeat_stop",
                    {
                        "phase": "final",
                        "refused": refused,
                        "calls": calls,
                        "iteration": state.iteration,
                    },
                )
        return decision


def repeat_stopped(state: PipelineState) -> Optional[Dict[str, Any]]:
    """호스트용 — 이 턴이 반복 거부로 끝났으면 기록을, 아니면 None."""
    rec = state.shared.get(REPEAT_STOP_KEY)
    if isinstance(rec, dict) and rec.get("stopped"):
        return rec
    return None
