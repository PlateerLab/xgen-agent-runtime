"""도구가 낸 **결과**가 [전체로그]까지 온다.

프로드 실증 (2026-09-10)
------------------------
사용자가 Agent-XGeny 의 [전체로그]를 열면 도구 줄은 있는데 펼쳐도 "결과" 칸이
아예 없었다 — Bash 도, Write 도, 에이전트가 만든 도구도. 딱 하나, 워크플로우가
LangChain 도구로 감싸 넘긴 것만 결과가 보였다.

원인은 두 줄이 맞물린 것이다:

  1. ``tool.call_complete`` 는 **실패 사유만** 실었다. "성공 결과는 크고 모델이
     이미 받는다" 는 판단이었는데, 모델에게는 맞지만 **로그에게는 아니다.**
  2. 그래서 호스트는 ``result_sink`` 로 떨어졌다. 그 sink 는 호스트가 감싼
     LangChain 도구만 채운다 — 런타임 자체 도구·제작 도구·MCP 로 노출한 도구는
     아무도 채우지 않으므로 전부 빈 문자열이 됐다.

빈 문자열은 오류를 내지 않는다. 그래서 아무도 몰랐다.
"""
from __future__ import annotations

from typing import Any, Dict, List

from xgen_agent_runtime.stages.s10_tool.artifact.default.executors import (
    _emit_call_complete,
)


def _capture() -> tuple[List[tuple], Any]:
    seen: List[tuple] = []

    def on_event(name: str, data: Dict[str, Any]) -> None:
        seen.append((name, dict(data)))

    return seen, on_event


def test_a_successful_tool_carries_its_result():
    seen, on_event = _capture()
    _emit_call_complete(
        on_event,
        {"tool_use_id": "t1", "tool_name": "Bash"},
        {"content": '{"datas":[{"closePrice":"71,000"}]}'},
        147,
    )
    name, data = seen[0]
    assert name == "tool.call_complete"
    assert data["result"] == '{"datas":[{"closePrice":"71,000"}]}', (
        "결과가 사건에 실리지 않으면 호스트는 sink 로 떨어지고, "
        "sink 를 채우지 않는 도구는 [전체로그]에서 결과가 빈 칸이 된다"
    )
    assert "error" not in data


def test_a_failed_tool_still_carries_its_reason():
    seen, on_event = _capture()
    _emit_call_complete(
        on_event,
        {"tool_use_id": "t2", "tool_name": "ForgeTool"},
        {"is_error": True, "content": "등록 전 테스트에 실패했습니다"},
        171,
    )
    _, data = seen[0]
    assert data["is_error"] is True
    assert "등록 전 테스트" in data["error"]
    # 실패에 결과 칸을 겹쳐 싣지 않는다 — 같은 말이 두 칸에 나오면 어느 쪽이
    # 사실인지 읽는 사람이 판단해야 한다.
    assert "result" not in data


def test_a_big_result_is_cut_at_the_source():
    """이벤트는 여러 소비자를 지난다 — 잘라 보내는 편이 낫다."""
    seen, on_event = _capture()
    _emit_call_complete(
        on_event, {"tool_use_id": "t3", "tool_name": "Read"}, {"content": "x" * 50_000}, 10,
    )
    _, data = seen[0]
    assert 0 < len(data["result"]) <= 8000


def test_an_empty_result_stays_empty_not_a_lie():
    seen, on_event = _capture()
    _emit_call_complete(on_event, {"tool_use_id": "t4", "tool_name": "Write"}, {}, 16)
    _, data = seen[0]
    assert data["result"] == ""


class TestTheHostPrefersTheEventOverTheSink:
    """호스트는 **사건이 싣고 온 것**을 먼저 쓴다.

    sink 만 믿으면 sink 를 채우지 않는 도구가 전부 빈 칸이 된다 — 그게 이 버그였다.
    """

    @staticmethod
    def _pick(event_data: Dict[str, Any], sink: Dict[str, str]) -> str:
        # host/runner.py 의 선택 규칙과 같은 식.
        return (
            str(event_data.get("error") or "")
            or str(event_data.get("result") or "")
            or (sink or {}).get(event_data.get("name", ""), "")
        )

    def test_the_events_result_wins(self):
        assert self._pick({"name": "Bash", "result": "from-event"}, {"Bash": "from-sink"}) == "from-event"

    def test_the_sink_is_still_a_fallback(self):
        """구버전 런타임이 섞여 돌 수 있다 — 사건에 결과가 없으면 sink 를 본다."""
        assert self._pick({"name": "Bash"}, {"Bash": "from-sink"}) == "from-sink"

    def test_a_failure_reason_beats_both(self):
        assert self._pick(
            {"name": "Bash", "error": "boom", "result": "partial"}, {"Bash": "s"},
        ) == "boom"

    def test_no_result_anywhere_is_empty_not_a_crash(self):
        assert self._pick({"name": "Bash"}, {}) == ""
