"""도구 사건은 **어느 호출인지**를 싣는다 — ``tool_use_id`` 와 ``run_id`` (같은 값).

회귀 배경: 러너가 파이프라인 사건의 ``tool_use_id`` 를 버리고 이름·입력만
넘겼다. 호스트(xgen-workflow)는 ``run_id`` 로 도구 사건을 메시지에 모으는데
그 값이 늘 없어서 채팅의 도구 칩·[전체 로그]·제작 도구 카드가 전부 비었다.
이름만으로는 같은 도구를 동시에 두 번 부른 호출의 시작과 끝을 맞출 수 없다.

CLI 경로(``api.cli_tool_call`` / ``api.tool_result``)는 걸린 시간도 없었다 —
시작 시각을 id 별로 잡아 ``duration_ms`` 를 채운다.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

from xgen_agent_runtime import Pipeline, PipelineConfig, PipelineState
from xgen_agent_runtime.host.runner import _tool_call_event, _tool_end_event, stream_turn


def _ev(type_: str, **data: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, data=data)


def _sleep(seconds: float) -> SimpleNamespace:
    """스크립트 안의 대기 표시 — 사건이 아니다."""
    return SimpleNamespace(type="__sleep__", data={"seconds": seconds})


def _drive(script: List[SimpleNamespace]) -> List[Dict[str, Any]]:
    """스크립트대로 사건을 흘리는 파이프라인으로 stream_turn 을 돌려 도구 사건만 모은다."""
    pipeline = Pipeline(PipelineConfig())

    async def _run_stream(_input: Any, _state: PipelineState):
        for item in script:
            if item.type == "__sleep__":
                await asyncio.sleep(item.data["seconds"])
                continue
            yield item
        yield _ev("pipeline.complete", status="completed", result="")

    pipeline.run_stream = _run_stream  # type: ignore[method-assign]
    out = list(stream_turn(pipeline, "hi", PipelineState(session_id="ids")))
    return [
        c["data"]
        for c in out
        if isinstance(c, dict)
        and c.get("type") == "agent_event"
        and c["data"].get("type") in {"tool_call", "tool_result", "tool_error"}
    ]


# ── 사건 모양 ─────────────────────────────────────────────────────────


def test_세_종류_모두_id_와_run_id_를_같은_값으로_싣는다():
    call = _tool_call_event("Bash", {"command": "ls"}, tool_use_id="toolu_1")
    ok = _tool_end_event("Bash", "a.txt", tool_use_id="toolu_1")
    err = _tool_end_event("Bash", "boom", is_error=True, tool_use_id="toolu_1")
    assert [e["type"] for e in (call, ok, err)] == ["tool_call", "tool_result", "tool_error"]
    for event in (call, ok, err):
        assert event["tool_use_id"] == "toolu_1"
        assert event["run_id"] == "toolu_1"


def test_id_가_없으면_키를_싣지_않는다():
    """빈 문자열을 실으면 소비자가 있음으로 읽고 이름 폴백을 건너뛴다."""
    for missing in (None, ""):
        for event in (
            _tool_call_event("Bash", {}, tool_use_id=missing),
            _tool_end_event("Bash", "x", tool_use_id=missing),
            _tool_end_event("Bash", "x", is_error=True, tool_use_id=missing),
        ):
            assert "tool_use_id" not in event
            assert "run_id" not in event
    # 옛 호출 모양(인자 없음)도 그대로 동작한다.
    assert "run_id" not in _tool_call_event("Bash", {})
    assert "run_id" not in _tool_end_event("Bash", "x")


def test_문자열이_아닌_id_는_문자열로_싣는다():
    event = _tool_call_event("Bash", {}, tool_use_id=42)
    assert event["tool_use_id"] == "42" and event["run_id"] == "42"


# ── 서버 러너 경로 (tool.call_start / tool.call_complete) ─────────────


def test_러너_경로는_파이프라인_사건의_id_를_옮긴다():
    events = _drive(
        [
            _ev("tool.call_start", tool_use_id="toolu_a", name="Bash", input={"command": "ls"}),
            _ev("tool.call_start", tool_use_id="toolu_b", name="Bash", input={"command": "pwd"}),
            _ev(
                "tool.call_complete",
                tool_use_id="toolu_b",
                name="Bash",
                is_error=False,
                duration_ms=7,
                result="/work",
            ),
            _ev(
                "tool.call_complete",
                tool_use_id="toolu_a",
                name="Bash",
                is_error=True,
                duration_ms=12,
                error="permission denied",
            ),
        ]
    )
    assert [(e["type"], e.get("run_id")) for e in events] == [
        ("tool_call", "toolu_a"),
        ("tool_call", "toolu_b"),
        ("tool_result", "toolu_b"),
        ("tool_error", "toolu_a"),
    ]
    for event in events:
        assert event["tool_use_id"] == event["run_id"]
    assert events[2]["duration_ms"] == 7 and events[2]["result"] == "/work"
    assert events[3]["duration_ms"] == 12 and events[3]["error"] == "permission denied"


def test_러너_경로에서_id_가_비면_키가_없다():
    events = _drive(
        [
            _ev("tool.call_start", tool_use_id="", name="Bash", input={}),
            _ev("tool.call_complete", name="Bash", is_error=False, duration_ms=1, result="ok"),
        ]
    )
    assert len(events) == 2
    assert all("tool_use_id" not in e and "run_id" not in e for e in events)


# ── CLI 경로 (api.cli_tool_call / api.tool_result source=cli) ─────────


def test_cli_경로는_id_와_걸린_시간을_싣는다():
    events = _drive(
        [
            _ev("api.cli_tool_call", id="t1", name="mcp__xgen__Bash", input={"command": "ls"}),
            _sleep(0.05),
            _ev(
                "api.tool_result",
                tool_use_id="t1",
                content=[{"type": "text", "text": "a.txt"}],
                is_error=False,
                source="cli",
            ),
        ]
    )
    call, result = events
    assert call["type"] == "tool_call" and call["tool_use_id"] == "t1" and call["run_id"] == "t1"
    assert "duration_ms" not in call
    assert result["type"] == "tool_result"
    assert result["tool_name"] == "mcp__xgen__Bash"
    assert result["tool_use_id"] == "t1" and result["run_id"] == "t1"
    assert isinstance(result["duration_ms"], int) and result["duration_ms"] >= 40


def test_cli_동시_호출은_id_별로_짝과_시간을_맞춘다():
    events = _drive(
        [
            _ev("api.cli_tool_call", id="slow", name="Bash", input={}),
            _sleep(0.15),
            _ev("api.cli_tool_call", id="fast", name="Bash", input={}),
            _ev("api.tool_result", tool_use_id="fast", content="done", is_error=False, source="cli"),
            _ev(
                "api.tool_result",
                tool_use_id="slow",
                content="not found",
                is_error=True,
                source="cli",
            ),
        ]
    )
    by_type = [(e["type"], e.get("run_id")) for e in events]
    assert by_type == [
        ("tool_call", "slow"),
        ("tool_call", "fast"),
        ("tool_result", "fast"),
        ("tool_error", "slow"),
    ]
    fast, slow = events[2], events[3]
    assert slow["error"] == "not found"
    assert slow["duration_ms"] >= 140
    assert fast["duration_ms"] < slow["duration_ms"]


def test_cli_id_가_없으면_키도_시간도_없다():
    events = _drive(
        [
            _ev("api.cli_tool_call", name="Bash", input={}),
            _ev("api.tool_result", content="ok", is_error=False, source="cli"),
        ]
    )
    assert len(events) == 2
    for event in events:
        assert "tool_use_id" not in event and "run_id" not in event
    assert "duration_ms" not in events[1]


def test_cli_시작을_못_본_결과는_id_만_싣고_턴은_계속된다():
    """시작 사건을 놓쳐도 결과를 버리거나 턴을 깨지 않는다."""
    events = _drive(
        [
            _ev("api.tool_result", tool_use_id="orphan", content="ok", is_error=False, source="cli"),
        ]
    )
    (result,) = events
    assert result["tool_name"] == "cli_tool"
    assert result["tool_use_id"] == "orphan" and result["run_id"] == "orphan"
    assert "duration_ms" not in result


def test_api_출처의_tool_result_는_도구_사건이_아니다():
    """API 경로의 tool_result 는 모델 요청일 뿐 — 실행 사건은 tool.call_* 가 낸다."""
    events = _drive(
        [_ev("api.tool_result", tool_use_id="x", content="ok", is_error=False, source="api")]
    )
    assert events == []
