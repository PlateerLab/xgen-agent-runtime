"""반복 거부 종료 (stages/s16_loop/repeat_stop.py, 4.45.0).

근거: 4.29.0 같은 호출·같은 결과 가드가 N번째부터 실행을 건너뛰어도 Qwen 은 안내를 무시하고
같은 호출을 수십 번 더 했다 — 벤치 033·087·086 에서 건너뛴 호출 95·94·32회, 각 입력 약 300만
토큰으로 턴 예산 hard 에 걸려서야 끝났다. 거부가 쌓이면 보고를 받고 턴을 끝낸다.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.stages.s10_tool import repeat_guard
from xgen_agent_runtime.stages.s16_loop.repeat_stop import (
    REFUSED_KEY,
    REPEAT_STOP_KEY,
    RepeatStop,
    note_refused,
    repeat_stopped,
)
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry


def _state(calls: int = 2, refused: int = 0) -> PipelineState:
    state = PipelineState(session_id="s", model="m")
    state.messages = [
        {"role": "user", "content": "해줘"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "same"}]},
    ]
    for _ in range(calls):
        state.turn_token_usage.append(TokenUsage(input_tokens=10, output_tokens=1))
    if refused:
        note_refused(state.shared, refused)
    return state


# ── 단위 ──────────────────────────────────────────────────────────────


def test_note_refused_accumulates() -> None:
    shared: Dict[str, Any] = {}
    assert note_refused(shared, 1) == 1
    assert note_refused(shared, 2) == 3
    assert note_refused(shared, -5) == 3  # 음수는 무시


def test_below_threshold_nothing_happens() -> None:
    state = _state(refused=2)
    assert RepeatStop(stop_after=3).apply(state, "continue") == "continue"
    assert "Stopped:" not in state.messages[-1]["content"][0]["content"]


def test_final_note_then_the_next_response_ends_the_turn() -> None:
    stop = RepeatStop(stop_after=3)
    state = _state(calls=5, refused=3)
    assert stop.apply(state, "continue") == "continue"
    body = state.messages[-1]["content"][0]["content"]
    assert "[Stopped: 3 of your tool calls were refused" in body and "Do not call any more tools" in body
    assert state.events[-1]["data"]["phase"] == "final"

    # 모델이 그래도 도구를 불렀다 → 그 한 번은 처리됐고, 여기서 끝낸다
    state.turn_token_usage.append(TokenUsage(input_tokens=10))
    assert stop.apply(state, "continue") == "complete"
    assert state.completion_signal == "REPEAT_STOP"
    assert repeat_stopped(state) and repeat_stopped(state)["refused"] == 3
    assert stop.apply(state, "continue") == "complete"  # 멱등


def test_natural_completion_is_left_alone_and_recorded_after_the_note() -> None:
    stop = RepeatStop(stop_after=1)
    state = _state(calls=2, refused=1)
    stop.apply(state, "continue")
    state.turn_token_usage.append(TokenUsage(input_tokens=10))
    assert stop.apply(state, "complete") == "complete"
    assert repeat_stopped(state) is not None  # 안내가 붙도록 기록은 남긴다


def test_no_refusals_never_touches_a_normal_turn() -> None:
    stop = RepeatStop()
    state = _state(calls=40)
    for _ in range(5):
        assert stop.apply(state, "continue") == "continue"
    assert repeat_stopped(state) is None


def test_begin_turn_resets() -> None:
    state = _state(refused=5)
    state.shared[REPEAT_STOP_KEY] = {"stopped": True}
    state.begin_turn()
    assert REFUSED_KEY not in state.shared and REPEAT_STOP_KEY not in state.shared


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RepeatStop(stop_after=0)


# ── 파이프라인 끝까지: 087 모양 재현 ──────────────────────────────────


class _Read(Tool):
    def __init__(self) -> None:
        self.executed = 0

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return "read"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.executed += 1
        return ToolResult(content="26\tparser = argparse.ArgumentParser()")  # 매번 같은 결과


class _Stubborn(BaseClient):
    """같은 Read 를 끝없이 부른다 — 건너뛰기 안내를 무시한다. 'Stopped:' 를 보면 보고한다(obeys)."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, *, obeys: bool, **kw: Any) -> None:
        super().__init__(**kw)
        self.obeys = obeys
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        usage = TokenUsage(input_tokens=100, output_tokens=5)
        if self.obeys and "[Stopped:" in str(request.messages[-1].get("content")):
            return APIResponse(content=[ContentBlock(type="text", text="보고: cli.py 수정 중 막힘. 남은 일: --sort 인자.")],
                               stop_reason="end_turn", usage=usage, model="fake")
        return APIResponse(
            content=[ContentBlock(type="tool_use", tool_use_id=f"t{len(self.requests)}", tool_name="Read",
                                  tool_input={"file_path": "cli.py", "offset": 25, "limit": 10})],
            stop_reason="tool_use", usage=usage, model="fake",
        )


def _run(obeys: bool, **kw: Any):
    reg = ToolRegistry()
    tool = _Read()
    reg.register(tool, core=True)
    client = _Stubborn(obeys=obeys, api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, registry=reg, max_iterations=200, turn_input_budget_tokens=None, **kw,
    )
    text = runner.run_turn(pipe, "cli.py 고쳐", PipelineState(session_id="s", model="m"))
    return text, client, tool


def test_pipeline_ends_the_087_loop_with_a_report() -> None:
    text, client, tool = _run(obeys=True)
    skip_at = repeat_guard.SAME_RESULT_SKIP_AT
    # (SKIP_AT-1)번 실행(같은 결과) → 다음 3번 거부 → 안내 → 보고 응답으로 끝
    assert tool.executed == skip_at - 1
    assert len(client.requests) == skip_at - 1 + 3 + 1, len(client.requests)
    assert "보고: cli.py 수정 중 막힘" in text and "[안내: 같은 작업이 반복되어" in text


def test_pipeline_ends_even_when_the_model_ignores_the_stop_note() -> None:
    text, client, tool = _run(obeys=False)
    # 안내 뒤 응답이 또 도구 호출이어도 그 응답으로 끝난다 — 200회까지 가지 않는다
    assert len(client.requests) <= repeat_guard.SAME_RESULT_SKIP_AT - 1 + 3 + 1
    assert "[안내: 같은 작업이 반복되어" in text


def test_without_repeat_stop_the_loop_runs_to_the_iteration_cap() -> None:
    """고치기 전 동작 재현 — 건너뛰기만으로는 끝나지 않는다(벤치에선 300만 토큰 예산이 끊었다)."""
    _, client, tool = _run(obeys=False, repeat_stop_after=None)
    assert len(client.requests) >= 50
    assert tool.executed == repeat_guard.SAME_RESULT_SKIP_AT - 1  # 나머지는 전부 건너뛰기
