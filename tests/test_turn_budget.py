"""턴 입력 토큰 예산 (stages/s16_loop/turn_budget.py).

근거: dev 28일 7,159턴 — 입력 100만 토큰을 넘는 턴 11개(0.15%)가 전체 입력의 24%, 그중
3분의 2는 답도 없이 멈춘(running) 턴. soft 에서 "마무리하라", hard 에서 "도구 없이
보고하라" 를 붙이고 그다음 응답으로 턴을 끝낸다.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.stages.s16_loop.turn_budget import (
    BUDGET_KEY,
    TurnInputBudget,
    budget_stopped,
    turn_input_tokens,
)
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry


def _state(*prompt_tokens: int) -> PipelineState:
    state = PipelineState(session_id="s", model="m")
    state.messages = [{"role": "user", "content": "해줘"}]
    for n in prompt_tokens:
        state.turn_token_usage.append(TokenUsage(input_tokens=n, output_tokens=10))
    return state


def _with_tool_result(state: PipelineState, text: str = "ok") -> PipelineState:
    state.messages += [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": text}]},
    ]
    return state


def test_turn_input_tokens_counts_cache_reads_and_creation() -> None:
    state = _state()
    state.turn_token_usage.append(
        TokenUsage(input_tokens=100, cache_read_input_tokens=900, cache_creation_input_tokens=50)
    )
    assert turn_input_tokens(state) == 1050


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        TurnInputBudget(soft_tokens=10, hard_tokens=10)
    with pytest.raises(ValueError):
        TurnInputBudget(soft_tokens=0, hard_tokens=10)


def test_below_soft_nothing_happens() -> None:
    b = TurnInputBudget(soft_tokens=100, hard_tokens=200)
    state = _with_tool_result(_state(40, 50))
    assert b.apply(state, "continue") == "continue"
    assert BUDGET_KEY not in state.shared or not state.shared[BUDGET_KEY]
    assert "Turn budget" not in state.messages[-1]["content"][0]["content"]


def test_soft_note_is_appended_to_the_last_tool_result_once() -> None:
    b = TurnInputBudget(soft_tokens=100, hard_tokens=200)
    state = _with_tool_result(_state(60, 50))
    assert b.apply(state, "continue") == "continue"
    body = state.messages[-1]["content"][0]["content"]
    assert body.startswith("ok\n\n[Turn budget: 110 of 200 input tokens used. Wrap up now")
    assert state.shared[BUDGET_KEY]["soft_calls"] == 2
    assert state.events[-1]["type"] == "loop.turn_budget" and state.events[-1]["data"]["phase"] == "soft"

    # 다음 반복 — 또 붙이지 않는다
    state.turn_token_usage.append(TokenUsage(input_tokens=20))
    _with_tool_result(state, "again")
    assert b.apply(state, "continue") == "continue"
    assert state.messages[-1]["content"][0]["content"] == "again"


def test_hard_note_then_the_next_response_ends_the_turn_even_if_it_calls_tools() -> None:
    b = TurnInputBudget(soft_tokens=100, hard_tokens=200)
    state = _with_tool_result(_state(150, 80))  # 230 ≥ hard, soft 를 건너뛰고 바로 final
    assert b.apply(state, "continue") == "continue"
    assert "Turn budget exhausted: 230 input tokens (limit 200). Do not call any more tools" in (
        state.messages[-1]["content"][0]["content"]
    )
    assert state.shared[BUDGET_KEY]["final_calls"] == 2

    # 모델이 그래도 도구를 불렀다 → 그 한 번은 실행됐고, 여기서 턴을 끝낸다
    state.turn_token_usage.append(TokenUsage(input_tokens=30))
    _with_tool_result(state, "last")
    assert b.apply(state, "continue") == "complete"
    assert state.completion_signal == "TURN_INPUT_BUDGET"
    rec = budget_stopped(state)
    assert rec and rec["used"] == 260 and rec["stopped"] is True
    assert state.events[-1]["data"]["phase"] == "stop"
    assert b.apply(state, "continue") == "complete"  # 멱등


def test_natural_completion_after_the_final_note_is_left_alone() -> None:
    b = TurnInputBudget(soft_tokens=100, hard_tokens=200)
    state = _with_tool_result(_state(150, 80))
    b.apply(state, "continue")
    state.turn_token_usage.append(TokenUsage(input_tokens=30))
    state.messages.append({"role": "assistant", "content": [{"type": "text", "text": "done"}]})
    assert b.apply(state, "complete") == "complete"
    assert budget_stopped(state) is not None  # 안내가 붙도록 기록은 남긴다


def test_note_goes_to_a_user_message_when_the_last_is_assistant() -> None:
    b = TurnInputBudget(soft_tokens=100, hard_tokens=200)
    state = _state(150, 80)
    state.messages.append({"role": "assistant", "content": [{"type": "text", "text": "working"}]})
    assert b.apply(state, "continue") == "continue"
    assert state.messages[-1]["role"] == "user" and "Turn budget exhausted" in state.messages[-1]["content"]


def test_begin_turn_resets_the_budget_record() -> None:
    state = _state(1)
    state.shared[BUDGET_KEY] = {"stopped": True}
    state.begin_turn()
    assert BUDGET_KEY not in state.shared


# ── 파이프라인 끝까지 ──────────────────────────────────────────────────


class _Ping(Tool):
    @property
    def name(self) -> str:
        return "ping"

    @property
    def description(self) -> str:
        return "ping"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content="pong")


class _Endless(BaseClient):
    """도구를 끝없이 부르는 모델. 호출당 입력 300 토큰. 예산 안내를 보면 보고하고 멈춘다."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, *, obeys: bool, **kw: Any) -> None:
        super().__init__(**kw)
        self.obeys = obeys
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        usage = TokenUsage(input_tokens=300, output_tokens=5)
        last = request.messages[-1] if request.messages else {}
        saw_final = "Turn budget exhausted" in str(last.get("content"))
        if saw_final and self.obeys:
            return APIResponse(
                content=[ContentBlock(type="text", text="보고: 3단계까지 끝냈고 2단계 남음.")],
                stop_reason="end_turn", usage=usage, model="fake",
            )
        return APIResponse(
            content=[ContentBlock(type="tool_use", tool_use_id=f"t{len(self.requests)}", tool_name="ping", tool_input={})],
            stop_reason="tool_use", usage=usage, model="fake",
        )


def _run(obeys: bool, budget: Any) -> tuple[str, _Endless]:
    reg = ToolRegistry()
    reg.register(_Ping(), core=True)
    client = _Endless(obeys=obeys, api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, registry=reg, max_iterations=50, turn_input_budget_tokens=budget,
        # 같은 ping 이 같은 결과를 내므로 4.45.0 반복 거부 종료가 먼저 끊는다 — 여기선 예산만 본다.
        repeat_stop_after=None,
    )
    text = runner.run_turn(pipe, "끝없이 해", PipelineState(session_id="s", model="m"))
    return text, client


def test_pipeline_ends_the_turn_with_a_report_and_a_notice() -> None:
    # soft 1,000 (4번째 호출 뒤) → hard 2,000 (7번째 호출 뒤) → 8번째 응답이 보고 → 끝
    text, client = _run(obeys=True, budget=(1_000, 2_000))
    assert len(client.requests) == 8, len(client.requests)
    assert "보고: 3단계까지 끝냈고" in text and "[안내: 이 턴의 토큰 예산(2,400 토큰)에 도달해" in text
    soft_seen = [i for i, r in enumerate(client.requests) if "Wrap up now" in str(r.messages[-1].get("content"))]
    assert soft_seen == [4], soft_seen  # 1,200 ≥ 1,000 인 시점의 다음 요청에 한 번


def test_pipeline_ends_even_when_the_model_ignores_the_final_note() -> None:
    text, client = _run(obeys=False, budget=(1_000, 2_000))
    assert len(client.requests) == 8  # 8번째 응답(도구 호출)까지 실행하고 끝 — 9번째는 없다
    assert "[안내: 이 턴의 토큰 예산(" in text


def test_no_budget_means_no_limit() -> None:
    text, client = _run(obeys=False, budget=None)
    assert len(client.requests) > 8  # max_iterations(50) 까지 간다
