"""4.26.0 — 자동 이어가기 상한, 취소 턴 usage, 반복 실패 차단, 프롬프트 캐시 opt-in.

실측 배경 (2026-09-16 dev):
* "이거해줘" 한 마디에 한 턴이 도구 120회+·입력 252만 토큰까지 갔다 —
  반복 한도에 닿은 슬라이스를 사용자 확인 없이 최대 20번 자동으로 이어 갔다.
* 도구 인자 타입 오류 하나로 같은 호출을 15번 반복했다.
* 사용자가 중지한 턴은 사용량이 기록되지 않았다.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict

from xgen_agent_runtime.core.continuation import ContinuationInput
from xgen_agent_runtime.core.pipeline import PipelineEvent
from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.stages.s10_tool import repeat_guard
from xgen_agent_runtime.stages.s10_tool.artifact.default.stage import ToolStage
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry


# ── 자동 이어가기 상한 ──────────────────────────────────────────────


class _AlwaysResumable:
    """매 슬라이스가 반복 한도에 닿아 resumable 로 끝나는 파이프라인."""

    def __init__(self) -> None:
        self.stream_runs = 0
        self.runs = 0

    async def run_stream(self, text: Any, state: PipelineState) -> AsyncIterator[PipelineEvent]:
        # 실제 Pipeline 과 같다: 새 턴만 begin_turn, CONTINUE_RUN 은 누적을 유지한다.
        if isinstance(text, ContinuationInput):
            state.begin_continuation_slice()
        else:
            state.begin_turn()
        self.stream_runs += 1
        u = TokenUsage(input_tokens=10, output_tokens=1)
        state.token_usage += u
        state.turn_token_usage.append(u)
        yield PipelineEvent(type="text.delta", data={"text": f"s{self.stream_runs}"})
        yield PipelineEvent(
            type="pipeline.complete",
            data={"result": "", "resumable": True, "termination_reason": "max_iterations_per_slice"},
        )

    async def run(self, text: Any, state: PipelineState) -> Any:
        self.runs += 1

        class _R:
            resumable = True
            success = False
            status = "suspended"
            termination_reason = "max_iterations_per_slice"
            text = ""
            error = None

        return _R()

    async def aclose(self) -> None:
        pass

    def _resolved_provider_name(self, state: PipelineState) -> str:
        return "fake"


def test_default_continuation_cap_is_two() -> None:
    assert runner.DEFAULT_MAX_CONTINUATION_SLICES == 2


def test_stream_turn_stops_after_default_cap_and_tells_the_user() -> None:
    pipe = _AlwaysResumable()
    out = list(runner.stream_turn(pipe, "이거해줘", PipelineState(session_id="s")))
    assert pipe.stream_runs == 1 + runner.DEFAULT_MAX_CONTINUATION_SLICES
    texts = [c for c in out if isinstance(c, str)]
    assert texts[-1] == runner.SUSPEND_NOTICE
    events = [c["data"]["type"] for c in out if isinstance(c, dict) and c.get("type") == "agent_event"]
    assert events.count("task_progress") == runner.DEFAULT_MAX_CONTINUATION_SLICES
    assert events[-1] == "task_suspended"
    usage = [c for c in out if isinstance(c, dict) and c.get("type") == "usage"]
    assert len(usage) == 1 and usage[0]["data"]["input_tokens"] == 30


def test_stream_turn_host_can_still_raise_the_cap() -> None:
    pipe = _AlwaysResumable()
    list(runner.stream_turn(pipe, "x", PipelineState(session_id="s"), max_continuation_slices=5))
    assert pipe.stream_runs == 6


def test_structured_output_stream_gets_no_notice_text() -> None:
    pipe = _AlwaysResumable()
    out = list(
        runner.stream_turn(pipe, "x", PipelineState(session_id="s"), output_schema={"type": "object"})
    )
    assert runner.SUSPEND_NOTICE not in out


def test_run_turn_uses_the_same_default_cap() -> None:
    pipe = _AlwaysResumable()
    result = runner.run_turn(pipe, "x", PipelineState(session_id="s"))
    assert pipe.runs == 1 + runner.DEFAULT_MAX_CONTINUATION_SLICES
    assert result.startswith("[SUSPENDED]")


# ── 닫힌 스트림의 usage sink ────────────────────────────────────────


def test_usage_sink_is_filled_when_consumer_closes_the_stream() -> None:
    pipe = _AlwaysResumable()
    sink: Dict[str, Any] = {}
    gen = runner.stream_turn(pipe, "x", PipelineState(session_id="s"), usage_sink=sink)
    assert next(gen) == "s1"
    gen.close()
    assert sink["input_tokens"] == 10 and sink["partial"] is True


def test_usage_sink_matches_the_usage_chunk_on_normal_end() -> None:
    pipe = _AlwaysResumable()
    sink: Dict[str, Any] = {}
    out = list(runner.stream_turn(pipe, "x", PipelineState(session_id="s"), usage_sink=sink))
    chunk = next(c for c in out if isinstance(c, dict) and c.get("type") == "usage")
    assert sink == chunk["data"] and "partial" not in sink


# ── 반복 실패 차단 ──────────────────────────────────────────────────


class _BadArgTool(Tool):
    def __init__(self) -> None:
        self.executions = 0

    @property
    def name(self) -> str:
        return "lotteimall_search"

    @property
    def description(self) -> str:
        return "search"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.executions += 1
        if isinstance(input.get("max_results"), str):
            return ToolResult(
                content=(
                    "ERROR invalid_input: Invalid input for 'lotteimall_search': "
                    f"'{input['max_results']}' is not of type 'integer' "
                    f'{{"request_id": "req_{self.executions:08d}abcdef"}}'
                ),
                is_error=True,
            )
        return ToolResult(content="ok")


class _OtherTool(_BadArgTool):
    @property
    def name(self) -> str:
        return "other"


def _round(stage: ToolStage, state: PipelineState, i: int, *, tool: str = "lotteimall_search",
           max_results: Any = "3") -> Dict[str, Any]:
    state.pending_tool_calls = [{
        "tool_use_id": f"u{i}",
        "tool_name": tool,
        # 검색어는 매번 다르다 — 실측 사고와 같은 모양.
        "tool_input": {"query": f"상품 {i}", "max_results": max_results},
    }]
    asyncio.run(stage.execute(None, state))
    return state.tool_results[0]


def _stage(*tools: Tool) -> ToolStage:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return ToolStage(registry=reg)


def test_same_error_with_changing_query_warns_at_three_and_blocks_at_five() -> None:
    tool = _BadArgTool()
    stage, state = _stage(tool), PipelineState(session_id="s")
    results = [_round(stage, state, i) for i in range(1, 8)]

    assert "[반복 실패" not in results[1]["content"]
    assert "[반복 실패 3회]" in results[2]["content"]
    assert "[반복 실패 4회]" in results[3]["content"]
    assert "[반복 실패 5회]" in results[4]["content"]
    # 5번 실행된 뒤로는 실행하지 않고 차단 결과를 돌려준다.
    assert tool.executions == repeat_guard.BLOCK_AT
    for r in results[5:]:
        assert r["is_error"] and r["content"].startswith("ERROR repeated_failure_blocked")
        assert r["tool_use_id"] in {"u6", "u7"}
    types = [e["type"] for e in state.events]
    assert "tool.repeat_failure" in types and "tool.repeat_blocked" in types


def test_success_resets_the_count() -> None:
    tool = _BadArgTool()
    stage, state = _stage(tool), PipelineState(session_id="s")
    _round(stage, state, 1)
    _round(stage, state, 2)
    assert _round(stage, state, 3, max_results=3)["content"] == "ok"
    assert "[반복 실패" not in _round(stage, state, 4)["content"]


def test_block_is_per_tool_and_other_tools_keep_running() -> None:
    bad, other = _BadArgTool(), _OtherTool()
    stage, state = _stage(bad, other), PipelineState(session_id="s")
    for i in range(1, 6):
        _round(stage, state, i)
    assert _round(stage, state, 6)["content"].startswith("ERROR repeated_failure_blocked")
    assert _round(stage, state, 7, tool="other", max_results=3)["content"] == "ok"


def test_mixed_batch_keeps_result_order() -> None:
    bad, other = _BadArgTool(), _OtherTool()
    stage, state = _stage(bad, other), PipelineState(session_id="s")
    for i in range(1, 6):
        _round(stage, state, i)
    state.pending_tool_calls = [
        {"tool_use_id": "a", "tool_name": "other", "tool_input": {"max_results": 1}},
        {"tool_use_id": "b", "tool_name": "lotteimall_search", "tool_input": {"max_results": "3"}},
        {"tool_use_id": "c", "tool_name": "other", "tool_input": {"max_results": 2}},
    ]
    asyncio.run(stage.execute(None, state))
    assert [r["tool_use_id"] for r in state.tool_results] == ["a", "b", "c"]
    assert state.tool_results[1]["content"].startswith("ERROR repeated_failure_blocked")


def test_normalize_error_ignores_volatile_ids_and_json_body() -> None:
    a = repeat_guard.normalize_error("ERROR x: '3' is not of type 'integer' {\"id\": \"req_1\"}")
    b = repeat_guard.normalize_error("ERROR x: '3' is not of type 'integer' {\"id\": \"req_2\"}")
    assert a == b and "{" not in a


# ── 프롬프트 캐시 opt-in ────────────────────────────────────────────


def _build(**kw: Any):
    from tests.test_host_runner_usage import _UsageClient

    return runner.build_pipeline(
        name="t", provider="anthropic", model="claude-sonnet-4-6", api_key="k",
        llm_client=_UsageClient(api_key="k"), stream=False, enable_compaction=False, **kw,
    )


def test_prompt_cache_is_off_by_default() -> None:
    assert _build().get_stage(5) is None


def test_prompt_cache_opt_in_registers_aggressive_strategy() -> None:
    stage = _build(enable_prompt_cache=True).get_stage(5)
    assert stage is not None
    assert stage.get_strategy_slots()["strategy"].strategy.name == "aggressive_cache"


def test_repeat_counts_do_not_leak_into_the_next_turn() -> None:
    tool = _BadArgTool()
    stage, state = _stage(tool), PipelineState(session_id="s")
    for i in range(1, 6):
        _round(stage, state, i)
    state.begin_turn()
    assert not _round(stage, state, 6)["content"].startswith("ERROR repeated_failure_blocked")


# ── 4.27.0: 입력 타입 자동 변환 · 호출별 usage · 효율 원칙 프롬프트 ──────


def test_coerce_input_fixes_obvious_string_numbers_and_booleans() -> None:
    from xgen_agent_runtime.tools.errors import coerce_input

    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "ratio": {"type": "number"},
            "exact": {"type": "boolean"},
            "ids": {"type": "array", "items": {"type": "integer"}},
            "nested": {"type": "object", "properties": {"n": {"type": "integer"}}},
        },
    }
    raw = {"query": "3", "max_results": "3", "ratio": "0.5", "exact": "True", "ids": ["1", 2],
           "nested": {"n": "7"}}
    fixed = coerce_input(schema, raw)
    assert fixed == {"query": "3", "max_results": 3, "ratio": 0.5, "exact": True, "ids": [1, 2],
                     "nested": {"n": 7}}
    assert raw["max_results"] == "3"  # 원본 불변


def test_coerce_input_leaves_ambiguous_values_for_the_validator() -> None:
    from xgen_agent_runtime.tools.errors import coerce_input

    schema = {"type": "object", "properties": {"n": {"type": "integer"}, "flag": {"type": "boolean"}}}
    raw = {"n": "three", "flag": "yes"}
    assert coerce_input(schema, raw) is raw
    union = {"type": "object", "properties": {"v": {"type": ["string", "integer"]}}}
    assert coerce_input(union, {"v": "3"}) == {"v": "3"}


def test_router_accepts_string_integer_after_coercion() -> None:
    tool = _BadArgTool()
    tool_schema = {"type": "object", "properties": {"query": {"type": "string"},
                                                    "max_results": {"type": "integer"}}}
    type(tool).input_schema = property(lambda self: tool_schema)
    try:
        stage, state = _stage(tool), PipelineState(session_id="s")
        result = _round(stage, state, 1, max_results="3")
        assert result["content"] == "ok" and not result.get("is_error")
    finally:
        type(tool).input_schema = property(lambda self: {"type": "object"})


class _Prov:
    def __init__(self, provider: str) -> None:
        self._p = provider

    def _resolved_provider_name(self, state: PipelineState) -> str:
        return self._p


def _usage_state() -> PipelineState:
    state = PipelineState(session_id="s")
    for it, cr in ((1000, 0), (200, 9000), (300, 9500)):
        state.turn_token_usage.append(
            TokenUsage(input_tokens=it, output_tokens=10, cache_read_input_tokens=cr)
        )
    return state


def test_usage_reports_calls_and_prompt_sizes_for_anthropic() -> None:
    data = runner.turn_usage(_Prov("anthropic"), _usage_state())
    assert data["calls"] == 3
    assert data["first_call_prompt_tokens"] == 1000
    assert data["max_call_prompt_tokens"] == 9800  # 캐시분을 따로 보고 → 더한다


def test_usage_prompt_sizes_do_not_double_count_openai_cache() -> None:
    data = runner.turn_usage(_Prov("openai"), _usage_state())
    assert data["max_call_prompt_tokens"] == 1000  # prompt_tokens 에 이미 포함


def test_efficiency_block_mentions_round_trips_and_batching() -> None:
    from xgen_agent_runtime.host._constants import EFFICIENCY_PROMPT_BLOCK

    text = EFFICIENCY_PROMPT_BLOCK.lower()
    assert "round trip" in text and "parallel" in text and "one script" in text
