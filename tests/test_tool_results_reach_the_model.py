"""도구를 쓴 턴은 도구 결과를 모델이 본 뒤에 끝나야 한다.

실측(2026-09-17): 모델이 한 응답에 텍스트와 도구 호출을 함께 내면서 텍스트에
``[COMPLETE]``/``[ERROR ...]``/``[BLOCKED]`` 같은 마커가 섞이면 — 예: 빌드 로그
"[ERROR] ..." 를 인용하며 파일을 고치는 도구를 부를 때 — s9 가 completion_signal 을
세우고, s10 이 도구를 실행한 뒤 pending_tool_calls 를 비우고, s14 는 그 빈 목록만
보고 "도구 안 썼음 → complete", s16 은 upstream 결정을 따라 턴을 끝냈다.
도구는 돌았는데 모델은 결과를 못 본 채 호출 직전에 써 둔 추측이 답이 됐다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock, TokenUsage
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry


class _Build(Tool):
    def __init__(self) -> None:
        self.runs = 0

    @property
    def name(self) -> str:
        return "build"

    @property
    def description(self) -> str:
        return "build"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.runs += 1
        return ToolResult(content="BUILD OK — 0 errors")


class _Client(BaseClient):
    """1번째 응답: 마커 섞인 텍스트 + 도구 호출. 2번째: 도구 결과를 본 최종 답."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, first_text: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.first_text = first_text
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        if len(self.requests) == 1:
            return APIResponse(
                content=[
                    ContentBlock(type="text", text=self.first_text),
                    ContentBlock(type="tool_use", tool_use_id="t1", tool_name="build", tool_input={}),
                ],
                stop_reason="tool_use",
                usage=usage,
                model="fake",
            )
        return APIResponse(
            content=[ContentBlock(type="text", text="빌드 결과를 확인했습니다: BUILD OK")],
            stop_reason="end_turn",
            usage=usage,
            model="fake",
        )


def _run(first_text: str) -> tuple[str, _Client, _Build]:
    reg = ToolRegistry()
    tool = _Build()
    reg.register(tool, core=True)
    client = _Client(first_text, api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client,
        stream=False, enable_compaction=False, registry=reg,
    )
    text = runner.run_turn(pipe, "빌드 고쳐줘", PipelineState(session_id="s", model="m"))
    return text, client, tool


def test_plain_text_with_tool_call_continues_to_see_the_result() -> None:
    text, client, tool = _run("빌드를 돌려 보겠습니다.")
    assert tool.runs == 1 and len(client.requests) == 2
    assert "BUILD OK" in text


def test_marker_in_text_does_not_end_the_turn_before_tool_results_are_seen() -> None:
    # 모델이 로그를 인용하며 "[ERROR] ..." 를 쓰고 같은 응답에서 도구를 부른 경우.
    for first in ("[ERROR] Type error in page.tsx — 고쳐서 다시 빌드합니다.",
                  "[COMPLETE] 수정 완료, 확인차 빌드합니다.",
                  "[BLOCKED] 권한 확인 후 빌드합니다."):
        text, client, tool = _run(first)
        assert tool.runs == 1, first
        assert len(client.requests) == 2, f"도구 결과를 모델이 못 봤다: {first!r}"
        assert "BUILD OK" in text, first


# ── Stage 14 (evaluate) 단독: 결과가 있으면 마커가 뭐라 하든 continue ──────────


def test_evaluate_strategies_continue_when_tool_results_are_fresh() -> None:
    import asyncio

    from xgen_agent_runtime.stages.s14_evaluate.artifact.adaptive.strategy import (
        BinaryClassifyEvaluation,
    )
    from xgen_agent_runtime.stages.s14_evaluate.artifact.default.strategies import (
        SignalBasedEvaluation,
    )

    for strategy in (SignalBasedEvaluation(), BinaryClassifyEvaluation()):
        for signal in ("complete", "error", "blocked", None):
            state = PipelineState(session_id="s", model="m")
            state.completion_signal = signal
            state.pending_tool_calls = []  # Stage 10 이 비운 뒤
            state.tool_results = [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
            assert state.has_fresh_tool_results
            result = asyncio.run(strategy.evaluate(state))
            assert result.decision == "continue", (type(strategy).__name__, signal)


def test_stale_signal_from_previous_iteration_is_cleared_by_parse() -> None:
    import asyncio

    from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock, TokenUsage
    from xgen_agent_runtime.stages.s09_parse.artifact.default.stage import ParseStage

    state = PipelineState(session_id="s", model="m")
    state.completion_signal, state.completion_detail = "error", "old"
    resp = APIResponse(content=[ContentBlock(type="text", text="정상 답")], stop_reason="end_turn",
                       usage=TokenUsage(input_tokens=1, output_tokens=1), model="m")
    asyncio.run(ParseStage().execute(resp, state))
    assert state.completion_signal is None and state.completion_detail is None
