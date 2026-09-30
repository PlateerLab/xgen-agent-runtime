"""호스트의 도구 결과 필터 (4.71.0) — SDK·CLI 가 같은 후처리를 받는다.

호스트(xgen-workflow)는 관리자 정책에 따라 외부 데이터 도구 결과의 개인정보·금칙어를 가린다.
필터(``ToolContext.result_filter``: ``async (tool, result) -> ToolResult``)는 Stage 10 의
``RegistryRouter.route`` 한 곳에서 도구가 돈 직후 — 큰 결과 파일 저장·미리보기·이벤트·반복
가드보다 먼저 — 적용된다. 지키는 것:

* SDK 파이프라인(Stage 10 루프)·Stage 6 내부 디스패처·CLI 도구 표면(``TurnToolSurface``)·
  ToolBatch 항목이 모두 걸러진 결과를 본다.
* 필터는 **Tool 인스턴스**를 받는다(이름 접두가 겹쳐도 종류로 판정할 수 있게).
* 오류 결과도 필터로 간다 — 무엇을 가릴지는 호스트가 정한다.
* 필터가 고장 나면(예외·엉뚱한 반환) 원래 결과를 쓴다(fail-open).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any, Dict, List

import pytest

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.host.tool_surface import TurnToolSurface
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock, TokenUsage
from xgen_agent_runtime.stages.s10_tool import RegistryRouter
from xgen_agent_runtime.stages.s10_tool.artifact.default.stage import ToolStage
from xgen_agent_runtime.stages.s10_tool.dispatcher import ToolDispatcher
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in.tool_batch_tool import ToolBatchTool
from xgen_agent_runtime.tools.registry import ToolRegistry

SECRET = "010-1234-5678"
MASK = "***-****-****"


class _Lookup(Tool):
    """외부 데이터 도구 흉내 — 결과에 개인정보가 섞여 온다."""

    external_data = True  # 호스트가 종류로 판정하는 표지(예시)

    def __init__(self, name: str = "crm_lookup", *, fail: bool = False) -> None:
        self._name = name
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "look a customer up"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"q": {"type": "string"}}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        if self._fail:
            return ToolResult(content=f"lookup failed for {SECRET}", is_error=True)
        return ToolResult(content=f"name=Kim phone={SECRET}")


class _Local(_Lookup):
    """같은 모양의 로컬 도구 — 필터가 종류로 건너뛴다."""

    external_data = False


class _Recorder:
    """필터 — 받은 (tool, result) 를 기록하고 외부 데이터 도구만 가린다."""

    def __init__(self) -> None:
        self.seen: List[tuple] = []

    async def __call__(self, tool: Tool, result: ToolResult) -> ToolResult:
        self.seen.append((tool, result))
        if not getattr(tool, "external_data", False):
            return result
        content = result.content
        if isinstance(content, str):
            content = content.replace(SECRET, MASK)
        return replace(result, content=content)


def _ctx(result_filter: Any) -> ToolContext:
    return ToolContext(session_id="s1", working_dir="/tmp", result_filter=result_filter)


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools:
        reg.register(tool, core=True)
    return reg


def _call(name: str = "crm_lookup") -> Dict[str, Any]:
    return {"tool_use_id": "tu_1", "tool_name": name, "tool_input": {"q": "kim"}}


# ── 라우터: 한 곳에서 적용 ─────────────────────────────────────────────


def test_router_applies_the_filter_and_passes_the_tool_instance() -> None:
    tool = _Lookup()
    rec = _Recorder()
    out = asyncio.run(RegistryRouter(_registry(tool)).route("crm_lookup", {"q": "k"}, _ctx(rec)))
    assert out.content == f"name=Kim phone={MASK}"
    assert rec.seen[0][0] is tool, "필터는 이름이 아니라 실행된 Tool 인스턴스를 받는다"
    assert rec.seen[0][0].name == "crm_lookup"
    assert rec.seen[0][1].content == f"name=Kim phone={SECRET}", "필터는 가공 전 결과를 받는다"


def test_filter_decides_by_kind_not_by_name() -> None:
    rec = _Recorder()
    local = _Local("crm_lookup_local")
    out = asyncio.run(
        RegistryRouter(_registry(local)).route("crm_lookup_local", {"q": "k"}, _ctx(rec))
    )
    assert SECRET in out.content
    assert rec.seen and rec.seen[0][0] is local


def test_error_results_go_through_the_filter_too() -> None:
    rec = _Recorder()
    out = asyncio.run(
        RegistryRouter(_registry(_Lookup(fail=True))).route("crm_lookup", {"q": "k"}, _ctx(rec))
    )
    assert out.is_error and SECRET not in str(out.content)
    assert rec.seen[0][1].is_error


def test_a_failing_filter_falls_back_to_the_original_result(caplog) -> None:
    async def _boom(tool: Tool, result: ToolResult) -> ToolResult:
        raise RuntimeError("policy engine down")

    with caplog.at_level(logging.WARNING):
        out = asyncio.run(
            RegistryRouter(_registry(_Lookup())).route("crm_lookup", {"q": "k"}, _ctx(_boom))
        )
    assert out.content == f"name=Kim phone={SECRET}" and not out.is_error
    assert "crm_lookup" in caplog.text and "filter" in caplog.text


@pytest.mark.parametrize("bad", [None, "not a ToolResult"])
def test_a_filter_returning_nothing_useful_keeps_the_original(bad) -> None:
    async def _odd(tool: Tool, result: ToolResult) -> Any:
        return bad

    out = asyncio.run(
        RegistryRouter(_registry(_Lookup())).route("crm_lookup", {"q": "k"}, _ctx(_odd))
    )
    assert out.content == f"name=Kim phone={SECRET}"


def test_a_sync_filter_is_accepted() -> None:
    def _sync(tool: Tool, result: ToolResult) -> ToolResult:
        return replace(result, content="[masked]")

    out = asyncio.run(
        RegistryRouter(_registry(_Lookup())).route("crm_lookup", {"q": "k"}, _ctx(_sync))
    )
    assert out.content == "[masked]"


def test_no_filter_means_no_change() -> None:
    out = asyncio.run(
        RegistryRouter(_registry(_Lookup())).route("crm_lookup", {"q": "k"}, _ctx(None))
    )
    assert out.content == f"name=Kim phone={SECRET}"


def test_calls_that_never_ran_do_not_reach_the_filter() -> None:
    rec = _Recorder()
    out = asyncio.run(RegistryRouter(_registry(_Lookup())).route("nope", {}, _ctx(rec)))
    assert out.is_error and rec.seen == []


# ── SDK 경로 ─────────────────────────────────────────────────────────


class _Client(BaseClient):
    """1번째 응답: 도구 호출. 2번째: 최종 답 — 두 번째 요청에 실린 도구 결과를 본다."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        if len(self.requests) == 1:
            return APIResponse(
                content=[
                    ContentBlock(
                        type="tool_use",
                        tool_use_id="t1",
                        tool_name="crm_lookup",
                        tool_input={"q": "kim"},
                    ),
                ],
                stop_reason="tool_use",
                usage=usage,
                model="fake",
            )
        return APIResponse(
            content=[ContentBlock(type="text", text="done")],
            stop_reason="end_turn",
            usage=usage,
            model="fake",
        )


def _tool_result_texts(messages: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                out.append(str(block.get("content")))
    return out


@pytest.mark.parametrize("with_tool_context", [True, False])
def test_sdk_pipeline_model_and_events_see_filtered_content(with_tool_context) -> None:
    rec = _Recorder()
    client = _Client(api_key="k")
    pipe = runner.build_pipeline(
        name="t",
        provider="openai",
        model="m",
        api_key="k",
        llm_client=client,
        stream=False,
        enable_compaction=False,
        registry=_registry(_Lookup()),
        tool_context=ToolContext(session_id="s", working_dir="/tmp") if with_tool_context else None,
        tool_result_filter=rec,
    )
    state = PipelineState(session_id="s", model="m")
    runner.run_turn(pipe, "look kim up", state)
    assert rec.seen, "SDK Stage 10 did not call the filter"
    results = _tool_result_texts(state.messages)
    assert results and all(SECRET not in r for r in results)
    assert any(MASK in r for r in results)
    # 모델에게 간 요청에도, 화면으로 간 이벤트 미리보기에도 원문이 없다.
    assert SECRET not in repr(client.requests[-1])
    completes = [e for e in state.events if e.get("type") == "tool.call_complete"]
    assert completes and SECRET not in repr(completes)


def test_stage6_internal_dispatcher_applies_the_filter() -> None:
    rec = _Recorder()
    stage = ToolStage(registry=_registry(_Lookup()), context=_ctx(rec))
    out = asyncio.run(ToolDispatcher(stage).dispatch(_call(), PipelineState(session_id="s1")))
    assert SECRET not in str(out.get("content")) and MASK in str(out.get("content"))


def test_filter_runs_before_large_result_persistence(tmp_path) -> None:
    """큰 결과는 파일로 옮겨진다 — 그 파일에도 원문이 남으면 안 된다."""

    class _Big(_Lookup):
        def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
            return ToolCapabilities(concurrency_safe=True, max_result_chars=100)

        async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(content=(f"phone={SECRET}\n" * 200))

    rec = _Recorder()
    ctx = ToolContext(
        session_id="s1", working_dir=str(tmp_path), storage_path=str(tmp_path), result_filter=rec
    )
    stage = ToolStage(registry=_registry(_Big()), context=ctx)
    results = asyncio.run(stage.dispatch_calls([_call()], PipelineState(session_id="s1")))
    assert SECRET not in str(results[0].get("content"))
    persisted = [p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file()]
    assert persisted, "the oversized result should have been persisted"
    assert all(SECRET not in text for text in persisted)


# ── CLI 경로 (TurnToolSurface) ───────────────────────────────────────


class _TurnLoop:
    def __init__(self) -> None:
        import threading

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


@pytest.fixture
def turn_loop():
    tl = _TurnLoop()
    yield tl
    tl.close()


def test_cli_surface_returns_the_same_filtered_result_as_sdk(turn_loop) -> None:
    rec = _Recorder()
    tool = _Lookup()
    surface = TurnToolSurface(
        registry=_registry(tool), tool_context=_ctx(rec), state=PipelineState(session_id="s1")
    )
    surface.bind_loop(turn_loop.loop)
    out = asyncio.run(surface.call("crm_lookup", {"q": "kim"}))
    assert out == {"content": [{"type": "text", "text": f"name=Kim phone={MASK}"}], "isError": False}
    assert rec.seen[0][0] is tool

    # SDK Stage 10 과 글자 하나까지 같다.
    stage = ToolStage(registry=_registry(_Lookup()), context=_ctx(_Recorder()))
    sdk = asyncio.run(stage.dispatch_calls([_call()], PipelineState(session_id="s1")))
    assert out["content"][0]["text"] == sdk[0]["content"]


def test_cli_surface_without_a_context_takes_a_filter_set_later(turn_loop) -> None:
    """run_tool_context 가 없는 턴 — 표면이 만든 컨텍스트에 실어도 실행에 닿는다."""
    rec = _Recorder()
    surface = TurnToolSurface(
        registry=_registry(_Lookup()), tool_context=None, state=PipelineState(session_id="s1")
    )
    surface.tool_context.result_filter = rec
    surface.bind_loop(turn_loop.loop)
    out = asyncio.run(surface.call("crm_lookup", {"q": "kim"}))
    assert MASK in out["content"][0]["text"] and rec.seen


# ── ToolBatch: 항목마다 걸러진다 ─────────────────────────────────────


def test_tool_batch_members_are_filtered_with_their_own_tool() -> None:
    rec = _Recorder()
    inner = _Lookup()
    batch = ToolBatchTool()
    stage = ToolStage(registry=_registry(inner, batch), context=_ctx(rec))
    call = {
        "tool_use_id": "tu_b",
        "tool_name": batch.name,
        "tool_input": {"tool": "crm_lookup", "inputs": [{"q": "a"}, {"q": "b"}]},
    }
    results = asyncio.run(stage.dispatch_calls([call], PipelineState(session_id="s1")))
    assert SECRET not in str(results[0].get("content"))
    seen_tools = [t for t, _ in rec.seen]
    assert seen_tools.count(inner) == 2, "each batch member reaches the filter as itself"
    assert any(t is batch for t in seen_tools), "the batch's own result is offered too"


# ── 종류 표지: 이름이 아니라 종류로 판정할 수 있게 ────────────────────


def test_tool_origin_tells_device_node_memory_and_builtin_tools_apart() -> None:
    from langchain_core.tools import StructuredTool

    from xgen_agent_runtime.host.device_tools import build_device_guide, build_device_tool
    from xgen_agent_runtime.host.memory_tools import build_memory_tools
    from xgen_agent_runtime.host.tools import adapt_tools
    from xgen_agent_runtime.tools import tool_origin
    from xgen_agent_runtime.tools.built_in.bash_tool import BashTool
    from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
    from xgen_agent_runtime.tools.mcp.adapter import MCPToolAdapter

    async def _call(tool: str, args: Dict[str, Any]) -> Any:
        return "ok"

    device = build_device_tool(
        server="local", tool="ReadFile", description="d", input_schema={}, call=_call
    )
    guide = build_device_guide(name="mcp_local_BrowserGuide", description="d", text="t")
    # 같은 접두의 노드 도구 — 이름만으로는 기기 도구와 구분되지 않는다.
    node = StructuredTool.from_function(
        func=lambda q="": "ok", name="mcp_local_lookup", description="node tool"
    )
    adapted = adapt_tools([node, {"name": "calc", "func": lambda **_: 1, "description": "c"}])
    assert adapted is not None

    assert tool_origin(device) == tool_origin(guide) == "device"
    assert {tool_origin(t) for t in build_memory_tools(object())} == {"memory"}
    assert {tool_origin(adapted.get(n)) for n in adapted.list_names()} == {"adapted"}
    assert tool_origin(ReadTool()) == tool_origin(BashTool()) == "builtin"
    assert MCPToolAdapter.__module__.startswith("xgen_agent_runtime.tools.mcp")
    assert tool_origin(_Lookup()) == ""  # 호스트가 만든 도구는 호스트가 안다
