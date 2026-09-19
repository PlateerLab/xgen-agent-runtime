"""ToolBatch — 같은 도구를 입력 목록으로 한 왕복에 실행."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_CLASSES, BUILT_IN_TOOL_FEATURES
from xgen_agent_runtime.tools.built_in.tool_batch_tool import MAX_ITEMS, ToolBatchTool
from xgen_agent_runtime.tools.registry import ToolRegistry
from xgen_agent_runtime.host.tool_exposure import is_turn_one


class _Search(Tool):
    def __init__(self, safe: bool = True) -> None:
        self.safe = safe
        self.in_flight = 0
        self.peak = 0

    @property
    def name(self) -> str:
        return "shop_search"

    @property
    def description(self) -> str:
        return "search"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"},
                                                 "limit": {"type": "integer"}}, "required": ["query"]}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=self.safe)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        if input["query"] == "boom":
            return ToolResult(content="ERROR upstream: 500", is_error=True)
        return ToolResult(content={"query": input["query"], "limit": input.get("limit"), "hits": 1})


def _run(reg: ToolRegistry, args: Dict[str, Any]) -> ToolResult:
    ctx = ToolContext(session_id="s")
    ctx.tool_registry = reg  # type: ignore[attr-defined]
    return asyncio.run(ToolBatchTool().execute(args, ctx))


def _reg(tool: Tool, *, core: bool = True) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool, core=core)
    reg.register(ToolBatchTool(), core=True)
    return reg


def test_runs_every_input_in_one_call_and_keeps_order() -> None:
    tool = _Search()
    res = _run(_reg(tool), {"tool": "shop_search", "inputs": [{"query": f"q{i}"} for i in range(6)]})
    body = json.loads(res.content)
    assert not res.is_error and body["calls"] == 6 and body["ok"] == 6
    assert [r["input"]["query"] for r in body["results"]] == [f"q{i}" for i in range(6)]
    assert tool.peak > 1  # concurrency-safe → 병렬


def test_inputs_go_through_normal_validation_and_coercion() -> None:
    res = _run(_reg(_Search()), {"tool": "shop_search", "inputs": [{"query": "a", "limit": "3"}, {"limit": 1}]})
    body = json.loads(res.content)
    assert body["results"][0]["ok"] and '"limit": 3' in body["results"][0]["result"]
    assert not body["results"][1]["ok"] and "query" in body["results"][1]["error"]


def test_partial_failure_is_not_an_error_but_all_failed_is() -> None:
    reg = _reg(_Search())
    partial = _run(reg, {"tool": "shop_search", "inputs": [{"query": "ok"}, {"query": "boom"}]})
    assert not partial.is_error and json.loads(partial.content)["failed"] == 1
    allbad = _run(reg, {"tool": "shop_search", "inputs": [{"query": "boom"}]})
    assert allbad.is_error


def test_non_concurrency_safe_tools_run_sequentially() -> None:
    tool = _Search(safe=False)
    _run(_reg(tool), {"tool": "shop_search", "inputs": [{"query": str(i)} for i in range(5)], "max_concurrency": 8})
    assert tool.peak == 1


def test_deferred_tool_can_be_batched_by_name() -> None:
    reg = _reg(_Search(), core=False)
    assert not reg.is_exposed("shop_search")
    res = _run(reg, {"tool": "shop_search", "inputs": [{"query": "x"}]})
    assert not res.is_error and reg.is_exposed("shop_search")


def test_rejects_unknown_nested_and_bad_inputs() -> None:
    reg = _reg(_Search())
    assert _run(reg, {"tool": "nope", "inputs": [{}]}).is_error
    assert _run(reg, {"tool": "ToolBatch", "inputs": [{}]}).is_error
    assert _run(reg, {"tool": "shop_search", "inputs": ["q"]}).is_error


def test_caps_item_count() -> None:
    res = _run(_reg(_Search()), {"tool": "shop_search", "inputs": [{"query": str(i)} for i in range(MAX_ITEMS + 5)]})
    assert json.loads(res.content)["calls"] == MAX_ITEMS


def test_registered_as_workflow_builtin_and_turn_one() -> None:
    assert BUILT_IN_TOOL_CLASSES["ToolBatch"] is ToolBatchTool
    assert "ToolBatch" in BUILT_IN_TOOL_FEATURES["workflow"]
    assert is_turn_one("ToolBatch")


# ── 스테이지 검사 우회 금지 (허용 목록·반복 차단·이벤트) ─────────────────


def _stage_run(reg: ToolRegistry, args: Dict[str, Any], *, state: Any = None, binding: Any = None):
    from xgen_agent_runtime.core.state import PipelineState
    from xgen_agent_runtime.stages.s10_tool.artifact.default.stage import ToolStage

    stage = ToolStage(registry=reg)
    if binding is not None:
        stage.tool_binding = binding
    state = state or PipelineState(session_id="s")
    state.pending_tool_calls = [{"tool_use_id": "outer", "tool_name": "ToolBatch", "tool_input": args}]
    asyncio.run(stage.execute(None, state))
    return state.tool_results[-1], state


def test_batch_cannot_reach_a_tool_the_stage_binding_disallows() -> None:
    from xgen_agent_runtime.tools.stage_binding import StageToolBinding

    tool = _Search()
    binding = StageToolBinding(stage_order=10, blocked={"shop_search"})
    result, _ = _stage_run(_reg(tool), {"tool": "shop_search", "inputs": [{"query": "x"}]}, binding=binding)
    assert result["is_error"] and "access_denied" in result["content"]
    assert tool.peak == 0


def test_batch_cannot_run_a_tool_blocked_by_repeat_guard() -> None:
    from xgen_agent_runtime.core.state import PipelineState
    from xgen_agent_runtime.stages.s10_tool import repeat_guard

    tool = _Search()
    state = PipelineState(session_id="s")
    state.shared["tool.repeat_error_blocked"] = {"shop_search": "ERROR upstream: 500"}
    result, _ = _stage_run(_reg(tool), {"tool": "shop_search", "inputs": [{"query": "x"}]}, state=state)
    assert result["is_error"] and result["content"].startswith("ERROR repeated_failure_blocked")
    assert tool.peak == 0
    assert repeat_guard.blocked_result({"tool_name": "shop_search"}, state.shared)


def test_each_item_emits_call_events_so_real_executions_are_visible() -> None:
    result, state = _stage_run(_reg(_Search()), {"tool": "shop_search", "inputs": [{"query": "a"}, {"query": "b"}]})
    assert not result.get("is_error")
    starts = [e["data"] for e in state.events if e["type"] == "tool.call_start"]
    names = [d["name"] for d in starts]
    assert names.count("ToolBatch") == 1 and names.count("shop_search") == 2
    inner_ids = [d["tool_use_id"] for d in starts if d["name"] == "shop_search"]
    assert all(i.startswith("ToolBatch-") for i in inner_ids) and len(set(inner_ids)) == 2
    completes = [e["data"] for e in state.events if e["type"] == "tool.call_complete" and e["data"]["name"] == "shop_search"]
    assert len(completes) == 2 and all("result" in d for d in completes)


def test_same_error_on_every_item_counts_once_per_batch() -> None:
    reg = _reg(_Search())
    bad = {"tool": "shop_search", "inputs": [{"limit": 1} for _ in range(10)]}  # query 누락 = 입력 오류
    result, state = _stage_run(reg, bad)
    body = json.loads(result["content"].split("\n")[0]) if result["content"].startswith("{") else None
    assert body is not None and body["failed"] == 10
    # 한 번의 시도로 센다 — 첫 배치에서 곧바로 차단되지 않는다.
    counts = state.shared["tool.repeat_error_counts"]
    assert max(counts.values()) == 1
    assert not state.shared.get("tool.repeat_error_blocked")


def test_adapted_tool_can_open_the_family_it_points_to() -> None:
    """커넥터 안내 도구처럼 LangChain 으로 들어온 문도 가리킨 도구를 실제로 연다."""
    from pydantic import create_model
    from langchain_core.tools import StructuredTool

    from xgen_agent_runtime.host.tools import OPENS_FAMILY_KEY, adapt_tools

    guide = StructuredTool.from_function(
        func=lambda: "Tools: mcp_local_BrowserTabs", name="mcp_local_BrowserGuide",
        description="guide", args_schema=create_model("G"),
    )
    guide.metadata = {OPENS_FAMILY_KEY: ["mcp_local_BrowserTabs"]}
    member = StructuredTool.from_function(
        func=lambda: "tabs", name="mcp_local_BrowserTabs", description="tabs", args_schema=create_model("T"),
    )
    reg = adapt_tools([guide, member], core=lambda n: n.endswith("Guide"))
    assert not reg.is_exposed("mcp_local_BrowserTabs")
    ctx = ToolContext(session_id="s")
    ctx.tool_registry = reg  # type: ignore[attr-defined]
    res = asyncio.run(reg.get("mcp_local_BrowserGuide").execute({}, ctx))
    assert reg.is_exposed("mcp_local_BrowserTabs")
    assert "Now callable: mcp_local_BrowserTabs" in res.content
