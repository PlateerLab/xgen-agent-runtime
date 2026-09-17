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
