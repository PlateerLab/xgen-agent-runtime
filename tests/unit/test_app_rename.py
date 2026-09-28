"""아티팩트 → 앱 개명(2026-09-28, 4.66.0) — 문은 이름 목록으로, 옛 이름으로 불러도 지금 도구가 돈다."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import RegistryRouter
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in.tool_batch_tool import ToolBatchTool
from xgen_agent_runtime.tools.gates import family_of, gate_of
from xgen_agent_runtime.tools.registry import ToolRegistry
from xgen_agent_runtime.tools.renamed import RENAMED_TOOLS, current_name

APP_FAMILY = ("AppCreate", "AppPublish", "AppStatus", "AppList", "AppDelete")


class _Echo(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test tool"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"slug": {"type": "string"}}}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=f"{self._name}:{input.get('slug', '')}")


def _registry(*names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for n in names:
        reg.register(_Echo(n))
    return reg


def _ctx(reg: ToolRegistry | None = None) -> ToolContext:
    ctx = ToolContext(session_id="test", working_dir="/tmp")
    if reg is not None:
        ctx.tool_registry = reg
    return ctx


def test_the_app_gate_opens_exactly_its_five_tools():
    assert gate_of("AppGuide") is not None
    registered = ["AppGuide", *APP_FAMILY, "AppendRows", "AppStoreLookup", "JobList"]
    # 사용자가 만든 App* 도구는 앱 가족이 아니다 — 이름 목록이라 끌려 들어가지 않는다.
    assert sorted(family_of("AppGuide", registered)) == sorted(APP_FAMILY)


def test_the_connector_prefixed_gate_opens_its_own_prefix():
    registered = ["mcp__connector__AppGuide", *(f"mcp__connector__{n}" for n in APP_FAMILY)]
    assert len(family_of("mcp__connector__AppGuide", registered)) == len(APP_FAMILY)


def test_old_names_map_to_the_new_ones_only_when_the_new_tool_exists():
    assert set(RENAMED_TOOLS.values()) == {"AppGuide", *APP_FAMILY}
    have = {"AppCreate", "mcp__connector__AppPublish"}.__contains__
    assert current_name("ArtifactCreate", have) == "AppCreate"
    assert current_name("mcp__connector__ArtifactPublish", have) == "mcp__connector__AppPublish"
    assert current_name("ArtifactDelete", have) is None, "지금 이름이 없으면 보내지 않는다"
    assert current_name("BrowserNavigate", have) is None


def test_the_router_runs_the_new_tool_for_an_old_name():
    router = RegistryRouter(_registry("AppCreate"))
    out = asyncio.run(router.route("ArtifactCreate", {"slug": "demo"}, _ctx()))
    assert not out.is_error and out.content == "AppCreate:demo"


def test_an_unknown_name_is_still_unknown():
    router = RegistryRouter(_registry("AppCreate"))
    out = asyncio.run(router.route("ArtifactSave", {}, _ctx()))
    assert out.is_error and "ArtifactSave" in str(out.content)


def test_tool_batch_runs_the_new_tool_for_an_old_name():
    reg = _registry("AppStatus")
    out = asyncio.run(
        ToolBatchTool().execute({"tool": "ArtifactStatus", "inputs": [{"slug": "a"}, {"slug": "b"}]}, _ctx(reg))
    )
    assert not out.is_error, out.content
    assert "AppStatus:a" in str(out.content) and "AppStatus:b" in str(out.content)
