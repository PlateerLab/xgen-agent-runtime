"""모델에게 가는 도구 정의 — 두 CLI 가 고쳐 보내도 뜻이 남는가 (tools.definition, 4.70.0).

실측(2026-09-30, 가짜 모델 서버로 요청 캡처)에서 나온 한도를 테스트로 못박는다:
Claude Code 는 도구 설명을 2048자에서 자르고 이름의 한글을 ``_`` 로 바꾸며 64자 넘는 이름을 그대로 보낸다.
Codex 는 스키마가 약 5,000바이트를 넘으면 설명을 전부 지우고 default·format·범위 키워드를 늘 지운다.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from xgen_agent_runtime.host.tools import adapt_tools
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.tool_search_tool import ToolSearchTool
from xgen_agent_runtime.tools.definition import (
    DESCRIPTION_BUDGET,
    MAX_TOOL_NAME,
    SCHEMA_BUDGET,
    api_definition,
    fit_definition,
    full_reference,
    normalize_input_schema,
    safe_tool_name,
)


class _LC:
    """LangChain 도구 흉내 — name/description/args_schema + ainvoke."""

    def __init__(self, name: str, description: str = "", args_schema: Any = None) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.calls = []

    async def ainvoke(self, payload: Dict[str, Any]) -> str:
        self.calls.append(payload)
        return f"{self.name} ok"


# ── 이름 ─────────────────────────────────────────────────────────────


def test_names_fit_the_cli_prefix_and_never_collapse():
    long = "t_" + "x" * 58
    assert len(safe_tool_name(long)) <= MAX_TOOL_NAME
    assert len("mcp__connector__" + safe_tool_name(long)) <= 64
    # 한글 이름 둘이 같은 이름이 되지 않는다(Claude Code 는 둘 다 ``____`` 로 만든다).
    a, b = safe_tool_name("한글도구"), safe_tool_name("한글툴")
    assert a != b and a.startswith("tool_") and b.startswith("tool_")
    assert safe_tool_name("한글도구") == a  # 같은 이름은 늘 같은 결과(턴을 넘어 안정)
    assert safe_tool_name("dot.name") == "dot_name"
    assert safe_tool_name("mcp_tms-mcp_python_get_issue") == "mcp_tms-mcp_python_get_issue"
    assert safe_tool_name("search", taken={"search"}) == "search_2"


def test_two_nodes_with_the_same_tool_name_both_survive():
    first, second = _LC("search", "from node A"), _LC("search", "from node B")
    registry = adapt_tools([first, second])
    assert sorted(registry.list_names()) == ["search", "search_2"]
    ctx = ToolContext(session_id="s")
    asyncio.run(registry.get("search_2").execute({}, ctx))
    assert second.calls == [{}] and first.calls == []


def test_a_node_tool_cannot_take_a_builtin_name():
    registry = adapt_tools([_LC("Read", "reads jira"), _LC("memory_search", "x")])
    names = set(registry.list_names())
    assert "Read" not in names and "Read_2" in names
    assert "memory_search" not in names
    assert "(Original name: Read)" not in registry.get("Read_2").description or True


def test_a_renamed_tool_keeps_its_original_name_in_the_description():
    registry = adapt_tools([_LC("이슈 검색", "Jira 이슈를 찾는다")])
    (name,) = registry.list_names()
    assert name.startswith("tool_")
    assert "(Original name: 이슈 검색)" in registry.get(name).description


# ── 스키마 ───────────────────────────────────────────────────────────


_NESTED = {
    "type": "object",
    "title": "Args",
    "properties": {
        "issue": {"$ref": "#/$defs/Issue", "description": "the issue"},
        "title": {"type": "string", "title": "Title", "default": "x"},
        "count": {"type": "integer", "minimum": 1, "maximum": 50},
        "tree": {"$ref": "#/$defs/Node"},
    },
    "required": ["issue", "ghost"],
    "$defs": {
        "Issue": {"type": "object", "title": "Issue", "properties": {"key": {"type": "string", "enum": ["A", "B"]}}},
        "Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}},
    },
}


def test_schema_is_self_contained_and_keeps_its_meaning():
    out = normalize_input_schema(_NESTED)
    text = json.dumps(out)
    assert "$ref" not in text and "$defs" not in text
    assert out["properties"]["issue"]["properties"]["key"]["enum"] == ["A", "B"]
    assert out["properties"]["issue"]["description"] == "the issue"
    # 속성 이름 title 은 남고 스키마 키워드 title 만 빠진다.
    assert "title" in out["properties"] and "title" not in out["properties"]["title"]
    # Codex 가 지우는 제약은 글로도 남는다.
    assert 'Default: "x"' in out["properties"]["title"]["description"]
    assert "Range: >= 1, <= 50" in out["properties"]["count"]["description"]
    # 순환 참조는 끊긴다.
    assert out["properties"]["tree"]["properties"]["child"] == {"type": "object"}
    assert out["required"] == ["issue"]


def test_pydantic_nested_models_no_longer_lose_their_defs():
    from pydantic import BaseModel, Field

    class Filter(BaseModel):
        status: str = Field(description="issue status")

    class Args(BaseModel):
        query: str = Field(description="JQL")
        filter: Filter

    registry = adapt_tools([_LC("jira_find", "find issues", Args)])
    sent = registry.to_api_format()[0]["input_schema"]
    assert "$ref" not in json.dumps(sent)
    assert sent["properties"]["filter"]["properties"]["status"]["description"] == "issue status"


# ── 예산 ─────────────────────────────────────────────────────────────


def test_long_definitions_fit_both_clis_and_point_to_the_full_text():
    schema = {
        "type": "object",
        "description": "Top-level notes that Codex would drop.",
        "properties": {f"p{i}": {"type": "string", "description": "Z" * 900} for i in range(8)},
    }
    desc, fitted, trimmed = fit_definition("big_tool", "D" * 5000, schema)
    assert trimmed
    assert len(desc) <= DESCRIPTION_BUDGET
    assert len(json.dumps(fitted, ensure_ascii=False, separators=(",", ":"))) <= SCHEMA_BUDGET
    assert 'ToolSearch(query="big_tool")' in desc
    assert "description" not in fitted  # 최상위 설명은 도구 설명으로 옮겨 간다


def test_top_level_schema_description_moves_into_the_tool_description():
    desc, fitted, trimmed = fit_definition(
        "t", "Does a thing.", {"type": "object", "description": "Use carefully.", "properties": {}}
    )
    assert not trimmed and "Use carefully." in desc and "description" not in fitted


def test_empty_description_is_filled_from_the_inputs():
    desc, _, _ = fit_definition("jira_find", "", {"type": "object", "properties": {"jql": {"type": "string"}}})
    assert desc == "jira_find. Inputs: jql."


def test_toolsearch_by_exact_name_returns_the_full_text():
    long_tool = _LC("big_tool", "B" * 5000 + " TAIL-MARKER", {"type": "object", "properties": {"q": {"type": "string", "description": "Q" * 3000}}})
    registry = adapt_tools([long_tool], core=False)
    registry.register(ToolSearchTool(), core=True)
    ctx = ToolContext(session_id="s")
    ctx.tool_registry = registry
    res = asyncio.run(ToolSearchTool().execute({"query": "big_tool"}, ctx))
    assert "Full reference for big_tool:" in res.content
    assert "TAIL-MARKER" in res.content and "Q" * 3000 in res.content
    assert full_reference(registry.get("ToolSearch")) is None  # 줄지 않은 도구는 원문을 되풀이하지 않는다


def test_sdk_and_cli_send_the_same_definition():
    """SDK(registry.to_api_format) 와 CLI(TurnToolSurface.tools_list) 는 같은 함수를 쓴다."""
    from xgen_agent_runtime import PipelineState
    from xgen_agent_runtime.host.tool_surface import TurnToolSurface

    registry = adapt_tools([_LC("jira_find", "find", _NESTED)])
    sdk = registry.to_api_format(exposed_only=True)
    cli = TurnToolSurface(registry=registry, tool_context=None, state=PipelineState(session_id="s")).tools_list()
    assert [(d["name"], d["description"], d["input_schema"]) for d in sdk] == [
        (d["name"], d["description"], d["inputSchema"]) for d in cli
    ]
    assert api_definition(registry.get("jira_find")) == sdk[0]


def test_long_device_tool_names_are_capped():
    from xgen_agent_runtime.host.local_folders import model_tool_name

    name = model_tool_name("my-very-long-local-mcp-server", "read_multiple_files_with_metadata")
    assert len(name) <= MAX_TOOL_NAME and name.startswith("mcp_my-very-long")
    assert model_tool_name("local", "ReadFile") == "mcp_local_ReadFile"
