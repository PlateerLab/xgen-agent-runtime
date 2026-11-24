import asyncio

import pytest

from xgen_agent_runtime.core.steer import SteerInbox
from xgen_agent_runtime.tools.base import ToolCapabilities, ToolContext
from xgen_agent_runtime.tools.built_in.tool_search_tool import ToolSearchTool
from xgen_agent_runtime.tools.registry import ToolRegistry
from xgen_agent_runtime.tools.base import build_tool


def test_steer_retry_returns_original_terminal_outcome():
    inbox = SteerInbox()
    message = inbox.offer("정정", client_msg_id="1", meta={"user_id": "first"})
    assert message
    assert inbox.offer("정정", client_msg_id="1", meta={"user_id": "first"}) is message
    assert inbox.take() == [message]
    inbox.resolve(message, True)
    other = inbox.offer("같은 ID", client_msg_id="1", meta={"user_id": "other"})
    assert other is not message
    inbox.close()
    assert inbox.offer("정정", client_msg_id="1", meta={"user_id": "first"}).wait(0) is True


@pytest.mark.parametrize("count", [10, 100, 500])
def test_discover_then_activate_in_large_catalog(count):
    registry = ToolRegistry()
    for i in range(count):
        name = f"SearchDrive{i}"
        registry.register(build_tool(name=name, description="Search Google Drive documents",
                                     input_schema={"type": "object"}, execute=lambda *_: None), core=False)
    context = ToolContext(tool_registry=registry)
    search = ToolSearchTool()
    result = asyncio.run(search.execute({"query": "구글 드라이브 문서 검색"}, context))
    assert len(result.metadata["results"]) == 10
    assert result.metadata["activated_schema_bytes"] == 0
    assert not registry.is_exposed("SearchDrive0")
    result = asyncio.run(search.execute({"query": "SearchDrive0"}, context))
    assert result.metadata["activated"] == ["SearchDrive0"]
    assert result.metadata["activated_schema_bytes"] > 0


@pytest.mark.parametrize("traits", [
    {"concurrency_safe": True}, {"read_only": True},
    {"read_only": True, "concurrency_safe": True, "mandatory_approval": True},
    {"read_only": True, "concurrency_safe": True, "destructive": True},
])
def test_only_independent_reads_can_overlap(traits):
    assert not ToolCapabilities(**traits).parallel_read


@pytest.mark.parametrize(("extension", "part"), [
    ("docx", "word/document.xml"), ("xlsx", "xl/workbook.xml"), ("pptx", "ppt/presentation.xml")
])
def test_office_deliverable_requires_its_main_part(extension, part):
    import io
    import zipfile
    from xgen_agent_runtime.stages.s16_loop.completion_review import describe_file, is_problem
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(part, "<document/>")
    assert not is_problem(describe_file("결과." + extension, buf.getvalue()))
    assert is_problem(describe_file("결과." + extension, b"incorrect file contents"))

@pytest.mark.parametrize(("query","expected"), [
    ("list documents", ["ListA", "ListB"]),
    ("ListB", ["ListB"]),
    ("ListA, ListB", ["ListA", "ListB"]),
    ("list nonexistent", []),
    ("문서", ["LocalDocs"]),
    ("*", []),
])
async def test_native_search_keeps_existing_activation(query, expected):
    registry = ToolRegistry()
    for name, description in (("ListA","list documents"), ("ListB","list documents"), ("LocalDocs","문서")):
        registry.register(build_tool(name=name, description=description, input_schema={"type":"object"}, execute=lambda *_: None), core=False)
    search = ToolSearchTool()
    result = await search.execute({"query":query}, ToolContext(tool_registry=registry))
    assert result.metadata.get("activated", []) == expected


def test_compaction_role_uses_host_approved_model_only_for_summary():
    from xgen_agent_runtime.core.state import PipelineState
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import LLMSummaryCompactor
    state = PipelineState(session_id="a", model="parent")
    for selected, expected in [(None, "parent"), ("small-approved", "small-approved")]:
        resolver, enabled = LLMSummaryCompactor(summary_model=selected)._resolve(state)
        assert enabled and resolver(state).model == expected
        assert state.model == "parent"
