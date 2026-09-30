"""CLI 백엔드의 도구 표면 = SDK 표면 (4.70.0).

CLI(claude_code·codex)는 도구를 MCP 로만 받는다. 호스트 브릿지는 턴 조립이 만든 **같은 레지스트리**를
:class:`TurnToolSurface` 로 광고·실행한다. 여기서 지키는 것:

* 광고 = SDK Stage 3 가 모델에게 보내는 목록(노출분·같은 스키마, 문 도달성·기록 복원 검사 포함).
* 실행 = SDK Stage 10 과 같은 함수(``ToolStage.dispatch_calls``) — 문이 방을 열고, 가드가 돈다.
* 실행은 턴 루프에서 — 다른 스레드의 서빙 루프에서 불러도 턴 루프로 넘긴다.
* 기기 도구 래퍼는 스키마를 그대로 주고(enum 보존), 이미지·실패·기계 혼동을 정확히 다룬다.
* codex·claude 네이티브 도구는 꺼진다(우리 표면만).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List

import pytest

from xgen_agent_runtime import PipelineState
from xgen_agent_runtime.host.device_tools import build_device_tool, to_tool_result
from xgen_agent_runtime.host.local_folders import SHARED_FOLDERS_KEY
from xgen_agent_runtime.host.tool_surface import TurnToolSurface, to_mcp_result
from xgen_agent_runtime.tools import ToolRegistry
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in.self_extend_guide_tool import (
    SELF_EXTEND_FAMILY,
    SelfExtendGuideTool,
)
from xgen_agent_runtime.tools.catalog import deferred_catalog_text


class _Echo(Tool):
    def __init__(self, name: str, *, schema: Dict[str, Any] | None = None) -> None:
        self._name = name
        self._schema = schema or {
            "type": "object",
            "properties": {"text": {"type": "string", "enum": ["a", "b"]}},
        }
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} does one thing."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._schema

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:  # noqa: A002
        self.calls.append(dict(input))
        return ToolResult(content=f"{self._name}:{input.get('text', '')}")


def _surface(registry: ToolRegistry, *, state: PipelineState | None = None, working_dir: str = "/work/sb"):
    ctx = ToolContext(session_id="s1", working_dir=working_dir, allowed_paths=[working_dir])
    return TurnToolSurface(registry=registry, tool_context=ctx, state=state or PipelineState(session_id="s1"))


class _TurnLoop:
    """턴 루프 흉내 — 별도 스레드의 이벤트 루프(러너가 on_loop 로 알려 주는 것)."""

    def __init__(self) -> None:
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


# ── 광고 ────────────────────────────────────────────────────────────


def test_tools_list_is_what_stage3_sends():
    registry = ToolRegistry()
    registry.register(_Echo("Read"), core=True)
    registry.register(_Echo("jira_search"), core=False)
    surface = _surface(registry)
    listed = surface.tools_list()
    sent = registry.to_api_format(exposed_only=True)
    assert [t["name"] for t in listed] == [t["name"] for t in sent] == ["Read"]
    assert listed[0]["inputSchema"] == sent[0]["input_schema"]
    assert listed[0]["inputSchema"]["properties"]["text"]["enum"] == ["a", "b"]
    assert listed[0]["description"] == sent[0]["description"]


def test_a_hidden_family_without_a_visible_gate_is_opened_like_stage3():
    """Stage 3 의 문 도달성 검사를 CLI 표면도 한다 — 숨긴 가족에 보이는 문이 없으면 연다."""
    registry = ToolRegistry()
    registry.register(_Echo("Read"), core=True)
    registry.register(_Echo("JobSchedule"), core=False)  # 문(JobGuide)이 없다
    surface = _surface(registry)
    assert "JobSchedule" in [t["name"] for t in surface.tools_list()]


# ── 실행 ────────────────────────────────────────────────────────────


def test_call_without_a_running_turn_is_an_error():
    registry = ToolRegistry()
    registry.register(_Echo("Read"), core=True)
    surface = _surface(registry)
    out = asyncio.run(surface.call("Read", {"text": "a"}))
    assert out["isError"] and "no longer running" in out["content"][0]["text"]


def test_call_runs_on_the_turn_loop_through_stage10(turn_loop):
    registry = ToolRegistry()
    tool = _Echo("Read")
    registry.register(tool, core=True)
    surface = _surface(registry)
    surface.bind_loop(turn_loop.loop)

    seen_loops: List[Any] = []
    orig = tool.execute

    async def _spy(input, context):  # noqa: A002
        seen_loops.append(asyncio.get_running_loop())
        return await orig(input, context)

    tool.execute = _spy  # type: ignore[method-assign]
    out = asyncio.run(surface.call("Read", {"text": "a"}))  # 서빙 루프(다른 스레드)
    assert out == {"content": [{"type": "text", "text": "Read:a"}], "isError": False}
    assert seen_loops == [turn_loop.loop]


def test_a_gate_called_over_the_bridge_opens_its_family(turn_loop):
    registry = ToolRegistry()
    registry.register(SelfExtendGuideTool(), core=True)
    for name in SELF_EXTEND_FAMILY:
        registry.register(_Echo(name), core=False)
    surface = _surface(registry)
    surface.bind_loop(turn_loop.loop)
    before = set(surface.exposed_names())
    assert not before & set(SELF_EXTEND_FAMILY)
    out = asyncio.run(surface.call("SelfExtendGuide", {}))
    assert not out["isError"]
    after = set(surface.exposed_names())
    assert set(SELF_EXTEND_FAMILY) <= after  # list_changed 가 이 차이를 알린다


def test_repeated_identical_failures_are_blocked_like_sdk(turn_loop):
    """반복 실패 가드는 Stage 10 의 것 — CLI 에서도 같은 호출을 무한히 돌리지 않는다."""

    class _Fail(_Echo):
        async def execute(self, input, context):  # noqa: A002
            self.calls.append(dict(input))
            return ToolResult(content="Error: boom", is_error=True)

    registry = ToolRegistry()
    tool = _Fail("Flaky")
    registry.register(tool, core=True)
    surface = _surface(registry)
    surface.bind_loop(turn_loop.loop)
    results = [asyncio.run(surface.call("Flaky", {"text": "a"})) for _ in range(6)]
    assert all(r["isError"] for r in results)
    assert len(tool.calls) < 6, "같은 실패를 막지 않고 매번 실행했다"


def test_a_sandbox_tool_given_a_device_path_is_told_so(turn_loop):
    """기계 혼동 안내(second_machine)도 같은 함수 안에 있다 — CLI 에서도 붙는다."""
    registry = ToolRegistry()
    registry.register(_Echo("Read", schema={"type": "object", "properties": {"file_path": {"type": "string"}}}), core=True)
    registry.register(_Echo("mcp_local_ReadFile"), core=True)
    registry.register(_Echo("mcp_local_ListDir"), core=True)
    state = PipelineState(session_id="s1")
    state.shared[SHARED_FOLDERS_KEY] = {
        "device": "Mac",
        "folders": [{"name": "proj", "path": "/Users/me/proj"}],
    }
    surface = _surface(registry, state=state)
    surface.bind_loop(turn_loop.loop)
    out = asyncio.run(surface.call("Read", {"file_path": "/Users/me/proj/a.md"}))
    text = "\n".join(b["text"] for b in out["content"])
    assert "[Wrong machine]" in text and "mcp_local_ReadFile" in text


def test_to_mcp_result_keeps_images():
    out = to_mcp_result(
        {
            "content": [
                {"type": "text", "text": "shot"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}},
            ],
            "is_error": False,
        }
    )
    assert out["content"][1] == {"type": "image", "data": "AAA", "mimeType": "image/jpeg"}


# ── 숨김 목록 ────────────────────────────────────────────────────────


def test_catalog_groups_members_under_their_visible_gate():
    registry = ToolRegistry()
    registry.register(_Echo("JobGuide"), core=True)
    registry.register(_Echo("JobSchedule"), core=False)
    registry.register(_Echo("JobList"), core=False)
    registry.register(_Echo("db_query_orders"), core=False)
    text = deferred_catalog_text(registry)
    assert "JobGuide opens: " in text
    assert "JobSchedule" in text and "JobList" in text
    assert "db_query_orders" in text


# ── 기기 도구 ────────────────────────────────────────────────────────


def _device(tool: str, calls: List[Any], payload: Dict[str, Any] | None = None, *, server: str = "local", schema=None):
    async def _call(raw, args):
        calls.append((raw, dict(args)))
        return payload or {"ok": True, "result": {"content": [{"type": "text", "text": "done"}]}}

    return build_device_tool(
        server=server,
        tool=tool,
        description=f"{tool} on the device",
        input_schema=schema
        or {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "close", "activate"]},
                "tabs": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["action"],
        },
        call=_call,
    )


def test_device_tool_keeps_the_schema_as_sent():
    tool = _device("BrowserTabs", [])
    schema = tool.to_api_format()["input_schema"]
    assert schema["properties"]["action"]["enum"] == ["list", "close", "activate"]
    assert schema["properties"]["tabs"]["items"] == {"type": "integer"}
    assert schema["required"] == ["action"]


def test_device_search_is_searchfiles_for_the_model_and_search_for_the_device():
    calls: List[Any] = []
    tool = _device("Search", calls, schema={"type": "object", "properties": {"query": {"type": "string"}}})
    assert tool.name == "mcp_local_SearchFiles"
    ctx = ToolContext(session_id="s", working_dir="/work/sb")
    res = asyncio.run(tool.execute({"query": "x"}, ctx))
    assert not res.is_error
    assert calls == [("Search", {"query": "x"})]


def test_device_tool_refuses_a_sandbox_path_before_calling_the_device():
    calls: List[Any] = []
    tool = _device("ReadFile", calls, schema={"type": "object", "properties": {"path": {"type": "string"}}})
    ctx = ToolContext(session_id="s", working_dir="/work/sandbox-1")
    res = asyncio.run(tool.execute({"path": "/work/sandbox-1/out.txt"}, ctx))
    assert res.is_error and "[Wrong machine]" in res.content
    assert calls == []


def test_device_results_keep_images_and_errors():
    img = to_tool_result(
        "mcp_local_Screenshot",
        {"ok": True, "result": {"content": [{"type": "image", "data": "QUJD", "mimeType": "image/png"}]}},
    )
    assert isinstance(img.content, list) and img.content[0]["source"]["data"] == "QUJD"
    err = to_tool_result(
        "mcp_local_Shell",
        {"ok": True, "result": {"isError": True, "content": [{"type": "text", "text": "exit 1"}]}},
    )
    assert err.is_error and "exit 1" in err.content
    down = to_tool_result("mcp_local_Shell", {"ok": False, "error": "device offline"})
    assert down.is_error and "device offline" in down.content


# ── CLI 네이티브 도구 끄기 ─────────────────────────────────────────────


def test_codex_host_only_turns_native_tools_off_and_replaces_instructions():
    from xgen_agent_runtime.llm_client.translators._codex import (
        CODEX_NATIVE_FEATURES_OFF,
        codex_argv,
        codex_mcp_overrides,
    )
    from xgen_agent_runtime.llm_client.types import APIRequest

    req = APIRequest(model="gpt-5", messages=[{"role": "user", "content": "hi"}], system="SYS")
    argv = codex_argv(req, host_tools_only=True, instructions_path="/tmp/i.md")
    joined = " ".join(argv)
    for feature in ("shell_tool", "unified_exec", "multi_agent", "view_image", "goals"):
        assert feature in CODEX_NATIVE_FEATURES_OFF
        assert f"features.{feature}=false" in joined
    assert 'web_search="disabled"' in joined
    assert 'model_instructions_file="/tmp/i.md"' in joined
    plain = " ".join(codex_argv(req))
    assert "features.shell_tool=false" not in plain  # 기본 경로는 그대로
    overrides = " ".join(
        codex_mcp_overrides({"mcpServers": {"connector": {"command": "python", "args": ["b.py"]}}})
    )
    assert "mcp_servers.connector.tool_timeout_sec=3600.0" in overrides


def test_codex_host_only_client_sends_the_system_prompt_once(tmp_path):
    from xgen_agent_runtime.llm_client.codex import CodexCLIClient
    from xgen_agent_runtime.llm_client.types import APIRequest

    client = CodexCLIClient(auth_mode="api_key", api_key="k", host_tools_only=True)
    req = APIRequest(model="gpt-5", messages=[{"role": "user", "content": "hi"}], system="SYS-PROMPT")
    stdin = client._build_stdin(req)
    stdin = stdin.decode("utf-8") if isinstance(stdin, bytes) else stdin
    assert "SYS-PROMPT" not in stdin
    path = client._instructions_tempfile(req)
    try:
        assert open(path, encoding="utf-8").read() == "SYS-PROMPT"
    finally:
        import os

        os.unlink(path)
    legacy = CodexCLIClient(auth_mode="api_key", api_key="k")
    legacy_stdin = legacy._build_stdin(req)
    assert b"SYS-PROMPT" in legacy_stdin if isinstance(legacy_stdin, bytes) else "SYS-PROMPT" in legacy_stdin
    assert legacy._instructions_tempfile(req) == ""


def test_build_codex_cli_client_defaults_to_host_tools_only(monkeypatch):
    from xgen_agent_runtime.host import runner
    import xgen_agent_runtime.llm_client.codex as codex_mod

    seen: Dict[str, Any] = {}

    class _Capture:
        def __init__(self, **kw: Any) -> None:
            seen.update(kw)

    monkeypatch.setattr(codex_mod, "CodexCLIClient", _Capture)
    runner.build_codex_cli_client(auth_mode="api_key", api_key="k")
    assert seen["host_tools_only"] is True


def test_claude_code_gets_no_native_tools(monkeypatch):
    from xgen_agent_runtime.host import runner
    import xgen_agent_runtime.llm_client.claude_code as cc

    seen: Dict[str, Any] = {}

    class _Capture:
        def __init__(self, **kw: Any) -> None:
            seen.update(kw)

    monkeypatch.setattr(cc, "ClaudeCodeCLIClient", _Capture)
    runner.build_cli_client(auth_mode="api_key", api_key="k")
    extra = list(seen.get("extra_args") or [])
    i = extra.index("--tools")
    assert extra[i + 1] == ""
