"""실행기가 대화의 연결 폴더(local_folders)를 따르는가 — 표면·턴 안내·기록 정리.

기기 도구는 호스트가 앱 카탈로그에서 만들어 준다(build_connector_mcp_tools). 실행기는
host.local_folders 규칙으로 이번 턴에 보일 것을 거르고, 이번 턴의 상태를 턴 안내에 쓰고,
폴더 도구가 없으면 기록 속 옛 호출을 평문으로 바꾸라고 Stage 6 에 알린다.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.host import local_folders as lf
from xgen_agent_runtime.host import runner as runner_mod
from xgen_agent_runtime.tools.base import Tool, ToolResult

from tests.test_host_turn_executor_gates import _FakeHost, _registry_names, _run


@pytest.fixture
def capture(monkeypatch):
    """build_pipeline 인자(registry)와 스트림에 넘어간 state 를 잡는다."""
    seen: Dict[str, Any] = {}

    def _fake_build_pipeline(**kw):
        seen.update(kw)
        return object()

    def _fake_stream_turn(*args, **kwargs):
        seen["stream_input"] = args[1]
        seen["stream_state"] = args[2]
        return iter([])

    monkeypatch.setattr(runner_mod, "build_pipeline", _fake_build_pipeline)
    monkeypatch.setattr(runner_mod, "stream_turn", _fake_stream_turn)
    return seen


class _DeviceTool(Tool):
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"device tool {self._name}"

    async def execute(self, input, context):  # noqa: A002
        return ToolResult(content="ok")


_CATALOG = [
    "mcp_local_LocalControl",
    "mcp_local_ReadFile",
    "mcp_local_Shell",
    "mcp_local_BrowserNavigate",
]


class _DeviceHost(_FakeHost):
    def __init__(self, catalog: List[str] = _CATALOG, platform: str = "", **kw: Any) -> None:
        super().__init__(**kw)
        self._catalog = catalog
        self._platform = platform

    def local_device_platform(self) -> str:
        return self._platform

    def build_connector_mcp_tools(self, *a, **k):
        return [_DeviceTool(n) for n in self._catalog]


def _state(capture: Dict[str, Any]):
    return capture["stream_state"]


def test_old_client_keeps_every_device_tool_and_gets_no_note(capture) -> None:
    _run(_DeviceHost(), capture, client_surface="connector")
    names = _registry_names(capture)
    assert set(_CATALOG) <= set(names)
    state = _state(capture)
    assert SharedKeys.TURN_NOTES not in state.shared
    assert SharedKeys.RETIRED_TOOL_CALLS not in state.shared


def test_no_folder_hides_folder_tools_says_so_and_retires_old_calls(capture) -> None:
    _run(_DeviceHost(), capture, client_surface="connector", local_folders=[])
    names = _registry_names(capture)
    assert "mcp_local_ReadFile" not in names
    assert "mcp_local_Shell" not in names
    assert "mcp_local_LocalControl" not in names
    assert "mcp_local_BrowserNavigate" in names  # 브라우저는 자기 설정이 정한다
    state = _state(capture)
    assert "No folder on the user's device" in "\n".join(state.shared[SharedKeys.TURN_NOTES])
    assert state.shared[SharedKeys.RETIRED_TOOL_CALLS] == lf.retired_calls_spec()


def test_connected_folders_put_folder_tools_on_the_first_screen(capture) -> None:
    _run(
        _DeviceHost(platform="darwin"),
        capture,
        client_surface="connector",
        local_folders=[{"id": "1", "name": "proj", "path": "/Users/me/proj"}],
    )
    registry = capture["registry"]
    names = registry.list_names()
    assert "mcp_local_LocalControl" not in names
    assert registry.is_core("mcp_local_ReadFile")
    assert registry.is_core("mcp_local_Shell")
    state = _state(capture)
    note = "\n".join(state.shared[SharedKeys.TURN_NOTES])
    assert "- proj: /Users/me/proj" in note
    assert "user's Mac" in note
    assert SharedKeys.RETIRED_TOOL_CALLS not in state.shared


def test_connected_but_app_offline_is_told_and_old_calls_retired(capture) -> None:
    _run(
        _DeviceHost(catalog=[]),
        capture,
        client_surface="connector",
        local_folders=[{"path": "/Users/me/proj"}],
    )
    state = _state(capture)
    assert "not reachable right now" in "\n".join(state.shared[SharedKeys.TURN_NOTES])
    assert state.shared[SharedKeys.RETIRED_TOOL_CALLS] == lf.retired_calls_spec()


def test_cli_backend_gets_the_same_note_from_the_same_catalog(capture) -> None:
    host = _DeviceHost(platform="win32")
    _run(
        host,
        capture,
        provider="claude_code",
        client_surface="connector",
        local_folders=[{"path": "C:\\work\\docs"}],
    )
    # CLI 는 파이프라인 registry 를 쓰지 않는다 — 같은 레지스트리가 표면 객체로 브릿지에 간다.
    assert capture.get("registry") is None
    surface = host.cli_params["_tool_surface"]
    assert surface.registry.is_core("mcp_local_ReadFile")
    assert "mcp_local_LocalControl" not in surface.registry.list_names()
    note = "\n".join(_state(capture).shared[SharedKeys.TURN_NOTES])
    assert "- docs: C:\\work\\docs" in note
    assert "Windows PC" in note
    assert "Shell and ShellJob" in note


def test_a_host_without_the_platform_hook_says_device(capture) -> None:
    class _NoPlatformHost(_FakeHost):  # 훅이 없는 호스트
        def build_connector_mcp_tools(self, *a, **k):
            return [_DeviceTool("mcp_local_ReadFile")]

    _run(
        _NoPlatformHost(),
        capture,
        client_surface="connector",
        local_folders=[{"path": "/p"}],
    )
    assert "user's device" in "\n".join(_state(capture).shared[SharedKeys.TURN_NOTES])


def test_mobile_catalog_follows_the_mobile_folder_set(capture) -> None:
    catalog = ["mcp_mobile_ReadFile", "mcp_mobile_Notify", "mcp_mobile_Location"]
    _run(_DeviceHost(catalog=catalog), capture, client_surface="connector", local_folders=[])
    names = _registry_names(capture)
    assert "mcp_mobile_ReadFile" not in names
    assert {"mcp_mobile_Notify", "mcp_mobile_Location"} <= set(names)


class _FolderDeviceHost(_DeviceHost):
    """폴더가 다른 기기(사무실 PC)에 있다 — 웹에서 보낸 턴."""

    def folder_device_info(self) -> Dict[str, Any]:
        return {"name": "사무실 PC", "platform": "win32", "online": True, "remote": True}


def test_a_turn_from_another_screen_names_the_folder_pc(capture) -> None:
    _run(
        _FolderDeviceHost(catalog=["mcp_local_ReadFile", "mcp_local_Shell"], platform="win32"),
        capture,
        local_folders=[{"id": "1", "name": "report", "path": "C:\\report"}],
    )
    note = "\n".join(_state(capture).shared[SharedKeys.TURN_NOTES])
    assert 'Windows PC "사무실 PC"' in note
    assert "another screen" in note
    assert "- report: C:\\report" in note


def test_copy_tools_are_folder_tools_and_the_note_points_to_them() -> None:
    """기기 폴더 ↔ sandbox 복사 도구는 폴더 도구다(폴더가 있을 때만 보이고, 폴더 기기로 간다).

    턴 안내는 옮기는 길을 가리킨다 — 글만 읽는 ReadFile 로는 문서·그림을 sandbox 로 가져올 수 없었다.
    """
    folders = [lf.LocalFolder(id="1", name="KakaoTalk", path="/KakaoTalk")]
    assert lf.is_folder_tool("mcp_mobile_CopyToWorkspace")
    assert lf.is_folder_tool("mcp_local_CopyFromWorkspace")
    assert lf.is_folder_tool("mcp__connector__mcp_local_CopyToWorkspace")
    assert lf.is_folder_tool("mcp_web_CopyToWorkspace")  # 웹 [폴더] 도 같은 두 도구를 올린다
    assert lf.is_folder_tool("mcp_web_CopyFromWorkspace")
    note = lf.turn_note(
        folders,
        available_tools=["mcp_mobile_ReadFile", "mcp_mobile_CopyToWorkspace", "mcp_mobile_CopyFromWorkspace"],
        platform="android",
    )
    assert "copy them into your workspace with mcp_mobile_CopyToWorkspace" in note
    assert "use mcp_mobile_CopyFromWorkspace" in note
    old_app = lf.turn_note(folders, available_tools=["mcp_mobile_ReadFile"], platform="android")
    assert "read it with one side's tool" in old_app
    assert "mcp_local_CopyToWorkspace" in lf.retired_device_tool_names()


def test_web_folder_note_points_to_the_copy_tools() -> None:
    folders = [lf.LocalFolder(id="1", name="docs", path="/docs")]
    note = lf.turn_note(
        folders,
        available_tools=["mcp_web_ReadFile", "mcp_web_CopyToWorkspace", "mcp_web_CopyFromWorkspace"],
        platform="web",
    )
    assert "copy them into your workspace with mcp_web_CopyToWorkspace" in note
    assert "use mcp_web_CopyFromWorkspace" in note
    assert "This device has no terminal; use the file tools." in note


def test_wrong_machine_note_names_the_copy_tool() -> None:
    from xgen_agent_runtime.stages.s10_tool.second_machine import device_file_tools

    tools = device_file_tools(["mcp_local_ReadFile", "mcp_local_CopyToWorkspace", "Read"])
    assert tools == "mcp_local_ReadFile / mcp_local_CopyToWorkspace"
