"""실행기가 사용자 기기 접속(host.user_pc)을 따르는가 — 표면·턴 안내·기록 정리.

호스트가 ``user_pc_connection`` 훅으로 접속을 주면, 연결 폴더는 UserPc 하나로 닿는다. 기기의 폴더
도구(파일·셸·복사·열기·옛 입구)는 모델에게 보이지 않고, 폴더와 무관한 기기 도구(브라우저 등)는 남는다.
훅이 없는 호스트는 예전 규칙 그대로다(tests/test_host_local_folders.py).
"""

from __future__ import annotations

from typing import Any, Dict, List

from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.host import local_folders as lf
from xgen_agent_runtime.host import user_pc as up
from xgen_agent_runtime.stages.s10_tool import second_machine as sm

from tests.test_host_local_folders import _DeviceHost, _state, capture  # noqa: F401
from tests.test_host_turn_executor_gates import _registry_names, _run

_CATALOG = [
    "mcp_local_LocalControl",
    "mcp_local_ReadFile",
    "mcp_local_Shell",
    "mcp_local_CopyToWorkspace",
    "mcp_local_Open",
    "mcp_local_BrowserNavigate",
]


async def _never(folder, action, args):  # pragma: no cover — 표면 시험에서는 부르지 않는다
    return {"ok": True}


class _PcHost(_DeviceHost):
    def __init__(self, folders: List[up.PcFolder], **kw: Any) -> None:
        super().__init__(catalog=_CATALOG, **kw)
        self._pc_folders = folders

    def user_pc_connection(self):
        return up.UserPcConnection(folders=list(self._pc_folders), call=_never)


_MAC = up.PcFolder(
    name="proj", path="/Users/me/proj", device_id="d1", device_name="MacBook",
    platform="darwin", shell="bash",
)
_WIN = up.PcFolder(
    name="docs", path="C:\\Docs", device_id="d2", device_name="Office",
    platform="win32", shell="Git Bash",
)


def test_connected_folders_give_one_tool_and_hide_folder_tools(capture) -> None:  # noqa: F811
    _run(_PcHost([_MAC, _WIN]), capture, client_surface="connector",
         local_folders=[{"path": "/Users/me/proj"}])
    registry = capture["registry"]
    names = registry.list_names()
    assert "UserPc" in names and registry.is_core("UserPc")
    for hidden in ("mcp_local_ReadFile", "mcp_local_Shell", "mcp_local_CopyToWorkspace",
                   "mcp_local_Open", "mcp_local_LocalControl"):
        assert hidden not in names
    assert "mcp_local_BrowserNavigate" in names
    state = _state(capture)
    note = "\n".join(state.shared[SharedKeys.TURN_NOTES])
    assert '"proj": on "MacBook" (macOS, bash)' in note
    assert '"docs": on "Office" (Windows, Git Bash)' in note
    assert "Two machines" not in note
    retired = state.shared[SharedKeys.RETIRED_TOOL_CALLS]
    assert "mcp_local_Shell" in retired["names"] and "UserPc" not in retired["names"]
    assert state.shared[lf.SHARED_FOLDERS_KEY]["user_pc"] is True


def test_no_connected_folder_means_no_tool_a_fact_and_old_calls_retired(capture) -> None:  # noqa: F811
    _run(_PcHost([]), capture, client_surface="connector", local_folders=[])
    names = _registry_names(capture)
    assert "UserPc" not in names
    assert "mcp_local_Shell" not in names
    state = _state(capture)
    note = "\n".join(state.shared[SharedKeys.TURN_NOTES])
    assert "No folder on the user's devices is connected" in note
    assert "UserPc" in state.shared[SharedKeys.RETIRED_TOOL_CALLS]["names"]


def test_a_web_turn_without_folders_gets_no_note(capture) -> None:  # noqa: F811
    _run(_PcHost([]), capture, client_surface="web", local_folders=None)
    assert SharedKeys.TURN_NOTES not in _state(capture).shared


def test_cli_backend_gets_the_same_tool_on_its_surface(capture) -> None:  # noqa: F811
    host = _PcHost([_WIN])
    _run(host, capture, provider="claude_code", client_surface="connector",
         local_folders=[{"path": "C:\\Docs"}])
    surface = host.cli_params["_tool_surface"]
    assert surface.registry.is_core("UserPc")
    assert "mcp_local_ReadFile" not in surface.registry.list_names()


def test_a_host_whose_hook_fails_falls_back_to_the_old_rules(capture) -> None:  # noqa: F811
    class _Broken(_DeviceHost):
        def user_pc_connection(self):
            raise RuntimeError("db down")

    _run(_Broken(catalog=_CATALOG, platform="darwin"), capture, client_surface="connector",
         local_folders=[{"name": "proj", "path": "/Users/me/proj"}])
    names = _registry_names(capture)
    assert "UserPc" not in names
    assert "mcp_local_ReadFile" in names  # 예전 규칙: 폴더 도구가 첫 화면에


# ── sandbox 도구가 연결 폴더의 경로를 받았을 때 ────────────────────────


def _call(tool: str, inp: Dict[str, Any], text: str):
    calls = [{"tool_use_id": "t1", "tool_name": tool, "tool_input": inp}]
    results = [{"tool_use_id": "t1", "content": text}]
    return calls, results


def test_a_sandbox_tool_on_a_device_path_is_told_which_folder_it_is() -> None:
    shared = {lf.SHARED_FOLDERS_KEY: up.shared_folder_facts([_MAC])}
    calls, results = _call("Read", {"file_path": "/Users/me/proj/a.txt"}, "File not found")
    assert sm.annotate(calls, results, ["Read", "UserPc"], shared) == 1
    text = results[0]["content"]
    assert 'inside the connected folder "proj" on the user\'s device "MacBook"' in text
    assert 'UserPc with folder="proj"' in text


def test_a_plain_not_found_in_the_sandbox_is_left_alone() -> None:
    """"없음" 결과에 기기를 보라고 미는 안내는 이 경로에 없다."""
    shared = {lf.SHARED_FOLDERS_KEY: up.shared_folder_facts([_MAC])}
    calls, results = _call("Glob", {"pattern": "**/x.pptx"}, "No files matching")
    assert sm.annotate(calls, results, ["Glob", "UserPc"], shared) == 0
    assert results[0]["content"] == "No files matching"
