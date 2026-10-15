"""사용자 기기 접속 도구 UserPc — 폴더 해석·판정·결과 모양·시한 (host.user_pc).

판정은 모델의 판단이 아니라 이 도구가 지킨다: 폴더가 여럿이면 어느 것인지 받아야 하고, 꺼진 기기·
명령을 못 받는 기기는 기다리지 않고 거절하고, 한 턴에서 응답하지 않은 기기는 다시 기다리지 않는다.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from xgen_agent_runtime.host import user_pc as up
from xgen_agent_runtime.tools.base import ToolContext


def _folder(name: str = "proj", **kw: Any) -> up.PcFolder:
    base = dict(
        name=name,
        path=f"/Users/me/{name}",
        device_id="dev-mac",
        device_name="MacBook",
        platform="darwin",
        shell="bash",
    )
    base.update(kw)
    return up.PcFolder(**base)


class _Recorder:
    def __init__(self, reply: Dict[str, Any] | None = None, delay: float = 0.0) -> None:
        self.calls: List[Tuple[str, str, Dict[str, Any]]] = []
        self.reply = reply or {"ok": True, "exit_code": 0, "stdout": "hi\n", "stderr": ""}
        self.delay = delay

    async def __call__(self, folder: up.PcFolder, action: str, args: Dict[str, Any]):
        self.calls.append((folder.name, action, dict(args)))
        if self.delay:
            await asyncio.sleep(self.delay)
        return dict(self.reply)


def _run(tool, **inp: Any):
    return asyncio.run(tool.execute(inp, ToolContext()))


# ── 폴더 해석 ─────────────────────────────────────────────────────


def test_one_folder_may_be_left_out() -> None:
    rec = _Recorder()
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=rec))
    result = _run(tool, command="ls")
    assert not result.is_error
    assert rec.calls == [("proj", "run", {"command": "ls", "cwd": "", "wait_s": 60})]
    assert result.content.startswith('[UserPc · "MacBook" · folder "proj"] exit 0')


def test_several_folders_need_a_name_and_the_error_lists_them() -> None:
    rec = _Recorder()
    folders = [_folder("a"), _folder("b", device_id="dev-win", device_name="Office", platform="win32")]
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=folders, call=rec))
    result = _run(tool, command="ls")
    assert result.is_error
    assert '"a", "b"' in result.content
    assert rec.calls == []
    ok = _run(tool, command="ls", folder="B")  # 대소문자만 다르면 그 폴더
    assert not ok.is_error
    assert rec.calls[-1][0] == "b"


def test_unknown_folder_is_refused_with_the_choices() -> None:
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=_Recorder()))
    result = _run(tool, command="ls", folder="nope")
    assert result.is_error and 'No connected folder named "nope"' in result.content


# ── 경로 ─────────────────────────────────────────────────────────


def test_paths_are_relative_to_the_folder_and_its_absolute_path_is_accepted() -> None:
    rec = _Recorder({"ok": True, "files": []})
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=rec))
    _run(tool, action="get", path="/Users/me/proj/docs/a.pptx")
    assert rec.calls[-1][2]["path"] == "docs/a.pptx"
    _run(tool, action="get", path="./docs//b.txt")
    assert rec.calls[-1][2]["path"] == "docs/b.txt"


def test_paths_outside_the_folder_are_refused_before_the_device_is_called() -> None:
    rec = _Recorder()
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=rec))
    for bad in ("../secret", "/Users/me/other/x", "a/../../x"):
        result = _run(tool, action="get", path=bad)
        assert result.is_error, bad
    assert rec.calls == []


def test_windows_folder_paths_compare_without_case() -> None:
    rec = _Recorder({"ok": True, "files": []})
    win = _folder("docs", path="C:\\Users\\Me\\Docs", platform="win32")
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[win], call=rec))
    _run(tool, action="get", path="c:\\users\\me\\docs\\Report.xlsx")
    assert rec.calls[-1][2]["path"] == "Report.xlsx"


# ── 판정 ─────────────────────────────────────────────────────────


def test_an_offline_device_is_refused_at_once() -> None:
    rec = _Recorder()
    tool = up.build_user_pc_tool(
        up.UserPcConnection(folders=[_folder(online=False)], call=rec)
    )
    result = _run(tool, command="ls")
    assert result.is_error and "not connected right now" in result.content
    assert rec.calls == []


def test_a_device_without_a_shell_still_moves_files() -> None:
    rec = _Recorder({"ok": True, "files": [{"from": "a.txt", "to": "/ws/a.txt", "size": 3}]})
    phone = _folder("Download", shell="", platform="android", device_name="Galaxy")
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[phone], call=rec))
    refused = _run(tool, command="ls")
    assert refused.is_error and "does not take commands" in refused.content
    got = _run(tool, action="get", path="a.txt")
    assert not got.is_error
    assert "a.txt -> /ws/a.txt (3 bytes)" in got.content


def test_a_device_that_did_not_answer_is_not_waited_for_again_in_the_turn(monkeypatch) -> None:
    monkeypatch.setattr(up, "CALL_GRACE_S", 0)
    rec = _Recorder(delay=5)
    conn = up.UserPcConnection(folders=[_folder()], call=rec)
    tool = up.build_user_pc_tool(conn)
    first = _run(tool, command="find .", timeout=1)
    assert first.is_error and "did not respond" in first.content
    second = _run(tool, command="ls")
    assert second.is_error and "earlier in this turn" in second.content
    assert len(rec.calls) == 1


def test_a_device_that_reports_no_response_is_remembered_too() -> None:
    rec = _Recorder({"ok": False, "code": "NO_RESPONSE", "error": "timed out"})
    conn = up.UserPcConnection(folders=[_folder()], call=rec)
    tool = up.build_user_pc_tool(conn)
    assert _run(tool, command="ls").is_error
    assert _run(tool, command="pwd").is_error
    assert len(rec.calls) == 1


def test_bad_input_is_answered_without_calling_the_device() -> None:
    rec = _Recorder()
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=rec))
    assert _run(tool).is_error  # run 인데 command 없음
    assert _run(tool, action="job").is_error
    assert _run(tool, action="put").is_error
    assert _run(tool, action="delete").is_error
    assert rec.calls == []


# ── 결과 ─────────────────────────────────────────────────────────


def test_a_running_command_comes_back_as_a_job() -> None:
    rec = _Recorder({"ok": True, "running": True, "job_id": "j7", "stdout": "building"})
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=rec))
    result = _run(tool, command="npm run build", timeout=5)
    assert not result.is_error
    assert "still running as job j7" in result.content
    assert 'action="job", job_id="j7"' in result.content
    poll = _run(tool, action="job", job_id="j7", stop=True)
    assert rec.calls[-1] == ("proj", "job", {"job_id": "j7", "stop": True, "wait_s": 10})
    assert not poll.is_error


def test_exit_code_and_stderr_are_shown_like_bash() -> None:
    rec = _Recorder({"ok": True, "exit_code": 1, "stdout": "", "stderr": "no match", "cwd": "~/p"})
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=rec))
    result = _run(tool, command="grep x f")
    assert result.is_error
    assert "cwd ~/p exit 1" in result.content
    assert "STDERR:\nno match" in result.content


def test_wait_is_clamped() -> None:
    rec = _Recorder()
    tool = up.build_user_pc_tool(up.UserPcConnection(folders=[_folder()], call=rec))
    _run(tool, command="x", timeout=99999)
    assert rec.calls[-1][2]["wait_s"] == up.MAX_RUN_WAIT_S
    _run(tool, command="x", timeout="soon")
    assert rec.calls[-1][2]["wait_s"] == up.DEFAULT_RUN_WAIT_S


# ── 안내·표면 ─────────────────────────────────────────────────────


def test_the_note_lists_facts_for_every_folder_and_device() -> None:
    note = up.turn_note(
        [
            _folder("proj"),
            _folder(
                "docs",
                path="C:\\Docs",
                device_id="w",
                device_name="Office",
                platform="win32",
                shell="Git Bash",
                online=False,
            ),
            _folder("Download", path="/Download", device_name="Galaxy", platform="android", shell=""),
        ]
    )
    assert '"proj": on "MacBook" (macOS, bash), path /Users/me/proj, online' in note
    assert '"docs": on "Office" (Windows), path C:\\Docs, offline' in note
    assert "no commands, file transfer only" in note
    assert "UserPc" in note
    assert up.turn_note([]) == ""


def test_folder_tools_are_dropped_and_other_device_tools_stay() -> None:
    class T:
        def __init__(self, name: str) -> None:
            self.name = name

    kept = up.without_folder_tools(
        [T(n) for n in (
            "mcp_local_ReadFile",
            "mcp_local_Shell",
            "mcp_local_CopyToWorkspace",
            "mcp_local_Open",
            "mcp_local_LocalControl",
            "mcp_local_BrowserNavigate",
            "mcp_local_McpListServers",
            "mcp_mobile_TakePhoto",
            "mcp_mobile_Location",
            "mcp_web_Search",
        )]
    )
    assert [t.name for t in kept] == [
        "mcp_local_BrowserNavigate",
        "mcp_local_McpListServers",
        "mcp_mobile_Location",
    ]


def test_history_retires_old_folder_tools_and_userpc_when_nothing_is_connected() -> None:
    with_folders = up.retired_calls_spec(has_folders=True)
    assert "mcp_local_Shell" in with_folders["names"]
    assert "UserPc" not in with_folders["names"]
    without = up.retired_calls_spec(has_folders=False)
    assert "UserPc" in without["names"]


@pytest.mark.parametrize("requested", ["", None])
def test_resolve_folder_needs_a_choice_only_when_ambiguous(requested) -> None:
    assert up.resolve_folder([_folder()], requested).name == "proj"
    with pytest.raises(ValueError):
        up.resolve_folder([_folder("a"), _folder("b")], requested)
