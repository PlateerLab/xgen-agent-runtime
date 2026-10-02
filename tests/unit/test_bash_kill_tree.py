"""호스트 Bash 는 취소·시간 초과 때 셸이 띄운 자식까지 끝낸다.

회귀(2026-10-02 XD 실측): sandbox 없이 이 PC 에서 도는 Bash 를 턴 취소로 멈추면 ``sleep 30`` 이 고아로
남았다. 취소(CancelledError)를 받아도 프로세스를 건드리지 않았고, 시간 초과 때는 셸(``/bin/sh``)만
죽인 뒤 그 자식이 쥔 출력 파이프가 닫히기를 기다려 시간 초과가 제때 돌아오지 않았다.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from xgen_agent_runtime.tools.base import HOST_IS_EXECUTION_TARGET, ToolContext
from xgen_agent_runtime.tools.built_in import bash_tool as bt

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX here")


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_dir=str(tmp_path),
        allowed_paths=[str(tmp_path)],
        extras={HOST_IS_EXECUTION_TARGET: True},
    )


#: 셸이 띄운 자식의 pid 를 적고 기다리는 명령 — 그 자식이 살아남는지가 이 시험의 대상이다.
_CHILD = "sleep 30 & echo $! > child.txt; wait"


def _alive(pid: int) -> bool:
    """프로세스가 아직 도는가. 좀비(끝났는데 거둬지지 않은 것)는 끝난 것으로 본다."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            return False
    return True


def _wait_gone(pid: int, seconds: float = 4.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


async def _wait_for_file(path: Path, seconds: float = 10.0) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        await asyncio.sleep(0.02)
    raise AssertionError(f"{path} never appeared")


@posix_only
def test_cancel_kills_the_shells_children(tmp_path):
    async def scenario() -> int:
        task = asyncio.ensure_future(bt.BashTool().execute({"command": _CHILD}, _ctx(tmp_path)))
        child = int(await _wait_for_file(tmp_path / "child.txt"))
        assert _alive(child)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return child

    child = asyncio.run(scenario())
    assert _wait_gone(child), "the shell's child outlived the cancelled turn"


@posix_only
def test_timeout_kills_the_shells_children(tmp_path):
    # 예전에는 셸만 죽이고 그 자식이 쥔 출력 파이프가 닫히기를 기다려, 1초 시간 초과가 30초 뒤에야 돌아왔다.
    started = time.monotonic()
    result = asyncio.run(bt.BashTool().execute({"command": _CHILD, "timeout": 1000}, _ctx(tmp_path)))
    assert time.monotonic() - started < 1 + bt._KILL_GRACE_S + 5
    assert result.is_error and "timed out" in str(result.content)
    child = int((tmp_path / "child.txt").read_text().strip())
    assert _wait_gone(child), "the shell's child outlived the timeout"


@posix_only
def test_children_that_ignore_sigterm_are_killed(tmp_path):
    async def scenario() -> int:
        task = asyncio.ensure_future(
            bt.BashTool().execute({"command": "trap '' TERM; " + _CHILD}, _ctx(tmp_path))
        )
        child = int(await _wait_for_file(tmp_path / "child.txt"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return child

    started = time.monotonic()
    child = asyncio.run(scenario())
    assert _wait_gone(child)
    assert time.monotonic() - started < bt._KILL_GRACE_S + 8


def test_normal_commands_still_return_their_output(tmp_path):
    result = asyncio.run(bt.BashTool().execute({"command": "echo ok"}, _ctx(tmp_path)))
    assert not result.is_error
    assert "ok" in str(result.content)


def test_spawn_kwargs_per_platform():
    assert bt._host_spawn_kwargs(platform="linux") == {"start_new_session": True}
    assert bt._host_spawn_kwargs(platform="darwin") == {"start_new_session": True}
    flags = bt._host_spawn_kwargs(platform="win32")["creationflags"]
    assert flags & 0x00000200  # CREATE_NEW_PROCESS_GROUP
    # 창 없는 엔진이 띄우는 PowerShell 이 명령마다 콘솔 창을 만들지 않게(4.83.2).
    assert flags & 0x08000000  # CREATE_NO_WINDOW


def test_windows_kill_uses_taskkill_for_the_tree(monkeypatch):
    calls = []

    class _Killer:
        async def wait(self):
            return 0

    async def fake_exec(*argv, **kwargs):
        calls.append(argv)
        assert kwargs.get("creationflags", 0) & 0x08000000  # taskkill 도 창을 만들지 않는다
        return _Killer()

    class _Proc:
        pid = 4321
        returncode = None

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    monkeypatch.setattr(bt.asyncio, "create_subprocess_exec", fake_exec)
    proc = _Proc()
    asyncio.run(bt._kill_process_tree(proc, platform="win32"))
    assert calls == [("taskkill", "/T", "/F", "/PID", "4321")]
    # taskkill 이 셸을 못 끝냈으면 셸만이라도 끝낸다.
    assert proc.returncode == -9


def test_finished_process_is_left_alone():
    class _Done:
        pid = 1
        returncode = 0

    asyncio.run(bt._kill_process_tree(_Done(), platform="linux"))
