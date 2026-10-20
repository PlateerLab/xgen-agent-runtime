"""Execution operations require an explicit backend capability."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from xgen_agent_runtime.tools._geny_sandbox import ExecResult
from xgen_agent_runtime.tools.base import (
    EXECUTION_BACKEND_UNAVAILABLE,
    HOST_IS_EXECUTION_TARGET,
    ToolContext,
)
from xgen_agent_runtime.tools.built_in.bash_tool import BashTool
from xgen_agent_runtime.tools.built_in.dev_tools import REPLTool
from xgen_agent_runtime.tools.built_in.worktree_tools import EnterWorktreeTool
from xgen_agent_runtime.tools.built_in.write_tool import WriteTool
from xgen_agent_runtime.tools.fs import LocalFS
from xgen_agent_runtime.skills.shell_blocks import execute_blocks


def _assert_backend_unavailable(result) -> None:
    assert result.is_error is True
    assert result.content["error"]["code"] == EXECUTION_BACKEND_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [None, False, "true", 1])
async def test_missing_or_non_boolean_marker_never_starts_host_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker
) -> None:
    calls: list[str] = []

    async def forbidden_shell(*args, **kwargs):
        calls.append("shell")
        raise AssertionError("host shell was started")

    async def forbidden_exec(*args, **kwargs):
        calls.append("exec")
        raise AssertionError("host interpreter was started")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", forbidden_shell)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_exec)
    extras = {} if marker is None else {HOST_IS_EXECUTION_TARGET: marker}
    context = ToolContext(working_dir=str(tmp_path), extras=extras)

    bash = await BashTool().execute(
        {"command": "printf unsafe > host-bash.txt", HOST_IS_EXECUTION_TARGET: True},
        context,
    )
    repl = await REPLTool().execute(
        {"expression": "open('host-repl.txt', 'w').write('unsafe')"}, context
    )

    _assert_backend_unavailable(bash)
    _assert_backend_unavailable(repl)
    assert calls == []
    assert not (tmp_path / "host-bash.txt").exists()
    assert not (tmp_path / "host-repl.txt").exists()


@pytest.mark.asyncio
async def test_file_only_filesystem_does_not_authorize_shell_or_repl(tmp_path: Path) -> None:
    context = ToolContext(
        working_dir=str(tmp_path),
        allowed_paths=[str(tmp_path)],
        fs=LocalFS(str(tmp_path), [str(tmp_path)]),
    )

    write = await WriteTool().execute(
        {"file_path": "file-only.txt", "content": "allowed"}, context
    )
    bash = await BashTool().execute({"command": "printf unsafe > bash.txt"}, context)
    repl = await REPLTool().execute(
        {"expression": "open('repl.txt', 'w').write('unsafe')"}, context
    )

    assert write.is_error is False
    assert (tmp_path / "file-only.txt").read_text() == "allowed"
    _assert_backend_unavailable(bash)
    _assert_backend_unavailable(repl)
    assert not (tmp_path / "bash.txt").exists()
    assert not (tmp_path / "repl.txt").exists()


@pytest.mark.asyncio
async def test_exact_trusted_local_marker_allows_local_file_shell_and_repl(tmp_path: Path) -> None:
    context = ToolContext(
        working_dir=str(tmp_path),
        allowed_paths=[str(tmp_path)],
        extras={HOST_IS_EXECUTION_TARGET: True},
    )

    write = await WriteTool().execute(
        {"file_path": "write.txt", "content": "write"}, context
    )
    bash = await BashTool().execute({"command": "printf bash > bash.txt"}, context)
    repl = await REPLTool().execute(
        {"expression": "open('repl.txt', 'w').write('repl')"}, context
    )

    assert write.is_error is False
    assert bash.is_error is False
    assert repl.is_error is False
    assert (tmp_path / "write.txt").read_text() == "write"
    assert (tmp_path / "bash.txt").read_text() == "bash"
    assert (tmp_path / "repl.txt").read_text() == "repl"


class _Sandbox:
    def __init__(self) -> None:
        self.workdir = "/workspace"
        self.extra_roots = []
        self.readonly_roots = []
        self.ensure_calls = 0
        self.exec_calls = 0

    async def ensure(self) -> None:
        self.ensure_calls += 1

    async def exec(self, argv, **kwargs):
        self.exec_calls += 1
        return ExecResult(rc=0, stdout=b"sandbox\n", stderr=b"")

    async def read_bytes(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    async def write_bytes(self, path: str, data: bytes) -> int:
        return len(data)


class _UnavailableSandbox(_Sandbox):
    async def ensure(self) -> None:
        self.ensure_calls += 1
        raise RuntimeError("CANARY_BACKEND_SECRET")


@pytest.mark.asyncio
async def test_attached_sandbox_remains_the_default_execution_backend() -> None:
    sandbox = _Sandbox()
    context = ToolContext(working_dir="/workspace", sandbox=sandbox)

    bash = await BashTool().execute({"command": "echo sandbox"}, context)
    repl = await REPLTool().execute({"expression": "print('sandbox')"}, context)

    assert bash.is_error is False
    assert repl.is_error is False
    assert sandbox.ensure_calls == 2
    assert sandbox.exec_calls == 2


@pytest.mark.asyncio
async def test_unavailable_sandbox_never_falls_back_to_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []

    async def forbidden_shell(*args, **kwargs):
        calls.append("shell")
        raise AssertionError("host shell was started")

    async def forbidden_exec(*args, **kwargs):
        calls.append("exec")
        raise AssertionError("host interpreter was started")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", forbidden_shell)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_exec)
    sandbox = _UnavailableSandbox()
    context = ToolContext(working_dir="/workspace", sandbox=sandbox)

    bash = await BashTool().execute({"command": "printf unsafe > bash.txt"}, context)
    repl = await REPLTool().execute(
        {"expression": "open('repl.txt', 'w').write('unsafe')"}, context
    )
    write = await WriteTool().execute(
        {"file_path": "write.txt", "content": "unsafe"}, context
    )
    shell_blocks = await execute_blocks(
        "```!\nprintf unsafe > skill.txt\n```",
        cwd=str(tmp_path),
        sandbox=sandbox,
    )

    _assert_backend_unavailable(bash)
    _assert_backend_unavailable(repl)
    assert write.is_error and EXECUTION_BACKEND_UNAVAILABLE in str(write.content)
    assert shell_blocks.any_failed
    assert EXECUTION_BACKEND_UNAVAILABLE in shell_blocks.outcomes[0].stderr
    assert sandbox.exec_calls == 0
    assert calls == []
    assert not any(tmp_path.iterdir())
    visible = str(bash.content) + str(repl.content) + str(write.content)
    visible += shell_blocks.outcomes[0].stderr
    visible += " ".join(record.getMessage() for record in caplog.records)
    assert "CANARY_BACKEND_SECRET" not in visible


@pytest.mark.asyncio
async def test_worktree_tool_refuses_host_git_without_trusted_local_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def forbidden_git(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("host git was started")

    monkeypatch.setattr(
        "xgen_agent_runtime.tools.built_in.worktree_tools._git", forbidden_git
    )
    result = await EnterWorktreeTool().execute(
        {"branch": "unsafe"}, ToolContext(working_dir=str(tmp_path))
    )

    _assert_backend_unavailable(result)
    assert calls == 0
    assert not (tmp_path / ".worktrees").exists()


@pytest.mark.asyncio
async def test_worktree_tool_never_bypasses_an_attached_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def forbidden_git(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("host git was started")

    monkeypatch.setattr(
        "xgen_agent_runtime.tools.built_in.worktree_tools._git", forbidden_git
    )
    context = ToolContext(
        working_dir=str(tmp_path),
        sandbox=_Sandbox(),
        extras={HOST_IS_EXECUTION_TARGET: True},
    )
    result = await EnterWorktreeTool().execute({"branch": "unsafe"}, context)

    _assert_backend_unavailable(result)
    assert calls == 0
    assert not (tmp_path / ".worktrees").exists()
