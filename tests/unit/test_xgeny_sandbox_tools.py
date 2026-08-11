"""``ToolContext.sandbox`` 가 설정되면 파일/셸 도구가 **거기서** 동작한다.

GAPT 시절의 docker-exec 흉내 대신, 프로토콜을 그대로 만족하는 로컬 구현
하나로 검증한다. 그게 요점이다 — 런타임은 :class:`XgenySandbox` 만 알고
그 뒤가 무엇인지 모른다. 프로덕션에서는 HTTP 클라이언트가 같은 자리에 들어간다.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from xgen_agent_runtime.tools._xgeny_sandbox import (
    ExecResult,
    SandboxPathError,
    XgenySandbox,
    sandbox_path,
    sb_read_bytes,
    sb_run,
    sb_write_bytes,
)
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.bash_tool import BashTool
from xgen_agent_runtime.tools.built_in.edit_tool import EditTool
from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
from xgen_agent_runtime.tools.built_in.write_tool import WriteTool


class LocalSandbox:
    """디렉터리 하나를 세션으로 삼는 :class:`XgenySandbox` 구현."""

    def __init__(self, root: Path) -> None:
        self.workdir = str(root)
        self.ensured = 0

    async def ensure(self) -> None:
        self.ensured += 1

    async def exec(self, argv, *, cwd=None, stdin=None, env=None, timeout_s=120.0):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd or self.workdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", ""), **dict(env or {})},
        )
        out, err = await asyncio.wait_for(proc.communicate(input=stdin), timeout=timeout_s)
        return ExecResult(rc=proc.returncode or 0, stdout=out, stderr=err)

    async def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return len(data)


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    return LocalSandbox(root)


@pytest.fixture
def ctx(sandbox, tmp_path):
    # working_dir 가 세션 루트와 **같은 문자열**이다 — 호스트가 양쪽 루트를
    # 맞추기 때문에 변환 계층이 필요 없다.
    return ToolContext(
        session_id="t",
        working_dir=sandbox.workdir,
        allowed_paths=[sandbox.workdir],
        sandbox=sandbox,
    )


class TestProtocolIsHonoured:
    def test_a_plain_object_satisfies_the_protocol(self, sandbox):
        """구현체가 상속을 요구받지 않는다 — 호스트가 자기 클래스를 낸다."""
        assert isinstance(sandbox, XgenySandbox)

    async def test_helpers_wake_the_session_first(self, sandbox):
        await sb_write_bytes(sandbox, "a.txt", b"x")
        await sb_read_bytes(sandbox, "a.txt")
        await sb_run(sandbox, "true")
        assert sandbox.ensured == 3, "ensure() 는 매 진입점에서 불려야 한다(멱등)"


class TestPathGuard:
    def test_relative_resolves_against_the_session_root(self, sandbox):
        assert sandbox_path(sandbox, "sub/x.txt") == f"{sandbox.workdir}/sub/x.txt"

    def test_escapes_are_refused(self, sandbox):
        for bad in ["../outside.txt", "/etc/passwd", "a/../../x"]:
            with pytest.raises(SandboxPathError):
                sandbox_path(sandbox, bad)

    def test_an_absolute_path_inside_the_session_passes_through(self, sandbox):
        inside = f"{sandbox.workdir}/deep/f.txt"
        assert sandbox_path(sandbox, inside) == inside

    async def test_a_foreign_workdir_degrades_to_the_root(self, sandbox):
        """세션과 무관한 workdir 로 모든 호출이 죽지 않는다 — chdir 실패로
        exec 를 통째로 잃었던 실패 모드를 되풀이하지 않는다."""
        rc, out, _ = await sb_run(sandbox, "pwd", workdir="/somewhere/else")
        assert rc == 0 and out.strip() == sandbox.workdir


class TestToolsRunInTheSandbox:
    async def test_write_then_read_round_trips(self, ctx, sandbox):
        await WriteTool().execute({"file_path": "note.md", "content": "안녕"}, ctx)
        assert (Path(sandbox.workdir) / "note.md").read_text(encoding="utf-8") == "안녕"

        r = await ReadTool().execute({"file_path": "note.md"}, ctx)
        assert "안녕" in str(r.content)

    async def test_edit_changes_the_sandbox_copy(self, ctx, sandbox):
        (Path(sandbox.workdir) / "e.txt").write_text("before", encoding="utf-8")
        await EditTool().execute(
            {"file_path": "e.txt", "old_string": "before", "new_string": "after"}, ctx
        )
        assert (Path(sandbox.workdir) / "e.txt").read_text(encoding="utf-8") == "after"

    async def test_bash_runs_there_not_here(self, ctx, sandbox):
        r = await BashTool().execute({"command": "pwd"}, ctx)
        assert sandbox.workdir in str(r.content)
        assert r.metadata.get("sandboxed") is True

    async def test_a_file_written_by_bash_is_visible_to_read(self, ctx, sandbox):
        """도구들이 **같은** 파일 트리를 본다 — 이게 깨지면 에이전트는 자기가
        방금 만든 파일을 못 읽는다."""
        await BashTool().execute({"command": "echo hi > made.txt"}, ctx)
        r = await ReadTool().execute({"file_path": "made.txt"}, ctx)
        assert "hi" in str(r.content)


class TestHostIsNotTouched:
    async def test_nothing_lands_on_the_host_cwd(self, ctx, sandbox, tmp_path, monkeypatch):
        """샌드박스가 붙어 있으면 호스트 파일시스템에는 아무것도 안 생긴다 —
        격리가 목적인데 조용히 호스트에 쓰면 격리가 없는 것과 같다."""
        host = tmp_path / "host"
        host.mkdir()
        monkeypatch.chdir(host)
        await WriteTool().execute({"file_path": "leak.txt", "content": "x"}, ctx)
        assert list(host.iterdir()) == []
        assert (Path(sandbox.workdir) / "leak.txt").exists()
