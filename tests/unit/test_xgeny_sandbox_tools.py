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
from xgen_agent_runtime.tools.built_in.dev_tools import REPLTool
from xgen_agent_runtime.tools.built_in.edit_tool import EditTool
from xgen_agent_runtime.tools.built_in.notebook_edit_tool import NotebookEditTool
from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
from xgen_agent_runtime.tools.built_in.write_tool import WriteTool


class LocalSandbox:
    """디렉터리 하나를 세션으로 삼는 :class:`XgenySandbox` 구현."""

    def __init__(self, root: Path, extra_roots=(), readonly_roots=()) -> None:
        self.workdir = str(root)
        # 호스트가 명시적으로 연 형제 트리 (사용자 클라우드 등).
        self.extra_roots = [str(r) for r in extra_roots]
        # 그중 읽기 전용인 것 (읽기로 공유받은 폴더).
        self.readonly_roots = [str(r) for r in readonly_roots]
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


class TestExplicitlyOpenedTrees:
    """에이전트는 자기 workspace 말고도 다룰 것이 있다 — 사용자 계정의 클라우드.

    그걸 workdir 안으로 밀어 넣으면 에이전트 산출물과 사용자 파일이 한 트리에
    섞이고, 한쪽의 삭제 전파가 다른 쪽을 지운다. 형제 트리로 두고 명시적으로 연다.
    """

    def test_a_sibling_tree_is_reachable_when_opened(self, tmp_path):
        cloud = tmp_path / "user" / "51" / "workspace"
        cloud.mkdir(parents=True)
        sb = LocalSandbox(tmp_path / "session", extra_roots=[str(cloud)])
        (tmp_path / "session").mkdir(exist_ok=True)
        assert sandbox_path(sb, str(cloud / "a.txt")) == str(cloud / "a.txt")

    def test_it_is_refused_when_not_opened(self, tmp_path):
        cloud = tmp_path / "user" / "51" / "workspace"
        sb = LocalSandbox(tmp_path / "session")
        (tmp_path / "session").mkdir(exist_ok=True)
        with pytest.raises(SandboxPathError):
            sandbox_path(sb, str(cloud / "a.txt"))

    def test_opening_one_tree_does_not_open_the_rest(self, tmp_path):
        """열어 준 것만 열린다 — 상위 디렉터리가 통째로 열리면 안 된다."""
        cloud = tmp_path / "user" / "51" / "workspace"
        other = tmp_path / "user" / "99" / "workspace"
        sb = LocalSandbox(tmp_path / "session", extra_roots=[str(cloud)])
        (tmp_path / "session").mkdir(exist_ok=True)
        with pytest.raises(SandboxPathError):
            sandbox_path(sb, str(other / "secret.txt"))

    async def test_tools_can_write_into_an_opened_tree(self, tmp_path):
        cloud = tmp_path / "user" / "51" / "workspace"
        cloud.mkdir(parents=True)
        root = tmp_path / "session"
        root.mkdir()
        sb = LocalSandbox(root, extra_roots=[str(cloud)])
        ctx = ToolContext(
            session_id="t", working_dir=str(root),
            allowed_paths=[str(root), str(cloud)], sandbox=sb,
        )
        await WriteTool().execute(
            {"file_path": str(cloud / "note.txt"), "content": "클라우드"}, ctx
        )
        assert (cloud / "note.txt").read_text(encoding="utf-8") == "클라우드"


class TestReadOnlyTrees:
    """읽기로 공유받은 폴더 — 읽을 수는 있지만 쓸 수 없다.

    ⚠ 이건 **보안 경계가 아니라 빠른 피드백**이다. 셸은 파일시스템에 직접
    쓰므로 이 검사를 지나가지 않는다. 진짜 관문은 인덱스 커밋이고, 거기서
    거부되면 원본에 반영되지 않는다.

    그래도 여기서 막는 이유: 커밋은 턴이 끝날 때다. 그때 처음 알면 에이전트는
    이미 고쳤다고 믿고 한참 더 일한 뒤다.
    """

    def _shared(self, tmp_path):
        shared = tmp_path / "user" / "7" / "workspace" / "공유폴더"
        shared.mkdir(parents=True)
        root = tmp_path / "session"
        root.mkdir(exist_ok=True)
        return shared, root

    def test_reading_is_allowed(self, tmp_path):
        shared, root = self._shared(tmp_path)
        sb = LocalSandbox(root, extra_roots=[str(shared)], readonly_roots=[str(shared)])
        assert sandbox_path(sb, str(shared / "a.txt")) == str(shared / "a.txt")

    def test_writing_is_refused(self, tmp_path):
        shared, root = self._shared(tmp_path)
        sb = LocalSandbox(root, extra_roots=[str(shared)], readonly_roots=[str(shared)])
        with pytest.raises(SandboxPathError, match="읽기 전용"):
            sandbox_path(sb, str(shared / "a.txt"), write=True)

    async def test_the_write_tool_refuses(self, tmp_path):
        shared, root = self._shared(tmp_path)
        sb = LocalSandbox(root, extra_roots=[str(shared)], readonly_roots=[str(shared)])
        ctx = ToolContext(
            session_id="t", working_dir=str(root),
            allowed_paths=[str(root), str(shared)], sandbox=sb,
        )
        result = await WriteTool().execute(
            {"file_path": str(shared / "x.txt"), "content": "몰래"}, ctx
        )
        assert result.is_error, "읽기 전용 트리에 썼다"
        assert not (shared / "x.txt").exists()

    def test_my_own_workdir_is_never_readonly(self, tmp_path):
        """자기 작업 폴더까지 잠기면 에이전트가 아무것도 못 한다."""
        shared, root = self._shared(tmp_path)
        sb = LocalSandbox(root, extra_roots=[str(shared)], readonly_roots=[str(shared)])
        assert sandbox_path(sb, "out.txt", write=True) == str(root / "out.txt")

    def test_a_writable_sibling_stays_writable(self, tmp_path):
        """읽기 전용 목록에 없는 형제 트리는 그대로 쓸 수 있어야 한다."""
        shared, root = self._shared(tmp_path)
        cloud = tmp_path / "user" / "51" / "workspace"
        cloud.mkdir(parents=True)
        sb = LocalSandbox(
            root, extra_roots=[str(shared), str(cloud)], readonly_roots=[str(shared)],
        )
        assert sandbox_path(sb, str(cloud / "a.txt"), write=True) == str(cloud / "a.txt")

    def test_no_readonly_list_means_everything_is_writable(self, tmp_path):
        """이 확장을 모르는 구현(속성 없음)에서 예전 동작 그대로."""
        shared, root = self._shared(tmp_path)
        sb = LocalSandbox(root, extra_roots=[str(shared)])
        del sb.readonly_roots
        assert sandbox_path(sb, str(shared / "a.txt"), write=True) == str(shared / "a.txt")


class _SpySandbox(LocalSandbox):
    """LocalSandbox that counts protocol calls — proves a tool went through
    the sandbox (read_bytes/write_bytes/exec) and not the host filesystem."""

    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        self.reads = 0
        self.writes = 0
        self.execs = 0

    async def read_bytes(self, path: str) -> bytes:
        self.reads += 1
        return await super().read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> int:
        self.writes += 1
        return await super().write_bytes(path, data)

    async def exec(self, argv, **kw):
        self.execs += 1
        return await super().exec(argv, **kw)


def _spy_ctx(root):
    sb = _SpySandbox(root)
    return sb, ToolContext(session_id="t", working_dir=sb.workdir,
                           allowed_paths=[sb.workdir], sandbox=sb)


_MIN_NB = (
    '{"cells": [{"cell_type": "code", "source": ["print(1)\\n"], '
    '"metadata": {}, "outputs": [], "execution_count": null}], '
    '"metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
)


class TestP0ToolsRouteToSandbox:
    """The tools just fixed (NotebookEdit / REPL / skill shell) must go through
    the sandbox protocol, never the serving pod's filesystem/interpreter."""

    async def test_notebookedit_reads_and_writes_via_sandbox(self, tmp_path):
        root = tmp_path / "session"
        root.mkdir()
        (root / "nb.ipynb").write_text(_MIN_NB, encoding="utf-8")
        sb, ctx = _spy_ctx(root)
        r = await NotebookEditTool().execute(
            {"file_path": "nb.ipynb",
             "operations": [{"op": "replace", "cell_index": 0, "new_source": "print(2)\n"}]},
            ctx,
        )
        assert not r.is_error, r.content
        # routed through the sandbox protocol, not host open()/os.replace
        assert sb.reads >= 1 and sb.writes >= 1
        import json
        nb = json.loads((root / "nb.ipynb").read_text(encoding="utf-8"))
        assert "print(2)" in "".join(nb["cells"][0]["source"])

    async def test_notebookedit_missing_file_is_a_clean_error(self, tmp_path):
        root = tmp_path / "session"
        root.mkdir()
        sb, ctx = _spy_ctx(root)
        r = await NotebookEditTool().execute(
            {"file_path": "nope.ipynb",
             "operations": [{"op": "replace", "cell_index": 0, "source": "x"}]},
            ctx,
        )
        assert r.is_error and "not found" in str(r.content)

    async def test_repl_runs_in_the_sandbox_not_the_pod(self, tmp_path):
        root = tmp_path / "session"
        root.mkdir()
        sb, ctx = _spy_ctx(root)
        # Write a file from Python — it must land in the SANDBOX cwd.
        r = await REPLTool().execute(
            {"expression": "open('proof.txt','w').write('sbx')"}, ctx
        )
        assert sb.execs >= 1
        assert (root / "proof.txt").read_text(encoding="utf-8") == "sbx"

    async def test_skill_shell_block_runs_in_the_sandbox(self, tmp_path):
        from xgen_agent_runtime.skills.shell_blocks import execute_blocks

        root = tmp_path / "session"
        root.mkdir()
        sb = _SpySandbox(root)
        summary = await execute_blocks(
            "before !`echo hi > marker.txt && echo done` after",
            cwd=sb.workdir, sandbox=sb,
        )
        assert sb.execs >= 1
        assert (root / "marker.txt").exists()
        assert "done" in summary.rendered_body
