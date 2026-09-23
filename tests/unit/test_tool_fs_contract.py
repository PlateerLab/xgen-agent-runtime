"""파일시스템 포트 계약 — **같은 테스트를 로컬·러너 두 백엔드에 돌린다** (4.53.0).

도구가 러너/로컬을 각자 고르던 동안 두 경로는 서로 다르게 동작했다 (경로 탈출
예외 종류부터 달랐다). 이 파일이 "두 백엔드는 같은 의미로 동작한다" 의 정의다.
여기 없는 동작에 기대는 도구는 이관할 때 이 파일에 먼저 계약을 추가한다.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from xgen_agent_runtime.tools._xgeny_sandbox import ExecResult
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.fs import (
    FsAccessError,
    LocalFS,
    RunnerFS,
    ToolFileSystem,
    tool_fs,
)


class _Session:
    """디렉터리 하나를 세션으로 삼는 XgenySandbox (test_xgeny_sandbox_tools 와 같은 모양)."""

    def __init__(self, root: Path, extra_roots=(), readonly_roots=()) -> None:
        self.workdir = str(root)
        self.extra_roots = [str(r) for r in extra_roots]
        self.readonly_roots = [str(r) for r in readonly_roots]

    async def ensure(self) -> None:
        return None

    async def exec(self, argv, *, cwd=None, stdin=None, env=None, timeout_s=120.0):
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd or self.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", **dict(env or {})},
        )
        out, err = await proc.communicate()
        return ExecResult(rc=proc.returncode or 0, stdout=out, stderr=err)

    async def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return len(data)


class _SessionWithExists(_Session):
    """실제 세션(러너 HTTP 클라이언트·커넥터)처럼 ``exists`` 를 1급으로 가진 세션."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.exists_calls = 0

    async def exists(self, path: str) -> bool:
        self.exists_calls += 1
        return Path(path).exists()


@pytest.fixture(params=["local", "runner", "runner+exists"])
def fs(request, tmp_path) -> ToolFileSystem:
    root = tmp_path / "ws"
    root.mkdir()
    if request.param == "local":
        return LocalFS(str(root), allowed_paths=[str(root)])
    if request.param == "runner":
        return RunnerFS(_Session(root), str(root))
    return RunnerFS(_SessionWithExists(root), str(root))


def _run(coro):
    return asyncio.run(coro)


def _root(fs) -> Path:
    return Path(fs.working_dir)


# ── 두 백엔드가 똑같이 지켜야 하는 것 ────────────────────────────────


class TestContract:
    def test_write_then_read_round_trips(self, fs):
        n = _run(fs.write_bytes("note.md", "안녕".encode()))
        assert n == len("안녕".encode())
        assert _run(fs.read_bytes("note.md")) == "안녕".encode()

    def test_write_creates_parent_directories(self, fs):
        _run(fs.write_bytes("a/b/c.txt", b"x"))
        assert (_root(fs) / "a/b/c.txt").read_bytes() == b"x"

    def test_relative_paths_resolve_against_the_working_dir(self, fs):
        assert fs.resolve("sub/x.txt") == str(_root(fs) / "sub/x.txt")

    def test_reading_a_missing_file_is_file_not_found(self, fs):
        with pytest.raises(FileNotFoundError):
            _run(fs.read_bytes("nope.txt"))

    def test_escaping_the_tree_is_the_same_error_on_both(self, fs):
        """예전엔 로컬 PermissionError / 러너 SandboxPathError(RuntimeError) 로 갈렸다."""
        with pytest.raises(FsAccessError):
            fs.resolve("../../etc/passwd")
        with pytest.raises(PermissionError):  # 기존 로컬 호출부가 잡던 것도 그대로 잡힌다
            _run(fs.read_bytes("../../etc/passwd"))

    def test_empty_path_is_a_value_error(self, fs):
        with pytest.raises(ValueError):
            fs.resolve("")

    def test_exists(self, fs):
        assert _run(fs.exists("nope.txt")) is False
        _run(fs.write_bytes("d/f.bin", b"12345"))
        assert _run(fs.exists("d/f.bin")) is True
        assert _run(fs.exists("d")) is True  # 디렉터리도 "있다"

    def test_materialize_gives_a_readable_local_copy(self, fs):
        _run(fs.write_bytes("doc.docx", b"PK\x03\x04body"))

        async def go():
            async with fs.materialize("doc.docx") as local:
                return Path(local).read_bytes()

        assert _run(go()) == b"PK\x03\x04body"

    def test_materialize_of_a_missing_file_is_file_not_found(self, fs):
        async def go():
            async with fs.materialize("nope.docx"):
                pass

        with pytest.raises(FileNotFoundError):
            _run(go())

    def test_commit_lands_the_engine_output_in_the_workspace(self, fs, tmp_path):
        produced = tmp_path / "engine-out.pdf"
        produced.write_bytes(b"%PDF-1.7")
        _run(fs.commit(produced, "결과물/보고서.pdf"))
        assert _run(fs.read_bytes("결과물/보고서.pdf")) == b"%PDF-1.7"


# ── 러너만의 약속 ────────────────────────────────────────────────────


class TestRunnerOnly:
    def test_materialize_never_uses_the_workspace_path(self, tmp_path):
        """파드 사본이 워크스페이스와 같은 경로 문자열을 쓰면, 지난 턴의 잔재가 러너의
        새 파일을 조용히 가린다("방금 고쳤는데 옛날 내용이 나온다"). 구조적으로 막는다."""
        root = tmp_path / "ws"
        root.mkdir()
        fs = RunnerFS(_Session(root), str(root))
        _run(fs.write_bytes("r.docx", b"new"))

        async def go():
            async with fs.materialize("r.docx") as local:
                return str(local), Path(local).exists()

        local, existed = _run(go())
        assert existed
        assert not local.startswith(str(root))
        assert not Path(local).exists()  # 끝나면 치운다

    def test_native_exists_is_used_and_no_shell_is_spawned(self, tmp_path):
        """셸 ``test -f`` 는 붙지 않은 세션에서 루트를 추측해 "없음" 을 답했다(프로드 실증).
        세션이 exists 를 가지면 그것만 쓴다."""
        root = tmp_path / "ws"
        root.mkdir()
        session = _SessionWithExists(root)

        async def no_exec(*a, **k):
            raise AssertionError("exists 는 셸을 띄우면 안 된다")

        session.exec = no_exec
        fs = RunnerFS(session, str(root))
        assert _run(fs.exists("x.txt")) is False
        assert session.exists_calls == 1

    def test_the_session_is_woken_before_the_path_is_resolved(self, tmp_path):
        """붙지 않은 세션의 workdir 은 예상값이다. 먼저 깨워야 실제 루트로 푼다
        (sb_read_bytes 와 같은 순서)."""
        real = tmp_path / "real"
        real.mkdir()
        (real / "a.txt").write_text("ok")

        class _Lazy(_Session):
            async def ensure(self) -> None:
                self.workdir = str(real)  # 붙는 순간 실제 루트를 알게 된다

        fs = RunnerFS(_Lazy(tmp_path / "guess"), "")
        assert _run(fs.read_bytes("a.txt")) == b"ok"

    def test_readonly_roots_refuse_writes(self, tmp_path):
        root = tmp_path / "ws"
        shared = tmp_path / "shared"
        root.mkdir()
        shared.mkdir()
        (shared / "a.txt").write_text("x")
        fs = RunnerFS(_Session(root, extra_roots=[shared], readonly_roots=[shared]), str(root))
        assert _run(fs.read_bytes(str(shared / "a.txt"))) == b"x"
        with pytest.raises(FsAccessError):
            _run(fs.write_bytes(str(shared / "a.txt"), b"y"))


# ── 백엔드 선택은 한 곳 ──────────────────────────────────────────────


class TestSelection:
    def test_no_session_means_local(self, tmp_path):
        assert isinstance(tool_fs(ToolContext(working_dir=str(tmp_path))), LocalFS)

    def test_a_session_means_runner(self, tmp_path):
        ctx = ToolContext(working_dir=str(tmp_path), sandbox=_Session(tmp_path))
        assert isinstance(tool_fs(ctx), RunnerFS)

    def test_an_injected_fs_wins(self, tmp_path):
        mine = LocalFS(str(tmp_path))
        ctx = ToolContext(working_dir=str(tmp_path), sandbox=_Session(tmp_path), fs=mine)
        assert tool_fs(ctx) is mine

    def test_both_backends_satisfy_the_protocol(self, tmp_path):
        assert isinstance(LocalFS(str(tmp_path)), ToolFileSystem)
        assert isinstance(RunnerFS(_Session(tmp_path)), ToolFileSystem)


# ── 로컬 쓰기는 원자적이고 권한을 지킨다 ─────────────────────────────


class TestLocalAtomicWrite:
    def test_an_existing_files_mode_survives_a_rewrite(self, tmp_path):
        """mkstemp 은 0600 으로 만든다 — 그대로 rename 하면 스크립트가 실행 권한을 잃는다."""
        script = tmp_path / "run.sh"
        script.write_text("echo a")
        script.chmod(0o755)
        _run(LocalFS(str(tmp_path)).write_bytes("run.sh", b"echo b"))
        assert script.stat().st_mode & 0o777 == 0o755
        assert script.read_text() == "echo b"

    def test_no_temp_file_is_left_behind(self, tmp_path):
        _run(LocalFS(str(tmp_path)).write_bytes("a.txt", b"x"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]

    def test_a_failed_write_keeps_the_original(self, tmp_path, monkeypatch):
        target = tmp_path / "keep.txt"
        target.write_text("original")

        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            _run(LocalFS(str(tmp_path)).write_bytes("keep.txt", b"half-written"))
        assert target.read_text() == "original"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]
