"""Read·Write·Edit 는 러너와 로컬에서 **같은 말**을 한다 (4.54.0, 파일시스템 포트 2단계).

두 분기를 따로 구현하던 동안 실제로 갈라져 있던 것들 — 여기 테스트가 각각을 고정한다:

* 이미지: 로컬은 ``[Image file …]``, 러너는 바이너리 검사에 걸려 ``[Binary file …]``.
* Edit 성공 문구: 로컬은 절대 경로, 러너는 모델이 준 표기.
* 경로 탈출: 로컬은 ``Access denied …``, 러너는 ``Read error: …`` (RuntimeError 가 일반
  except 로 샜다).
* 쓰기 가드 장부: 러너 Read 는 모델 표기 하나만 적어서, 상대 경로로 읽고 절대 경로로
  쓰면 **러너에서만** 거절됐다.
* CRLF: 로컬 Edit 은 파일 전체의 줄 끝을 조용히 LF 로 바꿨고, 러너는 아예 안 맞았다.

그리고 4.51.0 의 회귀 — Write 가 자기가 쓴 파일을 장부에 안 올려 **방금 만든 파일을
다시 쓰면 거절**했다.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from xgen_agent_runtime.tools._xgeny_sandbox import ExecResult
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.edit_tool import EditTool
from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
from xgen_agent_runtime.tools.built_in.write_tool import WriteTool


class _Session:
    def __init__(self, root: Path) -> None:
        self.workdir = str(root)
        self.extra_roots: list = []
        self.readonly_roots: list = []

    async def ensure(self) -> None:
        return None

    async def exec(self, argv, *, cwd=None, stdin=None, env=None, timeout_s=120.0):
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd or self.workdir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "")},
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


class _Env:
    def __init__(self, root: Path, ctx: ToolContext, shared: dict) -> None:
        self.root = root
        self.ctx = ctx
        self.shared = shared

    def run(self, tool, payload):
        result = asyncio.run(tool.execute(payload, self.ctx))
        self.shared.update(result.state_mutations or {})  # 파이프라인이 하는 일
        return result


@pytest.fixture(params=["local", "runner"])
def env(request, tmp_path) -> _Env:
    root = tmp_path / "ws"
    root.mkdir()
    shared: dict = {}
    view = SimpleNamespace(shared=shared)
    if request.param == "local":
        ctx = ToolContext(
            session_id="t", working_dir=str(root), allowed_paths=[str(root)], state_view=view
        )
    else:
        ctx = ToolContext(
            session_id="t", working_dir=str(root), sandbox=_Session(root), state_view=view
        )
    return _Env(root, ctx, shared)


class TestSameAnswerOnBothBackends:
    def test_an_image_is_named_as_an_image(self, env):
        (env.root / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        r = env.run(ReadTool(), {"file_path": "shot.png"})
        assert r.content.startswith("[Image file: shot.png"), r.content

    def test_escaping_is_an_access_error_not_a_read_error(self, env):
        for tool, payload in (
            (ReadTool(), {"file_path": "../../etc/passwd"}),
            (EditTool(), {"file_path": "../../etc/passwd", "old_string": "a", "new_string": "b"}),
            (WriteTool(), {"file_path": "../../etc/x", "content": "y"}),
        ):
            r = env.run(tool, payload)
            assert r.is_error
            assert not r.content.startswith("Read error"), (tool.name, r.content)

    def test_success_messages_name_the_absolute_path(self, env):
        env.run(WriteTool(), {"file_path": "a.txt", "content": "hello"})
        w = env.run(WriteTool(), {"file_path": "b.txt", "content": "x"})
        e = env.run(EditTool(), {"file_path": "a.txt", "old_string": "hello", "new_string": "bye"})
        assert str(env.root / "b.txt") in w.content
        assert str(env.root / "a.txt") in e.content

    def test_missing_file_names_the_absolute_path(self, env):
        r = env.run(ReadTool(), {"file_path": "nope.md"})
        assert r.is_error and str(env.root / "nope.md") in r.content

    def test_read_numbers_lines_from_one(self, env):
        (env.root / "c.py").write_text("a\nb\n")
        assert env.run(ReadTool(), {"file_path": "c.py"}).content == "1\ta\n2\tb"


class TestWriteGuardLedger:
    def test_rewriting_a_file_you_just_wrote_is_allowed(self, env):
        """4.51.0 회귀: Write 가 장부에 안 올려서 Write→Write 가 거절됐다."""
        assert not env.run(WriteTool(), {"file_path": "report.md", "content": "v1"}).is_error
        r = env.run(WriteTool(), {"file_path": "report.md", "content": "v2"})
        assert not r.is_error, r.content
        assert (env.root / "report.md").read_text() == "v2"

    def test_writing_after_editing_is_allowed(self, env):
        (env.root / "e.txt").write_text("old")
        env.run(ReadTool(), {"file_path": "e.txt"})
        env.run(EditTool(), {"file_path": "e.txt", "old_string": "old", "new_string": "mid"})
        env.shared.clear()  # Read 기록을 지워도 Edit 기록만으로 통과해야 한다
        env.run(EditTool(), {"file_path": "e.txt", "old_string": "mid", "new_string": "mid2"})
        assert not env.run(WriteTool(), {"file_path": "e.txt", "content": "new"}).is_error

    def test_relative_read_covers_an_absolute_write(self, env):
        """예전엔 러너에서만 거절됐다 (러너 Read 가 모델 표기 하나만 적었다)."""
        (env.root / "r.txt").write_text("before")
        env.run(ReadTool(), {"file_path": "r.txt"})
        r = env.run(WriteTool(), {"file_path": str(env.root / "r.txt"), "content": "after"})
        assert not r.is_error, r.content

    def test_an_unread_existing_file_is_still_protected(self, env):
        (env.root / "user.docx").write_bytes(b"user content")
        r = env.run(WriteTool(), {"file_path": "user.docx", "content": "clobber"})
        assert r.is_error and "have not read it" in r.content
        assert (env.root / "user.docx").read_bytes() == b"user content"


class TestCrlf:
    def test_lf_old_string_matches_a_crlf_file_and_keeps_its_line_endings(self, env):
        (env.root / "win.txt").write_bytes(b"line1\r\nline2\r\nline3\r\n")
        r = env.run(
            EditTool(),
            {"file_path": "win.txt", "old_string": "line1\nline2", "new_string": "A\nB"},
        )
        assert not r.is_error, r.content
        assert (env.root / "win.txt").read_bytes() == b"A\r\nB\r\nline3\r\n"

    def test_an_lf_file_is_untouched_by_the_crlf_logic(self, env):
        (env.root / "unix.txt").write_bytes(b"x\ny\n")
        env.run(EditTool(), {"file_path": "unix.txt", "old_string": "x\ny", "new_string": "p\nq"})
        assert (env.root / "unix.txt").read_bytes() == b"p\nq\n"
