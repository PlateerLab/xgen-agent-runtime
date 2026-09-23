"""Glob·Grep 은 러너와 로컬에서 **같은 코드**를 돌린다 (파일시스템 포트 2단계).

예전엔 두 곳이 서로 다른 프로그램이었다 — 러너는 셸 ``for f in {pattern}`` / ``grep -E``,
로컬은 ``Path.glob`` / Python ``re``. 그래서:

* ``Grep("order_id = \\d+")`` 가 로컬은 찾고 러너는 **"No matches"** (grep -E 에 \\d 없음).
* 경로는 러너 상대(``src/a.py``·``./src/a.py``), 로컬 절대. 출력 모양·순서도 갈림.
* 러너 Glob 은 패턴을 셸에 따옴표 없이 넣어서 ``$(…)`` 가 **실행됐다**. Glob 은 read_only
  라 PLAN 모드에서도 확인 없이 돈다.

이제 ``_search.py`` 한 파일이 로컬(import)과 러너(``python3 -c``)에서 그대로 돈다.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from xgen_agent_runtime.tools._xgeny_sandbox import ExecResult
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.glob_tool import GlobTool
from xgen_agent_runtime.tools.built_in.grep_tool import GrepTool


class _Session:
    """세션 러너처럼 argv 를 **셸 없이** 실행한다. python3 는 이 인터프리터로 매핑."""

    def __init__(self, root: Path) -> None:
        self.workdir = str(root)
        self.extra_roots: list = []
        self.readonly_roots: list = []

    async def ensure(self) -> None:
        return None

    async def exec(self, argv, *, cwd=None, stdin=None, env=None, timeout_s=120.0):
        argv = [sys.executable if a == "python3" else a for a in argv[:1]] + list(argv[1:])
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd or self.workdir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C.UTF-8"},
        )
        out, err = await proc.communicate()
        return ExecResult(rc=proc.returncode or 0, stdout=out, stderr=err)

    async def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> int:
        Path(path).write_bytes(data)
        return len(data)


@pytest.fixture
def ws(tmp_path) -> Path:
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src/app.py").write_text("order_id = 12345\ndef handle(x):\n    return x\n")
    (root / "src/util.py").write_text("def helper():\n    pass\n")
    (root / "README.md").write_text("# 기능요구사항\nversion 2.0\n")
    (root / "node_modules").mkdir()
    (root / "node_modules/dep.py").write_text("def vendored():\n    pass\n")
    return root


def _both(ws: Path, tool, payload):
    local = ToolContext(session_id="t", working_dir=str(ws), allowed_paths=[str(ws)])
    runner = ToolContext(session_id="t", working_dir=str(ws), sandbox=_Session(ws))
    a = asyncio.run(tool.execute(payload, local))
    b = asyncio.run(tool.execute(payload, runner))
    return a, b


CASES = [
    (GlobTool, {"pattern": "**/*.py"}),
    (GlobTool, {"pattern": "*.md"}),
    (GlobTool, {"pattern": "*.xyz"}),
    (GlobTool, {"pattern": "*.py", "path": "src"}),
    (GrepTool, {"pattern": "def "}),
    (GrepTool, {"pattern": "def ", "output_mode": "content"}),
    (GrepTool, {"pattern": r"order_id = \d+", "output_mode": "content"}),
    (GrepTool, {"pattern": r"\w+\(", "output_mode": "count"}),
    (GrepTool, {"pattern": "DEF", "case_insensitive": True}),
    (GrepTool, {"pattern": "기능", "output_mode": "content"}),
    (GrepTool, {"pattern": "return", "output_mode": "content", "context": 1}),
    (GrepTool, {"pattern": "def", "glob": "util.py"}),
]


@pytest.mark.parametrize("tool_cls,payload", CASES, ids=[f"{c.__name__}:{p}" for c, p in CASES])
def test_both_backends_give_the_same_answer(ws, tool_cls, payload):
    a, b = _both(ws, tool_cls(), payload)
    assert (a.is_error, a.content) == (b.is_error, b.content)


def test_perl_style_regex_finds_the_line_on_the_runner(ws):
    """grep -E 시절 러너는 \\d 를 몰라 "No matches" 였다."""
    _, runner = _both(ws, GrepTool(), {"pattern": r"order_id = \d+", "output_mode": "content"})
    assert "order_id = 12345" in runner.content


def test_paths_are_absolute_on_the_runner(ws):
    _, runner = _both(ws, GlobTool(), {"pattern": "**/*.py"})
    for line in runner.content.splitlines():
        assert line.startswith(str(ws)), line


def test_noise_directories_are_skipped(ws):
    _, runner = _both(ws, GrepTool(), {"pattern": "vendored"})
    assert runner.content.startswith("No matches")


@pytest.mark.parametrize(
    "payload",
    ["$(touch {m})", "*.py; do :; done; touch {m}; for f in x", "`touch {m}`"],
    ids=["command-substitution", "close-the-loop", "backticks"],
)
def test_a_glob_pattern_is_never_run_as_a_shell_command(ws, payload):
    marker = ws / "INJECTED"
    runner = ToolContext(session_id="t", working_dir=str(ws), sandbox=_Session(ws))
    asyncio.run(GlobTool().execute({"pattern": payload.format(m=marker)}, runner))
    assert not marker.exists()


def test_an_invalid_regex_is_reported_before_reaching_the_runner(ws):
    a, b = _both(ws, GrepTool(), {"pattern": "("})
    assert a.is_error and b.is_error and "Invalid regex" in b.content
