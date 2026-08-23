"""Glob/Grep 호스트 경로도 allowed_paths 가드를 받는다 (커넥터 로컬 턴: PC 전역 열거 차단)."""
import asyncio
from pathlib import Path

from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.glob_tool import GlobTool
from xgen_agent_runtime.tools.built_in.grep_tool import GrepTool


def _ctx(ws: Path) -> ToolContext:
    return ToolContext(sandbox=None, working_dir=str(ws), allowed_paths=[str(ws)])


def test_glob_and_grep_reject_paths_outside_allowed(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("needle here")
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "secret.txt").write_text("needle secret")
    ctx = _ctx(ws)
    r = asyncio.run(GlobTool().execute({"pattern": "*.txt", "path": str(outside)}, ctx))
    assert r.is_error and "outside allowed" in r.content
    r = asyncio.run(GrepTool().execute({"pattern": "needle", "path": str(outside)}, ctx))
    assert r.is_error and "outside allowed" in r.content
    r = asyncio.run(GlobTool().execute({"pattern": "../outside/*.txt"}, ctx))
    # 상대 패턴으로 밖을 훑는 결과는 필터된다
    assert "secret.txt" not in (r.content or "")


def test_glob_and_grep_work_inside_allowed(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("needle here")
    ctx = _ctx(ws)
    r = asyncio.run(GlobTool().execute({"pattern": "*.txt"}, ctx))
    assert not r.is_error and "a.txt" in r.content
    r = asyncio.run(GrepTool().execute({"pattern": "needle"}, ctx))
    assert not r.is_error and "a.txt" in r.content


def test_display_result_keeps_tail_marker():
    from xgen_agent_runtime.host import runner

    text = "x" * 6000 + "\n[[download:/results/report.docx]]"
    shown = runner._display_result(text)
    assert len(shown) < 6000
    assert shown.endswith("[[download:/results/report.docx]]")
    assert "chars truncated" in shown
    assert runner._display_result("short") == "short"
