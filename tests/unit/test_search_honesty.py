"""Grep·Glob 은 **찾은 것을 정직하게 말한다** (2026-10-23 감사).

예전 결함(재현됨): 앞 200개 파일만 찾고 2MiB 넘는 파일을 건너뛰면서 아무 말이 없었고, content
모드에서 첫 파일의 매치가 300개를 넘으면 "No matches" 를 돌려줬다(count 모드는 400개). 중괄호 glob
(도구 설명의 예 ``*.{ts,tsx}``)은 아무것도 찾지 못했다. 한국어 출력은 러너 stdout 상한(200KB)을 넘어
JSON 이 깨졌다. 모델은 "없다" 고 믿고 지우거나 고친다.
"""

from __future__ import annotations

import json

from xgen_agent_runtime.tools.built_in import _search

#: 러너 exec 의 stdout 상한(xgen-workflow-sandbox app/runner/exec.py STDOUT_CAP).
RUNNER_STDOUT_CAP = 200_000


def _grep(base, pattern, **kw):
    return _search.run({"op": "grep", "base": str(base), "pattern": pattern, **kw})


def test_many_matches_in_one_file_are_shown_not_denied(tmp_path):
    (tmp_path / "big.py").write_text("\n".join(f"call_api({i})" for i in range(400)))
    assert _grep(tmp_path, "call_api", output_mode="count")["text"].startswith("400 matches in 1 files")
    first = _grep(tmp_path, "call_api", output_mode="content")["text"]
    assert "No matches" not in first
    assert first.count(":> ") == 300
    assert "matches 1-300 of 400 in 1 files; call Grep again with offset=300" in first
    second = _grep(tmp_path, "call_api", output_mode="content", offset=300)["text"]
    assert second.count(":> ") == 100 and "call_api(399)" in second
    assert "matches 301-400 of 400" in second


def test_every_file_is_searched(tmp_path):
    for i in range(250):
        (tmp_path / f"f{i:03d}.txt").write_text("nothing here\n")
    (tmp_path / "f230.txt").write_text("needle\n")
    out = _grep(tmp_path, "needle", output_mode="files")["text"]
    assert out.strip().endswith("f230.txt")


def test_large_files_are_read_line_by_line_not_skipped(tmp_path):
    big = tmp_path / "big.log"
    big.write_text("x" * 100 + "\n" * 1 + ("filler line\n" * 300_000) + "needle at the end\n")
    assert big.stat().st_size > 2 * 1024 * 1024
    out = _grep(tmp_path, "needle", output_mode="content")["text"]
    assert "needle at the end" in out


def test_brace_globs_match(tmp_path):
    (tmp_path / "a.ts").write_text("export const k = 1\n")
    (tmp_path / "b.tsx").write_text("export const k = 2\n")
    (tmp_path / "c.js").write_text("export const k = 3\n")
    out = _grep(tmp_path, "export", glob="*.{ts,tsx}", output_mode="files")["text"]
    assert "a.ts" in out and "b.tsx" in out and "c.js" not in out
    glob = _search.run({"op": "glob", "base": str(tmp_path), "pattern": "*.{ts,tsx}"})["text"]
    assert "a.ts" in glob and "b.tsx" in glob and "c.js" not in glob


def test_what_was_not_searched_is_said(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("needle\n")
    (tmp_path / "blob.bin").write_bytes(b"needle\0\0\0")
    (tmp_path / "a.txt").write_text("other\n")
    out = _grep(tmp_path, "needle", output_mode="files")["text"]
    assert "No matches" in out
    assert "not searched: node_modules/ (1)" in out
    assert "1 binary file(s) not searched" in out
    inside = _grep(tmp_path / "node_modules", "needle", output_mode="files")["text"]
    assert "dep.js" in inside


def test_korean_output_fits_the_runner_and_keeps_its_note(tmp_path):
    line = "주문 번호 처리 로직 " * 40
    (tmp_path / "ko.txt").write_text("\n".join(f"{i} {line}" for i in range(300)))
    result = _grep(tmp_path, "주문", output_mode="content")
    wire = json.dumps(result, ensure_ascii=False).encode("utf-8")
    assert len(wire) < RUNNER_STDOUT_CAP
    text = result["text"]
    assert "of 300 in 1 files; call Grep again with offset=" in text


def test_glob_pages_long_lists(tmp_path):
    for i in range(520):
        (tmp_path / f"g{i:03d}.md").write_text("x")
    out = _search.run({"op": "glob", "base": str(tmp_path), "pattern": "*.md"})["text"]
    assert "files 1-500 of 520" in out and "offset=500" in out
    rest = _search.run({"op": "glob", "base": str(tmp_path), "pattern": "*.md", "offset": 500})["text"]
    assert rest.count(".md") == 20 and "files 501-520 of 520" in rest
