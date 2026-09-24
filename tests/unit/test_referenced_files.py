"""요청에 이름이 나온 작업 폴더 파일을 첫 턴에 붙인다 (host/referenced_files.py)."""

import asyncio

from xgen_agent_runtime.host.referenced_files import candidates, collect
from xgen_agent_runtime.tools.fs import LocalFS


def _run(text, root, **kw):
    return asyncio.run(collect(text, LocalFS(str(root)), **kw))


def test_candidates_cover_absolute_relative_and_bare_names():
    text = "Inputs: `/ws/in/a.json`, in/calendars/alex.json and **sales.csv**. See https://x.com/a.html"
    c = candidates(text)
    assert "/ws/in/a.json" in c and "in/calendars/alex.json" in c and "sales.csv" in c
    assert not any("x.com" in x for x in c)


def test_existing_text_files_are_attached_and_witnessed(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.json").write_text('{"k": 1}', encoding="utf-8")
    (tmp_path / "회의록.md").write_text("# 회의", encoding="utf-8")
    pre = _run(f"read {tmp_path}/in/a.json and 회의록.md, then write out/r.txt", tmp_path)
    assert '{"k": 1}' in pre.block and "# 회의" in pre.block
    assert str(tmp_path / "in" / "a.json") in pre.witnessed
    assert "out/r.txt" not in pre.block  # 없는 파일(만들 결과물)은 조용히 건너뛴다


def test_outside_the_working_folder_is_ignored(tmp_path):
    outside = tmp_path.parent / "outside_secret_plan.txt"
    outside.write_text("nope", encoding="utf-8")
    pre = _run(f"look at {outside}", tmp_path)
    assert pre.block == ""


def test_secretish_names_are_never_attached(tmp_path):
    (tmp_path / ".env").write_text("TOKEN=abc", encoding="utf-8")
    (tmp_path / "credentials.json").write_text("{}", encoding="utf-8")
    pre = _run("check .env and credentials.json", tmp_path)
    assert "TOKEN=abc" not in pre.block and pre.attached == []


def test_binary_files_are_named_not_attached(tmp_path):
    (tmp_path / "policy.pdf").write_bytes(b"%PDF-1.4\x00\x01binary")
    pre = _run("apply policy.pdf", tmp_path)
    assert "policy.pdf" in pre.block and "binary .pdf" in pre.block and "%PDF" not in pre.block
    assert pre.witnessed == []


def test_large_files_are_truncated_and_not_witnessed(tmp_path):
    (tmp_path / "big.log").write_text("x" * 50_000, encoding="utf-8")
    pre = _run("summarize big.log", tmp_path, per_file=1_000)
    assert "truncated" in pre.block and str(tmp_path / "big.log") in pre.attached
    assert pre.witnessed == []  # 일부만 봤다 — 덮어쓰기 허락은 주지 않는다


def test_total_budget_stops_attaching(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(str(i) * 900, encoding="utf-8")
    pre = _run("use f0.txt f1.txt f2.txt", tmp_path, per_file=1_000, total=1_500)
    assert len(pre.attached) == 2 and "budget used up" in pre.block


def test_nothing_named_means_no_block(tmp_path):
    assert _run("안녕하세요, 오늘 일정 알려줘", tmp_path).block == ""
