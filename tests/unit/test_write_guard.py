"""읽지 않은 파일을 말없이 덮어쓰지 않는다 (4.51.0).

참고 하네스의 기본 계약이다 — 파일을 바꾸려면 먼저 읽는다. 우리에겐 없어서
`Write` 가 이미 있는 파일을 내용도 모른 채 잘라내고 새로 쓸 수 있었다. 사용자가
올린 문서·이전 세션 산출물이 그렇게 사라져도 **로그에는 흔적이 없다**(성공한
쓰기다). 그래서 이 구멍은 기록으로 셀 수 없고 계약으로만 막는다.

마찰 크기는 실측했다 (dev 30일): Write 1,016건 중 **841건이 그 대화에서 처음
건드리는 경로** — 대부분 새 파일이라 그대로 통과한다. Edit 은 372건 중 371건이
이미 내용을 알고 있어(Read 또는 자기가 Write) 손대지 않았다.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace



from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in._file_witness import (
    MAX_ENTRIES,
    WITNESSED_KEY,
    is_witnessed,
    witnessed_mutation,
)
from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
from xgen_agent_runtime.tools.built_in.write_tool import WriteTool


def _ctx(tmp_path, shared=None):
    return ToolContext(
        session_id="t",
        working_dir=str(tmp_path),
        state_view=SimpleNamespace(shared=shared if shared is not None else {}),
    )


def _run(tool, payload, ctx):
    return asyncio.run(tool.execute(payload, ctx))


class TestLedger:
    def test_records_both_spellings(self):
        view = SimpleNamespace(shared={})
        mut = witnessed_mutation(view, "a.txt", "/abs/a.txt")
        assert mut[WITNESSED_KEY] == ["a.txt", "/abs/a.txt"]

    def test_reread_does_not_duplicate(self):
        view = SimpleNamespace(shared={WITNESSED_KEY: ["a", "b"]})
        assert witnessed_mutation(view, "a")[WITNESSED_KEY] == ["b", "a"]

    def test_forgets_the_oldest_beyond_the_cap(self):
        view = SimpleNamespace(shared={WITNESSED_KEY: [str(i) for i in range(MAX_ENTRIES)]})
        book = witnessed_mutation(view, "new")[WITNESSED_KEY]
        assert len(book) == MAX_ENTRIES
        assert book[-1] == "new" and "0" not in book

    def test_unknown_state_view_is_not_witnessed(self):
        assert is_witnessed(None, "/a") is False


class TestWriteGuard:
    def test_new_file_is_written_without_reading(self, tmp_path):
        """마찰을 만들지 않는 자리 — 실측상 Write 의 대부분이다."""
        target = tmp_path / "new.txt"
        r = _run(WriteTool(), {"file_path": str(target), "content": "hi"}, _ctx(tmp_path))
        assert r.is_error is False
        assert target.read_text() == "hi"

    def test_existing_file_is_not_clobbered_blindly(self, tmp_path):
        target = tmp_path / "기능요구사항.docx"
        target.write_text("사용자가 올린 내용")
        r = _run(WriteTool(), {"file_path": str(target), "content": "덮어씀"}, _ctx(tmp_path))
        assert r.is_error is True
        assert "have not read it" in r.content
        assert target.read_text() == "사용자가 올린 내용"  # 그대로다

    def test_reading_first_allows_the_write(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("before")
        shared: dict = {}
        ctx = _ctx(tmp_path, shared)
        read = _run(ReadTool(), {"file_path": str(target)}, ctx)
        shared.update(read.state_mutations)  # 파이프라인이 하는 일
        r = _run(WriteTool(), {"file_path": str(target), "content": "after"}, ctx)
        assert r.is_error is False
        assert target.read_text() == "after"

    def test_an_empty_file_is_not_protected(self, tmp_path):
        """잃을 내용이 없다 — 막을 이유도 없다."""
        target = tmp_path / "empty.txt"
        target.write_text("")
        r = _run(WriteTool(), {"file_path": str(target), "content": "x"}, _ctx(tmp_path))
        assert r.is_error is False

    def test_the_refusal_names_the_next_move(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("x")
        r = _run(WriteTool(), {"file_path": str(target), "content": "y"}, _ctx(tmp_path))
        assert "Read it first" in r.content
        assert "Edit" in r.content
        assert "Nothing was written" in r.content

    def test_a_relative_path_read_covers_the_resolved_write(self, tmp_path):
        target = tmp_path / "rel.txt"
        target.write_text("before")
        shared: dict = {}
        ctx = _ctx(tmp_path, shared)
        read = _run(ReadTool(), {"file_path": "rel.txt"}, ctx)
        assert read.is_error is False
        shared.update(read.state_mutations)
        r = _run(WriteTool(), {"file_path": str(target), "content": "after"}, ctx)
        assert r.is_error is False
