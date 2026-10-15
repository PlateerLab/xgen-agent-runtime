"""ParseDocument — 문서의 글 위주 요소를 뽑는다(doc2chunk 추출 단계, 4.74.0).

문서 편집 도구(edit2docs 기반 Doc*)를 걷어 내고 남긴 유일한 문서 도구다. 결과는 원본의 전체 구조가
아니라 글 중심 표현이라는 것을 모델에게 말해야 한다.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("xgen_doc2chunk")

from xgen_agent_runtime.tools.base import ToolContext  # noqa: E402
from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_CLASSES, BUILT_IN_TOOL_FEATURES  # noqa: E402
from xgen_agent_runtime.tools.built_in.parse_document_tool import ParseDocumentTool  # noqa: E402


def _run(tool_input, tmp_path):
    ctx = ToolContext(session_id="s", working_dir=str(tmp_path))
    return asyncio.run(ParseDocumentTool().execute(tool_input, ctx))


def test_it_replaces_the_editing_tools():
    assert BUILT_IN_TOOL_FEATURES["parsing"] == ["ParseDocument"]
    assert "documents" not in BUILT_IN_TOOL_FEATURES
    assert not [name for name in BUILT_IN_TOOL_CLASSES if name.startswith("Doc")]
    desc = ParseDocumentTool().description
    assert "NOT the full document structure" in desc and "cannot be used to rebuild or edit" in desc


def test_a_word_document_gives_its_text_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_heading("분기 보고", level=1)
    doc.add_paragraph("매출이 12% 늘었다.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "지역", "매출"
    table.cell(1, 0).text, table.cell(1, 1).text = "서울", "120"
    doc.save(tmp_path / "report.docx")

    out = _run({"file_path": "report.docx"}, tmp_path)
    assert not out.is_error, out.content
    assert out.content.startswith("[ParseDocument: report.docx — text-centric extraction")
    assert "not the full document structure" in out.content
    assert "매출이 12% 늘었다." in out.content and "서울" in out.content and "120" in out.content


def test_a_spreadsheet_gives_its_sheets(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "요약"
    sheet.append(["품목", "수량"])
    sheet.append(["사과", 3])
    book.save(tmp_path / "items.xlsx")

    out = _run({"file_path": "items.xlsx"}, tmp_path)
    assert not out.is_error, out.content
    assert "요약" in out.content and "사과" in out.content


def test_long_results_come_in_parts(tmp_path):
    (tmp_path / "long.csv").write_text("a,b\n" + "\n".join(f"{i},{'x' * 40}" for i in range(400)), encoding="utf-8")
    first = _run({"file_path": "long.csv", "limit": 500}, tmp_path)
    assert "[chars 0-500 of" in first.content and "offset=500" in first.content
    second = _run({"file_path": "long.csv", "offset": 500, "limit": 500}, tmp_path)
    assert "[chars 500-1000 of" in second.content


def test_failures_are_told_plainly(tmp_path):
    (tmp_path / "bin.xyz").write_bytes(b"\x00\x01")
    assert "unsupported file type: .xyz" in _run({"file_path": "bin.xyz"}, tmp_path).content
    assert _run({"file_path": "missing.pdf"}, tmp_path).is_error
    assert _run({"path": "missing.pdf"}, tmp_path).is_error, "옛 인자 이름(path)은 라우터가 file_path 로 옮긴다"
