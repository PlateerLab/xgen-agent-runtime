from __future__ import annotations

from xgen_agent_runtime.tools.base import HOST_IS_EXECUTION_TARGET, ToolContext
from xgen_agent_runtime.tools.built_in.bash_tool import BashTool


def _context(tmp_path) -> ToolContext:
    return ToolContext(
        working_dir=str(tmp_path),
        allowed_paths=[str(tmp_path)],
        extras={HOST_IS_EXECUTION_TARGET: True},
    )


async def test_csv_contract_rejects_inconsistent_rows(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": "printf 'a,b\\n1,2,3\\n' > out.csv",
            "artifact_contracts": [
                {
                    "path": "out.csv",
                    "format": "csv",
                    "columns": ["a", "b"],
                    "exact_rows": 1,
                    "allowed_values": {"a": ["1", "2"]},
                    "unique_by": ["a"],
                }
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is True
    assert "ARTIFACT VALIDATION FAILED" in result.content
    assert "inconsistent column counts at lines 2" in result.content
    assert result.metadata["exit_code"] == 0
    assert result.metadata["artifact_validation"]["ok"] is False


async def test_csv_contract_accepts_standard_writer_output(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": (
                "python3 - <<'PY'\n"
                "import csv\n"
                "with open('out.csv', 'w', newline='') as f:\n"
                "    csv.writer(f).writerows([['a','b'], ['1','2']])\n"
                "PY"
            ),
            "artifact_contracts": [
                {
                    "path": "out.csv",
                    "format": "csv",
                    "columns": ["a", "b"],
                    "exact_rows": 1,
                }
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is False
    assert "ARTIFACT VALIDATION OK" in result.content
    assert "1 data rows, 2 columns" in result.content
    assert result.artifacts == {"validated": ["out.csv"]}


async def test_csv_contract_rejects_disallowed_and_duplicate_values(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": "printf 'id,status\\na,ok\\na,unknown\\n' > out.csv",
            "artifact_contracts": [
                {
                    "path": "out.csv",
                    "format": "csv",
                    "allowed_values": {"status": ["ok", "warn"]},
                    "unique_by": ["id"],
                }
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is True
    assert "column 'status' has disallowed value 'unknown'" in result.content
    assert "duplicate key ('a',)" in result.content


async def test_csv_contract_rejects_duplicate_header_names(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": "printf 'id,id\\na,b\\n' > out.csv",
            "artifact_contracts": [{"path": "out.csv", "format": "csv"}],
        },
        _context(tmp_path),
    )

    assert result.is_error is True
    assert "CSV header has duplicate names: id" in result.content


async def test_json_and_text_constraints_report_precise_failures(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": (
                "printf '{\"items\": []}' > out.json; "
                "printf 'do not mention secret-token' > notes.md"
            ),
            "artifact_contracts": [
                {
                    "path": "out.json",
                    "format": "json",
                    "required_keys": ["items", "summary"],
                    "array_lengths": {"/items": 2},
                },
                {
                    "path": "notes.md",
                    "format": "text",
                    "forbidden_strings": ["secret-token"],
                    "required_strings": ["do not mention"],
                },
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is True
    assert "missing top-level JSON keys: summary" in result.content
    # 개수는 실패 사유가 아니라 참고 — 실패는 구조(키 누락)·금지 문구 때문이다.
    assert "note: out.json: 0 items at '/items' (contract said 2)" in result.content
    assert "forbidden string is present: 'secret-token'" in result.content


async def test_json_pointer_array_length_accepts_nested_arrays(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": "printf '{\"payload\":{\"items\":[1,2]}}' > out.json",
            "artifact_contracts": [
                {
                    "path": "out.json",
                    "format": "json",
                    "array_lengths": {"/payload/items": 2},
                }
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is False
    assert "ARTIFACT VALIDATION OK" in result.content


async def test_required_keys_apply_to_each_object_in_a_json_array(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": "printf '[{\"id\":1},{\"id\":2}]' > out.json",
            "artifact_contracts": [
                {
                    "path": "out.json",
                    "format": "json",
                    "required_keys": ["id"],
                    "array_lengths": {"": 2},
                }
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is False
    assert "ARTIFACT VALIDATION OK" in result.content


async def test_contract_rejects_paths_outside_the_workspace(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": "printf ok > out.txt",
            "artifact_contracts": [{"path": "../out.txt", "format": "text"}],
        },
        _context(tmp_path),
    )

    assert result.is_error is True
    assert "invalid workspace-relative artifact path" in result.content


async def test_contract_report_is_returned_without_masking_command_failure(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": "printf ok > out.txt; exit 7",
            "artifact_contracts": [{"path": "out.txt", "format": "text"}],
        },
        _context(tmp_path),
    )

    assert result.is_error is True
    assert result.metadata["exit_code"] == 7
    assert "ARTIFACT VALIDATION OK" in result.content


async def test_bash_without_contract_preserves_existing_behavior(tmp_path) -> None:
    result = await BashTool().execute(
        {"command": "printf ok"},
        _context(tmp_path),
    )

    assert result.is_error is False
    assert result.content == "ok"
    assert "artifact_validation" not in result.metadata
    assert result.artifacts == {}


# ── 2026-09-26 dev 실사용 점검에서 나온 세 가지 ────────────────────────────


async def test_csv_with_utf8_bom_passes_header_check(tmp_path) -> None:
    """엑셀 호환 CSV(utf-8-sig)의 BOM 은 표준 parser 처럼 떼고 본다 — 예전엔 헤더 불일치로 실패."""
    result = await BashTool().execute(
        {
            "command": (
                "python3 - <<'PY'\n"
                "import csv\n"
                "with open('out.csv', 'w', newline='', encoding='utf-8-sig') as f:\n"
                "    csv.writer(f).writerows([['조항', '변경'], ['제2조', '24개월']])\n"
                "PY"
            ),
            "artifact_contracts": [{"path": "out.csv", "format": "csv", "columns": ["조항", "변경"]}],
        },
        _context(tmp_path),
    )

    assert result.is_error is False, result.content
    assert "ARTIFACT VALIDATION OK" in result.content


async def test_row_count_mismatch_is_a_note_not_a_failure(tmp_path) -> None:
    """요청에 없는 개수를 모델이 지어내면 데이터를 늘리며 헛돈다(dev: 4→9행, Bash 7회) — 실패로 보지 않는다."""
    result = await BashTool().execute(
        {
            "command": "printf 'a,b\\n1,2\\n3,4\\n' > out.csv",
            "artifact_contracts": [{"path": "out.csv", "format": "csv", "columns": ["a", "b"], "exact_rows": 5}],
        },
        _context(tmp_path),
    )

    assert result.is_error is False, result.content
    assert "ARTIFACT VALIDATION OK" in result.content
    assert "note: out.csv: 2 data rows (contract said 5)" in result.content
    assert "never add or remove rows just to match" in result.content


async def test_binary_output_is_skipped_and_text_outputs_are_still_checked(tmp_path) -> None:
    """xlsx 같은 바이너리는 검사 대상이 아니다 — 같이 적은 CSV 는 그대로 검사한다(예전엔 전체 실패)."""
    result = await BashTool().execute(
        {
            "command": (
                "printf 'PK\\003\\004\\000\\000binary' > report.xlsx; "
                "printf 'a,b\\n1,2\\n' > out.csv"
            ),
            "artifact_contracts": [
                {"path": "report.xlsx", "format": "text"},
                {"path": "out.csv", "format": "csv", "columns": ["a", "b"]},
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is False, result.content
    assert "out.csv: csv, 1 data rows, 2 columns" in result.content
    assert "note: report.xlsx: not checked — not a text file" in result.content


async def test_binary_skip_does_not_hide_a_real_csv_failure(tmp_path) -> None:
    result = await BashTool().execute(
        {
            "command": (
                "printf 'PK\\003\\004\\000binary' > report.xlsx; "
                "printf 'a,b\\n1,2,3\\n' > out.csv"
            ),
            "artifact_contracts": [
                {"path": "report.xlsx", "format": "text"},
                {"path": "out.csv", "format": "csv"},
            ],
        },
        _context(tmp_path),
    )

    assert result.is_error is True
    assert "inconsistent column counts" in result.content
    assert "note: report.xlsx: not checked" in result.content
