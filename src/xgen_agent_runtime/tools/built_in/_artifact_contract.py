"""Deterministic post-command validation for model-declared artifacts.

The model supplies structural expectations alongside a Bash action.  This
module enforces only those expectations with standard parsers; it contains no
task names, business vocabulary, or benchmark answers.  Validation is bounded
and executes through the same filesystem port as the command's workspace.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from xgen_agent_runtime.tools.fs import tool_fs

MAX_CONTRACTS = 8
MAX_VALIDATION_BYTES = 512 * 1024


@dataclass(frozen=True)
class ArtifactContractReport:
    ok: bool
    message: str
    checked: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": list(self.checked),
            "errors": list(self.errors),
        }


#: 개수(행 수·배열 길이)는 실패로 보지 않고 알리기만 한다. 요청에 개수가 없는데도 모델이 계약에 개수를
#: 지어 넣고, 불일치가 나면 데이터와 기대 개수를 같이 늘리며 헛돌았다(2026-09-26 dev: 계약서 비교에서 4→9행,
#: Bash 7회·185초. Codex 실험실에서도 요청에 없던 exact_rows=9). 형식 검사(파싱·헤더·열 수·허용값·중복)는
#: 객관적이라 그대로 실패로 둔다.
_COUNT_NOTE = (
    "Counts are informational: change the data only if the request itself states this count; "
    "never add or remove rows just to match the contract."
)

#: BOM(엑셀 호환 CSV 에 흔하다)은 표준 parser 처럼 떼고 본다(utf-8-sig). 예전에는 첫 헤더가 '\ufeff조항'
#: 이 되어 헤더 불일치로 실패했다.
_BOM = "\ufeff"


def _slash(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def _relative_contract_path(path: Any, root: str) -> str | None:
    candidate = _slash(path).strip()
    normalized_root = _slash(root).rstrip("/")
    if normalized_root and candidate.startswith(normalized_root + "/"):
        candidate = candidate[len(normalized_root) + 1 :]
    elif candidate == normalized_root:
        return None
    elif candidate.startswith("/") or (len(candidate) >= 2 and candidate[1] == ":"):
        return None
    candidate = candidate.removeprefix("./")
    pure = PurePosixPath(candidate)
    if not candidate or candidate == "." or ".." in pure.parts:
        return None
    return pure.as_posix()


def _string_list(contract: Mapping[str, Any], key: str) -> list[str]:
    value = contract.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_text_constraints(
    path: str,
    content: str,
    contract: Mapping[str, Any],
    errors: list[str],
) -> None:
    for required in _string_list(contract, "required_strings"):
        if required not in content:
            errors.append(f"{path}: required string is missing: {required!r}")
    for forbidden in _string_list(contract, "forbidden_strings"):
        if forbidden in content:
            errors.append(f"{path}: forbidden string is present: {forbidden!r}")


def _json_pointer(document: Any, pointer: str) -> tuple[bool, Any]:
    if pointer == "":
        return True, document
    if not pointer.startswith("/"):
        return False, None
    current = document
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return False, None
    return True, current


def _validate_json(
    path: str,
    content: str,
    contract: Mapping[str, Any],
    errors: list[str],
    notes: list[str],
) -> str:
    try:
        document = json.loads(content, object_pairs_hook=_duplicate_rejecting_object)
    except (ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: invalid JSON: {exc}")
        return "json"

    required_keys = _string_list(contract, "required_keys")
    if required_keys:
        if isinstance(document, dict):
            missing = [key for key in required_keys if key not in document]
            if missing:
                errors.append(f"{path}: missing top-level JSON keys: {', '.join(missing)}")
        elif isinstance(document, list) and all(isinstance(item, dict) for item in document):
            for index, item in enumerate(document):
                missing = [key for key in required_keys if key not in item]
                if missing:
                    errors.append(
                        f"{path}: JSON item {index} is missing keys: {', '.join(missing)}"
                    )
        else:
            errors.append(f"{path}: required_keys needs an object or an array of objects")

    array_lengths = contract.get("array_lengths")
    if isinstance(array_lengths, dict):
        for pointer, expected in array_lengths.items():
            if not isinstance(pointer, str) or not isinstance(expected, int):
                errors.append(f"{path}: array_lengths must map JSON Pointers to integers")
                continue
            found, value = _json_pointer(document, pointer)
            if not found:
                errors.append(f"{path}: JSON Pointer not found: {pointer!r}")
            elif not isinstance(value, list):
                errors.append(f"{path}: JSON Pointer {pointer!r} does not select an array")
            elif len(value) != expected:
                notes.append(
                    f"{path}: {len(value)} items at {pointer!r} (contract said {expected}). {_COUNT_NOTE}"
                )
    return "json"


def _validate_csv(
    path: str,
    content: str,
    contract: Mapping[str, Any],
    errors: list[str],
    notes: list[str],
) -> str:
    try:
        rows = list(csv.reader(io.StringIO(content, newline=""), strict=True))
    except csv.Error as exc:
        errors.append(f"{path}: invalid CSV: {exc}")
        return "csv"
    if not rows:
        errors.append(f"{path}: CSV is empty")
        return "csv"

    width = len(rows[0])
    duplicate_headers = sorted({name for name in rows[0] if rows[0].count(name) > 1})
    if duplicate_headers:
        errors.append(f"{path}: CSV header has duplicate names: {', '.join(duplicate_headers)}")
    inconsistent = [index + 1 for index, row in enumerate(rows) if len(row) != width]
    if inconsistent:
        shown = ", ".join(str(index) for index in inconsistent[:5])
        suffix = "..." if len(inconsistent) > 5 else ""
        errors.append(f"{path}: CSV rows have inconsistent column counts at lines {shown}{suffix}")

    columns = _string_list(contract, "columns")
    if columns and rows[0] != columns:
        errors.append(f"{path}: CSV header mismatch; expected {columns!r}, found {rows[0]!r}")

    header_index = {name: index for index, name in enumerate(rows[0])}
    allowed_values = contract.get("allowed_values")
    if isinstance(allowed_values, dict):
        for column, raw_allowed in allowed_values.items():
            if column not in header_index:
                errors.append(f"{path}: allowed_values names an absent CSV column: {column!r}")
                continue
            if not isinstance(raw_allowed, list) or not all(
                isinstance(value, str) for value in raw_allowed
            ):
                errors.append(f"{path}: allowed_values[{column!r}] must be a string list")
                continue
            allowed = set(raw_allowed)
            index = header_index[column]
            for line, row in enumerate(rows[1:], start=2):
                if len(row) > index and row[index] not in allowed:
                    errors.append(
                        f"{path}: line {line} column {column!r} has disallowed value {row[index]!r}"
                    )

    unique_by = _string_list(contract, "unique_by")
    if unique_by:
        absent = [column for column in unique_by if column not in header_index]
        if absent:
            errors.append(f"{path}: unique_by names absent CSV columns: {', '.join(absent)}")
        else:
            seen: dict[tuple[str, ...], int] = {}
            indices = [header_index[column] for column in unique_by]
            for line, row in enumerate(rows[1:], start=2):
                if any(len(row) <= index for index in indices):
                    continue
                key = tuple(row[index] for index in indices)
                if key in seen:
                    errors.append(
                        f"{path}: duplicate key {key!r} for unique_by {unique_by!r} "
                        f"at lines {seen[key]} and {line}"
                    )
                else:
                    seen[key] = line

    data_rows = max(0, len(rows) - 1)
    exact_rows = contract.get("exact_rows")
    min_rows = contract.get("min_rows")
    max_rows = contract.get("max_rows")
    if isinstance(exact_rows, int) and data_rows != exact_rows:
        notes.append(f"{path}: {data_rows} data rows (contract said {exact_rows}). {_COUNT_NOTE}")
    if isinstance(min_rows, int) and data_rows < min_rows:
        notes.append(
            f"{path}: {data_rows} data rows (contract said at least {min_rows}). {_COUNT_NOTE}"
        )
    if isinstance(max_rows, int) and data_rows > max_rows:
        notes.append(
            f"{path}: {data_rows} data rows (contract said at most {max_rows}). {_COUNT_NOTE}"
        )
    return f"csv, {data_rows} data rows, {width} columns"


async def validate_artifact_contracts(
    contracts: Sequence[Mapping[str, Any]],
    context: Any,
) -> ArtifactContractReport:
    """Validate model-declared output contracts through the active filesystem."""
    if not contracts:
        return ArtifactContractReport(ok=True, message="ARTIFACT VALIDATION OK: no contracts")
    if len(contracts) > MAX_CONTRACTS:
        error = f"too many artifact contracts: {len(contracts)} > {MAX_CONTRACTS}"
        return ArtifactContractReport(
            ok=False,
            message=f"ARTIFACT VALIDATION FAILED\n- {error}",
            errors=(error,),
        )

    fs = tool_fs(context)
    try:
        await fs.exists(".")
        root = fs.resolve(".")
    except Exception as exc:  # noqa: BLE001 — surface a compact tool error
        error = f"workspace inspection failed: {type(exc).__name__}: {exc}"
        return ArtifactContractReport(
            ok=False,
            message=f"ARTIFACT VALIDATION FAILED\n- {error}",
            errors=(error,),
        )

    errors: list[str] = []
    normalized: list[tuple[Mapping[str, Any], str]] = []
    for contract in contracts:
        path = _relative_contract_path(contract.get("path"), root)
        if path is None:
            errors.append(f"invalid workspace-relative artifact path: {contract.get('path')!r}")
        else:
            normalized.append((contract, path))

    try:
        selected = await fs.search(
            {
                "op": "read_texts",
                "base": root,
                "paths": [path for _contract, path in normalized],
                "max_files": MAX_CONTRACTS,
                "max_bytes": MAX_VALIDATION_BYTES,
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface a compact tool error
        error = f"workspace inspection failed: {type(exc).__name__}: {exc}"
        return ArtifactContractReport(
            ok=False,
            message=f"ARTIFACT VALIDATION FAILED\n- {error}",
            errors=(error,),
        )

    if not selected.get("ok") or not selected.get("eligible"):
        reason = str(selected.get("reason") or "workspace_unavailable")
        path = str(selected.get("path") or "").strip()
        error = f"bounded artifact read was unavailable: {reason}"
        if path:
            error += f" ({path})"
        return ArtifactContractReport(
            ok=False,
            message=f"ARTIFACT VALIDATION FAILED\n- {error}",
            errors=(error,),
        )

    files = {
        str(entry.get("path")): str(entry.get("content"))
        for entry in selected.get("files", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and isinstance(entry.get("content"), str)
    }
    skipped = {
        str(entry.get("path"))
        for entry in selected.get("skipped", []) or []
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    checked: list[str] = []
    summaries: list[str] = []
    notes: list[str] = []
    for contract, path in normalized:
        if path in skipped:
            notes.append(
                f"{path}: not checked — not a text file (contracts check JSON, CSV and text only)"
            )
            continue
        content = files.get(path)
        if content is None:
            errors.append(f"{path}: required artifact is missing or is not regular UTF-8 text")
            continue
        if content.startswith(_BOM):
            content = content[len(_BOM) :]

        checked.append(path)
        before = len(errors)
        _validate_text_constraints(path, content, contract, errors)
        artifact_format = str(contract.get("format") or "text").lower()
        summary = "text"
        if artifact_format == "json":
            summary = _validate_json(path, content, contract, errors, notes)
        elif artifact_format == "csv":
            summary = _validate_csv(path, content, contract, errors, notes)
        elif artifact_format != "text":
            errors.append(f"{path}: unsupported artifact format: {artifact_format!r}")
        if len(errors) == before:
            summaries.append(f"{path}: {summary}")

    note_lines = "".join(f"\n- note: {item}" for item in notes)
    if errors:
        message = "ARTIFACT VALIDATION FAILED\n" + "\n".join(f"- {item}" for item in errors)
        return ArtifactContractReport(
            ok=False,
            message=message + note_lines,
            checked=tuple(checked),
            errors=tuple(errors),
        )
    message = "ARTIFACT VALIDATION OK" + "".join(f"\n- {item}" for item in summaries)
    return ArtifactContractReport(ok=True, message=message + note_lines, checked=tuple(checked))


__all__ = [
    "ArtifactContractReport",
    "MAX_CONTRACTS",
    "validate_artifact_contracts",
]
