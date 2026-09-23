"""ReadTool — read file contents with line numbers."""

from __future__ import annotations

import mimetypes
from pathlib import PurePosixPath
from typing import Any, Dict

from xgen_agent_runtime.tools.built_in._file_witness import witnessed_mutation
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.fs import tool_fs

_DEFAULT_LIMIT = 2000


class ReadTool(Tool):
    """Read a file from the local filesystem.

    Returns content with line numbers (cat -n format).
    Detects binary files and refuses to read them (except images).
    """

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return (
            "Read a file from the filesystem. Returns content with line numbers. "
            "Use offset and limit to read specific portions of large files."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the file to read (starts with `/`). "
                        "A relative path is resolved against your working folder."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Default: 0.",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines to read. Default: {_DEFAULT_LIMIT}.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["file_path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Pure filesystem read — no side effects, safe to parallelise.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        offset = input.get("offset", 0)
        limit = input.get("limit", _DEFAULT_LIMIT)

        # 러너든 로컬이든 한 길로 읽는다 — 백엔드 선택은 tool_fs 한 곳에서만 일어난다.
        # 예전엔 분기마다 따로 구현해서 같은 파일에 두 곳이 다르게 답했다(이미지는
        # 로컬만 "Image file", 러너는 바이너리 검사에 걸려 "Binary file").
        fs = tool_fs(context)
        try:
            resolved = fs.resolve(file_path)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)
        name = PurePosixPath(resolved).name

        try:
            raw = await fs.read_bytes(file_path)
        except FileNotFoundError:
            return ToolResult(content=f"File not found: {resolved}", is_error=True)
        except IsADirectoryError:
            return ToolResult(
                content=f"Cannot read directory: {resolved}. Use Bash with 'ls' instead.",
                is_error=True,
            )
        except PermissionError as e:
            return ToolResult(content=str(e), is_error=True)
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"Read error: {e}", is_error=True)

        mime, _ = mimetypes.guess_type(name)
        if mime and mime.startswith("image/"):
            return ToolResult(content=f"[Image file: {name}, {len(raw)} bytes, type={mime}]")

        if b"\x00" in raw[:8192]:
            return ToolResult(content=f"[Binary file: {name}, {len(raw)} bytes]")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except Exception:
                return ToolResult(content=f"[Binary file: {name}, {len(raw)} bytes]")

        lines = text.splitlines(keepends=True)
        total = len(lines)
        selected = lines[offset : offset + limit]
        if not selected and total > 0:
            return ToolResult(
                content=f"Offset {offset} is beyond file end ({total} lines).", is_error=True
            )

        # 줄 번호는 1부터 (cat -n 과 같은 모양)
        output = "\n".join(
            f"{i}\t{line.rstrip()}" for i, line in enumerate(selected, start=offset + 1)
        )
        if offset + limit < total:
            output += f"\n\n... ({total - offset - limit} more lines, {total} total)"

        return ToolResult(
            content=output,
            # 이 세션에서 내용을 본 파일로 기록한다(_file_witness). 모델이 준 표기와
            # 해석된 절대 경로 둘 다 — Write 가 어느 쪽으로 와도 알아본다. 예전엔 러너
            # 분기만 모델 표기 하나를 적어서, 상대 경로로 읽고 절대 경로로 쓰면 러너에서만
            # 거절됐다.
            state_mutations=witnessed_mutation(context.state_view, file_path, resolved),
        )
