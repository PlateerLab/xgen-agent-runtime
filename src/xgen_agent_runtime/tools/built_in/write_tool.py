"""WriteTool — create or overwrite a file."""

from __future__ import annotations

from typing import Any, Dict

from xgen_agent_runtime.tools.built_in._file_witness import (
    is_witnessed,
    refusal,
    witnessed_mutation,
)
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.fs import tool_fs


class WriteTool(Tool):
    """Write content to a file, creating parent directories as needed.

    Overwrites existing files. For partial modifications, use EditTool instead.
    """

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates parent directories if needed. "
            "Overwrites existing files. For partial edits, use the Edit tool."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the file to write (starts with `/`). "
                        "A relative path is resolved against your working folder."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        content = input.get("content", "")

        fs = tool_fs(context)
        try:
            resolved = fs.resolve(file_path, write=True)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)

        # 읽지 않은 기존 파일을 말없이 덮어쓰지 않는다 (_file_witness, 4.51.0).
        # 새 파일은 그대로 통과한다 — 막으려는 것은 "내용을 모른 채 지우는 일" 뿐이다.
        view = context.state_view
        if not (is_witnessed(view, file_path) or is_witnessed(view, resolved)):
            try:
                existing = await fs.read_bytes(file_path)
            except FileNotFoundError:
                existing = b""
            except IsADirectoryError:
                return ToolResult(
                    content=f"Cannot write: {resolved} is a directory.", is_error=True
                )
            except Exception:  # noqa: BLE001 — 확인 실패로 쓰기를 막지 않는다(예전 러너 동작)
                existing = b""
            if existing:
                return ToolResult(content=refusal(resolved), is_error=True)

        try:
            n = await fs.write_bytes(file_path, content.encode("utf-8"))
        except PermissionError as e:
            return ToolResult(content=str(e), is_error=True)
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"Write error: {e}", is_error=True)

        return ToolResult(
            content=f"Successfully wrote {n} bytes to {resolved}",
            # 방금 쓴 파일은 에이전트가 내용을 안다 — 장부에 올린다. 4.51.0 은 이걸 빠뜨려서
            # **자기가 방금 만든 파일을 다시 쓰면 거절**했다(Write→Write). 실측상 "자기가 쓴
            # 걸 다시 씀" 이 월 54건인데, 그 전부가 막힐 뻔했다.
            state_mutations=witnessed_mutation(view, file_path, resolved),
        )
