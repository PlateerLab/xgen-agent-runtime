"""WriteTool — create or overwrite a file."""

from __future__ import annotations

from typing import Any, Dict

from xgen_agent_runtime.tools.built_in._file_witness import is_witnessed, refusal
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in._path_guard import resolve_and_validate


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

        # Sandbox: write the file into the agent's XGeny session.
        if context.sandbox is not None:
            from xgen_agent_runtime.tools._xgeny_sandbox import sb_write_bytes

            wd = context.working_dir or "/workspace"

            # 읽지 않은 파일을 말없이 덮어쓰지 않는다 (_file_witness).
            if not is_witnessed(context.state_view, file_path):
                from xgen_agent_runtime.tools._xgeny_sandbox import sb_read_bytes

                try:
                    existing = await sb_read_bytes(context.sandbox, file_path, workdir=wd)
                except Exception:  # noqa: BLE001 — 없는 파일이면 그대로 새로 쓴다
                    existing = b""
                if existing:
                    return ToolResult(content=refusal(file_path), is_error=True)

            try:
                n = await sb_write_bytes(
                    context.sandbox, file_path, content.encode("utf-8"), workdir=wd
                )
                return ToolResult(content=f"Successfully wrote {n} bytes to {file_path}")
            except PermissionError as e:
                return ToolResult(content=str(e), is_error=True)
            except Exception as e:  # noqa: BLE001
                return ToolResult(content=f"Write error: {e}", is_error=True)

        try:
            resolved = resolve_and_validate(file_path, context.working_dir, context.allowed_paths)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)

        # 읽지 않은 파일을 말없이 덮어쓰지 않는다 (_file_witness). 새 파일은 그대로
        # 통과한다 — 막으려는 것은 "내용을 모른 채 지우는 일" 뿐이다.
        if (
            resolved.exists()
            and resolved.is_file()
            and resolved.stat().st_size > 0
            and not is_witnessed(context.state_view, file_path)
            and not is_witnessed(context.state_view, str(resolved))
        ):
            return ToolResult(content=refusal(str(resolved)), is_error=True)

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            size = resolved.stat().st_size
            return ToolResult(content=f"Successfully wrote {size} bytes to {resolved}")
        except OSError as e:
            return ToolResult(content=f"Write error: {e}", is_error=True)
