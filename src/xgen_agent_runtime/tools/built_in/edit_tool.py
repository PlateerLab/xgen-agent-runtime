"""EditTool — perform exact string replacements in files."""

from __future__ import annotations

from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in._file_witness import witnessed_mutation
from xgen_agent_runtime.tools.fs import tool_fs


class EditTool(Tool):
    """Replace exact string occurrences in a file.

    By default, old_string must appear exactly once (for safety).
    Set replace_all=True to replace every occurrence.
    """

    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return (
            "Perform exact string replacements in a file. "
            "old_string must be unique in the file unless replace_all is true."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the file to modify (starts with `/`). "
                        "A relative path is resolved against your working folder."
                    ),
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find and replace.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false).",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = input.get("file_path", "")
        old_string = input.get("old_string", "")
        new_string = input.get("new_string", "")
        replace_all = input.get("replace_all", False)

        if not old_string:
            return ToolResult(content="old_string must not be empty", is_error=True)
        if old_string == new_string:
            return ToolResult(content="old_string and new_string must be different", is_error=True)

        # 치환 로직은 **한 벌**이다. 예전엔 러너·로컬 분기에 따로 있었고 서로 다르게 답했다
        # (성공 문구의 경로 표기, 경로 탈출이 한쪽은 "Access denied", 한쪽은 "Read error").
        fs = tool_fs(context)
        try:
            resolved = fs.resolve(file_path, write=True)
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)

        try:
            raw = await fs.read_bytes(file_path)
        except FileNotFoundError:
            return ToolResult(content=f"File not found: {resolved}", is_error=True)
        except IsADirectoryError:
            return ToolResult(content=f"Cannot edit a directory: {resolved}", is_error=True)
        except PermissionError as e:
            return ToolResult(content=str(e), is_error=True)
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"Read error: {e}", is_error=True)
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            return ToolResult(content=f"Read error: {e}", is_error=True)

        old, new = _match_line_endings(content, old_string, new_string)
        count = content.count(old)
        if count == 0:
            return ToolResult(
                content="old_string not found in file. Ensure the string matches exactly, including whitespace and indentation.",
                is_error=True,
            )
        if not replace_all and count > 1:
            return ToolResult(
                content=f"old_string appears {count} times in file. Provide more context to make it unique, or set replace_all=true.",
                is_error=True,
            )
        new_content = content.replace(old, new, count if replace_all else 1)

        try:
            await fs.write_bytes(file_path, new_content.encode("utf-8"))
        except PermissionError as e:
            return ToolResult(content=str(e), is_error=True)
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"Write error: {e}", is_error=True)

        return ToolResult(
            content=f"Successfully edited {resolved} ({count} replacement{'s' if count > 1 else ''})",
            # 방금 고친 파일은 내용을 안다(old_string 이 정확히 맞아야 고칠 수 있다).
            state_mutations=witnessed_mutation(context.state_view, file_path, resolved),
        )


def _match_line_endings(content: str, old: str, new: str) -> tuple:
    """CRLF 파일에서 LF 로 쓴 old_string 을 맞춰 준다 — 파일의 줄 끝은 그대로 둔다.

    모델은 거의 항상 ``\n`` 으로 쓴다. 파일이 ``\r\n`` 이면 그대로는 한 번도 안
    맞는다. 예전 로컬 분기는 ``read_text()`` 가 줄 끝을 ``\n`` 으로 바꿔 읽고 그대로
    저장해서 "맞기는 했지만 **파일 전체의 줄 끝을 조용히 바꿔** 버렸고", 러너 분기는
    바이트 그대로라 아예 안 맞았다. 사용자 PC 커넥터에는 윈도우도 있다.

    파일이 CRLF 이고 old_string 에 CR 이 없을 때만, old/new 의 ``\n`` 을 ``\r\n`` 으로
    바꿔 맞춘다. 그 밖엔 손대지 않는다.
    """
    if "\r\n" in content and "\r" not in old and "\n" in old:
        crlf_old = old.replace("\n", "\r\n")
        if content.count(crlf_old):
            return crlf_old, new.replace("\r\n", "\n").replace("\n", "\r\n")
    return old, new
