"""GrepTool — search file contents with regex."""

from __future__ import annotations

import re
from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.fs import tool_fs


class GrepTool(Tool):
    """Search file contents using regular expressions.

    Supports filtering by file glob pattern and multiple output modes.
    """

    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents with regex. Supports filtering by glob pattern. "
            "Output modes: 'content' (matching lines), 'files' (file paths only), 'count' (match counts)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}').",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files", "count"],
                    "description": "Output mode. Default: 'files'.",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context to show around matches (for 'content' mode).",
                    "minimum": 0,
                    "maximum": 10,
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search. Default: false.",
                },
            },
            "required": ["pattern"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Read-only ripgrep — safe to fan out when the LLM issues
        # several grep calls in a single turn.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        pattern_str = input.get("pattern", "")
        if not pattern_str:
            return ToolResult(content="pattern must not be empty", is_error=True)
        try:
            re.compile(pattern_str)  # 러너까지 가기 전에 여기서 알려 준다
        except re.error as e:
            return ToolResult(content=f"Invalid regex: {e}", is_error=True)

        # 러너든 로컬이든 **같은 Python re** 로 찾는다(_search.py). 예전 러너 분기는
        # ``grep -E`` 라서 ``\d`` 같은 표현을 모르고 "No matches" 를 돌려줬다 —
        # 파일에 있는 것을 없다고 말하는, 조용히 틀린 답이었다.
        fs = tool_fs(context)
        try:
            base = fs.resolve(input.get("path", "") or ".")
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)
        result = await fs.search(
            {
                "op": "grep",
                "base": base,
                "pattern": pattern_str,
                "glob": input.get("glob"),
                "output_mode": input.get("output_mode", "files"),
                "context": input.get("context", 0),
                "case_insensitive": bool(input.get("case_insensitive", False)),
            }
        )
        return ToolResult(content=result["text"], is_error=not result["ok"])
