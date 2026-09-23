"""GlobTool — find files by pattern matching."""

from __future__ import annotations

from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.fs import tool_fs


class GlobTool(Tool):
    """Find files matching a glob pattern.

    Searches from the given directory (or working_dir) and returns
    matching file paths sorted by modification time (newest first).
    """

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts'). "
            "Returns matching file paths sorted by modification time."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files against.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to working directory.",
                },
            },
            "required": ["pattern"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Directory-walk glob — no side effects, safe to fan out.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = input.get("pattern", "")
        if not pattern:
            return ToolResult(content="pattern must not be empty", is_error=True)

        # 러너든 로컬이든 **같은 코드**(_search.py)로 찾는다 — 예전 러너 분기는 패턴을
        # 셸에 따옴표 없이 끼워 넣어 ``$(…)`` 가 실행됐고, 경로를 상대로 돌려줬다.
        fs = tool_fs(context)
        try:
            base = fs.resolve(input.get("path", "") or ".")
        except (PermissionError, ValueError) as e:
            return ToolResult(content=str(e), is_error=True)
        result = await fs.search({"op": "glob", "base": base, "pattern": pattern})
        return ToolResult(content=result["text"], is_error=not result["ok"])
