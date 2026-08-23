"""GlobTool — find files by pattern matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in._path_guard import resolve_and_validate

_MAX_RESULTS = 500


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
        search_path = input.get("path", "") or context.working_dir

        if not pattern:
            return ToolResult(content="pattern must not be empty", is_error=True)

        # Sandbox: expand the glob inside the container (bash globstar).
        if context.sandbox is not None:
            import shlex

            from xgen_agent_runtime.tools._xgeny_sandbox import sb_run

            wd = context.working_dir or "/workspace"
            spath = input.get("path", "") or "."
            # globstar makes ** recurse; nullglob makes a no-match expand to
            # nothing; print only regular files, newest first.
            cmd = (
                f"shopt -s globstar nullglob dotglob; cd {shlex.quote(spath)} 2>/dev/null || exit 0; "
                f'for f in {pattern}; do [ -f "$f" ] && printf \'%s\\t%s\\n\' "$(stat -c %Y "$f" 2>/dev/null)" "$f"; done '
                f"| sort -rn | cut -f2- | head -n {_MAX_RESULTS}"
            )
            rc, out, _err = await sb_run(context.sandbox, cmd, workdir=wd)
            out = out.strip()
            if not out:
                return ToolResult(content=f"No files matching '{pattern}' in {search_path}")
            return ToolResult(content=out)

        # 호스트 경로(sandbox 없음): 검색 루트도 allowed_paths 안이어야 한다 — Read/Write 와
        # 같은 가드. 커넥터 로컬 턴에서 PC 전역 열거를 막는다(감사 #11 후속).
        try:
            base = resolve_and_validate(search_path, context.working_dir, context.allowed_paths)
        except PermissionError as e:
            return ToolResult(content=str(e), is_error=True)
        except ValueError:
            base = Path(context.working_dir or ".")
        if not base.is_dir():
            return ToolResult(content=f"Directory not found: {search_path}", is_error=True)

        try:
            matches = list(base.glob(pattern))
        except Exception as e:
            return ToolResult(content=f"Glob error: {e}", is_error=True)

        # Filter to files only, sort by mtime descending
        files = [m for m in matches if m.is_file()]
        # 심볼릭 링크로 허용 경로 밖을 가리키는 항목은 제외한다.
        if context.allowed_paths:
            roots = [Path(ap).resolve() for ap in context.allowed_paths]

            def _inside(m: Path) -> bool:
                try:
                    r = m.resolve()
                except Exception:  # noqa: BLE001
                    return False
                return any(r == root or root in r.parents for root in roots)

            files = [m for m in files if _inside(m)]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not files:
            return ToolResult(content=f"No files matching '{pattern}' in {search_path}")

        truncated = len(files) > _MAX_RESULTS
        if truncated:
            files = files[:_MAX_RESULTS]

        output = "\n".join(str(f) for f in files)
        if truncated:
            output += f"\n\n... (showing {_MAX_RESULTS} of {len(matches)} matches)"

        return ToolResult(content=output)
