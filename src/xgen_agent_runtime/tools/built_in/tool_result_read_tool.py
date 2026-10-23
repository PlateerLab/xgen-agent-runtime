"""ToolResultRead — read the full original of a tool result that was clipped (2026-10-23).

A result too large for the model's context window reaches the model clipped (head and tail, with a
marker naming its id); the full body is kept in the host's tool-result store
(:func:`xgen_agent_runtime.stages.s10_tool.persistence.persisted_path`). A result shortened later in
the turn to make room (``core.context_prune``) keeps its original in the state
(:mod:`xgen_agent_runtime.core.originals`). This tool reads any part of either by character offset.

Before, an oversized result became a 480-character preview and the note said the full body was "on
the server; your file tools cannot open it". The model either worked from the preview as if it were
the result, or re-ran the tool hoping for a smaller output. Clipping must come with a real way to
read what was clipped.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

#: Characters returned when the call does not say (the window fit still bounds what the model reads).
DEFAULT_LIMIT = 40_000


class ToolResultReadTool(Tool):
    @property
    def name(self) -> str:
        return "ToolResultRead"

    @property
    def description(self) -> str:
        return (
            "Read the full original of a tool result that was clipped to fit your context window. "
            "A clipped result says so and names its id; pass that id and a character offset to read "
            "any part of it (the result says the total size and the next offset)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The id named in the clipped result's notice.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start from. Default 0.",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"How many characters to return. Default {DEFAULT_LIMIT}.",
                    "minimum": 1,
                },
            },
            "required": ["id"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Reading a stored result again must not re-store and re-clip itself under the tool cap:
        # the window fit alone bounds it (with its own marker when even one page is too big).
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
            max_result_chars=0,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        from xgen_agent_runtime.core.originals import original_tool_result
        from xgen_agent_runtime.stages.s10_tool.persistence import _render_content, persisted_path

        result_id = str(input.get("id") or "").strip()
        if not result_id:
            return ToolResult(
                content="id is required (it is named in the clipped result's notice)", is_error=True
            )

        # 1) The store: the full body of a result clipped when it came back (the fullest copy there is).
        storage = getattr(context, "storage_path", None)
        path = persisted_path(str(storage), result_id) if storage else ""
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    envelope = json.load(fh)
            except (OSError, ValueError) as exc:
                return ToolResult(
                    content=f"The stored result could not be read: {exc}", is_error=True
                )
            body = envelope.get("display_text")
            if not isinstance(body, str) or not body:
                body = _render_content(envelope.get("content"))
            return self._page(
                body,
                result_id,
                envelope.get("tool_name") or "tool",
                bool(envelope.get("is_error")),
                input,
            )

        # 2) The state: a result shortened later in this turn to make room keeps its original there.
        state = getattr(context, "state_view", None)
        kept = original_tool_result(state, result_id)
        if kept is not None:
            return self._page(
                _render_content(kept.get("content")),
                result_id,
                _tool_name_of(state, result_id),
                bool(kept.get("is_error")),
                input,
            )

        return ToolResult(
            content=(
                f"No stored result with id {result_id!r} on this server. A result from an earlier "
                "turn is in this conversation's record: read it with ConversationHistory "
                "(view='search' with a phrase from it, then view='turn'). Otherwise re-run the "
                "tool with a narrower query (a line range, a filter) to get the part you need."
            ),
            is_error=True,
        )

    @staticmethod
    def _page(
        body: str, result_id: str, tool_name: str, is_error: bool, input: Dict[str, Any]
    ) -> ToolResult:
        total = len(body)
        try:
            offset = max(0, int(input.get("offset") or 0))
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = max(1, int(input.get("limit") or DEFAULT_LIMIT))
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        if offset >= total:
            return ToolResult(
                content=f"[{tool_name} result {result_id}: {total} chars; offset {offset} is past the end]"
            )
        end = min(total, offset + limit)
        part = body[offset:end]
        tail = f"next offset={end}" if end < total else "end of the result"
        header = f"[{tool_name} result {result_id}: chars {offset}-{end} of {total}; {tail}]"
        return ToolResult(content=f"{header}\n{part}", is_error=is_error)


def _tool_name_of(state: Any, tool_use_id: str) -> str:
    for message in reversed(list(getattr(state, "messages", None) or [])):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") == tool_use_id
            ):
                return str(block.get("name") or "tool")
    return "tool"
