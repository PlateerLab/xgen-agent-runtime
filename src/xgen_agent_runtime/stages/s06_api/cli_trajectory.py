"""What a CLI backend did inside one call, as conversation messages (2026-10-23).

CLI backends (claude_code, codex) run their own tool loop inside the subprocess. Their terminal
response carries only text: the tool calls are consumed so Stage 10 does not dispatch them again
(``llm_client.translators._cli.StreamJsonAccumulator.finalize``). Stage 6 then recorded that text as
the turn's only assistant message, and the tool calls and their results were in no record at all.
The next turn did not know which files the agent had read or written, or what a command printed,
although the user had watched every call go by.

This collects the stream's own events in order (text, ``tool_use``, ``tool_result``) and gives them
back as the assistant/user messages an API backend would have produced, so the state, short-term
memory, the window and ConversationHistory hold what really happened. The final message keeps only
the text written after the last tool result; the text written between calls sits where it was said.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["CLI_TRAJECTORY_KEY", "CliTrajectory"]

#: ``state.shared`` slot for the trajectory of the CLI call in flight (reset per attempt).
CLI_TRAJECTORY_KEY = "_cli_trajectory"

#: The final message when the CLI ended right after a tool result without writing anything more.
_ENDED_AFTER_TOOLS = "[The answer ended after the tool results above, with no further text.]"


class CliTrajectory:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.tools = 0
        self._text: List[str] = []
        self._assistant: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []
        #: Token streams send a tool's input as JSON fragments after its ``tool_use`` start.
        self._input_json: List[str] = []
        self._input_for: Optional[Dict[str, Any]] = None

    def _close_input(self) -> None:
        block, raw = self._input_for, "".join(self._input_json)
        self._input_for, self._input_json = None, []
        if block is None or not raw.strip() or block.get("input"):
            return
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"_raw": raw}
        block["input"] = parsed if isinstance(parsed, dict) else {"value": parsed}

    def _flush_results(self) -> None:
        if self._results:
            self.messages.append({"role": "user", "content": self._results})
            self._results = []

    def _flush_text(self) -> None:
        text = "".join(self._text)
        self._text = []
        if text.strip():
            self._assistant.append({"type": "text", "text": text})

    def _flush_assistant(self) -> None:
        if self._assistant:
            self.messages.append({"role": "assistant", "content": self._assistant})
            self._assistant = []

    def text(self, chunk: str, *, separate: bool = False) -> None:
        """``separate``: each chunk is a whole message (CLI message granularity), not a token."""
        if not chunk:
            return
        self._close_input()
        if self._results:
            self._flush_results()
        if separate and self._text:
            self._text.append("\n\n")
        self._text.append(chunk)

    def tool_use(self, tool_use_id: Any, name: Any, tool_input: Any) -> None:
        self._close_input()
        self._flush_results()
        self._flush_text()
        block = {
            "type": "tool_use",
            "id": str(tool_use_id or f"cli_{self.tools}"),
            "name": str(name or "tool"),
            "input": tool_input if isinstance(tool_input, dict) else {"value": tool_input},
        }
        self._assistant.append(block)
        self._input_for = block
        self.tools += 1

    def input_delta(self, fragment: str) -> None:
        if self._input_for is not None and fragment:
            self._input_json.append(str(fragment))

    def block_stop(self) -> None:
        self._close_input()

    def tool_result(self, tool_use_id: Any, content: Any, is_error: bool) -> None:
        self._close_input()
        self._flush_text()
        self._flush_assistant()
        if isinstance(content, list):
            body: Any = [b for b in content if isinstance(b, dict)] or ""
        else:
            body = "" if content is None else str(content)
        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": str(tool_use_id or ""),
            "content": body,
        }
        if is_error:
            block["is_error"] = True
        self._results.append(block)

    def finish(self) -> Tuple[List[Dict[str, Any]], str]:
        """(messages before the final answer, the final answer's text)."""
        self._close_input()
        tail = "".join(self._text)
        self._text = []
        self._flush_results()
        # A call still waiting for its result when the CLI ended: the window's pair repair answers it.
        self._flush_assistant()
        return list(self.messages), tail

    def final_content(
        self, response_blocks: List[Dict[str, Any]], tail: str
    ) -> List[Dict[str, Any]]:
        """The final assistant message: the response's non-text blocks plus the text after the last tool."""
        kept = [b for b in response_blocks if isinstance(b, dict) and b.get("type") != "text"]
        text = tail if tail.strip() else _ENDED_AFTER_TOOLS
        return kept + [{"type": "text", "text": text}]


def take(state: Any) -> Optional[CliTrajectory]:
    shared = getattr(state, "shared", None)
    if not isinstance(shared, dict):
        return None
    traj = shared.pop(CLI_TRAJECTORY_KEY, None)
    return traj if isinstance(traj, CliTrajectory) else None
