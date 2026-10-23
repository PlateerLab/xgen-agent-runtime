"""Assignments: the user's instructions, kept as memory notes tagged ``assignment`` (2026-10-23).

What the user asks the agent to do, and what they say matters, is kept in memory and nowhere else.
The agent keeps it there itself when it judges it worth keeping: a note (``memory_write``) tagged
``assignment`` with the user's words, the conditions they set and what was done (files, places,
results), rewritten as the work changes and tagged ``done`` when it is finished or called off.

The point is to find the work again later: "the edit we did last week, change one thing". Such a
request is answered through memory, on demand, like any other recall:

1. the note is found by the memory search the turn already runs, or by the agent
   (``memory_search``, ``memory_list(tag="assignment")``);
2. ``memory_read`` gives the note and the conversations it was written in (``written_in``, stamped
   by memory on every write, not by the model);
3. ``ConversationHistory(conversation=...)`` reads that conversation word for word, when the host
   can open it for this user.

Assignment notes are deliberately *not* put in the prompt on every turn. The conversation in
progress is already in the context as it happened (the short-term window); repeating its
instructions from a note doubles them, and a note written earlier can lag behind what was just
said. Open assignments from other conversations would also ride along in every unrelated turn.
(4.99.0 did inject them on every turn; 4.100.0 stopped it.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

__all__ = ["ASSIGNMENT_TAG", "WRITTEN_IN_KEY", "stamp_written_in", "written_in"]

#: The tag that makes a note an assignment.
ASSIGNMENT_TAG = "assignment"
#: Frontmatter key memory stamps on every note it writes: the conversations it was written in.
WRITTEN_IN_KEY = "written_in"
#: Conversations kept per note (the most recent).
MAX_WRITTEN_IN = 50


def written_in(frontmatter: Any) -> List[Dict[str, str]]:
    value = (frontmatter or {}).get(WRITTEN_IN_KEY) if isinstance(frontmatter, dict) else None
    if not isinstance(value, list):
        return []
    return [
        {"conversation": str(v.get("conversation") or ""), "at": str(v.get("at") or "")}
        for v in value
        if isinstance(v, dict) and v.get("conversation")
    ]


def stamp_written_in(frontmatter: Any, conversation: str, *, at: str = "") -> Dict[str, Any]:
    """``frontmatter`` with this conversation recorded as the latest one the note was written in."""
    out: Dict[str, Any] = dict(frontmatter) if isinstance(frontmatter, dict) else {}
    if not conversation:
        return out
    at = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = [e for e in written_in(out) if e["conversation"] != conversation]
    entries.append({"conversation": conversation, "at": at})
    out[WRITTEN_IN_KEY] = entries[-MAX_WRITTEN_IN:]
    return out
