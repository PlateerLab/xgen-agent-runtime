"""Assignments: the user's instructions, kept as memory notes tagged ``assignment`` (2026-10-23).

What the user asks the agent to do, and what they say matters, belongs in memory and nowhere else.
The agent keeps it there itself, when it judges it worth keeping: a note (``memory_write``) tagged
``assignment`` with the user's words, the conditions they set and what was delivered, rewritten as
the work changes and tagged ``done`` (or ``dropped``) when it is finished or called off.

Memory's part is to bring those notes back. Ordinary notes are recalled by search, keyed on the
latest message, and a "continue" or "fix that" matches nothing. So the open assignment notes are
loaded on every turn, whatever the message says, and shown in their own section
(:class:`~xgen_agent_runtime.stages.s03_system.artifact.default.builders.AssignmentsBlock`). They
share the memory budget (a share of the model's window); a note that does not fit whole is cut with
a marker naming it, and notes left out are listed by name so ``memory_read`` can open them.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

__all__ = [
    "ASSIGNMENT_TAG",
    "CLOSED_TAGS",
    "is_open_assignment",
    "render_assignments",
]

#: The tag that makes a note an assignment.
ASSIGNMENT_TAG = "assignment"
#: Tags that close one (finished, or called off by the user).
CLOSED_TAGS = ("done", "dropped")
#: Notes read in full per turn at most; the rest are listed by name.
MAX_NOTES_READ = 30


def _tags(tags: Iterable[str]) -> set:
    return {str(t or "").strip().lower() for t in tags or ()}


def is_open_assignment(tags: Iterable[str]) -> bool:
    found = _tags(tags)
    return ASSIGNMENT_TAG in found and not (found & set(CLOSED_TAGS))


def render_assignments(notes: Sequence[Tuple[str, str, str]], max_chars: int) -> str:
    """``notes`` = (filename, title, body), most recently updated first → the section body.

    Each note shows its filename, so the agent can rewrite that very note when the work changes.
    """
    from xgen_agent_runtime.memory.pinned import render_pinned

    return render_pinned(
        (
            (filename, f"{title or filename} (note: {filename})", body)
            for filename, title, body in notes
        ),
        max_chars,
    )
