"""Pinned notes as one prompt block: every provider renders them the same way (2026-10-23).

Pinned notes (the ``critical`` category: the fact ledger, identity, prohibitions, the evergreen
summary) are injected into every turn. The providers each budgeted them on their own, and both lost
facts without a word: the XGEN DB provider returned nothing at all when its first note was longer
than the budget, and the file provider dropped every note past the budget unseen. A fact the agent
was told to always keep then simply was not there, and nothing said so.

Here: the fact ledger first, then importance, then recency (the caller orders); one oversized note
cannot take the whole budget (each gets at most half); a note that does not fit whole is cut with
a marker naming the note, and the notes left out are listed by name so ``memory_read`` can open them.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

__all__ = ["LEDGER_FILENAME", "render_pinned"]

#: The fact ledger note: always first.
LEDGER_FILENAME = "__facts__.md"
#: A note is never given more than this share of the budget.
_PER_NOTE_SHARE = 0.5
#: Below this much room, a note is listed rather than cut to a stub.
_MIN_PIECE = 200


def render_pinned(notes: Iterable[Tuple[str, str, str]], max_chars: int) -> str:
    """``notes`` = ``(filename, title, body)`` in priority order → the block (``""`` if empty)."""
    if max_chars <= 0:
        return ""
    items = [(f or "", t or "", (b or "").strip()) for f, t, b in notes]
    items = [(f, t, b) for f, t, b in items if b]
    if not items:
        return ""
    per_note = max(_MIN_PIECE, int(max_chars * _PER_NOTE_SHARE))
    parts: List[str] = []
    left_out: List[str] = []
    used = 0
    for filename, title, body in items:
        piece = f"## {title}\n{body}" if title else body
        room = max_chars - used - (2 if parts else 0)
        limit = min(per_note, room)
        if len(piece) > limit:
            if limit < _MIN_PIECE:
                left_out.append(filename or title)
                continue
            marker = (
                f"\n…[this note is cut here: {limit:,} of {len(piece):,} chars shown; "
                f'memory_read("{filename}") shows all of it]'
            )
            piece = piece[: max(0, limit - len(marker))].rstrip() + marker
        parts.append(piece)
        used += len(piece) + (2 if len(parts) > 1 else 0)
    if left_out:
        names = ", ".join(n for n in left_out[:20] if n)
        more = f" and {len(left_out) - 20} more" if len(left_out) > 20 else ""
        parts.append(
            f"[{len(left_out)} more pinned note(s) did not fit here: {names}{more}. "
            "Read them with memory_read before relying on what they say.]"
        )
    return "\n\n".join(parts)
