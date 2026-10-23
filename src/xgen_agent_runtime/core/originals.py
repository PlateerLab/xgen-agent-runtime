"""Originals of messages the context reducers shortened in place (2026-10-23).

The deterministic reducers (:mod:`xgen_agent_runtime.core.context_prune`: de-duplication, trimming
old tool results and inputs, removing images; the guard's last-resort fits) rewrite blocks of
``state.messages`` **in place**, so that the next request fits the model's window. Short-term memory
records those same message objects only when the turn ends. Before this module, what memory kept of
the turn was therefore the shortened copy: a result the tool really returned was saved as "600 chars
… [trimmed]", and every later turn (the short-term window, ConversationHistory, the summary) read the
cut as if it were what happened.

A reducer now calls :func:`keep_original` before it changes a message. The recorders read
:func:`original_of`, so memory keeps what was really said and returned, and ToolResultRead reads a
trimmed result's full text from here while the turn is still running
(:func:`original_tool_result`). The request copy is still the shortened one: only what goes to the
model is reduced, never the record.

Entries hold the message object itself, so its ``id`` cannot be reused while the entry lives. They
are dropped once the message has been recorded (:func:`forget_originals`) or is no longer anywhere in
the state.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "ORIGINALS_KEY",
    "keep_original",
    "original_of",
    "originals_of",
    "original_tool_result",
    "forget_originals",
]

#: ``state.shared`` slot: ``{id(message): (message, original_copy)}``.
ORIGINALS_KEY = "_context_originals"


def _ledger(state: Any, *, create: bool) -> Optional[Dict[int, Tuple[Any, Any]]]:
    shared = getattr(state, "shared", None)
    if not isinstance(shared, dict):
        return None
    ledger = shared.get(ORIGINALS_KEY)
    if not isinstance(ledger, dict):
        if not create:
            return None
        ledger = {}
        shared[ORIGINALS_KEY] = ledger
    return ledger


def _live_ids(state: Any) -> set:
    """Messages still reachable from the state: the conversation and the not-yet-recorded queue."""
    ids = {id(m) for m in (getattr(state, "messages", None) or [])}
    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, dict):
        from xgen_agent_runtime.core.compaction import PENDING_RECORDS_KEY

        pending = metadata.get(PENDING_RECORDS_KEY)
        if isinstance(pending, dict):
            for queue in pending.values():
                ids.update(id(m) for m in (queue or []))
    return ids


def keep_original(state: Any, message: Any) -> None:
    """Copy ``message`` before a reducer changes it. The first copy wins (it is the original)."""
    if state is None or not isinstance(message, dict):
        return
    ledger = _ledger(state, create=True)
    if ledger is None or id(message) in ledger:
        return
    # A message that left the state (summarized and recorded) needs no original any more.
    if ledger:
        live = _live_ids(state)
        for key in [k for k in ledger if k not in live]:
            ledger.pop(key, None)
    ledger[id(message)] = (message, copy.deepcopy(message))


def original_of(state: Any, message: Any) -> Any:
    """The message as it was before any reducer changed it (the message itself when unchanged)."""
    ledger = _ledger(state, create=False)
    if not ledger:
        return message
    entry = ledger.get(id(message))
    if entry is None or entry[0] is not message:
        return message
    return entry[1]


def originals_of(state: Any, messages: Iterable[Any]) -> List[Any]:
    return [original_of(state, m) for m in messages]


def original_tool_result(state: Any, tool_use_id: str) -> Optional[Dict[str, Any]]:
    """The original ``tool_result`` block for ``tool_use_id`` if a reducer shortened it."""
    ledger = _ledger(state, create=False)
    if not ledger or not tool_use_id:
        return None
    for _message, original in ledger.values():
        content = original.get("content") if isinstance(original, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and str(block.get("tool_use_id") or "") == tool_use_id
            ):
                return block
    return None


def forget_originals(state: Any, messages: Iterable[Any]) -> None:
    """Drop the entries of messages that have now been recorded."""
    ledger = _ledger(state, create=False)
    if not ledger:
        return
    for m in messages:
        entry = ledger.get(id(m))
        if entry is not None and entry[0] is m:
            ledger.pop(id(m), None)
