"""Skill gateways — the door that actually opens its room.

## The contract

Our tool surface is hierarchical: basic verbs are visible from turn 1, and
everything else sits behind **one door per family** (``DelegationGuide``,
``JobGuide``, ``BrowserGuide``, ``DocGuide``). ``tool_exposure`` states it
plainly — "DelegateTask/SubAgent*/Task* 는 이 문 뒤에".

Behind the door only works if opening it *lets you in*. It did not. A guide
returned its map and nothing else, so the family stayed deferred and the only
thing that could actually expose a member was ``ToolSearch`` — which no guide
ever mentions. The result was a surface that contradicted its own instructions:

    DelegationGuide  ->  "1. DelegateTask(task)  ONE VERB, the default."
    DelegateTask     ->  Error: No such tool available

The model did exactly what it was told and hit a wall. Worse, the failure is
client-side on CLI backends (Claude Code rejects unknown names locally), so no
amount of server leniency could rescue the call.

So: **calling a guide opens its family.** The guide is the search step for its
own room — ``ToolSearch`` stays what it always was, the way into the long tail
(connected API / DB / MCP nodes, hundreds of schemas nobody should read up
front).

## Why activation lives here and not in each guide

Three guides in three modules doing the same best-effort dance is three places
to forget the ``try``. One helper, one behaviour: never raise, and always say
what opened — a guide that silently fails to open its room is the bug this
module exists to end.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List

logger = logging.getLogger(__name__)

#: Appended to a guide's answer when it just opened its family. The model reads
#: this in the same result that told it which verb to use, so the two halves of
#: the instruction ("use DelegateTask" / "DelegateTask is callable now") arrive
#: together instead of one turn apart.
_OPENED_TEMPLATE = "\n\nNow callable: {names}."


def open_family(context: Any, names: Iterable[str]) -> List[str]:
    """Activate this gateway's members on the turn's registry.

    Returns the names that went from deferred to callable (already-visible
    members are not listed — nothing changed for them). Never raises: a guide
    that cannot reach the registry still has a map to hand back, and losing the
    map because activation failed would be a worse trade.
    """
    registry = getattr(context, "tool_registry", None)
    activate = getattr(registry, "activate", None)
    if not callable(activate):
        return []
    opened: List[str] = []
    for name in names:
        try:
            if activate(name):
                opened.append(name)
        except Exception:  # noqa: BLE001 — one bad name never costs the rest
            logger.debug("skill gateway: activate(%r) failed", name, exc_info=True)
    return opened


def with_opened(content: str, opened: List[str]) -> str:
    """Append the "now callable" line when the door actually opened something."""
    if not opened:
        return content
    return content + _OPENED_TEMPLATE.format(names=", ".join(sorted(opened)))
