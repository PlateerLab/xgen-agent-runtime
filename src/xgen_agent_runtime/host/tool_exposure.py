"""How many tools the agent sees at once.

Our tool surface is **hierarchical**, not flat. The agent always sees its basic
tools — web, files, shell, delegation, memory, the search tool of every attached
knowledge source. Everything beyond that (connected API / DB / MCP nodes, which
can be hundreds of schemas) is announced by name and one line, and the agent
pulls in the full schema with ``ToolSearch`` when it actually needs it.

That is the point: a tool list is a map, not an inventory. Sending every schema
up front spends the context window on things the turn will never call, and a
model that reads a hundred near-identical schemas picks worse than one that
reads five and drills into the right one.

Two settings:

``hierarchy``
    The default. Basic tools visible, the rest discovered on demand.

``flat``
    Every connected schema up front. An escape hatch for models that cannot
    drive a discovery step; it costs tokens on every request.

Older workflows stored ``all`` (everything up front) or ``search`` (defer).
Both now resolve to ``hierarchy`` — the hierarchy is the platform's behaviour,
and an agent that wants the flat surface says so explicitly.
"""

from __future__ import annotations

#: Basic tools stay visible; the rest is discovered with ToolSearch.
HIERARCHY = "hierarchy"
#: Every connected tool schema is sent up front.
FLAT = "flat"

#: Values that mean "send everything up front".
_FLAT_ALIASES = frozenset({FLAT, "all_upfront", "upfront"})


def normalize_exposure(value: object) -> str:
    """Resolve a stored ``tool_exposure`` to ``hierarchy`` or ``flat``.

    Unknown values resolve to ``hierarchy`` rather than raising: an exposure
    setting is a preference, and a typo in it should not stop a turn from
    running.
    """
    text = str(value or "").strip().lower()
    return FLAT if text in _FLAT_ALIASES else HIERARCHY


def sends_every_schema(value: object) -> bool:
    """Should connected tool nodes be registered as immediately visible?

    Only the flat surface says yes. Under the hierarchy they are registered
    deferred, and ``ToolSearch`` activates the ones a turn actually needs.
    """
    return normalize_exposure(value) == FLAT
