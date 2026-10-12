"""``/compact``: summarize the conversation so far at the next model request."""

from __future__ import annotations

from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class CompactCommand(SlashCommand):
    """Queue a manual compaction on the live pipeline state.

    Runtime compaction belongs to Stage 2 (``core.compaction.run_compaction``): it consumes
    ``SharedKeys.CONTEXT_COMPACTION_REQUEST`` at the start of the next pass and summarizes
    synchronously, ``trigger="manual"``, even when the context is not yet near the window. If the
    summary model fails the history is left unchanged (a manual request is soft pressure).
    The command only places that request; the host passes the state as
    ``ctx.extras["state"]``. (Earlier versions looked for a "Stage 19 summarize strategy",
    which no pipeline registers, so the command always reported failure.)
    """

    name = "compact"
    description = "Summarize the conversation so far before the next model request."
    category = SlashCategory.CONTROL

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        state = (ctx.extras or {}).get("state")
        if state is None or not isinstance(getattr(state, "shared", None), dict):
            return SlashResult(content="No active conversation state to compact.", success=False)
        if getattr(state, "_context_compactor", None) is None:
            # Stage 2 would drop the request silently (compaction is off for this agent, e.g. a
            # CLI backend that manages its own context).
            return SlashResult(
                content="Context compaction is turned off for this agent.", success=False
            )
        request = {"source": "manual", "iteration": int(getattr(state, "iteration", 0) or 0)}
        state.shared[SharedKeys.CONTEXT_COMPACTION_REQUEST] = request
        add_event = getattr(state, "add_event", None)
        if callable(add_event):
            add_event("context.compaction_requested", dict(request))
        return SlashResult(
            content="✓ The conversation will be summarized before the next model request."
        )


__all__ = ["CompactCommand"]
