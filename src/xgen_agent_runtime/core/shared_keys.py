"""Central catalogue of well-known keys for ``state.shared``.

Cycle 20260424 executor uplift — Phase 1 Week 2 Checkpoint 3.

``state.shared`` is a ``Dict[str, Any]`` — free-form by design so stages
and host plugins can stash arbitrary data. Without a convention two
unrelated components may collide on the same key.

This module provides:
- Canonical string constants for keys produced / consumed by executor
  core and officially-blessed host plugins (e.g. Geny's creature
  state). Use these instead of literal strings.
- Namespacing helper for third-party plugins: ``plugin_key("namespace",
  "key") → "plugin.namespace.key"``.

All constants are deliberately stable strings — renaming one is a
breaking change and requires a major bump.

See ``executor_uplift/09_design_extension_interface.md`` §4.
"""

from __future__ import annotations

from typing import Final


class SharedKeys:
    """Well-known keys for ``PipelineState.shared``."""

    # ── Executor core -------------------------------------------------

    TOOL_CALL_ID: Final = "executor.current_tool_call_id"
    """Identifier of the tool_use block currently being processed."""

    PRIMARY_PROVIDER: Final = "primary_provider"
    """**Retired (4.71.0)** — no longer written.

    It carried the resolved Stage 6 provider for sub-agent factories to
    inherit; sub-agent orchestration was removed, so the executor no
    longer publishes it. The constant stays (stable-string contract —
    removal is a major-version change) so imports keep working; hosts
    that need the provider read ``state.llm_client.provider``."""

    SKILL_CTX: Final = "executor.current_skill_ctx"
    """Context for an in-flight Skill invocation (Phase 3)."""

    PERMISSION_CACHE: Final = "executor.permission_cache"
    """Map of ``(tool_name, input_hash) → PermissionDecision`` to avoid
    re-checking the matrix for identical inputs in the same turn."""

    FILE_WITNESSED: Final = "executor.file_witnessed"
    """Paths whose contents the agent has observed in this session.

    File tools use this ledger to refuse blind overwrites of existing
    content. It is executor state because Stage 10 owns applying tool
    mutations; the canonical namespace prevents the mutation filter from
    silently dropping it.
    """

    WORKSPACE_FAST_PATH: Final = "executor.workspace_fast_path"
    """Latest small-workspace routing decision and structural metrics."""

    TOOL_REVIEW_FLAGS: Final = "executor.tool_review_flags"
    """List of annotations emitted by Stage 11 Tool Review (Phase 9)."""

    TASKS_NEW_THIS_TURN: Final = "executor.tasks_new_this_turn"
    """**Retired (4.71.0)** — Stage 12 / 13 were removed; nothing writes it.
    Kept only so imports don't break (stable-string contract)."""

    TASKS_BY_STATUS: Final = "executor.tasks_by_status"
    """**Retired (4.71.0)** — Stage 12 / 13 were removed; nothing writes it.
    Kept only so imports don't break (stable-string contract)."""

    HITL_REQUEST: Final = "executor.hitl_request"
    """Present when Stage 15 HITL should block for approval (Phase 9)."""

    HITL_DECISION: Final = "executor.hitl_decision"
    """Set by Pipeline.resume(token, decision) — consumed by Stage 16
    Loop to decide continue / error."""

    TURN_SUMMARY: Final = "executor.turn_summary"
    """SummaryRecord written by Stage 19 Summarize (Phase 9)."""

    LAST_CHECKPOINT_ID: Final = "executor.last_checkpoint_id"
    """Most recent persist id from Stage 20 Persist (Phase 9)."""

    CONTEXT_COMPACTION_REQUEST: Final = "context.compaction_requested"
    """Request-boundary compaction requested by a loop token dimension."""

    # ── Memory -------------------------------------------------------

    MEMORY_CONTEXT_CHUNKS: Final = "memory.context_chunks"
    """Retrieved chunks injected by Stage 2 Context's retriever."""

    MEMORY_NEEDS_REFLECTION: Final = "memory.needs_reflection"
    """Boolean flag for deferred reflection — legacy Geny path."""

    TURN_NOTES: Final = "executor.turn_notes"
    """Host-written notes about THIS turn's situation (list of strings).

    Rendered by TurnNotesBlock into the volatile tail of the system
    prompt, so with the default turn_context placement they ride next
    to the latest user message and are never persisted to history. Use it
    for state that can change between turns of one conversation (e.g. the
    folders on the user's device connected to this conversation): a stale
    copy in history would contradict the current state."""

    TOOL_MAP: Final = "executor.tool_map"
    """The memory-map surface applied this turn (``tool_exposure="map"``).

    ``{"open": [...], "hidden": [...], "core": {...}, "activated": [...]}`` from
    :func:`host.tool_exposure.apply_tool_map`. Stage 3 reads it to fall back to the
    pre-map surface when the map's tools miss twice, then sets ``"fallen_back": True``."""

    RETIRED_TOOL_CALLS: Final = "executor.retired_tool_calls"
    """Tools absent this turn whose earlier calls must not look callable.

    ``{"names": [...], "reason": str}``. Stage 6 rewrites matching
    ``tool_use``/``tool_result`` pairs in the request copy into one
    plain-text line each (``core.message_repair.retire_tool_calls``), so the
    model neither calls a tool that is gone nor trusts its old results as
    current. History itself is untouched.

    The host sets it for the device folder tools when this conversation
    has no folder connected (``host.local_folders``)."""

    # ── Geny (host plugin) -------------------------------------------

    GENY_CREATURE_STATE: Final = "geny.creature_state"
    """Tamagotchi creature state snapshot hydrated at turn start."""

    GENY_MUTATION_BUFFER: Final = "geny.mutation_buffer"
    """Pending creature-state mutations accumulated by game tools."""

    GENY_CREATURE_ROLE: Final = "geny.creature_role"
    """Creature role classifier bound per-session (vtuber etc.)."""

    # ── Helpers ------------------------------------------------------

    @staticmethod
    def plugin_key(namespace: str, key: str) -> str:
        """Build a namespaced key for third-party plugins.

        Example::

            SharedKeys.plugin_key("myplugin", "state") == "plugin.myplugin.state"

        Plugin authors should always use this helper — direct literal
        strings in user code risk collisions with a future executor
        release that claims the same name.
        """
        if not namespace or not namespace.isidentifier():
            raise ValueError(f"plugin namespace must be a valid identifier: {namespace!r}")
        if not key:
            raise ValueError("plugin key cannot be empty")
        return f"plugin.{namespace}.{key}"
