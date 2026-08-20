"""Stage 10 — the per-dispatch ToolContext must carry the sandbox.

Regression guard for the class of bug where ``ToolStage.build_dispatch_
context`` (and ``ToolSandbox.execute_tool``) rebuild a fresh ``ToolContext``
per tool call and silently drop the host-attached runtime handles — most
critically ``sandbox``. When ``sandbox`` is dropped, ``context.sandbox`` is
``None`` on every real dispatch, so Bash / Read / Write / Edit / Glob / Grep
fall to their local-subprocess path and run on the SERVING POD instead of the
agent's isolated XGeny session. Every XGeny agent, regardless of provider, is
supposed to execute its built-in tools inside its own sandbox — this test
locks that contract at the exact seam the earlier tests bypassed (they built
ToolContext directly and called ``router.route``, never exercising the
rebuild).
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.stages.s10_tool.artifact.default.stage import ToolStage
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.sandbox import SandboxConfig, ToolSandbox


class _FakeSandbox:
    """Stand-in for an XGeny sandbox session (identity is all that matters)."""

    def __init__(self) -> None:
        self.workdir = "/workspace"


def _events_sink(name, payload):  # pragma: no cover - identity callback
    return None


# ─────────────────────────────────────────────────────────────────
# build_dispatch_context — the decisive seam
# ─────────────────────────────────────────────────────────────────


class TestBuildDispatchContextCarriesRuntimeHandles:
    def test_sandbox_is_propagated(self):
        sandbox = _FakeSandbox()
        stage = ToolStage(context=ToolContext(session_id="s", sandbox=sandbox))
        state = PipelineState(session_id="s")

        ctx = stage.build_dispatch_context(state)

        # THE regression: the per-dispatch context must be the same live
        # sandbox handle the host attached, not None.
        assert ctx.sandbox is sandbox

    def test_event_emit_and_parent_tool_use_id_are_propagated(self):
        stage = ToolStage(
            context=ToolContext(
                session_id="s",
                event_emit=_events_sink,
                parent_tool_use_id="toolu_parent",
            )
        )
        state = PipelineState(session_id="s")

        ctx = stage.build_dispatch_context(state)

        assert ctx.event_emit is _events_sink
        assert ctx.parent_tool_use_id == "toolu_parent"

    def test_sandbox_none_stays_none_without_a_session(self):
        stage = ToolStage(context=ToolContext(session_id="s"))
        state = PipelineState(session_id="s")

        ctx = stage.build_dispatch_context(state)

        # No over-reach: absent a session the handle stays None (the pod
        # path), it is not fabricated.
        assert ctx.sandbox is None

    def test_previously_working_fields_still_propagate(self):
        # Guard against a regression in the OTHER direction — the rebuild
        # must keep carrying the handles it already carried.
        sentinel_env = {"FOO": "bar"}
        stage = ToolStage(
            context=ToolContext(
                session_id="s",
                working_dir="/work",
                env_vars=sentinel_env,
                permission_mode="auto",
            )
        )
        state = PipelineState(session_id="s")

        ctx = stage.build_dispatch_context(state)

        assert ctx.working_dir == "/work"
        assert ctx.env_vars == sentinel_env
        assert ctx.permission_mode == "auto"


# ─────────────────────────────────────────────────────────────────
# ToolSandbox.execute_tool — the second rebuild site
# ─────────────────────────────────────────────────────────────────


class _CtxCapturingTool(Tool):
    def __init__(self) -> None:
        self.seen_context: ToolContext | None = None

    @property
    def name(self):
        return "capture_ctx"

    @property
    def description(self):
        return "captures the ToolContext it is handed"

    @property
    def input_schema(self):
        return {"type": "object"}

    async def execute(self, input, context):
        self.seen_context = context
        return ToolResult(content="ok")


class TestToolSandboxPreservesContext:
    @pytest.mark.asyncio
    async def test_env_var_enrichment_keeps_sandbox(self):
        # When SandboxConfig.env_vars is set, execute_tool rebuilds the
        # context to layer the env in. That rebuild must not drop sandbox.
        sandbox = _FakeSandbox()
        tool = _CtxCapturingTool()
        wrapper = ToolSandbox(SandboxConfig(env_vars={"INJECTED": "1"}))
        ctx = ToolContext(session_id="s", sandbox=sandbox, env_vars={"BASE": "1"})

        await wrapper.execute_tool(tool, {}, ctx)

        assert tool.seen_context is not None
        assert tool.seen_context.sandbox is sandbox
        # env was still layered in (both base and injected present)
        assert tool.seen_context.env_vars.get("BASE") == "1"
        assert tool.seen_context.env_vars.get("INJECTED") == "1"

    @pytest.mark.asyncio
    async def test_context_passthrough_when_no_env_override(self):
        sandbox = _FakeSandbox()
        tool = _CtxCapturingTool()
        wrapper = ToolSandbox(SandboxConfig())  # no env_vars → no rebuild
        ctx = ToolContext(session_id="s", sandbox=sandbox)

        await wrapper.execute_tool(tool, {}, ctx)

        assert tool.seen_context is ctx
        assert tool.seen_context.sandbox is sandbox
