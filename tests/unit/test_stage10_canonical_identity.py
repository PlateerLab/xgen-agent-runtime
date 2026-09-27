"""The server-issued identity survives real tool dispatch without leaking to model state."""

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.session import CanonicalSessionIdentity
from xgen_agent_runtime.stages.s10_tool.artifact.default.stage import ToolStage
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry
from xgen_agent_runtime.tools.sandbox import SandboxConfig, ToolSandbox


def _identity() -> CanonicalSessionIdentity:
    return CanonicalSessionIdentity(
        tenant_id="system",
        user_id=42,
        platform_session_id=uuid4(),
        device_id=uuid4(),
        agent_session_id=uuid4(),
        platform_type="web",
    )


class _CaptureTool(Tool):
    def __init__(self) -> None:
        self.contexts: list[ToolContext] = []

    @property
    def name(self) -> str:
        return "capture_identity"

    @property
    def description(self) -> str:
        return "Capture dispatch context"

    @property
    def input_schema(self) -> dict:
        return {"type": "object"}

    async def execute(self, input: dict, context: ToolContext) -> ToolResult:
        self.contexts.append(context)
        return ToolResult(content="ok")


def test_identity_requires_typed_server_ids_and_is_immutable() -> None:
    identity = _identity()
    with pytest.raises(FrozenInstanceError):
        identity.user_id = 99
    with pytest.raises(ValueError, match="platform_session_id"):
        CanonicalSessionIdentity(
            tenant_id="system", user_id=42, platform_session_id="unverified",
            device_id=uuid4(), agent_session_id=uuid4(), platform_type="web",
        )
    with pytest.raises(ValueError, match="user_id"):
        CanonicalSessionIdentity(
            tenant_id="system", user_id=True, platform_session_id=uuid4(),
            device_id=uuid4(), agent_session_id=uuid4(), platform_type="web",
        )


@pytest.mark.asyncio
async def test_identity_reaches_each_tool_call_without_changing_local_session_or_model_state() -> None:
    identity = _identity()
    tool = _CaptureTool()
    registry = ToolRegistry()
    registry.register(tool)
    stage = ToolStage(
        registry=registry,
        context=ToolContext(session_id="host-local-id", canonical_identity=identity),
    )
    state = PipelineState(session_id="execution-123")
    state.pending_tool_calls = [
        {"tool_use_id": f"tu_{index}", "tool_name": tool.name, "tool_input": {}}
        for index in range(2)
    ]

    await stage.execute(None, state)

    assert len(tool.contexts) == 2
    for context in tool.contexts:
        assert context.canonical_identity is identity
        assert context.session_id == "execution-123"
        assert context.metadata == {}
        assert context.extras == {}
        assert context.env_vars is None
    assert identity.agent_session_id.hex not in repr(state.messages)
    assert identity.platform_session_id.hex not in repr(state.messages)
    assert "canonical_identity" not in repr(tool.contexts[0])


@pytest.mark.asyncio
async def test_sandbox_environment_enrichment_keeps_identity() -> None:
    identity = _identity()
    tool = _CaptureTool()
    wrapper = ToolSandbox(SandboxConfig(env_vars={"INJECTED": "1"}))
    context = ToolContext(session_id="execution-123", canonical_identity=identity)

    await wrapper.execute_tool(tool, {}, context)

    assert tool.contexts[0].canonical_identity is identity
    assert tool.contexts[0].env_vars == {"INJECTED": "1"}
    assert "canonical_identity" not in tool.contexts[0].env_vars
    assert ToolContext().canonical_identity is None
