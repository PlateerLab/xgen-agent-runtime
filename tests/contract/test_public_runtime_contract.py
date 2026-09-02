"""Compatibility pins for the host-facing runtime contract.

Harness internals are expected to evolve.  The call boundary and wire-shaped
dataclasses are not: hosts construct these objects directly and persist event
envelopes between releases.  Keep this file intentionally boring and explicit
so an interface change is reviewed as a compatibility decision, not absorbed
as an incidental refactor.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import Any

import xgen_agent_runtime
from xgen_agent_runtime import Pipeline
from xgen_agent_runtime.core.result import PipelineResult
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.events.types import PipelineEvent
from xgen_agent_runtime.llm_client.types import APIRequest, APIResponse, ContentBlock


def _public_field_names(dataclass_type: type[Any]) -> list[str]:
    """Return host-visible dataclass fields in positional order."""

    return [field.name for field in fields(dataclass_type) if not field.name.startswith("_")]


def test_pipeline_entrypoint_signatures_are_stable() -> None:
    """Hosts must not need an adapter when harness internals change."""

    assert str(inspect.signature(Pipeline.run)) == (
        "(self, input: 'Any', state: 'Optional[PipelineState]' = None, *, "
        "overrides: 'Optional[ModelOverrides]' = None) -> 'PipelineResult'"
    )
    assert str(inspect.signature(Pipeline.run_stream)) == (
        "(self, input: 'Any', state: 'Optional[PipelineState]' = None, *, "
        "overrides: 'Optional[ModelOverrides]' = None) -> "
        "'AsyncIterator[PipelineEvent]'"
    )


def test_wire_and_result_dataclass_fields_are_stable() -> None:
    """Pin field names and order because positional callers exist in the wild."""

    assert _public_field_names(PipelineEvent) == [
        "type",
        "stage",
        "iteration",
        "timestamp",
        "data",
        "session_id",
        "run_id",
        "seq",
    ]
    assert _public_field_names(APIRequest) == [
        "model",
        "messages",
        "max_tokens",
        "system",
        "temperature",
        "top_p",
        "top_k",
        "tools",
        "tool_choice",
        "stop_sequences",
        "stream",
        "thinking",
        "response_format",
        "session_hint",
        "mcp_config",
        "metadata",
    ]
    assert _public_field_names(ContentBlock) == [
        "type",
        "text",
        "tool_use_id",
        "tool_name",
        "tool_input",
        "thinking_text",
        "raw",
    ]
    assert _public_field_names(APIResponse) == [
        "content",
        "stop_reason",
        "usage",
        "model",
        "message_id",
        "raw",
    ]
    assert _public_field_names(PipelineResult) == [
        "text",
        "output",
        "success",
        "error",
        "iterations",
        "token_usage",
        "turn_token_usage",
        "total_cost_usd",
        "cache_metrics",
        "thinking_history",
        "events",
        "session_id",
        "pipeline_id",
        "model",
        "metadata",
        "state",
    ]


def test_pipeline_state_public_fields_are_stable() -> None:
    """New harness bookkeeping belongs in private state, not the host schema."""

    assert _public_field_names(PipelineState) == [
        "session_id",
        "pipeline_id",
        "system",
        "messages",
        "iteration",
        "max_iterations",
        "current_stage",
        "stage_history",
        "stream",
        "single_turn",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "tools",
        "tool_choice",
        "stop_sequences",
        "tools_version",
        "thinking_enabled",
        "thinking_budget_tokens",
        "thinking_type",
        "thinking_display",
        "thinking_history",
        "token_usage",
        "turn_token_usage",
        "total_cost_usd",
        "session_cost_usd",
        "cost_budget_usd",
        "cache_metrics",
        "memory_refs",
        "context_window_budget",
        "loop_decision",
        "completion_signal",
        "completion_detail",
        "pending_tool_calls",
        "tool_results",
        "tool_dispatcher",
        "delegate_requests",
        "agent_results",
        "evaluation_score",
        "evaluation_feedback",
        "final_text",
        "final_output",
        "last_api_response",
        "created_at",
        "updated_at",
        "metadata",
        "shared",
        "events",
        "llm_client",
        "credentials",
        "subagent_registry",
        "session_runtime",
    ]


def test_primary_contract_types_remain_package_root_exports() -> None:
    expected = {
        "Pipeline": Pipeline,
        "PipelineState": PipelineState,
        "PipelineResult": PipelineResult,
        "PipelineEvent": PipelineEvent,
        "APIRequest": APIRequest,
        "APIResponse": APIResponse,
        "ContentBlock": ContentBlock,
    }

    for name, value in expected.items():
        assert getattr(xgen_agent_runtime, name) is value
        assert name in xgen_agent_runtime.__all__
