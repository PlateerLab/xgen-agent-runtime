"""Pipeline execution result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.core.state import PipelineState, TokenUsage, CacheMetrics
from xgen_agent_runtime.core.run_status import RunStatus, TerminationReason


@dataclass
class PipelineResult:
    """Final result of a pipeline execution."""

    # Output
    text: str = ""
    output: Optional[Any] = None

    # Execution summary
    success: bool = True
    error: Optional[str] = None
    iterations: int = 0

    # Token & Cost
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    turn_token_usage: List[TokenUsage] = field(default_factory=list)
    total_cost_usd: float = 0.0
    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)

    # Thinking
    thinking_history: List[Dict[str, Any]] = field(default_factory=list)

    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    session_id: str = ""
    pipeline_id: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # The PipelineState the run actually used (2.2.0, audit §3.3).
    # When a host calls ``run(input)`` without passing a state, the
    # pipeline builds one internally — and before this field existed
    # that state (with the full conversation history) was simply
    # dropped, which is how GAPT shipped a prod amnesia bug. Hosts
    # recover it here and pass it to the next ``run()`` to continue
    # the conversation. repr-suppressed (it drags clients/credentials)
    # and excluded from equality; treat as a runtime handle, NOT part
    # of the serializable result payload.
    state: Optional[PipelineState] = field(default=None, repr=False, compare=False)

    @property
    def status(self) -> str:
        """Execution-slice status; only ``completed`` means task success."""
        if self.state is not None:
            return self.state.run_status
        return RunStatus.COMPLETED.value if self.success else RunStatus.FAILED.value

    @property
    def termination_reason(self) -> Optional[str]:
        """Machine-readable reason the slice stopped."""
        if self.state is not None:
            return self.state.termination_reason
        return None if self.success else TerminationReason.ERROR.value

    @property
    def resumable(self) -> bool:
        """Whether a host should schedule a continuation slice."""
        return bool(self.state is not None and self.state.resumable)

    @property
    def checkpoint_id(self) -> Optional[str]:
        """Durable checkpoint associated with a suspended slice, if any."""
        return self.state.checkpoint_id if self.state is not None else None

    @classmethod
    def from_state(cls, state: PipelineState) -> PipelineResult:
        """Create a result from final pipeline state."""
        is_error = state.run_status == RunStatus.FAILED.value
        is_complete = state.run_status == RunStatus.COMPLETED.value
        return cls(
            text=state.final_text,
            output=state.final_output,
            success=is_complete,
            error=state.completion_detail if is_error else None,
            iterations=state.iteration,
            token_usage=state.token_usage,
            turn_token_usage=list(state.turn_token_usage),
            total_cost_usd=state.total_cost_usd,
            cache_metrics=state.cache_metrics,
            thinking_history=list(state.thinking_history),
            events=list(state.events),
            session_id=state.session_id,
            pipeline_id=state.pipeline_id,
            model=state.model,
            metadata=dict(state.metadata),
            state=state,
        )

    @classmethod
    def error_result(cls, error: str, state: Optional[PipelineState] = None) -> PipelineResult:
        """Create an error result."""
        if state:
            state.mark_failed(error)
            result = cls.from_state(state)
            result.error = error
            return result
        return cls(success=False, error=error)
