"""Stage 7: Token — interface definitions."""

from __future__ import annotations

from abc import abstractmethod

from xgen_agent_runtime.core.stage import Strategy
from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.stages.s06_api.types import APIResponse


class TokenTracker(Strategy):
    """Base interface for token tracking."""

    @abstractmethod
    def track(self, response: APIResponse, state: PipelineState) -> TokenUsage:
        """Track token usage from an API response."""
        ...


class CostCalculator(Strategy):
    """Base interface for cost calculation."""

    @abstractmethod
    def calculate(self, usage: TokenUsage, model: str) -> float:
        """Calculate cost in USD from token usage."""
        ...
