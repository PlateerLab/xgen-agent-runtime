"""Stage 13: Loop — interface definitions."""

from __future__ import annotations

from abc import abstractmethod

from xgen_agent_runtime.core.stage import Strategy
from xgen_agent_runtime.core.state import PipelineState


class LoopDecision:
    CONTINUE = "continue"
    COMPLETE = "complete"
    ERROR = "error"
    ESCALATE = "escalate"


class LoopController(Strategy):
    """Base interface for loop control decisions."""

    @abstractmethod
    def decide(self, state: PipelineState) -> str:
        """Decide whether to continue looping.

        Returns: "continue", "complete", "error", or "escalate"
        """
        ...
