"""History compactors — backward-compatible re-export wrapper."""

from xgen_agent_runtime.stages.s02_context.interface import HistoryCompactor
from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
    TruncateCompactor,
    SummaryCompactor,
    SlidingWindowCompactor,
)

__all__ = [
    "HistoryCompactor",
    "TruncateCompactor",
    "SummaryCompactor",
    "SlidingWindowCompactor",
]
