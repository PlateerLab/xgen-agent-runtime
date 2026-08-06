"""Stage 2: Context — default artifact."""

from xgen_agent_runtime.stages.s02_context.artifact.default.stage import ContextStage
from xgen_agent_runtime.stages.s02_context.artifact.default.strategies import (
    SimpleLoadStrategy,
    HybridStrategy,
    ProgressiveDisclosureStrategy,
)
from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
    TruncateCompactor,
    SummaryCompactor,
    LLMSummaryCompactor,
    SlidingWindowCompactor,
)
from xgen_agent_runtime.stages.s02_context.artifact.default.retrievers import (
    MCPResourceRetriever,
    NullRetriever,
    StaticRetriever,
)

Stage = ContextStage

__all__ = [
    "Stage",
    "ContextStage",
    "SimpleLoadStrategy",
    "HybridStrategy",
    "ProgressiveDisclosureStrategy",
    "TruncateCompactor",
    "SummaryCompactor",
    "LLMSummaryCompactor",
    "SlidingWindowCompactor",
    "NullRetriever",
    "StaticRetriever",
    "MCPResourceRetriever",
]
