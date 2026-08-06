"""Stage 8: Think — default artifact."""

from xgen_agent_runtime.stages.s08_think.artifact.default.stage import ThinkStage
from xgen_agent_runtime.stages.s08_think.artifact.default.processors import (
    PassthroughProcessor,
    ExtractAndStoreProcessor,
    ThinkingFilterProcessor,
)
from xgen_agent_runtime.stages.s08_think.artifact.default.budget import (
    AdaptiveThinkingBudget,
    StaticThinkingBudget,
    apply_thinking_budget,
    make_planner,
)

Stage = ThinkStage

__all__ = [
    "Stage",
    "ThinkStage",
    "PassthroughProcessor",
    "ExtractAndStoreProcessor",
    "ThinkingFilterProcessor",
    "AdaptiveThinkingBudget",
    "StaticThinkingBudget",
    "apply_thinking_budget",
    "make_planner",
]
