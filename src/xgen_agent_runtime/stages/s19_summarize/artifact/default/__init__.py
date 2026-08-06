"""Default artifact for Stage 19: Summarize (S9b.4)."""

from xgen_agent_runtime.stages.s19_summarize.artifact.default.importance import (
    FixedImportance,
    HeuristicImportance,
)
from xgen_agent_runtime.stages.s19_summarize.artifact.default.stage import SummarizeStage
from xgen_agent_runtime.stages.s19_summarize.artifact.default.summarizers import (
    NoSummarizer,
    RuleBasedSummarizer,
)

Stage = SummarizeStage

__all__ = [
    "FixedImportance",
    "HeuristicImportance",
    "NoSummarizer",
    "RuleBasedSummarizer",
    "Stage",
    "SummarizeStage",
]
