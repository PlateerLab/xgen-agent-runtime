"""Default artifact for Stage 12: Evaluate."""

from xgen_agent_runtime.stages.s14_evaluate.artifact.default.stage import EvaluateStage
from xgen_agent_runtime.stages.s14_evaluate.artifact.default.strategies import (
    SignalBasedEvaluation,
    CriteriaBasedEvaluation,
    AgentEvaluation,
    NoScorer,
    WeightedScorer,
)

Stage = EvaluateStage

__all__ = [
    "Stage",
    "EvaluateStage",
    "SignalBasedEvaluation",
    "CriteriaBasedEvaluation",
    "AgentEvaluation",
    "NoScorer",
    "WeightedScorer",
]
