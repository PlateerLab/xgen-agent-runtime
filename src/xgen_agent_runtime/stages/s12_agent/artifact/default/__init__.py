"""Default artifact for Stage 11: Agent."""

from xgen_agent_runtime.stages.s12_agent.artifact.default.stage import AgentStage
from xgen_agent_runtime.stages.s12_agent.artifact.default.orchestrators import (
    SingleAgentOrchestrator,
    DelegateOrchestrator,
    EvaluatorOrchestrator,
    DefaultSubPipelineFactory,
)

Stage = AgentStage

__all__ = [
    "Stage",
    "AgentStage",
    "SingleAgentOrchestrator",
    "DelegateOrchestrator",
    "EvaluatorOrchestrator",
    "DefaultSubPipelineFactory",
]
