"""Agent orchestrators — backward-compatible re-exports."""

from xgen_agent_runtime.stages.s12_agent.interface import AgentOrchestrator, SubPipelineFactory
from xgen_agent_runtime.stages.s12_agent.types import AgentResult
from xgen_agent_runtime.stages.s12_agent.artifact.default.orchestrators import (
    SingleAgentOrchestrator,
    DelegateOrchestrator,
    EvaluatorOrchestrator,
    DefaultSubPipelineFactory,
)

__all__ = [
    "AgentOrchestrator",
    "SubPipelineFactory",
    "AgentResult",
    "SingleAgentOrchestrator",
    "DelegateOrchestrator",
    "EvaluatorOrchestrator",
    "DefaultSubPipelineFactory",
]
