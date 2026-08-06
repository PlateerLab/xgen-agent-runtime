"""Stage 4: Guard — default artifact."""

from xgen_agent_runtime.stages.s04_guard.artifact.default.stage import GuardStage
from xgen_agent_runtime.stages.s04_guard.artifact.default.guards import (
    TokenBudgetGuard,
    CostBudgetGuard,
    IterationGuard,
    PermissionGuard,
)

Stage = GuardStage

__all__ = [
    "Stage",
    "GuardStage",
    "TokenBudgetGuard",
    "CostBudgetGuard",
    "IterationGuard",
    "PermissionGuard",
]
