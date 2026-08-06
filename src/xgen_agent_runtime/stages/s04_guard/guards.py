"""Guard implementations — backward-compatible re-exports.

Concrete implementations have moved to:
  xgen_agent_runtime.stages.s04_guard.artifact.default.guards

ABCs and infrastructure live in:
  xgen_agent_runtime.stages.s04_guard.interface

Data types live in:
  xgen_agent_runtime.stages.s04_guard.types
"""

from xgen_agent_runtime.stages.s04_guard.types import GuardResult
from xgen_agent_runtime.stages.s04_guard.interface import Guard, GuardChain
from xgen_agent_runtime.stages.s04_guard.artifact.default.guards import (
    TokenBudgetGuard,
    CostBudgetGuard,
    IterationGuard,
    PermissionGuard,
)

__all__ = [
    "GuardResult",
    "Guard",
    "GuardChain",
    "TokenBudgetGuard",
    "CostBudgetGuard",
    "IterationGuard",
    "PermissionGuard",
]
