"""Run/slice terminal semantics for resumable agent execution.

An invocation of :class:`Pipeline` is an execution *slice*.  Reaching a
slice guard (iterations, wall clock, tool calls) is not the same thing as
finishing the user's task.  These enums keep that distinction explicit
without changing the wire shape of the existing result dataclass.
"""

from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    """Outcome of one pipeline execution slice."""

    RUNNING = "running"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TerminationReason(str, Enum):
    """Stable machine-readable reason attached to a non-running slice."""

    MODEL_COMPLETED = "model_completed"
    MAX_ITERATIONS_PER_SLICE = "max_iterations_per_slice"
    MAX_TOOL_CALLS_PER_SLICE = "max_tool_calls_per_slice"
    WALL_CLOCK_BUDGET = "wall_clock_budget"
    COST_BUDGET = "cost_budget"
    TOKEN_BUDGET = "token_budget"
    CONTEXT_LIMIT = "context_limit"
    CONTEXT_COMPACTION_FAILED = "context_compaction_failed"
    USER_INPUT_REQUIRED = "user_input_required"
    ERROR = "error"
    CANCELLED = "cancelled"


__all__ = ["RunStatus", "TerminationReason"]
