"""Memory strategies — backward-compatible re-exports."""

from xgen_agent_runtime.stages.s18_memory.interface import MemoryUpdateStrategy
from xgen_agent_runtime.stages.s18_memory.artifact.default.strategies import (
    AppendOnlyStrategy,
    NoMemoryStrategy,
    ReflectiveStrategy,
    StructuredReflectiveStrategy,
)

__all__ = [
    "MemoryUpdateStrategy",
    "AppendOnlyStrategy",
    "NoMemoryStrategy",
    "ReflectiveStrategy",
    "StructuredReflectiveStrategy",
]
