"""Default artifact for Stage 15: Memory."""

from xgen_agent_runtime.stages.s18_memory.artifact.default.stage import MemoryStage
from xgen_agent_runtime.stages.s18_memory.artifact.default.strategies import (
    AppendOnlyStrategy,
    NoMemoryStrategy,
    ReflectiveStrategy,
)
from xgen_agent_runtime.stages.s18_memory.artifact.default.persistence import (
    InMemoryPersistence,
    FilePersistence,
)

Stage = MemoryStage

__all__ = [
    "Stage",
    "MemoryStage",
    "AppendOnlyStrategy",
    "NoMemoryStrategy",
    "ReflectiveStrategy",
    "InMemoryPersistence",
    "FilePersistence",
]
