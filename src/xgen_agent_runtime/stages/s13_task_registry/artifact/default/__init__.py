"""Default artifact for Stage 13: Task Registry (S9b.2)."""

from xgen_agent_runtime.stages.s13_task_registry.artifact.default.file_backed_registry import (
    FileBackedRegistry,
)
from xgen_agent_runtime.stages.s13_task_registry.artifact.default.policies import (
    EagerWaitPolicy,
    FireAndForgetPolicy,
    TimedWaitPolicy,
)
from xgen_agent_runtime.stages.s13_task_registry.artifact.default.registry import (
    InMemoryRegistry,
)
from xgen_agent_runtime.stages.s13_task_registry.artifact.default.stage import (
    TaskRegistryStage,
)

Stage = TaskRegistryStage

__all__ = [
    "EagerWaitPolicy",
    "FileBackedRegistry",
    "FireAndForgetPolicy",
    "InMemoryRegistry",
    "Stage",
    "TaskRegistryStage",
    "TimedWaitPolicy",
]
