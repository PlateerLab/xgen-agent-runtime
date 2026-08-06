"""Default artifact for Stage 10: Tool."""

from xgen_agent_runtime.stages.s10_tool.artifact.default.stage import ToolStage
from xgen_agent_runtime.stages.s10_tool.artifact.default.executors import (
    SequentialExecutor,
    ParallelExecutor,
)
from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import RegistryRouter

Stage = ToolStage

__all__ = [
    "Stage",
    "ToolStage",
    "SequentialExecutor",
    "ParallelExecutor",
    "RegistryRouter",
]
