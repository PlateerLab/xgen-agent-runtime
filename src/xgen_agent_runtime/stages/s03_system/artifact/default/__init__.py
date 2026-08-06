"""Stage 3: System — default artifact."""

from xgen_agent_runtime.stages.s03_system.artifact.default.stage import SystemStage
from xgen_agent_runtime.stages.s03_system.artifact.default.builders import (
    StaticPromptBuilder,
    MutablePromptBuilder,
    ComposablePromptBuilder,
    PersonaBlock,
    RulesBlock,
    DateTimeBlock,
    MemoryContextBlock,
    ToolInstructionsBlock,
    CustomBlock,
)

Stage = SystemStage

__all__ = [
    "Stage",
    "SystemStage",
    "StaticPromptBuilder",
    "MutablePromptBuilder",
    "ComposablePromptBuilder",
    "PersonaBlock",
    "RulesBlock",
    "DateTimeBlock",
    "MemoryContextBlock",
    "ToolInstructionsBlock",
    "CustomBlock",
]
