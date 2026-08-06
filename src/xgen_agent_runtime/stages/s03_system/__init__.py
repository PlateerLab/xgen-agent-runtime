"""Stage 3: System — assemble system prompt."""

from xgen_agent_runtime.stages.s03_system.interface import PromptBuilder, PromptBlock
from xgen_agent_runtime.stages.s03_system.artifact.default import (
    SystemStage,
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
from xgen_agent_runtime.stages.s03_system.persona import (
    DynamicPersonaPromptBuilder,
    PersonaProvider,
    PersonaResolution,
)

__all__ = [
    "SystemStage",
    "PromptBuilder",
    "PromptBlock",
    "StaticPromptBuilder",
    "MutablePromptBuilder",
    "ComposablePromptBuilder",
    "PersonaBlock",
    "RulesBlock",
    "DateTimeBlock",
    "MemoryContextBlock",
    "ToolInstructionsBlock",
    "CustomBlock",
    "DynamicPersonaPromptBuilder",
    "PersonaProvider",
    "PersonaResolution",
]
