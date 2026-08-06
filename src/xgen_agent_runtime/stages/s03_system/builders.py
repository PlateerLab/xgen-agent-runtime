"""Prompt builders — backward-compatible re-export wrapper."""

from xgen_agent_runtime.stages.s03_system.interface import PromptBuilder, PromptBlock
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

__all__ = [
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
]
