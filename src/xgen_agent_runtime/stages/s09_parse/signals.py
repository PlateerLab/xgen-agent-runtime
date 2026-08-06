"""Completion signal detection. Backward-compatible re-exports."""

from xgen_agent_runtime.stages.s09_parse.interface import (
    CompletionSignal,
    CompletionSignalDetector,
)
from xgen_agent_runtime.stages.s09_parse.artifact.default.signals import (
    RegexDetector,
    StructuredDetector,
    HybridDetector,
)

__all__ = [
    "CompletionSignal",
    "CompletionSignalDetector",
    "RegexDetector",
    "StructuredDetector",
    "HybridDetector",
]
