"""Memory retrievers — backward-compatible re-export wrapper."""

from xgen_agent_runtime.stages.s02_context.interface import MemoryRetriever
from xgen_agent_runtime.stages.s02_context.types import MemoryChunk
from xgen_agent_runtime.stages.s02_context.artifact.default.retrievers import (
    NullRetriever,
    StaticRetriever,
)

__all__ = [
    "MemoryChunk",
    "MemoryRetriever",
    "NullRetriever",
    "StaticRetriever",
]
