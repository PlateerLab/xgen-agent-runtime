"""Stage 2: Context — interface definitions (ABCs only)."""

from __future__ import annotations

from abc import abstractmethod
from typing import List

from xgen_agent_runtime.core.stage import Strategy
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.stages.s02_context.types import MemoryChunk


class ContextStrategy(Strategy):
    """Base interface for context collection."""

    @abstractmethod
    async def build_context(self, state: PipelineState) -> None:
        """Build context by modifying state (loading history, memory, etc.)."""


class HistoryCompactor(Strategy):
    """Base interface for compacting conversation history when budget is exceeded."""

    # Set True by compactors that write their own compaction snapshot to a
    # memory provider (e.g. a host wrapper that records to a vault). When
    # True, the centralised :func:`xgen_agent_runtime.core.compaction.run_compaction`
    # helper skips its own ``provider.record_compaction`` call so the
    # snapshot is not persisted twice. Plain in-library compactors leave
    # this False and rely on the helper for persistence.
    persists_own_compaction: bool = False

    @abstractmethod
    async def compact(self, state: PipelineState) -> None:
        """Compact history in state.messages to fit within budget."""


class MemoryRetriever(Strategy):
    """Base interface for memory retrieval."""

    @abstractmethod
    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        """Retrieve relevant memory chunks for a query."""
