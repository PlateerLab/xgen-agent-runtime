"""xgen-agent-runtime — memory subsystem.

Two layers live here:

1. **`xgen_agent_runtime.memory.provider` (Phase 1+, runtime path)** —
   the unified `MemoryProvider` Protocol plus its 7 layer handles
   and supporting domain dataclasses. Every concrete memory
   implementation, including `EphemeralMemoryProvider` in
   `xgen_agent_runtime.memory.providers.ephemeral`, conforms to this
   contract. Stages 2 (Context) and 15 (Memory) consume the provider;
   they no longer talk to layer-specific strategies directly.

2. **Legacy adapter (`GenyMemoryRetriever` / `GenyMemoryStrategy` /
   `GenyPersistence`)** — duck-typed wrappers around Geny's
   `SessionMemoryManager`. These are kept *only* as validation
   fixtures for Phase 3 (C7 — adapter parity). They are NOT the
   operating path and should not be wired into new code.

Public alias::

    from xgen_agent_runtime.memory import (
        MemoryProvider,
        EphemeralMemoryProvider,
        Layer, Capability, Scope, Importance,
    )
"""

# ── Phase 1+ unified contract ───────────────────────────────────────
from xgen_agent_runtime.memory.provider import (
    BackendInfo,
    Capability,
    CostEvent,
    CostModel,
    CuratedHandle,
    EmbeddingDescriptor,
    ExecutionSummary,
    GlobalHandle,
    Importance,
    IndexHandle,
    Insight,
    Layer,
    LTMHandle,
    MemoryDescriptor,
    MemoryEvent,
    MemoryHooks,
    MemoryProvider,
    MemorySnapshot,
    InteractionFields,
    Note,
    NoteDraft,
    NoteGraph,
    NoteMeta,
    NoteOutline,
    NotePatch,
    NoteRef,
    NoteSummary,
    NotesHandle,
    OutlineNode,
    RecordReceipt,
    ReflectionContext,
    ReindexPlan,
    RetrievalQuery,
    RetrievalResult,
    Scope,
    STMHandle,
    Turn,
    VectorHandle,
)
from xgen_agent_runtime.memory.embedding import (
    EmbeddingClient,
    EmbeddingError,
    LocalHashEmbeddingClient,
    create_embedding_client,
)
from xgen_agent_runtime.memory.providers import (
    EphemeralMemoryProvider,
    FileMemoryProvider,
    SQLMemoryProvider,
)
from xgen_agent_runtime.memory.composite import CompositeMemoryProvider, LayerRouting
from xgen_agent_runtime.memory.factory import MemoryProviderFactory

# ── Stage 2 / Stage 18 generic plumbing ─────────────────────────────
# Provider-driven retriever + strategy. Hosts attach a MemoryProvider
# (typically a CompositeMemoryProvider) and pass a MemoryHooks bag
# carrying retrieval policy + post-write callbacks.
from xgen_agent_runtime.memory.retriever import MemoryAwareRetriever
from xgen_agent_runtime.memory.vector import DocumentChunk, QdrantVectorStore
from xgen_agent_runtime.memory.facts import (
    FACT_EXTRACTION_SCHEMA,
    FACTS_CATEGORY,
    FACTS_FILENAME,
    MEMORY_ENGINE_SYSTEM_PROMPT,
    Fact,
    FactExtraction,
    FactExtractionReport,
    FactLedger,
    LedgerState,
    build_fact_extraction_instruction,
    render_ledger_markdown,
)
from xgen_agent_runtime.memory.rollup import (
    EVERGREEN_SCHEMA,
    SEGMENT_DIGEST_SCHEMA,
    MemoryRollup,
    RollupReport,
    build_evergreen_instruction_structured,
    build_segment_instruction_structured,
    render_evergreen,
    render_segment_digest,
    build_segment_instruction,
)
from xgen_agent_runtime.memory.strategy import ProviderDrivenStrategy
from xgen_agent_runtime.memory.presets import GenyPresets

__all__ = [
    "DocumentChunk",
    "QdrantVectorStore",
    "Fact",
    "FactExtraction",
    "FactExtractionReport",
    "FactLedger",
    "LedgerState",
    "FACT_EXTRACTION_SCHEMA",
    "FACTS_FILENAME",
    "FACTS_CATEGORY",
    "MEMORY_ENGINE_SYSTEM_PROMPT",
    "build_fact_extraction_instruction",
    "render_ledger_markdown",
    "SEGMENT_DIGEST_SCHEMA",
    "EVERGREEN_SCHEMA",
    "build_segment_instruction_structured",
    "build_evergreen_instruction_structured",
    "render_segment_digest",
    "render_evergreen",
    # semantic compaction / rollup
    "MemoryRollup",
    "RollupReport",
    "build_segment_instruction",
    # contract
    "MemoryProvider",
    "MemoryDescriptor",
    "MemoryHooks",
    "MemoryEvent",
    "Layer",
    "Capability",
    "Scope",
    "Importance",
    "BackendInfo",
    "EmbeddingDescriptor",
    "CostEvent",
    "CostModel",
    "Note",
    "NoteMeta",
    "NoteDraft",
    "NotePatch",
    "NoteRef",
    "NoteGraph",
    "NoteSummary",
    "NoteOutline",
    "OutlineNode",
    "InteractionFields",
    "Turn",
    "ExecutionSummary",
    "RecordReceipt",
    "Insight",
    "ReflectionContext",
    "RetrievalQuery",
    "RetrievalResult",
    "MemorySnapshot",
    "ReindexPlan",
    "STMHandle",
    "LTMHandle",
    "NotesHandle",
    "VectorHandle",
    "CuratedHandle",
    "GlobalHandle",
    "IndexHandle",
    # providers
    "EphemeralMemoryProvider",
    "FileMemoryProvider",
    "SQLMemoryProvider",
    "CompositeMemoryProvider",
    "LayerRouting",
    "MemoryProviderFactory",
    # embedding
    "EmbeddingClient",
    "EmbeddingError",
    "LocalHashEmbeddingClient",
    "create_embedding_client",
    # stage 2/18 generic plumbing
    "MemoryAwareRetriever",
    "ProviderDrivenStrategy",
    "GenyPresets",
]
