"""Stage 15: Memory — update and persist memory."""

from xgen_agent_runtime.stages.s18_memory.stage import MemoryStage
from xgen_agent_runtime.stages.s18_memory.strategies import (
    MemoryUpdateStrategy,
    AppendOnlyStrategy,
    NoMemoryStrategy,
    ReflectiveStrategy,
    StructuredReflectiveStrategy,
)
from xgen_agent_runtime.stages.s18_memory.persistence import (
    ConversationPersistence,
    InMemoryPersistence,
    FilePersistence,
)
from xgen_agent_runtime.stages.s18_memory.insight import (
    INSIGHTS_KEY,
    PENDING_INSIGHTS_KEY,
    coerce_insight,
    drain_pending_insights,
    insights_to_dicts,
    list_recorded_insights,
    record_insight,
)

__all__ = [
    "MemoryStage",
    "MemoryUpdateStrategy",
    "AppendOnlyStrategy",
    "NoMemoryStrategy",
    "ReflectiveStrategy",
    "StructuredReflectiveStrategy",
    "ConversationPersistence",
    "InMemoryPersistence",
    "FilePersistence",
    "INSIGHTS_KEY",
    "PENDING_INSIGHTS_KEY",
    "coerce_insight",
    "drain_pending_insights",
    "insights_to_dicts",
    "list_recorded_insights",
    "record_insight",
]
