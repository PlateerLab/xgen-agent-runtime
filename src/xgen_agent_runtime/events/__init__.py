"""Event system for real-time pipeline observability."""

from xgen_agent_runtime.events.bus import EventBus
from xgen_agent_runtime.events.catalog import (
    EVENT_CATALOG_VERSION,
    PAYLOADS,
    RETIRED_EVENT_TYPES,
    EventTypes,
    known_event_types,
)
from xgen_agent_runtime.events.types import PipelineEvent

__all__ = [
    "EVENT_CATALOG_VERSION",
    "EventBus",
    "EventTypes",
    "PAYLOADS",
    "PipelineEvent",
    "RETIRED_EVENT_TYPES",
    "known_event_types",
]
