"""Concrete `MemoryProvider` implementations.

Shipped so far:
    - `EphemeralMemoryProvider` (Phase 1) — in-memory reference.
    - `FileMemoryProvider` (Phase 2a) — disk-persistent, Geny-compatible.
    - `SQLMemoryProvider` (Phase 2c) — SQLite (Postgres adapter pending).

Coming next:
    - `CompositeMemoryProvider` (Phase 2d) — per-layer backend routing.
"""

from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider
from xgen_agent_runtime.memory.providers.file import FileMemoryProvider
from xgen_agent_runtime.memory.providers.sql import SQLMemoryProvider

__all__ = ["EphemeralMemoryProvider", "FileMemoryProvider", "SQLMemoryProvider"]
