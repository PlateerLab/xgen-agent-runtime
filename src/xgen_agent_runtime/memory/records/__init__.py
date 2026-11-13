"""Memory records (4.108.0): typed, versioned memory with provenance — written by extract → reconcile,
selected per turn by assemble, kept sharp by consolidate. See ``model`` for the why."""

from xgen_agent_runtime.memory.records.assemble import Assembly, assemble, projects_in_play
from xgen_agent_runtime.memory.records.consolidate import consolidate, find_clusters
from xgen_agent_runtime.memory.records.extract import Candidate, IndexedTurn, parse_candidates
from xgen_agent_runtime.memory.records.migrate import MIGRATED_TAG, migrate_legacy
from xgen_agent_runtime.memory.records.model import (
    AUTHORITIES,
    HISTORY_CATEGORY,
    KINDS,
    RECORD_TAG,
    STATUSES,
    Evidence,
    MemoryRecord,
    authority_rank,
    now_iso,
)
from xgen_agent_runtime.memory.records.owner import (
    OWNER_ACTIONS,
    OwnerActionError,
    owner_action,
    owner_create,
    record_summary,
    records_overview,
)
from xgen_agent_runtime.memory.records.pipeline import RecordsPassReport, RecordsPipeline
from xgen_agent_runtime.memory.records.reconcile import Reconciler
from xgen_agent_runtime.memory.records.store import STATE_CATEGORY, STATE_FILENAME, RecordStore

__all__ = [
    "AUTHORITIES",
    "Assembly",
    "Candidate",
    "Evidence",
    "HISTORY_CATEGORY",
    "IndexedTurn",
    "KINDS",
    "MIGRATED_TAG",
    "MemoryRecord",
    "OWNER_ACTIONS",
    "OwnerActionError",
    "RECORD_TAG",
    "RecordStore",
    "RecordsPassReport",
    "RecordsPipeline",
    "Reconciler",
    "STATE_CATEGORY",
    "STATE_FILENAME",
    "STATUSES",
    "assemble",
    "authority_rank",
    "consolidate",
    "find_clusters",
    "migrate_legacy",
    "now_iso",
    "owner_action",
    "owner_create",
    "parse_candidates",
    "projects_in_play",
    "record_summary",
    "records_overview",
]
