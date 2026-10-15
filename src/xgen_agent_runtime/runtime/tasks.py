"""Background task records + registries (relocated in 4.71.0).

These types used to live in ``stages/s13_task_registry`` — Stage 13 was
the pipeline stage that drained sub-agent delegations into a registry.
Sub-agent orchestration was removed in 4.71.0 and the stage with it, but
the *registry* is still the storage half of the background-task runtime
(:class:`~xgen_agent_runtime.runtime.task_runner.BackgroundTaskRunner`,
the cron daemon, the ``/tasks`` slash command). It lives here now,
next to its only consumers, with no stage coupling.

* :class:`TaskStatus` / :class:`TaskRecord` / :class:`TaskFilter` — the
  record model.
* :class:`TaskRegistry` — storage ABC (records + per-task output bytes).
* :class:`InMemoryRegistry` — process-lifetime backend.
* :class:`FileBackedRegistry` — single-process durable backend
  (``registry.jsonl`` + ``outputs/<task_id>.bin``).
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Record model ─────────────────────────────────────────────────────


class TaskStatus(str, enum.Enum):
    """Lifecycle states for a registered task.

    PENDING  — registered but not yet started.
    RUNNING  — execution in progress.
    DONE     — completed successfully.
    FAILED   — completed with an error.
    CANCELLED — externally cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = frozenset({TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED})


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TaskRecord:
    """A single registered task. Mutable so the runner can update status in place.

    ``payload`` carries whatever the executor for ``kind`` needs — e.g.
    ``{"command": ...}`` for ``local_bash``, or a cron job's payload.
    ``result`` is filled when the task reaches a terminal status.
    ``output_path`` is set by registries that persist streaming output
    to external storage (file / blob) so callers can locate the bytes
    without going through the registry.
    """

    task_id: str
    kind: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    iteration_seen: int = 0
    output_path: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    def mark(self, status: TaskStatus, *, result: Any = None, error: Optional[str] = None) -> None:
        """Transition the task. Sets started_at / completed_at automatically."""
        if status == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = _now()
        if status in _TERMINAL_STATUSES:
            self.completed_at = _now()
            if result is not None:
                self.result = result
            if error is not None:
                self.error = error
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "iteration_seen": self.iteration_seen,
            "output_path": self.output_path,
        }


@dataclass
class TaskFilter:
    """Filter applied to :meth:`TaskRegistry.list_filtered`.

    Fields combine with AND. ``None`` values are no-ops. ``limit``
    caps the result set after sorting (most-recent ``created_at`` first).
    """

    status: Optional[TaskStatus] = None
    kind: Optional[str] = None
    created_after: Optional[datetime] = None
    limit: Optional[int] = None


# ── Registry ABC ─────────────────────────────────────────────────────


class TaskRegistry(ABC):
    """Storage backend for :class:`TaskRecord` instances.

    Until 4.70.0 this was a Stage 13 ``Strategy``; it is a plain ABC now.
    ``name`` / ``description`` stay as (overridable) properties so
    existing backends that defined them keep working unchanged.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def register(self, record: TaskRecord) -> None:
        """Insert a new record. Re-registering the same task_id replaces it."""
        ...

    @abstractmethod
    def get(self, task_id: str) -> Optional[TaskRecord]:
        """Return the record by id, or None if unknown."""
        ...

    @abstractmethod
    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        """Mutate the named task's status. Returns the record (None if unknown)."""
        ...

    @abstractmethod
    def list_all(self) -> List[TaskRecord]:
        """Snapshot of every record currently in the registry."""
        ...

    @abstractmethod
    def remove(self, task_id: str) -> bool:
        """Drop the named task. Returns False if the id is unknown."""
        ...

    def by_status(self) -> Dict[str, List[TaskRecord]]:
        """Group records by status value (string keys)."""
        out: Dict[str, List[TaskRecord]] = {}
        for record in self.list_all():
            out.setdefault(record.status.value, []).append(record)
        return out

    # ── Optional: filtering + streaming output ────────────────────────
    #
    # Defaults provide a working in-memory implementation on top of
    # ``list_all``. Backends that persist tasks to disk / DB should
    # override for efficiency. Backends that have no concept of output
    # streams keep the default no-ops; tools that try to ``read_output``
    # will get an empty bytes payload.

    def list_filtered(self, filter: TaskFilter) -> List[TaskRecord]:
        """Return records matching ``filter``, ordered by ``created_at`` desc.

        The default implementation builds on ``list_all`` and is suitable
        for in-memory backends. Persistent backends (Postgres / Redis)
        should override to push the filter into the query layer.
        """
        rows = self.list_all()
        if filter.status is not None:
            rows = [r for r in rows if r.status == filter.status]
        if filter.kind is not None:
            rows = [r for r in rows if r.kind == filter.kind]
        if filter.created_after is not None:
            rows = [r for r in rows if r.created_at >= filter.created_after]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        if filter.limit is not None:
            rows = rows[: filter.limit]
        return rows

    async def append_output(self, task_id: str, chunk: bytes) -> None:
        """Append output bytes for a task. No-op by default.

        Backends override to persist to memory / disk / blob storage.
        ``chunk`` may be partial — callers may invoke this many times
        per task.
        """
        return None

    async def read_output(
        self,
        task_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> bytes:
        """Return previously appended output bytes from ``offset``.

        Returns ``b""`` when no output is recorded for ``task_id``
        (or when the backend does not support output storage).
        """
        return b""

    async def stream_output(self, task_id: str) -> AsyncIterator[bytes]:
        """Async-iterate output chunks until the task reaches a terminal status.

        The default implementation yields what is currently buffered
        and returns once the record is terminal. Backends that maintain
        an ``asyncio.Event`` per task should override to deliver chunks
        as they arrive (no polling).
        """
        offset = 0
        while True:
            chunk = await self.read_output(task_id, offset)
            if chunk:
                yield chunk
                offset += len(chunk)
            record = self.get(task_id)
            if record is None or record.is_terminal:
                # Drain any final bytes the producer wrote between the
                # last read and the terminal transition.
                tail = await self.read_output(task_id, offset)
                if tail:
                    yield tail
                return


# ── In-memory backend ────────────────────────────────────────────────


class InMemoryRegistry(TaskRegistry):
    """Process-lifetime task store.

    Suitable for single-process deployments. Hosts that need durable
    task state can plug their own :class:`TaskRegistry` (e.g. backed
    by Postgres / Redis) — the runner doesn't care about the backend.

    Output streaming is supported via per-task ``bytearray`` buffers
    plus an ``asyncio.Event`` so :meth:`stream_output` wakes on each
    :meth:`append_output` rather than polling.
    """

    def __init__(self) -> None:
        self._records: Dict[str, TaskRecord] = {}
        self._outputs: Dict[str, bytearray] = {}
        self._output_events: Dict[str, asyncio.Event] = {}

    @property
    def name(self) -> str:
        return "in_memory"

    @property
    def description(self) -> str:
        return "In-memory task registry (process lifetime)"

    def register(self, record: TaskRecord) -> None:
        self._records[record.task_id] = record
        # Buffer is created lazily on first append_output, but we set
        # up the event eagerly so consumers waiting on stream_output
        # before the first chunk arrives don't race.
        self._output_events.setdefault(record.task_id, asyncio.Event())

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._records.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        record = self._records.get(task_id)
        if record is None:
            return None
        record.mark(status, result=result, error=error)
        # On terminal transition, wake any stream_output consumers so
        # they can drain the tail and exit instead of waiting forever.
        if record.is_terminal:
            event = self._output_events.get(task_id)
            if event is not None:
                event.set()
        return record

    def list_all(self) -> List[TaskRecord]:
        return list(self._records.values())

    def remove(self, task_id: str) -> bool:
        existed = self._records.pop(task_id, None) is not None
        self._outputs.pop(task_id, None)
        event = self._output_events.pop(task_id, None)
        if event is not None:
            event.set()
        return existed

    # ── Output streaming ──────────────────────────────────────────────

    async def append_output(self, task_id: str, chunk: bytes) -> None:
        if not chunk:
            return
        buf = self._outputs.setdefault(task_id, bytearray())
        buf.extend(chunk)
        event = self._output_events.setdefault(task_id, asyncio.Event())
        # Wake all current waiters, then immediately rearm so the next
        # append_output can wake fresh waiters without us holding state.
        event.set()
        event.clear()

    async def read_output(
        self,
        task_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> bytes:
        buf = self._outputs.get(task_id)
        if buf is None or offset >= len(buf):
            return b""
        end = len(buf) if limit is None else min(offset + limit, len(buf))
        return bytes(buf[offset:end])

    async def stream_output(self, task_id: str) -> AsyncIterator[bytes]:
        offset = 0
        while True:
            chunk = await self.read_output(task_id, offset)
            if chunk:
                yield chunk
                offset += len(chunk)
                # Loop back to immediately drain anything else queued.
                continue
            record = self._records.get(task_id)
            if record is None:
                return
            if record.is_terminal:
                # One last drain in case bytes arrived between the read
                # above and the terminal transition.
                tail = await self.read_output(task_id, offset)
                if tail:
                    yield tail
                return
            event = self._output_events.get(task_id)
            if event is None:
                # Record exists but no event registered (manual mutation
                # bypassed register). Bail to avoid hanging.
                return
            try:
                # Cap individual waits so a runaway producer / forgotten
                # terminal transition never hangs the consumer forever.
                await asyncio.wait_for(event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


# ── File-backed backend ──────────────────────────────────────────────
#
# Layout under ``root``:
#
#     root/
#       registry.jsonl     — one TaskRecord per line (last write wins
#                            per ``task_id``; ``{"_deleted": true}``
#                            lines are tombstones)
#       outputs/<task_id>.bin
#                          — raw appended output bytes
#
# The registry loads ``registry.jsonl`` lazily on first access and keeps
# an in-memory cache. Mutations append a fresh JSON line so a
# crash-then-restart resumes from the most recent write.

_TOMBSTONE_KEY = "_deleted"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _record_from_jsonable(data: Dict[str, Any]) -> TaskRecord:
    return TaskRecord(
        task_id=data["task_id"],
        kind=data.get("kind", ""),
        payload=dict(data.get("payload") or {}),
        status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
        created_at=_parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
        started_at=_parse_dt(data.get("started_at")),
        completed_at=_parse_dt(data.get("completed_at")),
        result=data.get("result"),
        error=data.get("error"),
        iteration_seen=int(data.get("iteration_seen") or 0),
        output_path=data.get("output_path"),
    )


def _load_with_tombstones(path: Path) -> Dict[str, TaskRecord]:
    """Replay ``registry.jsonl``: last write per task_id wins, tombstones delete.

    Corrupt / partial lines (e.g. a torn write at the tail after a
    crash) are skipped with a warning instead of failing the load.
    """
    cache: Dict[str, TaskRecord] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning(
                "file_backed_registry_skipped_corrupt_line",
                extra={"path": str(path), "error": str(exc)},
            )
            continue
        if data.get(_TOMBSTONE_KEY):
            cache.pop(data.get("task_id", ""), None)
            continue
        try:
            record = _record_from_jsonable(data)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "file_backed_registry_skipped_bad_record",
                extra={"path": str(path), "error": str(exc)},
            )
            continue
        cache[record.task_id] = record
    return cache


class FileBackedRegistry(TaskRegistry):
    """Durable single-process task registry.

    Mutations append to ``registry.jsonl``; on load, the latest line
    per ``task_id`` wins (tombstones remove) so corrupted / partial
    writes at the tail are tolerated. Output bytes for each task are
    stored as a side file under ``outputs/<task_id>.bin`` and read with
    normal file seek / read.

    Suitable for self-hosted deployments where Postgres / Redis is
    overkill but :class:`InMemoryRegistry` loses too much on restart.
    For multi-process / clustered deployments, plug a real DB-backed
    registry instead.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._registry_path = self._root / "registry.jsonl"
        self._outputs_dir = self._root / "outputs"
        self._cache: Dict[str, TaskRecord] = {}
        self._loaded = False
        self._mutate_lock = asyncio.Lock()
        self._output_events: Dict[str, asyncio.Event] = {}

    @property
    def name(self) -> str:
        return "file_backed"

    @property
    def description(self) -> str:
        return "Durable single-process task registry (jsonl append + side files for output)"

    # ── Loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        self._cache = _load_with_tombstones(self._registry_path)
        self._loaded = True

    def _append_line(self, record: TaskRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str)
        with self._registry_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # ── TaskRegistry protocol ────────────────────────────────────────

    def register(self, record: TaskRecord) -> None:
        self._ensure_loaded()
        self._cache[record.task_id] = record
        self._append_line(record)
        self._output_events.setdefault(record.task_id, asyncio.Event())

    def get(self, task_id: str) -> Optional[TaskRecord]:
        self._ensure_loaded()
        return self._cache.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        self._ensure_loaded()
        record = self._cache.get(task_id)
        if record is None:
            return None
        record.mark(status, result=result, error=error)
        self._append_line(record)
        if record.is_terminal:
            event = self._output_events.get(task_id)
            if event is not None:
                event.set()
        return record

    def list_all(self) -> List[TaskRecord]:
        self._ensure_loaded()
        return list(self._cache.values())

    def remove(self, task_id: str) -> bool:
        self._ensure_loaded()
        if task_id not in self._cache:
            return False
        del self._cache[task_id]
        # Tombstone the line so reload doesn't resurrect it.
        with self._registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": task_id, _TOMBSTONE_KEY: True}) + "\n")
        # Drop side output file.
        out_path = self._output_path_for(task_id)
        if out_path.exists():
            out_path.unlink()
        event = self._output_events.pop(task_id, None)
        if event is not None:
            event.set()
        return True

    # ── Output streaming ─────────────────────────────────────────────

    def _output_path_for(self, task_id: str) -> Path:
        # task_id is registered by callers; we still defang directory traversal
        # in case a backend swap injects untrusted ids later.
        safe = task_id.replace("/", "_").replace("..", "_")
        return self._outputs_dir / f"{safe}.bin"

    async def append_output(self, task_id: str, chunk: bytes) -> None:
        if not chunk:
            return
        self._ensure_loaded()
        path = self._output_path_for(task_id)
        async with self._mutate_lock:
            with path.open("ab") as handle:
                handle.write(chunk)
            event = self._output_events.setdefault(task_id, asyncio.Event())
            event.set()
            event.clear()

    async def read_output(
        self,
        task_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> bytes:
        path = self._output_path_for(task_id)
        if not path.exists():
            return b""
        with path.open("rb") as handle:
            handle.seek(offset)
            if limit is None:
                return handle.read()
            return handle.read(limit)

    async def stream_output(self, task_id: str) -> AsyncIterator[bytes]:
        offset = 0
        while True:
            chunk = await self.read_output(task_id, offset)
            if chunk:
                yield chunk
                offset += len(chunk)
                continue
            record = self._cache.get(task_id)
            if record is None:
                return
            if record.is_terminal:
                tail = await self.read_output(task_id, offset)
                if tail:
                    yield tail
                return
            event = self._output_events.get(task_id)
            if event is None:
                return
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


__all__ = [
    "FileBackedRegistry",
    "InMemoryRegistry",
    "TaskFilter",
    "TaskRecord",
    "TaskRegistry",
    "TaskStatus",
]
