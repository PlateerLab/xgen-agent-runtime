"""Record storage over any ``NotesHandle`` — one note per record, history notes per old version.

Riding the notes infrastructure means every provider (ephemeral, file, sql, a host's own store)
holds records with no new tables, and search, vector index, graph and the memory browser see them.

* A record is a note ``mem-<id>.md`` in its kind's category; the record itself is one JSON scalar in
  frontmatter (``memory_record_json``); the body is the human-readable text (what search indexes).
* ``put`` never overwrites blindly: it compares the stored version with the one the caller read
  (two background passes on two pods can touch the same record) and archives the version it
  replaces as ``memh-<id>-v<n>.md`` (category ``memory-history``) — every change can be shown and
  reverted.
* Reading many records: a store may offer ``read_category(category) -> List[Note]`` (one query);
  otherwise notes are listed and read concurrently.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from xgen_agent_runtime.memory.records.model import (
    HISTORY_CATEGORY,
    KINDS,
    RECORD_KEY,
    RECORD_TAG,
    MemoryRecord,
    category_for,
    filename_for,
    normalize_name,
    now_iso,
    slugify,
)

logger = logging.getLogger(__name__)

__all__ = ["RecordStore", "STATE_FILENAME"]

#: Records pipeline state: per-session extraction cursors, pass counters, migration status.
STATE_FILENAME = "__memory_records__.md"
STATE_CATEGORY = "memory-system"
_STATE_KEY = "memory_records_state_json"
_READ_CONCURRENCY = 16


def _importance(value: str) -> Any:
    from xgen_agent_runtime.memory.provider import Importance

    try:
        return Importance(value)
    except ValueError:
        return Importance.MEDIUM


def _record_of(note: Any) -> Optional[MemoryRecord]:
    fm = getattr(note, "frontmatter", None) or {}
    return MemoryRecord.from_json(fm.get(RECORD_KEY))


def _body_of(record: MemoryRecord) -> str:
    """What search and the vector index see: the text, then the names it goes by."""
    names = [record.subject] + [a for a in record.aliases if a]
    lines = [record.text]
    if record.headline and record.headline != record.text:
        lines.insert(0, record.headline)
    tail = " · ".join(n for n in names if n)
    if record.key:
        tail = f"{tail} · {record.key}" if tail else record.key
    if tail:
        lines.append(f"({tail})")
    return "\n\n".join(lines)


def _title_of(record: MemoryRecord) -> str:
    if record.headline:
        return record.headline[:200]
    base = f"{record.subject}: {record.key}" if record.key else record.subject
    return (base or record.text)[:200]


def _tags_of(record: MemoryRecord) -> List[str]:
    tags = [RECORD_TAG, f"kind:{record.kind}", f"status:{record.status}"]
    if record.project:
        tags.append(f"project:{slugify(record.project)}")
    return tags


class RecordStore:
    def __init__(self, provider: Any) -> None:
        if provider is None:
            raise ValueError("RecordStore requires a provider")
        self._provider = provider

    def _notes(self) -> Any:
        return self._provider.notes()

    # ── reading ─────────────────────────────────────────────────────

    async def _read_notes(self, category: str) -> List[Any]:
        notes = self._notes()
        fast = getattr(notes, "read_category", None)
        if callable(fast):
            try:
                return list(await fast(category))
            except Exception:  # noqa: BLE001 — fall back to list + read
                logger.debug("records: read_category failed; reading one by one", exc_info=True)
        metas = await notes.list(category=category)
        sem = asyncio.Semaphore(_READ_CONCURRENCY)

        async def _one(meta: Any) -> Any:
            async with sem:
                try:
                    return await notes.read(meta.ref.filename)
                except Exception:  # noqa: BLE001
                    return None

        got = await asyncio.gather(*(_one(m) for m in metas))
        return [n for n in got if n is not None]

    async def load(
        self,
        *,
        kinds: Optional[Sequence[str]] = None,
        live_only: bool = True,
    ) -> List[MemoryRecord]:
        """Records of ``kinds`` (all kinds by default). Unreadable notes are skipped, never fatal."""
        out: List[MemoryRecord] = []
        for kind in kinds or KINDS:
            try:
                notes = await self._read_notes(category_for(kind))
            except Exception:  # noqa: BLE001
                logger.warning("records: could not read %s", kind, exc_info=True)
                continue
            for note in notes:
                rec = _record_of(note)
                if rec is None or rec.kind != kind:
                    continue
                if live_only and not rec.live:
                    continue
                out.append(rec)
        return out

    async def get(self, record_id: str) -> Optional[MemoryRecord]:
        try:
            note = await self._notes().read(filename_for(record_id))
        except Exception:  # noqa: BLE001
            return None
        return _record_of(note) if note is not None else None

    async def counts(self) -> Dict[str, Dict[str, int]]:
        """``{kind: {status: n}}`` from note metadata only (tags) — cheap, for the memory index."""
        out: Dict[str, Dict[str, int]] = {}
        notes = self._notes()
        for kind in KINDS:
            try:
                metas = await notes.list(category=category_for(kind))
            except Exception:  # noqa: BLE001
                continue
            bucket: Dict[str, int] = {}
            for meta in metas:
                tags = list(getattr(meta, "tags", None) or [])
                if RECORD_TAG not in tags:
                    continue
                status = next((t[7:] for t in tags if t.startswith("status:")), "active")
                bucket[status] = bucket.get(status, 0) + 1
            if bucket:
                out[kind] = bucket
        return out

    # ── ids ─────────────────────────────────────────────────────────

    async def new_id(self, record: MemoryRecord) -> str:
        """Readable, stable, unique: ``<kind>-<subject>[-<key>]`` (+ ``-2`` … on collision)."""
        base = record.kind[0] + "-" + slugify(record.subject or record.text, limit=28)
        if record.key:
            base += "-" + slugify(record.key, limit=20)
        candidate, n = base, 1
        while await self.get(candidate) is not None:
            n += 1
            candidate = f"{base}-{n}"
        return candidate

    # ── writing ─────────────────────────────────────────────────────

    async def put(
        self,
        record: MemoryRecord,
        *,
        expected_version: Optional[int],
        reason: str = "",
    ) -> bool:
        """Write ``record``. ``expected_version`` is the version the caller read (``None`` = new).

        Returns False when the stored record moved on meanwhile (the caller re-reads and decides
        again) or when the write failed. The replaced version is archived first.
        """
        from xgen_agent_runtime.memory.provider import NoteDraft

        current = await self.get(record.id)
        if expected_version is None:
            if current is not None:
                return False
            record.version = 1
            record.recorded_at = record.recorded_at or now_iso()
        else:
            if current is None or current.version != expected_version:
                return False
            await self._archive(current)
            record.version = current.version + 1
        record.updated_at = now_iso()
        if reason:
            record.change_reason = reason
        draft = NoteDraft(
            title=_title_of(record),
            body=_body_of(record),
            importance=_importance(record.importance),
            tags=_tags_of(record),
            category=record.category,
            filename=record.filename,
            frontmatter={RECORD_KEY: record.to_json()},
        )
        try:
            await self._notes().write(draft)
        except Exception:  # noqa: BLE001
            logger.warning("records: write failed (%s)", record.id, exc_info=True)
            return False
        return True

    async def _archive(self, record: MemoryRecord) -> None:
        from xgen_agent_runtime.memory.provider import NoteDraft

        try:
            await self._notes().write(
                NoteDraft(
                    title=f"{_title_of(record)} (v{record.version})",
                    body=record.text,
                    importance=_importance("low"),
                    tags=[RECORD_TAG, "history", f"record:{record.id}"],
                    category=HISTORY_CATEGORY,
                    filename=f"memh-{record.id}-v{record.version}.md",
                    frontmatter={RECORD_KEY: record.to_json()},
                )
            )
        except Exception:  # noqa: BLE001 — losing an archive copy must not lose the change itself
            logger.warning(
                "records: history write failed (%s v%d)", record.id, record.version, exc_info=True
            )

    async def history(self, record_id: str) -> List[MemoryRecord]:
        """Previous versions, oldest first (the current version is ``get``)."""
        notes = self._notes()
        try:
            metas = await notes.list(category=HISTORY_CATEGORY)
        except Exception:  # noqa: BLE001
            return []
        prefix = f"memh-{record_id}-v"
        out: List[MemoryRecord] = []
        for meta in metas:
            name = meta.ref.filename
            if not name.startswith(prefix):
                continue
            note = await notes.read(name)
            rec = _record_of(note) if note is not None else None
            if rec is not None and rec.id == record_id:
                out.append(rec)
        return sorted(out, key=lambda r: r.version)

    async def revert(
        self, record_id: str, version: int, *, reason: str = ""
    ) -> Optional[MemoryRecord]:
        """Make ``version`` current again (as a new version — reverting is itself a change)."""
        current = await self.get(record_id)
        if current is None:
            return None
        old = next((r for r in await self.history(record_id) if r.version == int(version)), None)
        if old is None:
            return None
        restored = MemoryRecord.from_dict(old.to_dict())
        assert restored is not None
        restored.change_reason = reason or f"reverted to v{version}"
        ok = await self.put(
            restored, expected_version=current.version, reason=restored.change_reason
        )
        return restored if ok else None

    # ── pipeline state ──────────────────────────────────────────────

    async def load_state(self) -> Dict[str, Any]:
        try:
            note = await self._notes().read(STATE_FILENAME)
        except Exception:  # noqa: BLE001
            return {}
        if note is None:
            return {}
        raw = (getattr(note, "frontmatter", None) or {}).get(_STATE_KEY)
        try:
            data = json.loads(raw) if isinstance(raw, str) else {}
        except ValueError:
            data = {}
        return data if isinstance(data, dict) else {}

    async def save_state(self, state: Dict[str, Any]) -> bool:
        from xgen_agent_runtime.memory.provider import NoteDraft

        body = (
            "Memory records pipeline state (machine-maintained): extraction cursors per conversation, "
            "pass counters, migration status, the last turn's memory selection."
        )
        try:
            await self._notes().write(
                NoteDraft(
                    title="Memory system state",
                    body=body,
                    importance=_importance("low"),
                    tags=["memory-system"],
                    category=STATE_CATEGORY,
                    filename=STATE_FILENAME,
                    frontmatter={
                        _STATE_KEY: json.dumps(state, ensure_ascii=False, separators=(",", ":"))
                    },
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("records: state write failed", exc_info=True)
            return False
        return True


def same_identity(a: MemoryRecord, b: MemoryRecord) -> bool:
    """Same kind and the same subject (by any name) and key."""
    if a.kind != b.kind or normalize_name(a.key) != normalize_name(b.key):
        return False
    return bool(set(a.names()) & set(b.names()))


def names_in(text: str, records: Iterable[MemoryRecord]) -> List[MemoryRecord]:
    """Records whose subject or an alias appears in ``text`` (normalized substring)."""
    hay = normalize_name(text)
    if not hay:
        return []
    out = []
    for r in records:
        if any(len(n) >= 2 and n in hay for n in r.names()):
            out.append(r)
    return out
