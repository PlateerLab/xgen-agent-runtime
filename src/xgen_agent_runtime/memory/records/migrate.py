"""Move the pre-records memory over — once, and then whatever lands in the old pinned category later.

Sources: the fact ledger (``__facts__.md``, active facts), the evergreen (``__evergreen__.md``) and
every other note in the pinned ``critical`` category (notes the agent pinned, identity, prohibitions).

With a model: the notes are split into records through the normal extract → reconcile path (authority
``migrated``), so duplicates across the ledger, the evergreen and pinned notes collapse into one record.
Without one, or when that fails: a deterministic conversion that loses nothing — each ledger fact
becomes one record, each other note one record carrying its whole body (shown by its title when long).

The old notes are not deleted or rewritten; they are tagged ``migrated-to-records`` and leave the
always-injected set (the retriever injects records instead).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from xgen_agent_runtime.memory.records.extract import (
    CANDIDATE_SCHEMA,
    IndexedTurn,
    batch_turns,
    build_catalog,
    build_migration_instruction,
    parse_candidates,
)
from xgen_agent_runtime.memory.records.match import coverage
from xgen_agent_runtime.memory.records.model import MemoryRecord, now_iso, slugify
from xgen_agent_runtime.memory.records.reconcile import (
    CompleteStructured,
    Reconciler,
    ReconcileReport,
)
from xgen_agent_runtime.memory.records.store import RecordStore

logger = logging.getLogger(__name__)

__all__ = ["MIGRATED_TAG", "PINNED_CATEGORY", "migrate_legacy"]

PINNED_CATEGORY = "critical"
MIGRATED_TAG = "migrated-to-records"
_LEDGER = "__facts__.md"
#: Share of an old entry's words that must reappear in the records made from it.
_MIN_COVERAGE = 0.6
_EVERGREEN = "__evergreen__.md"
#: Model calls' worth of old memory moved per pass; a large old memory moves over across passes.
MIGRATION_BATCHES_PER_PASS = 4
#: Ledger kinds → record kinds for the deterministic path.
_FACT_KIND = {
    "identity": "fact",
    "preference": "rule",
    "relationship": "fact",
    "commitment": "decision",
    "context": "state",
    "knowledge": "fact",
}


async def _legacy_notes(provider: Any) -> List[Any]:
    notes = provider.notes()
    try:
        metas = await notes.list(category=PINNED_CATEGORY)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for meta in metas:
        if MIGRATED_TAG in (getattr(meta, "tags", None) or []):
            continue
        try:
            note = await notes.read(meta.ref.filename)
        except Exception:  # noqa: BLE001
            note = None
        if note is not None and str(getattr(note, "body", "") or "").strip():
            out.append(note)
    return out


def _ledger_facts(note: Any) -> List[Any]:
    from xgen_agent_runtime.memory.facts import Fact
    import json as _json

    fm = getattr(note, "frontmatter", None) or {}
    raw = fm.get("facts_json")
    rows = []
    if isinstance(raw, str) and raw.strip():
        try:
            rows = _json.loads(raw)
        except ValueError:
            rows = []
    facts = [Fact.from_dict(r) for r in rows if isinstance(r, dict)]
    return [f for f in facts if f is not None and f.status == "active"]


def _entry_key(note: Any, fact: Any) -> str:
    """Stable per source version: a source edited after it was migrated is a new entry."""
    if fact is not None:
        return f"ledger:{fact.id}:{fact.updated or fact.created or ''}"
    at = note.updated_at.isoformat() if getattr(note, "updated_at", None) else ""
    return f"note:{note.ref.filename}:{at}"


def _origin(note: Any, fact: Any) -> str:
    return f"migration:ledger:{fact.id}" if fact is not None else f"migration:{note.ref.filename}"


def _entries(notes: Sequence[Any]) -> List[Tuple[IndexedTurn, Any, Any]]:
    """Every source as one 'memory' turn — ledger facts one by one, other notes whole.

    Returns ``(turn, note, fact_or_None)`` so a failed batch can be converted one by one.
    """
    out: List[Tuple[IndexedTurn, Any, Any]] = []
    for note in notes:
        name = note.ref.filename
        at = note.updated_at.isoformat() if getattr(note, "updated_at", None) else ""
        if name == _LEDGER:
            for f in _ledger_facts(note):
                turn = IndexedTurn(
                    len(out) + 1,
                    "memory",
                    f"(ledger fact {f.id}, {f.kind}, {f.importance}) {f.statement}",
                    f.updated or at,
                )
                out.append((turn, note, f))
            continue
        title = str(getattr(note, "title", "") or name)
        out.append(
            (
                IndexedTurn(len(out) + 1, "memory", f"(note “{title}”)\n{note.body.strip()}", at),
                note,
                None,
            )
        )
    return out


def _deterministic_record(note: Any, f: Any) -> MemoryRecord:
    if f is not None:
        subject, _, key = f.id.partition(".")
        return MemoryRecord(
            id="",
            kind=_FACT_KIND.get(f.kind, "fact"),
            subject=subject or f.id,
            key=key,
            text=f.statement,
            authority="migrated",
            importance=f.importance
            if f.importance in ("critical", "high", "medium", "low")
            else "medium",
            confidence=0.85,
            valid_from=f.created or "",
            confirmed_at=f.updated or f.created or "",
            origin=_origin(note, f),
        )
    name = note.ref.filename
    title = str(getattr(note, "title", "") or name).strip()
    body = str(note.body or "").strip()
    return MemoryRecord(
        id="",
        kind="fact" if name == _EVERGREEN else "rule",
        subject=title or slugify(name),
        text=body,
        headline=_headline(title, body) if len(body) > 200 else "",
        authority="migrated",
        importance="high",
        confidence=0.85,
        confirmed_at=note.updated_at.isoformat() if getattr(note, "updated_at", None) else "",
        origin=_origin(note, None),
    )


def _headline(title: str, body: str) -> str:
    """The title and the note's first line — what a long migrated note says, in one line."""
    first = next((ln.strip(" #-*\t") for ln in body.splitlines() if ln.strip(" #-*\t")), "")
    first = " ".join(first.split())
    if len(first) > 140:
        first = first[:140].rstrip() + "…"
    if title and first and not first.startswith(title):
        return f"{title}: {first}"
    return first or title


async def _deterministic(store: RecordStore, sources: Sequence[Tuple[Any, Any]]) -> int:
    """One record per ledger fact, one per other note. Loses nothing, and is idempotent.

    A source converted before (an interrupted pass, a note edited after migration) updates the record
    it made — found by ``origin`` — instead of adding a second one.
    """
    by_origin: Dict[str, MemoryRecord] = {}
    for r in await store.load(live_only=False):
        if r.origin.startswith("migration:") and (r.origin not in by_origin or r.live):
            by_origin[r.origin] = r
    written = 0
    for note, f in sources:
        rec = _deterministic_record(note, f)
        reason = (
            "migrated from the fact ledger"
            if f is not None
            else f"migrated from {note.ref.filename}"
        )
        old = by_origin.get(rec.origin)
        if old is not None:
            if old.text.strip() == rec.text.strip() or not old.live:
                continue  # already there (or retired since — a retirement is not undone by a re-run)
            old.text, old.headline = rec.text, rec.headline or old.headline
            old.change_reason = f"source edited: {reason}"
            written += int(
                await store.put(old, expected_version=old.version, reason=old.change_reason)
            )
            continue
        rec.id = await store.new_id(rec)
        if await store.put(rec, expected_version=None, reason=reason):
            written += 1
            by_origin[rec.origin] = rec
    return written


async def _mark(provider: Any, notes: Sequence[Any]) -> None:
    from xgen_agent_runtime.memory.provider import NotePatch

    for note in notes:
        tags = list(getattr(note, "tags", None) or [])
        if MIGRATED_TAG in tags:
            continue
        try:
            await provider.notes().update(note.ref.filename, NotePatch(tags=tags + [MIGRATED_TAG]))
        except Exception:  # noqa: BLE001
            logger.warning(
                "records: could not tag %s as migrated", note.ref.filename, exc_info=True
            )


async def migrate_legacy(
    provider: Any,
    store: RecordStore,
    *,
    complete_structured: Optional[CompleteStructured],
    done: Optional[Set[str]] = None,
    max_batches: int = MIGRATION_BATCHES_PER_PASS,
) -> Dict[str, Any]:
    """Migrate whatever old pinned memory is not migrated yet, at most ``max_batches`` model calls' worth.

    ``done`` holds the entries already migrated (the pipeline keeps it in its state), so a large old
    memory moves over across several passes and an interrupted pass resumes where it stopped. A note is
    tagged migrated once every entry of it is done; its keys then leave ``done``.
    """
    done = done if done is not None else set()
    notes = await _legacy_notes(provider)
    report: Dict[str, Any] = {"notes": len(notes), "mode": "", "records": 0, "remaining": 0}
    if not notes:
        return report
    entries = _entries(notes)
    # Keys first: tagging a note changes its time (and, in a store that hands out its own objects, the note).
    keys = {id(note): [_entry_key(n, f) for _t, n, f in entries if n is note] for note in notes}
    pending = [e for e in entries if _entry_key(e[1], e[2]) not in done]
    if complete_structured is not None and pending:
        live = await store.load()
        totals = ReconcileReport()
        try:
            for _ in range(max(0, max_batches)):
                if not pending:
                    break
                batch, _ = batch_turns([t for t, _n, _f in pending], limit=30_000)
                data = await complete_structured(
                    build_migration_instruction(
                        catalog=build_catalog(live, "\n".join(t.text for t in batch)), turns=batch
                    ),
                    CANDIDATE_SCHEMA,
                )
                candidates = parse_candidates(data, batch)
                if not candidates and any(len(t.text) > 40 for t in batch):
                    raise RuntimeError("migration produced no records")
                started = now_iso()
                rep = await Reconciler(
                    store, complete_structured=complete_structured, session="migration"
                ).run(candidates, live)
                if rep is None:
                    raise RuntimeError("migration judging failed")
                for k in ("added", "confirmed", "updated", "resolved", "contested"):
                    setattr(totals, k, getattr(totals, k) + getattr(rep, k))
                batch_done, pending = pending[: len(batch)], pending[len(batch) :]
                # Nothing may be lost in the split: an entry whose words did not make it into this batch's
                # records is also kept whole, one by one.
                written = "\n".join(
                    r.text + " " + r.subject for r in live if (r.updated_at or "") >= started
                )
                missed = [
                    (n, f) for t, n, f in batch_done if coverage(written, t.text) < _MIN_COVERAGE
                ]
                if missed:
                    report["kept_whole"] = int(report.get("kept_whole") or 0) + len(missed)
                    totals.added += await _deterministic(store, missed)
                done.update(_entry_key(n, f) for _t, n, f in batch_done)
            report.update(
                mode="model", records=totals.added, merged=totals.confirmed + totals.updated
            )
        except Exception as exc:  # noqa: BLE001 — what is done stays (valid records); the rest one by one
            logger.warning(
                "records: model migration stopped (%s); converting the rest one by one", exc
            )
            report.update(mode="model+deterministic", records=totals.added)
            report["records"] += await _deterministic(store, [(n, f) for _t, n, f in pending])
            done.update(_entry_key(n, f) for _t, n, f in pending)
            pending = []
    elif pending:
        report["mode"] = "deterministic"
        report["records"] = await _deterministic(store, [(n, f) for _t, n, f in pending])
        done.update(_entry_key(n, f) for _t, n, f in pending)
        pending = []
    report["remaining"] = len(pending)
    finished = [note for note in notes if all(k in done for k in keys[id(note)])]
    await _mark(provider, finished)
    for note in finished:
        done.difference_update(keys[id(note)])
    report["finished_notes"] = len(finished)
    report["at"] = now_iso()
    return report
