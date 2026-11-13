"""What the agent's owner does to records from outside a conversation (the memory browser).

The owner is the user: what they write here carries the user's authority (``user_directive`` for a
rule, ``user_statement`` otherwise), and every change is a new version with the previous one kept in
history. These functions are the only way a host should change a record — editing the note itself
would leave the record (kept in the note's frontmatter) and its text apart.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from xgen_agent_runtime.memory.records.model import KINDS, Evidence, MemoryRecord, now_iso
from xgen_agent_runtime.memory.records.store import RecordStore

__all__ = [
    "OWNER_ACTIONS",
    "OwnerActionError",
    "owner_action",
    "owner_create",
    "record_summary",
    "records_overview",
]

OWNER_ACTIONS = ("correct", "retire", "restore", "resolve", "settle", "revert", "confirm")


class OwnerActionError(ValueError):
    """A request that cannot be applied (unknown record, wrong state, stale version)."""


def record_summary(r: MemoryRecord) -> Dict[str, Any]:
    return {
        "id": r.id,
        "filename": r.filename,
        "kind": r.kind,
        "subject": r.subject,
        "key": r.key,
        "headline": r.headline,
        "text": r.text,
        "project": r.project,
        "scope": r.scope,
        "applies_when": list(r.applies_when),
        "status": r.status,
        "authority": r.authority,
        "importance": r.importance,
        "version": r.version,
        "confirm_count": r.confirm_count,
        "confirmed_at": r.confirmed_at,
        "updated_at": r.updated_at,
        "recorded_at": r.recorded_at,
        "contested_with": list(r.contested_with),
        "superseded_by": r.superseded_by,
        "change_reason": r.change_reason,
        "origin": r.origin,
    }


def _as_owner(r: MemoryRecord, reason: str) -> None:
    r.authority = "user_directive" if r.kind == "rule" else "user_statement"
    r.confirmed_at = now_iso()
    r.add_evidence(
        Evidence(session="memory-browser", at=r.confirmed_at, role="user", quote=reason[:300])
    )
    r.change_reason = reason


async def owner_action(
    provider: Any,
    record_id: str,
    action: str,
    *,
    text: Optional[str] = None,
    headline: Optional[str] = None,
    version: Optional[int] = None,
    expected_version: Optional[int] = None,
    reason: str = "",
) -> Dict[str, Any]:
    """Apply one owner action. ``expected_version`` (what the browser showed) guards concurrent changes."""
    if action not in OWNER_ACTIONS:
        raise OwnerActionError(f"action must be one of {', '.join(OWNER_ACTIONS)}")
    store = RecordStore(provider)
    rec = await store.get(record_id)
    if rec is None:
        raise OwnerActionError(f"no record {record_id!r}")
    if expected_version is not None and rec.version != int(expected_version):
        raise OwnerActionError(
            f"{record_id} changed meanwhile (now v{rec.version}); reload and retry"
        )
    reason = (reason or "").strip() or f"{action} in the memory browser"

    if action == "revert":
        if version is None:
            raise OwnerActionError("version is required")
        restored = await store.revert(record_id, int(version), reason=reason)
        if restored is None:
            raise OwnerActionError(f"{record_id} has no version {version}, or it changed meanwhile")
        return {"status": "reverted", "record": record_summary(restored)}

    if action == "settle":
        return await _settle(store, rec, reason)

    if action == "correct":
        new_text = (text or "").strip()
        if not new_text:
            raise OwnerActionError("text is required")
        rec.text = new_text
        rec.headline = (headline or "").strip()
        if rec.status in ("retired", "superseded"):
            rec.status = "active"
    elif action == "retire":
        rec.status = "retired"
    elif action == "restore":
        if rec.live:
            raise OwnerActionError(f"{record_id} is already in force")
        rec.status = "active"
        rec.superseded_by = ""
    elif action == "resolve":
        if rec.kind != "decision":
            raise OwnerActionError("only a decision is resolved")
        rec.status = "resolved"
        rec.resolved_at = now_iso()
    elif action == "confirm":
        rec.confirm_count += 1
    _as_owner(rec, reason)
    if not await store.put(rec, expected_version=rec.version, reason=reason):
        raise OwnerActionError(f"{record_id} changed meanwhile; reload and retry")
    return {"status": action, "record": record_summary(rec)}


async def _settle(store: RecordStore, keep: MemoryRecord, reason: str) -> Dict[str, Any]:
    if not keep.contested_with:
        raise OwnerActionError(f"{keep.id} is not contested")
    superseded: List[str] = []
    for oid in list(keep.contested_with):
        other = await store.get(oid)
        if other is None or not other.live:
            continue
        other.status = "superseded"
        other.superseded_by = keep.id
        other.contested_with = [x for x in other.contested_with if x != keep.id]
        other.change_reason = f"settled in favour of {keep.id}: {reason}"
        if await store.put(other, expected_version=other.version, reason=other.change_reason):
            superseded.append(oid)
    keep.status = "active"
    keep.contested_with = []
    keep.supersedes = list(dict.fromkeys(keep.supersedes + superseded))
    _as_owner(keep, reason)
    if not await store.put(keep, expected_version=keep.version, reason=reason):
        raise OwnerActionError(f"{keep.id} changed meanwhile; reload and retry")
    return {"status": "settled", "record": record_summary(keep), "superseded": superseded}


async def owner_create(
    provider: Any,
    *,
    kind: str,
    subject: str,
    text: str,
    key: str = "",
    project: str = "",
    importance: str = "high",
) -> Dict[str, Any]:
    """A record the owner writes directly (a rule they want kept, a fact to remember)."""
    if kind not in KINDS:
        raise OwnerActionError(f"kind must be one of {', '.join(KINDS)}")
    if not subject.strip() or not text.strip():
        raise OwnerActionError("subject and text are required")
    store = RecordStore(provider)
    rec = MemoryRecord(
        id="",
        kind=kind,
        subject=subject.strip(),
        key=key.strip(),
        text=text.strip(),
        project=project.strip(),
        scope=f"project:{project.strip()}" if project.strip() else "agent",
        importance=importance if importance in ("critical", "high", "medium", "low") else "high",
        confidence=0.95,
        origin="memory-browser",
    )
    _as_owner(rec, "written in the memory browser")
    rec.valid_from = rec.confirmed_at
    rec.id = await store.new_id(rec)
    if not await store.put(rec, expected_version=None, reason="written in the memory browser"):
        raise OwnerActionError("could not write the record")
    return {"status": "created", "record": record_summary(rec)}


async def records_overview(provider: Any) -> Dict[str, Any]:
    """Counts per kind and status, the contested records, and where the background pass stands."""
    store = RecordStore(provider)
    counts = await store.counts()
    contested = [record_summary(r) for r in await store.load() if r.status == "contested"]
    state = await store.load_state()
    migration = dict(state.get("migration") or {})
    migration.pop("done_keys", None)
    return {
        "counts": counts,
        "contested": contested,
        "migration": migration,
        "last_pass": state.get("last_pass") or {},
        "last_consolidation_at": state.get("last_consolidation_at") or "",
        "conversations": len(state.get("cursors") or {}),
    }
