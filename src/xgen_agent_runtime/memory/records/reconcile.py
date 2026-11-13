"""Reconcile: each candidate against the few existing records it may be — add, confirm, update,
resolve, contest or skip.

The model sees one candidate with its neighbourhood (same subject or alias, similar text), never
the whole memory, and proposes an operation. The code holds the invariants the model cannot break:

* the target must be one of the neighbours it was shown;
* a weaker source never replaces a stronger one (an inference does not overwrite what the user said;
  a document cannot rewrite a rule). A *state* is the exception that proves it: states change, so a
  newer observation from a tool replaces an older value whoever stated it;
* one live state per subject + key — a "new" state for a key that already has a value is an update;
* two equally strong live records that disagree are both marked ``contested`` (the user decides),
  never silently resolved;
* nothing is deleted: replaced versions go to history (``RecordStore.put``).

If the judging call fails, nothing is written and the caller retries the turns on the next pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from xgen_agent_runtime.memory.records.extract import Candidate
from xgen_agent_runtime.memory.records.match import Scorer
from xgen_agent_runtime.memory.records.model import (
    as_int,
    Evidence,
    MemoryRecord,
    authority_rank,
    importance_rank,
    normalize_name,
    now_iso,
    slugify,
)
from xgen_agent_runtime.memory.records.store import RecordStore, same_identity

logger = logging.getLogger(__name__)

__all__ = ["JUDGE_SCHEMA", "ReconcileReport", "Reconciler", "record_from_candidate"]

CompleteStructured = Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]

#: Kinds that can be the same thing as each other (a "fact" the model filed may be a known "state").
_GROUPS = {
    "rule": ("rule", "procedure"),
    "procedure": ("procedure", "rule"),
    "fact": ("fact", "state"),
    "state": ("state", "fact"),
    "decision": ("decision",),
}
NEIGHBOURS = 6
#: Lexical similarity below this is not a neighbour (BM25 units over the group).
_MIN_SIMILAR = 1.5

_CONFIDENCE = {
    "user_directive": 0.95,
    "user_statement": 0.9,
    "migrated": 0.85,
    "tool_result": 0.8,
    "agent_inference": 0.5,
}

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate", "op", "target", "reason"],
                "properties": {
                    "candidate": {"type": "integer"},
                    "op": {"enum": ["new", "same", "update", "resolves", "conflict", "skip"]},
                    "target": {
                        "type": "string",
                        "description": "id of the existing record ('' for new / skip)",
                    },
                    "merged_text": {
                        "type": "string",
                        "description": "same/update: the record's full text after taking the candidate in "
                        "(keep every detail of both that is still true). '' to use the candidate's text as is.",
                    },
                    "merged_headline": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


@dataclass
class ReconcileReport:
    added: int = 0
    confirmed: int = 0
    updated: int = 0
    resolved: int = 0
    contested: int = 0
    skipped: int = 0
    refused: int = 0  # operations the invariants turned down
    conflicts_on_write: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def changes(self) -> int:
        return self.added + self.confirmed + self.updated + self.resolved + self.contested


def _evidence(c: Candidate, session: str) -> Evidence:
    t = c.turn
    return Evidence(
        session=session,
        at=(t.at if t else "") or now_iso(),
        role=(t.role if t else ""),
        quote=c.evidence,
    )


def record_from_candidate(c: Candidate, rid: str, session: str) -> MemoryRecord:
    at = (c.turn.at if c.turn else "") or now_iso()
    rec = MemoryRecord(
        id=rid,
        kind=c.kind,
        subject=c.subject,
        key=c.key,
        text=c.text,
        headline=c.headline,
        aliases=list(c.aliases),
        scope=f"project:{slugify(c.project)}" if c.project else "agent",
        project=c.project,
        applies_when=list(c.applies_when),
        authority=c.authority,
        status="resolved" if c.kind == "decision" and c.decision_status == "resolved" else "active",
        importance=c.importance,
        confidence=_CONFIDENCE.get(c.authority, 0.6),
        valid_from=at,
        confirmed_at=at,
        origin="migration" if c.authority == "migrated" else "extraction",
    )
    rec.add_evidence(_evidence(c, session))
    if rec.status == "resolved":
        rec.resolved_at = at
    return rec


def _may_replace(candidate_authority: str, target: MemoryRecord) -> bool:
    if target.kind == "state":
        return authority_rank(candidate_authority) >= 3
    return authority_rank(candidate_authority) >= authority_rank(target.authority)


def _dedupe(cands: Sequence[Candidate]) -> List[Candidate]:
    """Within one batch the same thing said twice is one candidate — the later wording wins."""
    keyed: Dict[str, int] = {}
    out: List[Optional[Candidate]] = []
    for c in cands:
        k = f"{c.kind}::{normalize_name(c.subject)}::{normalize_name(c.key)}::{normalize_name(c.text)[:80]}"
        if c.kind == "state":
            k = f"state::{normalize_name(c.subject)}::{normalize_name(c.key)}"
        if k in keyed:
            prev = out[keyed[k]]
            if prev is not None:
                c.aliases = list(dict.fromkeys(prev.aliases + c.aliases))
            out[keyed[k]] = None
        keyed[k] = len(out)
        out.append(c)
    return [c for c in out if c is not None]


class Reconciler:
    def __init__(
        self,
        store: RecordStore,
        *,
        complete_structured: Optional[CompleteStructured],
        session: str = "",
    ) -> None:
        self._store = store
        self._complete = complete_structured
        self._session = session
        self._scorers: Dict[Tuple[str, ...], Tuple[Tuple[Tuple[str, int], ...], Scorer]] = {}

    # ── neighbourhood ───────────────────────────────────────────────

    def _scorer(self, key: Tuple[str, ...], group: Sequence[MemoryRecord]) -> Scorer:
        """One index per kind group for the whole run, rebuilt only when the group changed."""
        sig = tuple((r.id, r.version) for r in group)
        cached = self._scorers.get(key)
        if cached is not None and cached[0] == sig:
            return cached[1]
        scorer = Scorer([f"{r.subject} {' '.join(r.aliases)} {r.key} {r.text}" for r in group])
        self._scorers[key] = (sig, scorer)
        return scorer

    def neighbours(self, c: Candidate, live: Sequence[MemoryRecord]) -> List[MemoryRecord]:
        group = [r for r in live if r.kind in _GROUPS.get(c.kind, (c.kind,))]
        if not group:
            return []
        names = {normalize_name(c.subject)} | {normalize_name(a) for a in c.aliases}
        names.discard("")
        exact: List[MemoryRecord] = []
        named: List[MemoryRecord] = []
        for r in group:
            if names & set(r.names()):
                if normalize_name(r.key) == normalize_name(c.key):
                    exact.append(r)
                else:
                    named.append(r)
        query = f"{c.subject} {' '.join(c.aliases)} {c.key} {c.text}"
        similar = [
            group[i]
            for i, s in self._scorer(tuple(sorted(_GROUPS.get(c.kind, (c.kind,)))), group).rank(
                query, min_score=_MIN_SIMILAR
            )[:NEIGHBOURS]
        ]
        out: List[MemoryRecord] = []
        for r in exact + named + similar:
            if r not in out:
                out.append(r)
        return out[:NEIGHBOURS]

    # ── judging ─────────────────────────────────────────────────────

    def _instruction(self, items: Sequence[Tuple[int, Candidate, List[MemoryRecord]]]) -> str:
        blocks: List[str] = []
        for i, c, near in items:
            lines = [
                f"CANDIDATE {i}: [{c.kind}] {c.subject}"
                + (f" / {c.key}" if c.key else "")
                + f" — from {c.authority}"
                + (f", turn at {c.turn.at[:16]}" if c.turn and c.turn.at else ""),
                f"  text: {c.text}",
            ]
            if c.kind == "decision":
                lines.append(f"  decision_status: {c.decision_status}")
            for r in near:
                seen = r.day(r.last_seen())
                lines.append(
                    f"  EXISTING {r.id}: [{r.kind}, {r.status}, {r.authority}{', last seen ' + seen if seen else ''}] "
                    f"{r.subject}" + (f" / {r.key}" if r.key else "") + f" — {r.text}"
                )
            blocks.append("\n".join(lines))
        return (
            "You keep an agent's long-term memory consistent. For every CANDIDATE decide how it relates to the "
            "EXISTING records listed under it:\n"
            "- new: it is something not yet remembered.\n"
            "- same: it restates an existing record (possibly with more detail — then give merged_text).\n"
            "- update: it is a newer value or a correction of an existing record (give merged_text if parts of "
            "the old text are still true).\n"
            "- resolves: it settles an existing open decision.\n"
            "- conflict: it contradicts an existing record and it is NOT clear that it replaces it.\n"
            "- skip: not worth keeping (noise, already covered by a stronger record).\n"
            "Use only the ids shown under that candidate. A later statement by the same source normally updates an "
            "earlier one; a contradiction between two different strong sources is a conflict. Keep the user's "
            "language and every still-true detail in merged_text. Output ONLY JSON matching the schema.\n\n"
            + "\n\n".join(blocks)
        )

    async def run(
        self, candidates: Sequence[Candidate], live: List[MemoryRecord]
    ) -> Optional[ReconcileReport]:
        """Apply ``candidates`` against ``live`` (updated in place). ``None`` = judging failed, nothing written."""
        report = ReconcileReport()
        cands = _dedupe(candidates)
        items = [(i, c, self.neighbours(c, live)) for i, c in enumerate(cands)]
        decisions: Dict[int, Dict[str, Any]] = {}
        to_judge = [it for it in items if it[2]]
        if to_judge:
            if self._complete is None:
                return None
            try:
                data = await self._complete(self._instruction(to_judge), JUDGE_SCHEMA)
            except Exception:  # noqa: BLE001
                logger.warning("records: judging failed", exc_info=True)
                return None
            rows = data.get("decisions") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                return None
            for row in rows:
                if isinstance(row, dict):
                    try:
                        decisions[as_int(row.get("candidate"))] = row
                    except (TypeError, ValueError):
                        continue
        for i, c, near in items:
            row = decisions.get(i) or {"op": "new", "target": "", "reason": "no neighbours"}
            await self._apply(c, near, row, live, report)
        return report

    # ── applying ────────────────────────────────────────────────────

    async def _apply(
        self,
        c: Candidate,
        near: List[MemoryRecord],
        row: Dict[str, Any],
        live: List[MemoryRecord],
        report: ReconcileReport,
    ) -> None:
        op = str(row.get("op") or "new")
        target_id = str(row.get("target") or "")
        target = next((r for r in near if r.id == target_id), None)
        reason = str(row.get("reason") or "")[:300]
        merged_text = str(row.get("merged_text") or "").strip()
        merged_headline = str(row.get("merged_headline") or "").strip()

        if op in ("same", "update", "resolves", "conflict") and target is None:
            # The model named something it was not shown — treat as new, unless it is clearly the same thing.
            report.refused += 1
            report.notes.append(f"unknown target {target_id!r} for {c.subject}")
            op = "new"
        # One live value per state key; one live record per exact identity.
        if op == "new":
            twin = next((r for r in near if r.kind == c.kind and same_identity(r, _probe(c))), None)
            if twin is not None and c.kind in ("state", "fact", "procedure"):
                op, target = (
                    ("update" if normalize_name(twin.text) != normalize_name(c.text) else "same"),
                    twin,
                )
        if op == "skip":
            report.skipped += 1
            return
        if op == "new":
            await self._add(c, live, report)
            return
        assert target is not None
        if op == "same":
            await self._write(
                target,
                lambda r: self._confirm(r, c, merged_text, merged_headline),
                live,
                report,
                "confirmed",
                reason or "said again",
            )
            return
        if op == "resolves":
            if target.kind != "decision":
                op = "update"
            else:

                def _resolve(r: MemoryRecord) -> None:
                    r.status = "resolved"
                    r.resolved_at = (c.turn.at if c.turn else "") or now_iso()
                    r.text = merged_text or c.text
                    if merged_headline or c.headline:
                        r.headline = merged_headline or c.headline
                    r.add_evidence(_evidence(c, self._session))

                await self._write(target, _resolve, live, report, "resolved", reason or "decided")
                return
        if op == "update":
            if not _may_replace(c.authority, target):
                if authority_rank(c.authority) < 3:
                    report.refused += 1
                    report.notes.append(
                        f"{c.authority} may not change {target.id} ({target.authority})"
                    )
                    return
                op = "conflict"
            else:

                def _update(r: MemoryRecord) -> None:
                    r.text = merged_text or c.text
                    r.headline = merged_headline or c.headline or r.headline
                    r.aliases = list(dict.fromkeys(r.aliases + c.aliases))[:12]
                    if (
                        authority_rank(c.authority) >= authority_rank(r.authority)
                        or r.kind == "state"
                    ):
                        r.authority = c.authority
                    r.valid_from = (c.turn.at if c.turn else "") or now_iso()
                    r.confirmed_at = r.valid_from
                    if importance_rank(c.importance) < importance_rank(r.importance):
                        r.importance = c.importance
                    if c.applies_when:
                        r.applies_when = list(c.applies_when)
                    if r.status == "contested":
                        r.status = "active"
                    r.add_evidence(_evidence(c, self._session))

                await self._write(target, _update, live, report, "updated", reason or "newer value")
                return
        if op == "conflict":
            if authority_rank(c.authority) < authority_rank(target.authority):
                report.refused += 1
                report.notes.append(
                    f"weaker {c.authority} contradicts {target.id}; kept {target.authority}"
                )
                return
            new = await self._add(c, live, report, contested_with=target.id)
            if new is not None:

                def _contest(r: MemoryRecord) -> None:
                    r.status = "contested"
                    r.contested_with = list(dict.fromkeys(r.contested_with + [new.id]))

                await self._write(target, _contest, live, report, "", reason or "contradicted")
                report.contested += 1
            return

    def _confirm(
        self, r: MemoryRecord, c: Candidate, merged_text: str, merged_headline: str
    ) -> None:
        r.confirm_count += 1
        r.confirmed_at = (c.turn.at if c.turn else "") or now_iso()
        r.aliases = list(dict.fromkeys(r.aliases + c.aliases))[:12]
        if merged_text and _may_replace(c.authority, r):
            r.text = merged_text
            if merged_headline:
                r.headline = merged_headline
        if not r.headline and c.headline:
            r.headline = c.headline
        if authority_rank(c.authority) > authority_rank(r.authority):
            r.authority = c.authority  # the user saying it again outranks the agent's guess
        r.add_evidence(_evidence(c, self._session))

    async def _add(
        self,
        c: Candidate,
        live: List[MemoryRecord],
        report: ReconcileReport,
        *,
        contested_with: str = "",
    ) -> Optional[MemoryRecord]:
        rec = record_from_candidate(c, "", self._session)
        rec.id = await self._store.new_id(rec)
        if contested_with:
            rec.status = "contested"
            rec.contested_with = [contested_with]
        if not await self._store.put(
            rec, expected_version=None, reason="new" if not contested_with else "contradicts"
        ):
            report.conflicts_on_write += 1
            return None
        live.append(rec)
        if not contested_with:
            report.added += 1
        return rec

    async def _write(
        self,
        target: MemoryRecord,
        change: Callable[[MemoryRecord], None],
        live: List[MemoryRecord],
        report: ReconcileReport,
        counter: str,
        reason: str,
    ) -> bool:
        """Change ``target`` and write it; if another writer moved it on, redo the change on the fresh copy once."""
        for attempt in range(2):
            base = target if attempt == 0 else await self._store.get(target.id)
            if base is None:
                break
            fresh = MemoryRecord.from_dict(base.to_dict())
            assert fresh is not None
            change(fresh)
            if await self._store.put(fresh, expected_version=base.version, reason=reason):
                for i, r in enumerate(live):
                    if r.id == fresh.id:
                        live[i] = fresh
                if counter:
                    setattr(report, counter, getattr(report, counter) + 1)
                return True
        report.conflicts_on_write += 1
        return False


def _probe(c: Candidate) -> MemoryRecord:
    return MemoryRecord(
        id="", kind=c.kind, subject=c.subject, key=c.key, text=c.text, aliases=list(c.aliases)
    )
