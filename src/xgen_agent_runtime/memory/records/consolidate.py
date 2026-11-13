"""Consolidation — the background tidy-up that keeps a growing memory sharp.

Writing reconciles one turn at a time; over months the same thing still gets said many ways, decisions
get settled without anyone saying "the decision is settled", and long records lack a headline. This
pass looks at small clusters of records at a time and asks the model what to do with each cluster:

* merge   — several records say the same thing → one canonical record (the others superseded by it)
* retire  — no longer true / no longer relevant (never a record the user stated or directed)
* resolve — an open decision that later records show was settled
* contest — records that contradict each other at equal standing (the user decides)
* keep

Bounded per run (a few clusters), rotating through the memory over passes, so its cost does not grow
with the memory. Every change is a new version with the previous one in history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from xgen_agent_runtime.memory.records.match import Scorer
from xgen_agent_runtime.memory.records.model import (
    as_int,
    MemoryRecord,
    authority_rank,
    normalize_name,
    now_iso,
)
from xgen_agent_runtime.memory.records.reconcile import CompleteStructured
from xgen_agent_runtime.memory.records.store import RecordStore

logger = logging.getLogger(__name__)

__all__ = ["CONSOLIDATE_SCHEMA", "ConsolidationReport", "consolidate", "find_clusters"]

CLUSTERS_PER_RUN = 6
CLUSTER_MAX = 6
HEADLINES_PER_RUN = 30
#: Texts longer than this get a headline (the line used when a section is short of room).
HEADLINE_NEEDED_OVER = 220
#: BM25 score above which two records of a kind group are likely the same thing.
_NEAR = 6.0
_GROUP = {
    "rule": "rule",
    "procedure": "rule",
    "fact": "fact",
    "state": "fact",
    "decision": "decision",
}

CONSOLIDATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["actions", "headlines"],
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cluster", "op", "ids", "reason"],
                "properties": {
                    "cluster": {"type": "integer"},
                    "op": {"enum": ["merge", "retire", "resolve", "contest", "keep"]},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "canonical": {"type": "string", "description": "merge: the id that remains"},
                    "text": {
                        "type": "string",
                        "description": "merge: full text of the remaining record; resolve: what was decided",
                    },
                    "headline": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "headlines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "headline"],
                "properties": {"id": {"type": "string"}, "headline": {"type": "string"}},
            },
        },
    },
}


@dataclass
class ConsolidationReport:
    clusters: int = 0
    merged: int = 0
    retired: int = 0
    resolved: int = 0
    contested: int = 0
    headlines: int = 0
    refused: int = 0
    notes: List[str] = field(default_factory=list)


#: Without a "changed since" set, near-duplicates are looked for among this many most recent records.
RECENT_FOCUS = 200


def find_clusters(
    records: Sequence[MemoryRecord],
    *,
    start: int = 0,
    focus: Optional[Set[str]] = None,
) -> List[List[MemoryRecord]]:
    """Small groups that may be one thing: same identity, or very similar text within a kind group.

    Open decisions join the group of records about the same subject (a later state may settle them).
    ``start`` rotates which part of the memory is looked at first. Near-duplicate wording is searched
    for the ``focus`` records only (those changed since the last run) against all — linear in the memory,
    not quadratic.
    """
    live = [r for r in records if r.live]
    if len(live) < 2:
        return []
    parent = {r.id: r.id for r in live}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_identity: Dict[Tuple[str, str], List[MemoryRecord]] = {}
    by_name: Dict[str, List[MemoryRecord]] = {}
    for r in live:
        by_identity.setdefault((_GROUP[r.kind], r.identity_key()), []).append(r)
        for n in r.names():
            by_name.setdefault(n, []).append(r)
    for group in by_identity.values():
        for r in group[1:]:
            union(group[0].id, r.id)
    # The same thing under another of its names (subject of one = alias of the other), same key.
    for named in by_name.values():
        firsts: Dict[Tuple[str, str], str] = {}
        for r in named:
            ident = (_GROUP[r.kind], normalize_name(r.key))
            if ident in firsts:
                union(firsts[ident], r.id)
            else:
                firsts[ident] = r.id
    # Open decisions with the records about the same subject.
    for r in live:
        if r.kind != "decision":
            continue
        for n in r.names():
            for other in by_name.get(n, []):
                if other.id != r.id and other.last_seen() > r.last_seen():
                    union(r.id, other.id)
    # Near-duplicate wording within a kind group.
    for g in ("rule", "fact", "decision"):
        members = [r for r in live if _GROUP[r.kind] == g]
        if len(members) < 2:
            continue
        scorer = Scorer([f"{r.subject} {r.key} {r.text}" for r in members])
        if focus is None:
            recent = sorted(members, key=lambda r: r.updated_at or r.recorded_at, reverse=True)[
                :RECENT_FOCUS
            ]
            wanted = {r.id for r in recent}
        else:
            wanted = focus
        for i, r in enumerate(members):
            if r.id not in wanted:
                continue
            for j, score in scorer.rank(f"{r.subject} {r.key} {r.text}", min_score=_NEAR)[:4]:
                if j != i:
                    union(r.id, members[j].id)
    clusters: Dict[str, List[MemoryRecord]] = {}
    for r in live:
        clusters.setdefault(find(r.id), []).append(r)
    out = [sorted(c, key=lambda r: r.id)[:CLUSTER_MAX] for c in clusters.values() if len(c) > 1]
    out.sort(key=lambda c: c[0].id)
    if out and start:
        k = start % len(out)
        out = out[k:] + out[:k]
    return out


def _instruction(
    clusters: Sequence[Sequence[MemoryRecord]], needs_headline: Sequence[MemoryRecord]
) -> str:
    blocks = []
    for i, cluster in enumerate(clusters):
        lines = [f"CLUSTER {i}:"]
        for r in cluster:
            day = r.day(r.last_seen())
            lines.append(
                f"  {r.id} [{r.kind}, {r.status}, {r.authority}{', last seen ' + day if day else ''}] "
                f"{r.subject}" + (f" / {r.key}" if r.key else "") + f" — {r.text}"
            )
        blocks.append("\n".join(lines))
    heads = "\n".join(f"  {r.id}: {r.text}" for r in needs_headline) or "  (none)"
    return (
        "You maintain an agent's long-term memory. For each CLUSTER of records decide:\n"
        "- merge: they say the same thing → name the canonical id and give the full merged text (every "
        "still-true detail of all of them, the user's language).\n"
        "- retire: records that are no longer true or no longer matter (later records show it).\n"
        "- resolve: an open decision that later records show was settled → text = what was decided.\n"
        "- contest: records that contradict each other and it is unclear which holds.\n"
        "- keep: they are different things.\n"
        "One action per decision you make; ids must be from that cluster. Never invent facts.\n"
        "Also write a one-line headline (under ~100 characters, same language) for each record under "
        "NEEDS HEADLINE. Output ONLY JSON matching the schema.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nNEEDS HEADLINE:\n{heads}\n"
    )


async def consolidate(
    store: RecordStore,
    *,
    complete_structured: CompleteStructured,
    records: Optional[List[MemoryRecord]] = None,
    rotation: int = 0,
    since: str = "",
) -> Optional[ConsolidationReport]:
    """One bounded consolidation run. ``None`` = the model call failed (nothing changed).

    ``since``: ISO time of the previous run — records changed after it are checked for near-duplicates.
    """
    live = records if records is not None else await store.load()
    focus = {r.id for r in live if (r.updated_at or r.recorded_at or "") > since} if since else None
    clusters = find_clusters(live, start=rotation, focus=focus)[:CLUSTERS_PER_RUN]
    needs_headline = [
        r for r in live if r.live and not r.headline and len(r.text) > HEADLINE_NEEDED_OVER
    ]
    needs_headline.sort(key=lambda r: (r.kind != "rule", r.id))
    needs_headline = needs_headline[:HEADLINES_PER_RUN]
    report = ConsolidationReport(clusters=len(clusters))
    if not clusters and not needs_headline:
        return report
    try:
        data = await complete_structured(_instruction(clusters, needs_headline), CONSOLIDATE_SCHEMA)
    except Exception:  # noqa: BLE001
        logger.warning("records: consolidation call failed", exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    by_id = {r.id: r for r in live}
    for action in data.get("actions") or []:
        if not isinstance(action, dict):
            continue
        try:
            ci = as_int(action.get("cluster"))
        except (TypeError, ValueError):
            continue
        if not 0 <= ci < len(clusters):
            continue
        members = {r.id for r in clusters[ci]}
        ids = [i for i in (action.get("ids") or []) if i in members]
        op = str(action.get("op") or "keep")
        reason = str(action.get("reason") or "")[:300]
        if op == "merge":
            await _merge(
                store, by_id, ids, str(action.get("canonical") or ""), action, reason, report
            )
        elif op == "retire":
            for rid in ids:
                await _retire(store, by_id, rid, reason, report)
        elif op == "resolve":
            for rid in ids:
                await _resolve(store, by_id, rid, str(action.get("text") or ""), reason, report)
        elif op == "contest":
            await _contest(store, by_id, ids, reason, report)
    for row in data.get("headlines") or []:
        if not isinstance(row, dict):
            continue
        rid, head = str(row.get("id") or ""), " ".join(str(row.get("headline") or "").split())
        r = by_id.get(rid)
        if r is None or not head or r.headline or r not in needs_headline:
            continue
        if await _change(
            store, by_id, r, lambda x: setattr(x, "headline", head[:200]), "headline added"
        ):
            report.headlines += 1
    return report


async def _change(
    store: RecordStore, by_id: Dict[str, MemoryRecord], r: MemoryRecord, fn: Any, reason: str
) -> bool:
    fresh = MemoryRecord.from_dict(r.to_dict())
    assert fresh is not None
    fn(fresh)
    if await store.put(fresh, expected_version=r.version, reason=reason):
        by_id[r.id] = fresh
        return True
    return False


async def _merge(
    store: RecordStore,
    by_id: Dict[str, MemoryRecord],
    ids: List[str],
    canonical: str,
    action: Dict[str, Any],
    reason: str,
    report: ConsolidationReport,
) -> None:
    group = [by_id[i] for i in ids if i in by_id and by_id[i].live]
    if len(group) < 2 or canonical not in {r.id for r in group}:
        report.refused += 1
        return
    keep = by_id[canonical]
    if len({_GROUP[r.kind] for r in group}) > 1:
        report.refused += 1
        report.notes.append(f"merge across kinds refused: {ids}")
        return
    strongest = max(authority_rank(r.authority) for r in group)
    if authority_rank(keep.authority) < strongest:
        # The record that remains must carry the strongest source's standing.
        keep = max(group, key=lambda r: (authority_rank(r.authority), r.confirm_count))
    text = str(action.get("text") or "").strip() or keep.text
    head = " ".join(str(action.get("headline") or "").split()) or keep.headline
    others = [r for r in group if r.id != keep.id]

    def _fold(x: MemoryRecord) -> None:
        x.text = text
        x.headline = head[:200]
        x.confirm_count = sum(r.confirm_count for r in group)
        x.aliases = list(
            dict.fromkeys(x.aliases + [a for r in others for a in [r.subject] + r.aliases if a])
        )[:12]
        x.supersedes = list(dict.fromkeys(x.supersedes + [r.id for r in others]))
        x.confirmed_at = max(r.last_seen() for r in group)
        for r in others:
            for ev in r.evidence:
                x.add_evidence(ev)
        x.status = "active"
        x.contested_with = [c for c in x.contested_with if c not in {r.id for r in others}]

    if not await _change(store, by_id, keep, _fold, reason or "merged duplicates"):
        return
    for r in others:

        def _sup(x: MemoryRecord, canon: str = keep.id) -> None:
            x.status = "superseded"
            x.superseded_by = canon

        await _change(store, by_id, r, _sup, f"merged into {keep.id}")
    report.merged += len(others)


async def _retire(
    store: RecordStore,
    by_id: Dict[str, MemoryRecord],
    rid: str,
    reason: str,
    report: ConsolidationReport,
) -> None:
    r = by_id.get(rid)
    if r is None or not r.live:
        return
    if r.authority in ("user_directive", "user_statement") and r.kind != "state":
        # What the user said is retired by the user (or replaced by something newer they said), not by a tidy-up.
        report.refused += 1
        report.notes.append(f"retire of user record {rid} refused")
        return
    if await _change(
        store, by_id, r, lambda x: setattr(x, "status", "retired"), reason or "no longer true"
    ):
        report.retired += 1


async def _resolve(
    store: RecordStore,
    by_id: Dict[str, MemoryRecord],
    rid: str,
    text: str,
    reason: str,
    report: ConsolidationReport,
) -> None:
    r = by_id.get(rid)
    if r is None or r.kind != "decision" or not r.live:
        return

    def _fn(x: MemoryRecord) -> None:
        x.status = "resolved"
        x.resolved_at = now_iso()
        if text.strip():
            x.text = text.strip()

    if await _change(store, by_id, r, _fn, reason or "settled"):
        report.resolved += 1


async def _contest(
    store: RecordStore,
    by_id: Dict[str, MemoryRecord],
    ids: List[str],
    reason: str,
    report: ConsolidationReport,
) -> None:
    group = [by_id[i] for i in ids if i in by_id and by_id[i].live]
    if len(group) < 2:
        return
    top = max(authority_rank(r.authority) for r in group)
    group = [r for r in group if authority_rank(r.authority) == top]
    if len(group) < 2:
        return  # a weaker record contradicting a stronger one is not a contest; the stronger stands
    for r in group:
        others = [o.id for o in group if o.id != r.id]

        def _fn(x: MemoryRecord, others: List[str] = others) -> None:
            x.status = "contested"
            x.contested_with = list(dict.fromkeys(x.contested_with + others))

        await _change(store, by_id, r, _fn, reason or "contradicts another record")
    report.contested += len(group)
