"""Assemble: what of the remembered goes into this turn — chosen whole, never cut, within budget.

Storage is unbounded; this block is not. It is filled in priority order and every record either
goes in whole, goes in as its one-line headline, or is *counted* in the memory index with how to
open it — nothing disappears without the agent being told.

1. Rules in force. Rules hold whether or not the turn "sounds related", so they are not ranked by
   similarity: all rules of the agent and of the projects in play are candidates; a rule whose
   ``applies_when`` matches the turn goes first, the others follow as headlines when room is short.
   Contested rules are always shown, both sides, with the instruction to ask the user.
2. The current state of the projects in play — their state values and open decisions, each with the
   date it was last confirmed (the agent sees how fresh a line is instead of treating old status as
   now). Which projects are in play: those the turn or the recent turns name; otherwise the most
   recently active one.
3. Facts and procedures related to this turn (lexical + vector when an index exists), plus the
   critical identity facts that must never be missing.
4. The index: how much of each kind exists and what was not shown, with the tool to open it.

The order inside each section is stable (importance, then id) so the slow-changing part of the
prompt keeps its cache from one turn to the next.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from xgen_agent_runtime.memory.records.match import Scorer, coverage
from xgen_agent_runtime.memory.records.model import (
    MemoryRecord,
    authority_rank,
    filename_for,
    importance_rank,
    normalize_name,
)
from xgen_agent_runtime.memory.records.store import RecordStore

logger = logging.getLogger(__name__)

__all__ = ["Assembly", "assemble"]

#: Shares of the budget per section; what a section leaves unused flows to the next.
_SHARES = {"rules": 0.45, "state": 0.30, "related": 0.25}
#: Old pinned notes not migrated yet may use this much (they are what used to fill the pinned layer).
_LEGACY_SHARE = 0.75
#: Room kept for the memory index at the end of the block.
_INDEX_RESERVE = 700
#: A record longer than this share of its section is shown by its headline.
_LONG_SHARE = 0.25
#: Related facts/procedures considered per turn.
_RELATED_CANDIDATES = 24
_RELATED_MIN_SCORE = 1.0
_PROJECTS_IN_PLAY = 2
_AUTH_SHORT = {
    "user_directive": "user",
    "user_statement": "user",
    "migrated": "earlier memory",
    "tool_result": "observed",
    "agent_inference": "agent's guess",
}


@dataclass
class Assembly:
    """``standing`` = rules, state, index (changes when memory or the projects in play change — it can sit in
    the cached prompt prefix); ``related`` = what this turn's words brought up (changes every turn)."""

    standing: str = ""
    related: str = ""
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(s for s in (self.standing, self.related) if s)


def _first_line(text: str, limit: int = 160) -> str:
    line = " ".join(str(text or "").split())
    if len(line) <= limit:
        return line
    cut = line[:limit]
    for sep in (". ", "。", "다. ", "; "):
        i = cut.rfind(sep)
        if i >= limit // 2:
            return cut[: i + len(sep)].rstrip()
    return cut.rstrip() + "…"


def _short_form(r: MemoryRecord) -> str:
    return r.headline or _first_line(r.text)


def _line(r: MemoryRecord, *, full: bool) -> str:
    body = r.text if full else _short_form(r)
    body = " ".join(body.split()) if not full else body.strip()
    meta: List[str] = []
    who = _AUTH_SHORT.get(r.authority, "")
    if who:
        meta.append(who)
    day = r.day(r.last_seen())
    if day:
        meta.append(
            ("as of " if r.kind == "state" else "since " if r.kind == "decision" else "") + day
        )
    label = ""
    if r.kind == "decision":
        label = "OPEN DECISION: "
    elif r.kind == "state" and r.key:
        label = f"{r.subject} — {r.key}: "
    if r.kind == "rule" and r.applies_when:
        label = f"(when {' / '.join(r.applies_when[:3])}) "
    more = "" if full or body == r.text.strip() else f' [full: memory_read("{filename_for(r.id)}")]'
    tail = f" ({', '.join(meta)})" if meta else ""
    return f"- [{r.id}] {label}{body}{tail}{more}"


def _line_estimate(r: MemoryRecord) -> int:
    """About what ``_line(r, full=True)`` takes, without rendering it (sizing whole sections)."""
    return len(r.id) + len(r.text) + len(r.subject) + len(r.key) + 40


def _fill(
    records: Sequence[MemoryRecord],
    budget: int,
    overhead: int = 0,
) -> Tuple[List[str], List[str], List[str], List[MemoryRecord], int]:
    """Lines for ``records`` (priority order) within ``budget`` — breadth first, then depth.

    First every record gets its one-line form, in priority order, while it fits; then, again in priority
    order, a record is shown whole where the room allows. So on a small budget the agent still sees one
    line for as many records as possible (a rule it cannot see is a rule it cannot follow), and on a large
    one everything is whole. What does not fit even as one line is left to the index.

    ``overhead`` = what the section costs around its lines (heading, separators, closing note); it is
    counted in ``used`` once a line is in. Returns (lines, ids_full, ids_short, left_out, used).
    """
    chosen: List[Tuple[MemoryRecord, str]] = []
    left: List[MemoryRecord] = []
    used = overhead
    for r in records:
        short = _line(r, full=False)
        if used + len(short) + 1 <= budget:
            chosen.append((r, short))
            used += len(short) + 1
        else:
            left.append(r)
    long_limit = max(240, int(budget * _LONG_SHARE))
    lines: List[str] = []
    full_ids: List[str] = []
    short_ids: List[str] = []
    for r, short in chosen:
        full = _line(r, full=True)
        if full == short:
            lines.append(full)
            full_ids.append(r.id)
            continue
        grow = len(full) - len(short)
        if len(full) <= long_limit and used + grow <= budget:
            lines.append(full)
            full_ids.append(r.id)
            used += grow
        else:
            lines.append(short)
            short_ids.append(r.id)
    return lines, full_ids, short_ids, left, (used if lines else 0)


def _allocate(
    budget: int,
    legacy_used: int,
    pools: Mapping[str, Sequence[MemoryRecord]],
    overheads: Dict[str, int],
) -> Dict[str, int]:
    """Room per section: its share, but a section that needs less gives the rest to the others.

    Need = every record of the section whole. Room a section cannot use goes to the sections that
    still need more, rules first (they are the user's instructions), then state, then related.
    Unmigrated pinned notes (``legacy_used``) come out of the rules' share first.
    """
    avail = max(0, budget - legacy_used)
    need = {
        k: (overheads[k] + sum(_line_estimate(r) for r in pool)) if pool else 0
        for k, pool in pools.items()
    }
    share = {k: int(budget * _SHARES[k]) for k in pools}
    share["rules"] = max(0, share["rules"] - legacy_used)
    room = {k: min(need[k], share[k]) for k in pools}
    over = sum(room.values()) - avail
    for k in ("related", "state", "rules"):  # old pinned notes took more than the rules' share
        if over <= 0:
            break
        cut = min(over, room[k])
        room[k] -= cut
        over -= cut
    left = max(0, avail - sum(room.values()))
    for k in ("rules", "state", "related"):
        extra = min(left, need[k] - room[k])
        room[k] += extra
        left -= extra
    return room


def _priority(r: MemoryRecord) -> Tuple[Any, ...]:
    return (
        0 if r.status == "contested" else 1,
        importance_rank(r.importance),
        -authority_rank(r.authority),
        -r.confirm_count,
        r.id,
    )


def _stable(records: Sequence[MemoryRecord]) -> List[MemoryRecord]:
    return sorted(records, key=lambda r: (importance_rank(r.importance), r.kind, r.id))


def projects_in_play(records: Sequence[MemoryRecord], context: str) -> List[str]:
    projects: Dict[str, str] = {}
    for r in records:
        if r.project:
            projects.setdefault(normalize_name(r.project), r.project)
    if not projects:
        return []
    hay = normalize_name(context)
    named = [p for n, p in projects.items() if n and n in hay]
    # A project is also named when one of its records' subjects is (the user says the app, not the project).
    if not named:
        for r in records:
            if r.project and any(len(n) >= 3 and n in hay for n in r.names()):
                if r.project not in named:
                    named.append(r.project)
    if named:
        return named[:_PROJECTS_IN_PLAY]
    latest: Dict[str, str] = {}
    for r in records:
        if r.project:
            seen = r.last_seen()
            if seen > latest.get(r.project, ""):
                latest[r.project] = seen
    return sorted(latest, key=lambda p: latest[p], reverse=True)[:1]


def _filename_of(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    name = str(meta.get("filename") or getattr(chunk, "key", "") or "")
    return name.split("#", 1)[0]


async def _related(
    store: RecordStore,
    provider: Any,
    query: str,
    exclude: set,
) -> List[MemoryRecord]:
    """Facts and procedures that relate to ``query``: notes keyword search + vector, record notes only."""
    names: List[str] = []
    try:
        hits = await provider.notes().search(query, limit=_RELATED_CANDIDATES)
        names += [_filename_of(h) for h in hits or []]
    except Exception:  # noqa: BLE001
        logger.debug("records: keyword search failed", exc_info=True)
    vector = None
    try:
        vector = provider.vector()
    except Exception:  # noqa: BLE001
        vector = None
    if vector is not None:
        try:
            hits = await asyncio.wait_for(
                vector.search(query, top_k=_RELATED_CANDIDATES), timeout=8.0
            )
            names += [_filename_of(h) for h in hits or []]
        except Exception:  # noqa: BLE001
            logger.debug("records: vector search failed", exc_info=True)
    out: List[MemoryRecord] = []
    seen = set()
    for name in names:
        if not name.startswith("mem-") or name in seen:
            continue
        seen.add(name)
        rid = name[len("mem-") : -len(".md")] if name.endswith(".md") else name[len("mem-") :]
        if rid in exclude:
            continue
        rec = await store.get(rid)
        if rec is not None and rec.live and rec.kind in ("fact", "procedure", "state"):
            out.append(rec)
    if not out:
        return out
    scorer = Scorer([f"{r.subject} {' '.join(r.aliases)} {r.key} {r.text}" for r in out])
    scores = dict(scorer.rank(query))
    # Keep what the lexical check also supports, or what both searches returned first.
    keep = [r for i, r in enumerate(out) if scores.get(i, 0.0) >= _RELATED_MIN_SCORE or i < 3]
    keep.sort(key=lambda r: -scores.get(out.index(r), 0.0))
    return keep


async def _critical_facts(store: RecordStore, provider: Any) -> List[MemoryRecord]:
    """Agent-wide critical facts (who the user is, how to address them …) — always, whatever the turn."""
    from xgen_agent_runtime.memory.provider import Importance

    try:
        metas = await provider.notes().list(category="facts", importance=Importance.CRITICAL)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for meta in metas:
        name = meta.ref.filename
        if not name.startswith("mem-"):
            continue
        rec = await store.get(name[len("mem-") : -len(".md")])
        if rec is not None and rec.live and rec.scope == "agent":
            out.append(rec)
    return out


async def _legacy_pinned(provider: Any, budget: int) -> Tuple[str, List[str]]:
    """Old pinned notes the records pass has not migrated yet (the first turns after an upgrade, or a
    note pinned by an older path) — shown as before, so nothing pinned goes missing in between."""
    from xgen_agent_runtime.memory.pinned import LEDGER_FILENAME, render_pinned
    from xgen_agent_runtime.memory.records.migrate import MIGRATED_TAG, PINNED_CATEGORY

    if budget <= 0:
        return "", []
    notes = provider.notes()
    try:
        metas = await notes.list(category=PINNED_CATEGORY)
    except Exception:  # noqa: BLE001
        return "", []
    metas = [m for m in metas if MIGRATED_TAG not in (getattr(m, "tags", None) or [])]
    if not metas:
        return "", []
    metas.sort(
        key=lambda m: (
            m.ref.filename != LEDGER_FILENAME,
            importance_rank(getattr(getattr(m, "importance", None), "value", "medium")),
        )
    )
    rows = []
    for m in metas:
        try:
            note = await notes.read(m.ref.filename)
        except Exception:  # noqa: BLE001
            note = None
        if note is not None:
            rows.append((note.ref.filename, note.title, note.body))
    return render_pinned(rows, budget), [r[0] for r in rows]


async def assemble(
    provider: Any,
    *,
    query: str,
    context: str = "",
    budget_chars: int,
    include: Sequence[str] = ("rules", "state", "related"),
) -> Assembly:
    """The memory block for one turn. ``context`` = recent turns (they decide which projects are in play).

    ``include`` limits what is assembled (a host's first-tier memory-map turn keeps only the rules).
    """
    if budget_chars <= 0:
        return Assembly()
    total_budget = budget_chars
    # The index closes the block and always fits: keep its room out of the sections' share.
    budget_chars = max(0, budget_chars - min(_INDEX_RESERVE, budget_chars // 4))
    store = RecordStore(provider)
    live = await store.load(kinds=("rule", "state", "decision"))
    # Counts for the index: the background pass keeps them in its state (listing every record each turn
    # is the most expensive part of a large memory's turn); a store without that state counts now.
    counts = (await store.load_state()).get("counts") or await store.counts()
    ctx = f"{query}\n{context}"
    in_play = projects_in_play(live, ctx)
    in_play_norm = {normalize_name(p) for p in in_play}

    def _in_scope(r: MemoryRecord) -> bool:
        return r.scope == "agent" or normalize_name(r.project) in in_play_norm

    report: Dict[str, Any] = {"budget": total_budget, "projects": in_play}
    sections: List[str] = []
    spare = 0

    # ── 0. old pinned notes not migrated yet (shown whole, as before; they share the rules' room) ──
    legacy_text, legacy_names = await _legacy_pinned(provider, int(budget_chars * _LEGACY_SHARE))
    if legacy_text:
        sections_out_legacy = "## Memory — pinned notes (being moved into records)\n" + legacy_text
    else:
        sections_out_legacy = ""
    report["legacy"] = legacy_names
    legacy_used = len(sections_out_legacy)

    # ── gather what each section could show ─────────────────────────
    rules = [r for r in live if r.kind == "rule" and _in_scope(r)] if "rules" in include else []

    def _applies(r: MemoryRecord) -> bool:
        return not r.applies_when or any(coverage(ctx, p) >= 0.5 for p in r.applies_when)

    rules.sort(key=lambda r: (0 if _applies(r) else 1, _priority(r)))
    state = (
        [
            r
            for r in live
            if r.kind in ("state", "decision")
            and r.status in ("active", "contested")
            and _in_scope(r)
        ]
        if "state" in include
        else []
    )
    state.sort(key=lambda r: (0 if r.kind == "decision" else 1, _priority(r)))
    # Records of the two sections above are shown there or counted in the index — not again here.
    placed = {r.id for r in rules} | {r.id for r in state}
    related: List[MemoryRecord] = []
    if "related" in include:
        for r in await _critical_facts(store, provider):
            if r.id not in placed:
                related.append(r)
        query_text = query.strip() or context[-500:]
        if query_text:
            related += await _related(
                store, provider, query_text[:500], placed | {x.id for x in related}
            )

    rules_head = "## Memory — rules in force (the user's standing instructions; follow them)"
    contest_note = "- Contested lines above contradict each other; do not pick one yourself — ask the user which holds."
    title = ", ".join(in_play) if in_play else "everywhere"
    state_head = f"## Memory — current state ({title}; dates = when last confirmed, check before relying on old ones)"
    related_head = "## Memory — related to this turn"
    overheads = {
        "rules": len(rules_head)
        + 3
        + (len(contest_note) + 1 if any(r.status == "contested" for r in rules) else 0),
        "state": len(state_head) + 3,
        "related": len(related_head) + 3,
    }
    pools = {"rules": rules, "state": state, "related": related}
    rooms = _allocate(budget_chars, legacy_used, pools, overheads)
    if sections_out_legacy:
        sections.append(sections_out_legacy)

    # ── 1. rules ─────────────────────────────────────────────────────
    lines, full_ids, short_ids, left, used = _fill(rules, rooms["rules"], overheads["rules"])
    if lines:
        order = {rid: i for i, rid in enumerate(r.id for r in _stable(rules))}
        lines.sort(key=lambda ln: order.get(ln[3 : ln.index("]")], 0))
        if any(r.status == "contested" and (r.id in full_ids or r.id in short_ids) for r in rules):
            lines.append(contest_note)
        sections.append(rules_head + "\n" + "\n".join(lines))
    report.update(rules=full_ids, rules_short=short_ids, rules_left=[r.id for r in left])
    spare = max(0, rooms["rules"] - used)

    # ── 2. state of the projects in play ────────────────────────────
    lines, full_ids, short_ids, left, used = _fill(
        state, rooms["state"] + spare, overheads["state"]
    )
    if lines:
        order = {rid: i for i, rid in enumerate(r.id for r in _stable(state))}
        lines.sort(key=lambda ln: order.get(ln[3 : ln.index("]")], 0))
        sections.append(state_head + "\n" + "\n".join(lines))
    report.update(state=full_ids, state_short=short_ids, state_left=[r.id for r in left])
    spare = max(0, rooms["state"] + spare - used)

    # ── 3. related facts / procedures ───────────────────────────────
    budget = rooms["related"] + spare
    lines, full_ids, short_ids, left, used = _fill(related, budget, overheads["related"])
    related_text = ""
    if lines:
        if left:
            more = ", ".join(r.id for r in left[:8]) + ("…" if len(left) > 8 else "")
            line = f"- {len(left)} more related not shown for room: {more}."
            if used + len(line) + 1 <= budget:
                lines.append(line)
        related_text = related_head + "\n" + "\n".join(lines)
    report.update(related=full_ids, related_short=short_ids, related_left=[r.id for r in left])

    # ── 4. index (closes the standing part) ─────────────────────────
    if not sections and not related_text and not any(sum(b.values()) for b in counts.values()):
        report["chars"] = 0
        return Assembly(report=report)  # nothing remembered yet: no block at all
    body = "\n\n".join(s for s in sections if s)
    room = max(
        total_budget - len(body) - len(related_text) - 2, min(_INDEX_RESERVE, total_budget // 4)
    )
    out = Assembly(
        standing=_join(body, _index(counts, report, live, in_play, limit=room)),
        related=related_text,
        report=report,
    )
    # The budget is a hard ceiling. Sections are filled inside it; what can still push the total over
    # is the index on a very small budget — then a one-line index, then none, then related goes.
    if len(out.text) > total_budget:
        out.standing = _join(body, _SHORT_INDEX)
    if len(out.text) > total_budget:
        out.standing = body
    if len(out.text) > total_budget:
        report.update(
            related_left=report["related"] + report["related_short"] + report["related_left"],
            related=[],
            related_short=[],
        )
        out.related = ""
    report["chars"] = len(out.text)
    return out


_SHORT_INDEX = "## Memory index\nMore is remembered: memory_record(action='list') / memory_search."


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def _index(
    counts: Dict[str, Dict[str, int]],
    report: Dict[str, Any],
    live: Sequence[MemoryRecord],
    in_play: Sequence[str],
    *,
    limit: int = 10_000,
) -> str:
    """What is remembered beyond this block, fitted to ``limit`` (ids first go, then other projects)."""

    def n(kind: str, *statuses: str) -> int:
        b = counts.get(kind, {})
        return sum(b.get(s, 0) for s in statuses) if statuses else sum(b.values())

    parts = [
        f"rules {n('rule', 'active', 'contested')}",
        f"facts {n('fact', 'active', 'contested')}",
        f"state {n('state', 'active', 'contested')}",
        f"open decisions {n('decision', 'active', 'contested')}",
        f"procedures {n('procedure', 'active', 'contested')}",
    ]
    contested = sum(b.get("contested", 0) for b in counts.values())
    if contested:
        parts.append(f"contested {contested}")
    not_shown: List[str] = []
    for key, label in (("rules_left", "rules"), ("state_left", "state lines")):
        if report.get(key):
            not_shown.append(
                f"{len(report[key])} more {label} ({', '.join(report[key][:8])}{'…' if len(report[key]) > 8 else ''})"
            )
    others = sorted({r.project for r in live if r.project and r.project not in in_play})
    counted = [
        f"{len(report[key])} more {label}"
        for key, label in (("rules_left", "rules"), ("state_left", "state lines"))
        if report.get(key)
    ]
    tail = (
        'Open any record with memory_read("mem-<id>.md"); find more with memory_search. '
        "Correct or retire a record with memory_record when the user says it changed."
    )
    head = ["## Memory index", "Remembered: " + " · ".join(parts) + "."]
    for shown_ids, projects in (
        (not_shown, others[:12]),
        (counted, others[:12]),
        (counted, []),
        ([], []),
    ):
        lines = list(head)
        if shown_ids:
            lines.append("Not shown for room: " + "; ".join(shown_ids) + ".")
        if projects:
            lines.append("Other projects with remembered state: " + ", ".join(projects) + ".")
        lines.append(tail)
        text = "\n".join(lines)
        if len(text) <= limit:
            return text
    return _SHORT_INDEX
