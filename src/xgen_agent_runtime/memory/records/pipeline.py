"""The records pass run after each turn (host distillation): migrate → extract → reconcile → consolidate.

Per-conversation cursors: the short-term store is per conversation, so "where extraction got to" is too.
One agent-wide cursor (the old ledger's) skipped turns: a conversation whose turns were older than the
newest turn already processed elsewhere was never extracted.

Failure policy: a failed model call writes nothing and leaves the cursor, so the same turns are tried
again next pass; after ``MAX_FAILURES`` consecutive failures on the same turns they are passed over with a
logged warning (a turn the model cannot handle must not block every later one).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.memory.records.consolidate import consolidate
from xgen_agent_runtime.memory.records.extract import (
    CANDIDATE_SCHEMA,
    IndexedTurn,
    batch_turns,
    build_catalog,
    build_extraction_instruction,
    parse_candidates,
)
from xgen_agent_runtime.memory.records.migrate import migrate_legacy
from xgen_agent_runtime.memory.records.model import now_iso
from xgen_agent_runtime.memory.records.reconcile import (
    CompleteStructured,
    Reconciler,
    ReconcileReport,
)
from xgen_agent_runtime.memory.records.store import RecordStore

logger = logging.getLogger(__name__)

__all__ = ["RecordsPassReport", "RecordsPipeline", "turns_for_extraction"]

STM_WINDOW = 400
MAX_BATCHES_PER_PASS = 3
MAX_FAILURES = 3
#: Consolidate every N passes, or sooner when this many records changed since the last run.
CONSOLIDATE_EVERY = 8
CONSOLIDATE_AFTER_CHANGES = 25
#: Tool results are memory evidence, not memory: an excerpt is enough to notice a state change.
TOOL_EXCERPT = 1500


@dataclass
class RecordsPassReport:
    ran: bool = False
    migrated: Dict[str, Any] = field(default_factory=dict)
    turns: int = 0
    candidates: int = 0
    reconcile: Dict[str, int] = field(default_factory=dict)
    consolidation: Dict[str, Any] = field(default_factory=dict)
    skipped_reason: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def changes(self) -> int:
        return sum(
            int(v)
            for k, v in self.reconcile.items()
            if k in ("added", "confirmed", "updated", "resolved", "contested")
        )


def _turn_iso(turn: Any) -> str:
    ts = getattr(turn, "timestamp", None)
    try:
        return ts.isoformat() if ts is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def turns_for_extraction(turns: List[Any]) -> List[IndexedTurn]:
    """STM turns → indexed turns: user/assistant text, tool results as short excerpts, runtime notes marked."""
    from xgen_agent_runtime.core.context_prune import is_runtime_note_content

    out: List[IndexedTurn] = []
    for t in turns:
        role = str(getattr(t, "role", "") or "user")
        content = getattr(t, "content", "") or ""
        at = _turn_iso(t)
        texts: List[str] = []
        tool_texts: List[str] = []
        if isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and str(b.get("text") or "").strip():
                    texts.append(str(b["text"]))
                elif b.get("type") == "tool_result":
                    raw = b.get("content")
                    if isinstance(raw, list):
                        raw = "\n".join(str(x.get("text", "")) for x in raw if isinstance(x, dict))
                    raw = str(raw or "").strip()
                    if raw:
                        tool_texts.append(
                            raw if len(raw) <= TOOL_EXCERPT else raw[:TOOL_EXCERPT] + " …[excerpt]"
                        )
        else:
            texts.append(str(content))
        text = "\n".join(x for x in texts if x.strip()).strip()
        if text:
            r = role if role in ("user", "assistant", "tool") else "assistant"
            if r == "user" and is_runtime_note_content(text):
                r = "runtime"
            out.append(IndexedTurn(len(out) + 1, r, text, at))
        for tt in tool_texts:
            out.append(IndexedTurn(len(out) + 1, "tool", tt, at))
    return out


class RecordsPipeline:
    def __init__(
        self,
        provider: Any,
        *,
        complete_structured: Optional[CompleteStructured],
        session_id: str,
    ) -> None:
        self._provider = provider
        self._complete = complete_structured
        self._session = session_id or "default"
        self._store = RecordStore(provider)

    @property
    def store(self) -> RecordStore:
        return self._store

    async def run(self) -> RecordsPassReport:
        report = RecordsPassReport()
        state = await self._store.load_state()
        state.setdefault("cursors", {})
        state.setdefault("failures", {})

        # ① the pre-records memory, once (and anything that lands in the old pinned category later)
        # A large old memory moves over across passes; ``done`` is where the last pass got to.
        raw_prev = state.get("migration")
        prev: Dict[str, Any] = dict(raw_prev) if isinstance(raw_prev, dict) else {}
        done = {str(k) for k in prev.get("done_keys") or []}
        try:
            mig = await migrate_legacy(
                self._provider, self._store, complete_structured=self._complete, done=done
            )
            if mig.get("notes"):
                report.migrated = mig
                prev = {
                    **{k: v for k, v in mig.items() if k != "done_keys"},
                    "first_at": prev.get("first_at") or mig.get("at"),
                    "total_records": int(prev.get("total_records") or 0)
                    + int(mig.get("records") or 0),
                }
        except Exception:  # noqa: BLE001
            logger.warning("records: migration failed", exc_info=True)
        prev["done_keys"] = sorted(done)
        prev.setdefault("at", now_iso())
        state["migration"] = prev
        # Saved now: a pass cut short later (timeout) must not redo what moved over.
        await self._store.save_state(state)

        # ② new turns of this conversation
        try:
            raw = await self._provider.stm().recent(n=STM_WINDOW)
        except Exception:  # noqa: BLE001
            raw = []
        cursor = str(state["cursors"].get(self._session, ""))
        fresh_raw = [t for t in raw or [] if not cursor or _turn_iso(t) > cursor]
        turns = turns_for_extraction(fresh_raw)
        report.turns = len(turns)
        totals = ReconcileReport()
        if self._complete is None:
            report.skipped_reason = "no_model"
        elif not turns:
            report.skipped_reason = "no_new_turns"
        else:
            live = await self._store.load()
            remaining = turns
            for _ in range(MAX_BATCHES_PER_PASS):
                if not remaining:
                    break
                batch, _more = batch_turns(remaining)
                # The cursor is a time: never split turns that share one (a message and its tool results).
                while (
                    len(batch) < len(remaining)
                    and batch[-1].at
                    and remaining[len(batch)].at == batch[-1].at
                ):
                    batch = remaining[: len(batch) + 1]
                last_at = max((t.at for t in batch if t.at), default="")
                ok = await self._extract_batch(batch, live, totals, report)
                if ok:
                    state["failures"].pop(self._session, None)
                    if last_at:
                        state["cursors"][self._session] = last_at
                    remaining = remaining[len(batch) :]
                    await self._store.save_state(state)  # each batch is final once written
                    continue
                n = int(state["failures"].get(self._session, 0)) + 1
                state["failures"][self._session] = n
                if n >= MAX_FAILURES and last_at:
                    logger.warning(
                        "records: passing over %d turns after %d failed passes (session=%s)",
                        len(batch),
                        n,
                        self._session,
                    )
                    report.notes.append(f"passed over {len(batch)} turns after {n} failures")
                    state["cursors"][self._session] = last_at
                    state["failures"].pop(self._session, None)
                break
            report.ran = True
        report.reconcile = {
            k: getattr(totals, k)
            for k in (
                "added",
                "confirmed",
                "updated",
                "resolved",
                "contested",
                "skipped",
                "refused",
                "conflicts_on_write",
            )
        }

        # ③ consolidation, bounded and periodic
        passes = int(state.get("passes", 0)) + 1
        state["passes"] = passes
        changed = int(state.get("changes_since_consolidation", 0)) + report.changes
        state["changes_since_consolidation"] = changed
        due = (
            passes - int(state.get("last_consolidation_pass", 0)) >= CONSOLIDATE_EVERY
            or changed >= CONSOLIDATE_AFTER_CHANGES
        )
        if due and self._complete is not None:
            rep = await consolidate(
                self._store,
                complete_structured=self._complete,
                rotation=passes,
                since=str(state.get("last_consolidation_at") or ""),
            )
            if rep is not None:
                report.consolidation = {
                    k: getattr(rep, k)
                    for k in (
                        "clusters",
                        "merged",
                        "retired",
                        "resolved",
                        "contested",
                        "headlines",
                        "refused",
                    )
                }
                state["last_consolidation_pass"] = passes
                state["last_consolidation_at"] = now_iso()
                state["changes_since_consolidation"] = 0
        try:
            state["counts"] = await self._store.counts()  # the per-turn index reads these
        except Exception:  # noqa: BLE001
            logger.debug("records: counting failed", exc_info=True)
        state["last_pass"] = {
            "at": now_iso(),
            "session": self._session,
            "reconcile": report.reconcile,
            "turns": report.turns,
        }
        await self._store.save_state(state)
        return report

    async def _extract_batch(
        self,
        batch: List[IndexedTurn],
        live: List[Any],
        totals: ReconcileReport,
        report: RecordsPassReport,
    ) -> bool:
        if not any(t.role in ("user", "tool") for t in batch):
            return True  # agent monologue alone holds nothing to remember; the cursor moves on
        catalog = build_catalog(live, "\n".join(t.text for t in batch))
        if self._complete is None:
            return False
        try:
            data = await self._complete(
                build_extraction_instruction(catalog=catalog, turns=batch), CANDIDATE_SCHEMA
            )
        except Exception:  # noqa: BLE001
            logger.warning("records: extraction call failed", exc_info=True)
            return False
        if not isinstance(data, dict):
            return False
        candidates = parse_candidates(data, batch)
        report.candidates += len(candidates)
        if not candidates:
            return True
        rep = await Reconciler(
            self._store, complete_structured=self._complete, session=self._session
        ).run(candidates, live)
        if rep is None:
            return False
        for k in (
            "added",
            "confirmed",
            "updated",
            "resolved",
            "contested",
            "skipped",
            "refused",
            "conflicts_on_write",
        ):
            setattr(totals, k, getattr(totals, k) + getattr(rep, k))
        report.notes.extend(rep.notes[:10])
        return True
