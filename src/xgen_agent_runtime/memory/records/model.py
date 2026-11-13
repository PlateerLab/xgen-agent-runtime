"""Memory records — the unit of durable memory (4.108.0).

Why records, not documents
--------------------------
The fact ledger and the evergreen were single markdown documents rewritten wholesale and injected
whole into every turn. Memory size was prompt size: past a few hundred facts the pinned budget cut
them (or dropped every other pinned note by name), a rewrite could lose anything, and nothing said
where a line came from or whether it was still true.

A record is one small, typed, versioned piece of memory with its provenance. Storage is unbounded;
what reaches the prompt is *selected* per turn (``assemble``). Nothing is overwritten: a change
writes a new version and keeps the previous one in history (``store``), so every change can be
explained and reverted.

Kinds — each has its own update rule and its own reason to be in a prompt:

* ``rule``      standing instruction, prohibition, procedure the user set. Only the user creates
                one; it applies whenever its scope does (not only when it looks "relevant").
* ``fact``      an attribute of something (entity, system, person). Same subject + key → new version.
* ``state``     the current value of something that changes (version in production, counts,
                where a job stands). Same key → latest value; older values are history.
* ``decision``  something decided, or waiting for a decision. Open until resolved — by an event,
                never by the clock.
* ``procedure`` how a task is done here (steps, pipeline). Versioned like a fact.

Episodes (what happened) stay where they were: conversation archives, execution cards, digests.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "KINDS",
    "AUTHORITIES",
    "STATUSES",
    "LIVE_STATUSES",
    "IMPORTANCES",
    "MemoryRecord",
    "Evidence",
    "as_int",
    "authority_rank",
    "category_for",
    "filename_for",
    "normalize_name",
    "now_iso",
    "slugify",
]

KINDS = ("rule", "fact", "state", "decision", "procedure")

#: Who said it, strongest first. A weaker source never silently overrides a stronger one.
AUTHORITIES = (
    "user_directive",  # the user told the agent to do / not do something
    "user_statement",  # the user stated it
    "migrated",  # carried over from the pre-records memory (the user's earlier notes and ledger)
    "tool_result",  # observed in a tool result / file / document the agent read
    "agent_inference",  # the agent concluded it; nobody confirmed
)
_AUTHORITY_RANK = {
    "user_directive": 5,
    "user_statement": 4,
    "migrated": 3,
    "tool_result": 3,
    "agent_inference": 1,
}

STATUSES = (
    "active",  # in force
    "contested",  # conflicts with another live record of equal standing; the user decides
    "superseded",  # replaced by a newer record (``superseded_by``)
    "retired",  # withdrawn (no longer true / no longer wanted)
    "resolved",  # a decision that was settled
)
#: Statuses the agent still has to take into account.
LIVE_STATUSES = ("active", "contested")

IMPORTANCES = ("critical", "high", "medium", "low")
_IMPORTANCE_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

#: Notes category per kind (the memory browser shows them as folders).
_CATEGORIES = {
    "rule": "rules",
    "fact": "facts",
    "state": "state",
    "decision": "decisions",
    "procedure": "procedures",
}
#: Previous versions live here — never searched, never injected, read only to explain or revert.
HISTORY_CATEGORY = "memory-history"
#: Frontmatter key holding the whole record as one JSON scalar (nested frontmatter does not survive
#: every store — the file provider stringified dict rows and emptied the old ledger once).
RECORD_KEY = "memory_record_json"
#: Tag on every record note — lets hosts and searches tell records from free notes.
RECORD_TAG = "memory-record"


def as_int(value: Any) -> int:
    """A model's number (3, 3.0, "3") as an int; ``ValueError`` when it is not one."""
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    return int(str(value).strip())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def authority_rank(authority: str) -> int:
    return _AUTHORITY_RANK.get(authority, 0)


def importance_rank(importance: str) -> int:
    return _IMPORTANCE_RANK.get(importance, 9)


def category_for(kind: str) -> str:
    return _CATEGORIES.get(kind, "facts")


_SLUG_RE = re.compile(r"[^0-9a-z가-힣]+")


def normalize_name(text: str) -> str:
    """Comparison form of a subject / key / alias: NFKC, lower case, punctuation and spaces folded.

    "Ver 1.35.0", "ver1.35.0" and "VER 1·35·0" compare equal; Korean is kept as is.
    """
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    return _SLUG_RE.sub("", t)


def slugify(text: str, *, limit: int = 40) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    t = _SLUG_RE.sub("-", t).strip("-")
    return t[:limit].strip("-") or "x"


def filename_for(record_id: str) -> str:
    return f"mem-{record_id}.md"


@dataclass
class Evidence:
    """Where a record came from: the session, when, whose turn, and a short quote."""

    session: str = ""
    at: str = ""
    role: str = ""
    quote: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "Evidence":
        d = d if isinstance(d, dict) else {}
        return cls(
            session=str(d.get("session") or ""),
            at=str(d.get("at") or ""),
            role=str(d.get("role") or ""),
            quote=str(d.get("quote") or ""),
        )


#: Evidence kept per record: the first sighting and the latest few (provenance, not content).
MAX_EVIDENCE = 6


@dataclass
class MemoryRecord:
    id: str
    kind: str
    subject: str
    text: str
    key: str = ""
    #: One line the agent can act on when there is no room for ``text``. Written by the model with
    #: the record, never cut from ``text``.
    headline: str = ""
    aliases: List[str] = field(default_factory=list)
    #: ``agent`` (everywhere) or ``project:<slug>``.
    scope: str = "agent"
    project: str = ""
    #: Rules only: when the rule applies, in the user's words ("보고서를 만들 때"). Empty = always.
    applies_when: List[str] = field(default_factory=list)
    authority: str = "agent_inference"
    status: str = "active"
    importance: str = "medium"
    confidence: float = 0.8
    #: When it became true (the turn that established it) — not when it was written.
    valid_from: str = ""
    recorded_at: str = ""
    updated_at: str = ""
    #: Last time a turn said it again (a duplicate confirms instead of adding).
    confirmed_at: str = ""
    confirm_count: int = 1
    evidence: List[Evidence] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    superseded_by: str = ""
    contested_with: List[str] = field(default_factory=list)
    resolved_at: str = ""
    #: Why the last change was made (for the history view): "merged duplicates", "user corrected" …
    change_reason: str = ""
    origin: str = "extraction"
    version: int = 1

    # ── derived ──────────────────────────────────────────────────────

    @property
    def live(self) -> bool:
        return self.status in LIVE_STATUSES

    @property
    def filename(self) -> str:
        return filename_for(self.id)

    @property
    def category(self) -> str:
        return category_for(self.kind)

    def names(self) -> List[str]:
        """Normalized subject and aliases — the record's identities for matching."""
        out = [normalize_name(self.subject)] + [normalize_name(a) for a in self.aliases]
        return [n for n in dict.fromkeys(out) if n]

    def identity_key(self) -> str:
        """subject + key — two live records with the same identity key are the same thing."""
        return f"{normalize_name(self.subject)}::{normalize_name(self.key)}"

    def add_evidence(self, ev: Evidence) -> None:
        if not (ev.quote or ev.session or ev.at):
            return
        items = [
            e for e in self.evidence if (e.session, e.at, e.quote) != (ev.session, ev.at, ev.quote)
        ]
        items.append(ev)
        if len(items) > MAX_EVIDENCE:
            items = [items[0]] + items[-(MAX_EVIDENCE - 1) :]
        self.evidence = items

    # ── (de)serialization ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Any) -> Optional["MemoryRecord"]:
        if not isinstance(d, dict):
            return None
        rid = str(d.get("id") or "").strip()
        kind = str(d.get("kind") or "").strip()
        text = str(d.get("text") or "").strip()
        if not rid or kind not in KINDS or not text:
            return None

        def _list(name: str) -> List[str]:
            v = d.get(name)
            return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

        status = str(d.get("status") or "active")
        authority = str(d.get("authority") or "agent_inference")
        importance = str(d.get("importance") or "medium")
        try:
            confidence = float(d.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        try:
            confirm_count = max(1, int(d.get("confirm_count", 1)))
        except (TypeError, ValueError):
            confirm_count = 1
        try:
            version = max(1, int(d.get("version", 1)))
        except (TypeError, ValueError):
            version = 1
        return cls(
            id=rid,
            kind=kind,
            subject=str(d.get("subject") or "").strip(),
            key=str(d.get("key") or "").strip(),
            text=text,
            headline=str(d.get("headline") or "").strip(),
            aliases=_list("aliases"),
            scope=str(d.get("scope") or "agent"),
            project=str(d.get("project") or ""),
            applies_when=_list("applies_when"),
            authority=authority if authority in AUTHORITIES else "agent_inference",
            status=status if status in STATUSES else "active",
            importance=importance if importance in IMPORTANCES else "medium",
            confidence=min(1.0, max(0.0, confidence)),
            valid_from=str(d.get("valid_from") or ""),
            recorded_at=str(d.get("recorded_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
            confirmed_at=str(d.get("confirmed_at") or ""),
            confirm_count=confirm_count,
            evidence=[
                Evidence.from_dict(e) for e in d.get("evidence") or [] if isinstance(e, dict)
            ],
            supersedes=_list("supersedes"),
            superseded_by=str(d.get("superseded_by") or ""),
            contested_with=_list("contested_with"),
            resolved_at=str(d.get("resolved_at") or ""),
            change_reason=str(d.get("change_reason") or ""),
            origin=str(d.get("origin") or "extraction"),
            version=version,
        )

    @classmethod
    def from_json(cls, raw: Any) -> Optional["MemoryRecord"]:
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            return cls.from_dict(json.loads(raw))
        except ValueError:
            return None

    # ── display ─────────────────────────────────────────────────────

    def day(self, iso: str) -> str:
        """``MM-DD`` of an ISO time (dates on the page tell the agent how fresh a line is)."""
        m = re.match(r"\d{4}-(\d{2})-(\d{2})", iso or "")
        return f"{m.group(1)}-{m.group(2)}" if m else ""

    def last_seen(self) -> str:
        return self.confirmed_at or self.updated_at or self.recorded_at
