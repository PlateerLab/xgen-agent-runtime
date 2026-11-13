"""기록 모드의 메모리 도구 — ``memory_pin`` (기록으로 고정) 과 ``memory_record`` (기록 조회 · 정정).

기록(memory/records)은 자동 증류가 대화에서 만든다. 여기 도구는 에이전트가 직접 손대는 길이다:

* ``memory_pin`` — 늘 알아야 할 것을 기록 하나로 남긴다. 같은 것(종류 + 대상 + 키)이 이미 있으면 새로
  만들지 않고 새 버전으로 고친다. 누가 말한 것인지는 코드가 정한다: 이번 대화의 마지막 사용자 말에
  고정할 내용이 들어 있으면 사용자의 것, 아니면 에이전트의 판단(``agent_inference``)이다. 에이전트의
  판단은 사용자의 기록을 덮지 못한다(증류가 사용자의 말을 찾으면 그때 사용자의 것으로 올라간다).
* ``memory_record`` — 목록 · 한 건 · 이력을 보고, 확인 · 정정 · 철회 · 결정 완료 · 되돌리기 · 충돌 정리를
  한다. 모든 변경은 새 버전이고 앞 버전은 이력에 남는다(되돌릴 수 있다). 사용자가 정한 기록(권위
  ``user_*``)을 바꾸거나 충돌을 정리하는 것은 사용자가 그렇게 하라고 했을 때만이다(``user_asked``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.memory.records import (
    KINDS,
    Evidence,
    MemoryRecord,
    RecordStore,
    authority_rank,
    now_iso,
)
from xgen_agent_runtime.memory.records.match import Scorer, coverage
from xgen_agent_runtime.memory.records.model import as_int

__all__ = ["latest_user_text", "memory_record_action", "pin_record"]

#: 고정할 내용(제목 말고 본문)의 낱말 중 이만큼이 마지막 사용자 말에 있으면 사용자의 것으로 본다.
_USER_COVERAGE = 0.6
_LIST_LIMIT = 30
_ACTIONS = ("list", "get", "history", "confirm", "correct", "retire", "resolve", "revert", "settle")


def _ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def latest_user_text(state: Any) -> str:
    """이번 대화의 마지막 사용자 말(도구 결과 · 런타임 알림 제외)."""
    from xgen_agent_runtime.core.context_prune import is_runtime_note_content

    for msg in reversed(list(getattr(state, "messages", None) or [])):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(b.get("text") or "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = ""
        if text.strip() and not is_runtime_note_content(text):
            return text
    return ""


def _session(ctx: Any) -> str:
    return str(
        getattr(ctx, "session_id", "")
        or getattr(getattr(ctx, "state_view", None), "session_id", "")
        or ""
    )


def _brief(r: MemoryRecord) -> Dict[str, Any]:
    return {
        "id": r.id,
        "kind": r.kind,
        "subject": r.subject,
        **({"key": r.key} if r.key else {}),
        **({"project": r.project} if r.project else {}),
        "status": r.status,
        "authority": r.authority,
        "importance": r.importance,
        "version": r.version,
        "text": r.headline if r.headline and len(r.text) > 300 else r.text,
        "last_seen": r.last_seen(),
        **({"contested_with": r.contested_with} if r.contested_with else {}),
    }


def _full(r: MemoryRecord) -> Dict[str, Any]:
    d = r.to_dict()
    d["filename"] = r.filename
    return d


def _is_user(r: MemoryRecord) -> bool:
    return r.authority.startswith("user_")


async def pin_record(provider: Any, tool_input: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """``memory_pin`` 의 기록 모드. 반환은 도구 결과 dict(오류면 ``{"error": ...}``)."""
    title = str(tool_input.get("title") or "").strip()
    content = str(tool_input.get("content") or "").strip()
    if not title or not content:
        return {"error": "title and content are required"}
    kind = str(tool_input.get("kind") or "rule").strip().lower()
    if kind not in KINDS:
        return {"error": f"kind must be one of {', '.join(KINDS)}"}
    key = str(tool_input.get("key") or "").strip()
    project = str(tool_input.get("project") or "").strip()

    state = getattr(ctx, "state_view", None)
    said = latest_user_text(state)
    by_user = bool(said) and coverage(said, content) >= _USER_COVERAGE
    if by_user:
        authority = "user_directive" if kind == "rule" else "user_statement"
    else:
        authority = "agent_inference"
    ev = Evidence(
        session=_session(ctx),
        at=now_iso(),
        role="user" if by_user else "assistant",
        quote=(said if by_user else content)[:300],
    )

    store = RecordStore(provider)
    probe = MemoryRecord(id="", kind=kind, subject=title, key=key, text=content)
    twin: Optional[MemoryRecord] = None
    for r in await store.load(kinds=[kind]):
        if r.identity_key() == probe.identity_key():
            twin = r
            break
    if twin is not None:
        if (
            authority_rank(authority) < authority_rank(twin.authority)
            and twin.text.strip() != content
        ):
            return {
                "status": "kept",
                "id": twin.id,
                "note": (
                    f"{twin.id} was set by {twin.authority}; your own inference does not replace it. "
                    "If the user asked for this change, say so in your reply and pin again with their words."
                ),
                "record": _brief(twin),
            }
        same = twin.text.strip() == content
        if not same:
            twin.text = content
            twin.headline = ""
        if authority_rank(authority) > authority_rank(twin.authority):
            twin.authority = authority
        twin.importance = "critical" if twin.importance == "critical" else "high"
        twin.confirm_count += 1
        twin.confirmed_at = ev.at
        twin.add_evidence(ev)
        twin.change_reason = "pinned again" if same else "pinned with a new text"
        ok = await store.put(twin, expected_version=twin.version, reason=twin.change_reason)
        if not ok:
            return {
                "error": f"{twin.id} changed meanwhile; read it with memory_record and try again"
            }
        return {"status": "confirmed" if same else "updated", "id": twin.id, "record": _brief(twin)}

    rec = MemoryRecord(
        id="",
        kind=kind,
        subject=title,
        key=key,
        text=content,
        project=project,
        scope=f"project:{project}" if project else "agent",
        authority=authority,
        importance="high",
        confidence=0.9 if by_user else 0.6,
        valid_from=ev.at,
        confirmed_at=ev.at,
        origin="memory_pin",
    )
    rec.add_evidence(ev)
    rec.id = await store.new_id(rec)
    if not await store.put(rec, expected_version=None, reason="pinned"):
        return {"error": "could not write the record"}
    return {"status": "pinned", "id": rec.id, "filename": rec.filename, "record": _brief(rec)}


async def memory_record_action(
    provider: Any, tool_input: Dict[str, Any], ctx: Any
) -> Dict[str, Any]:
    """``memory_record`` 도구 본체."""
    action = str(tool_input.get("action") or "list").strip().lower()
    if action not in _ACTIONS:
        return {"error": f"action must be one of {', '.join(_ACTIONS)}"}
    store = RecordStore(provider)
    if action == "list":
        return await _list(store, tool_input)

    rid = str(tool_input.get("id") or "").strip()
    if rid.startswith("mem-") and rid.endswith(".md"):
        rid = rid[4:-3]
    if not rid:
        return {"error": "id is required"}
    rec = await store.get(rid)
    if rec is None:
        return {"error": f"no record {rid!r} (memory_record(action='list') shows ids)"}
    if action == "get":
        return {"record": _full(rec), "previous_versions": max(0, rec.version - 1)}
    if action == "history":
        versions = await store.history(rid)
        return {
            "id": rid,
            "current_version": rec.version,
            "versions": [
                {
                    "version": v.version,
                    "updated_at": v.updated_at,
                    "status": v.status,
                    "authority": v.authority,
                    "text": v.text,
                    "change_reason": v.change_reason,
                }
                for v in versions
            ]
            + [
                {
                    "version": rec.version,
                    "updated_at": rec.updated_at,
                    "status": rec.status,
                    "authority": rec.authority,
                    "text": rec.text,
                    "change_reason": rec.change_reason,
                    "current": True,
                }
            ],
        }

    reason = str(tool_input.get("reason") or "").strip()
    if not reason:
        return {"error": "reason is required for a change (it is kept in the record's history)"}
    user_asked = bool(tool_input.get("user_asked"))
    if action != "confirm" and _is_user(rec) and not user_asked:
        return {
            "error": (
                f"{rid} was set by the user ({rec.authority}); change it only when the user asked "
                "you to (user_asked=true)"
            )
        }
    ev = Evidence(
        session=_session(ctx),
        at=now_iso(),
        role="user" if user_asked else "assistant",
        quote=reason[:300],
    )

    if action == "settle":
        return await _settle(store, rec, reason, ev, user_asked)
    if action == "revert":
        try:
            version = as_int(tool_input.get("version"))
        except (TypeError, ValueError):
            return {"error": "version (an earlier version number) is required"}
        restored = await store.revert(rid, version, reason=reason)
        if restored is None:
            return {"error": f"{rid} has no version {version}, or it changed meanwhile"}
        return {"status": "reverted", "record": _brief(restored)}

    if action == "confirm":
        rec.confirm_count += 1
        rec.confirmed_at = ev.at
    elif action == "correct":
        text = str(tool_input.get("text") or "").strip()
        if not text:
            return {"error": "text (the corrected record) is required"}
        rec.text = text
        rec.headline = str(tool_input.get("headline") or "").strip()
        rec.confirmed_at = ev.at
        if rec.status == "contested" and not rec.contested_with:
            rec.status = "active"
    elif action == "retire":
        rec.status = "retired"
    elif action == "resolve":
        if rec.kind != "decision":
            return {"error": "only a decision is resolved; retire other kinds"}
        rec.status = "resolved"
        rec.resolved_at = ev.at
    rec.add_evidence(ev)
    rec.change_reason = reason
    if not await store.put(rec, expected_version=rec.version, reason=reason):
        return {"error": f"{rid} changed meanwhile; get it again and retry"}
    return {
        "status": {"confirm": "confirmed", "correct": "corrected"}.get(action, action + "d"),
        "record": _brief(rec),
    }


async def _list(store: RecordStore, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(tool_input.get("kind") or "").strip().lower() or None
    status = str(tool_input.get("status") or "live").strip().lower()
    project = str(tool_input.get("project") or "").strip().lower()
    query = str(tool_input.get("query") or "").strip()
    if kind and kind not in KINDS:
        return {"error": f"kind must be one of {', '.join(KINDS)}"}
    records = await store.load(kinds=[kind] if kind else None, live_only=status == "live")
    if status not in ("live", "all"):
        records = [r for r in records if r.status == status]
    if project:
        records = [r for r in records if project in (r.project or "").lower()]
    if query:
        scorer = Scorer([f"{r.subject} {' '.join(r.aliases)} {r.key} {r.text}" for r in records])
        records = [records[i] for i, score in scorer.rank(query) if score > 0]
    else:
        records.sort(key=lambda r: r.last_seen(), reverse=True)
    total = len(records)
    return {
        "total": total,
        "filters": {
            "kind": kind,
            "status": status,
            "project": project or None,
            "query": query or None,
        },
        "records": [_brief(r) for r in records[:_LIST_LIMIT]],
        **(
            {"more": f"{total - _LIST_LIMIT} more — narrow with kind/project/query"}
            if total > _LIST_LIMIT
            else {}
        ),
    }


async def _settle(
    store: RecordStore, keep: MemoryRecord, reason: str, ev: Evidence, user_asked: bool
) -> Dict[str, Any]:
    """충돌 정리: ``keep`` 을 남기고 맞서던 기록은 ``keep`` 으로 대체한다. 사용자가 정한다."""
    if keep.status != "contested" and not keep.contested_with:
        return {"error": f"{keep.id} is not contested"}
    if not user_asked:
        return {
            "error": "a contest is settled by the user: ask them which one holds, then settle with user_asked=true"
        }
    others: List[str] = []
    for oid in list(keep.contested_with):
        other = await store.get(oid)
        if other is None or not other.live:
            continue
        other.status = "superseded"
        other.superseded_by = keep.id
        other.contested_with = [x for x in other.contested_with if x != keep.id]
        other.change_reason = f"settled in favour of {keep.id}: {reason}"
        if await store.put(other, expected_version=other.version, reason=other.change_reason):
            others.append(oid)
    keep.status = "active"
    keep.contested_with = []
    keep.supersedes = list(dict.fromkeys(keep.supersedes + others))
    keep.authority = "user_directive" if keep.kind == "rule" else "user_statement"
    keep.confirmed_at = ev.at
    keep.confirm_count += 1
    keep.add_evidence(ev)
    keep.change_reason = reason
    if not await store.put(keep, expected_version=keep.version, reason=reason):
        return {"error": f"{keep.id} changed meanwhile; get it again and retry"}
    return {"status": "settled", "kept": _brief(keep), "superseded": others}
