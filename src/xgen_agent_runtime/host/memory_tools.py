"""agents/geny 호스트 메모리 도구 — provider 에 바인딩된 자가-조회/기록 도구.

geny-executor 는 memory_read/list/search 류 도구를 built-in 으로 싣지 않는다
(Geny 제품이 호스트에서 등록). 여기서 동일 개념의 6개 도구를 **현재 턴의
provider 인스턴스에 클로저로 바인딩**해 만들어 registry 에 core 로 등록한다.
Geny 도구와 달리 session_id 인자가 없다 — 도구 자체가 이 에이전트의 vault
하나에 붙어 있기 때문.

자동 계층(MemoryAwareRetriever 의 Pinned Facts/Relevant Knowledge 주입,
ProviderDrivenStrategy 의 STM 기록)과 별개로, 에이전트가 스스로 기억을
쓰고/찾고/읽는 명시적 표면을 제공한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("editor.geny_bridge.memory_tools")

_IMPORTANCE_VALUES = {"low", "medium", "high", "critical"}
#: memory_pin 이 쓰는 always-inject 카테고리 (executor retriever 의 pinned 로드 대상).
PINNED_CATEGORY = "critical"


def _ok(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def _split_tags(tags: Any) -> List[str]:
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def _summary_dict(s: Any) -> Dict[str, Any]:
    return {
        "filename": getattr(s, "filename", ""),
        "title": getattr(s, "title", ""),
        "category": getattr(s, "category", ""),
        "tags": list(getattr(s, "tags", []) or []),
        "importance": getattr(s, "importance", "medium"),
        "char_count": getattr(s, "char_count", 0),
        "modified": getattr(s, "modified", ""),
        "first_paragraph": getattr(s, "first_paragraph", ""),
    }


#: 없는 노트를 물었을 때 돌려주는 후보 수 — 고르기엔 충분하고 프리픽스를 부풀리지 않는 선.
_MISS_SUGGESTIONS = 5
#: 후보를 찾을 때 훑는 노트 수 상한.
_MISS_SCAN_LIMIT = 200


async def _list_filenames(provider: Any, category: Optional[str]) -> List[str]:
    """금고가 실제로 가진 노트 이름 — 조회 실패는 빈 목록(진단이 본작업을 막지 않는다)."""
    try:
        summaries = await provider.index().list_notes(category=category, limit=_MISS_SCAN_LIMIT)
    except Exception:  # noqa: BLE001 — 힌트를 못 만들어도 오류 자체는 돌려줘야 한다
        logger.debug("memory_read miss: list_notes(category=%r) failed", category, exc_info=True)
        return []
    return [str(getattr(s, "filename", "")) for s in summaries if getattr(s, "filename", "")]


async def _read_miss_payload(provider: Any, filename: str) -> Dict[str, Any]:
    """ "없다" 만 말하지 않고 **금고가 가진 것**을 함께 보여 준다.

    실측 (2026-09-02~20 dev): ``Note not found`` 45건 / 14 대화 = 대화당 3.2회.
    모델이 파일명을 지어내고(``conversations/conn-wf_…__user__깃허브-레포-….md``),
    없다는 말만 듣고 또 지어낸다. 이름은 우리가 만든 규칙이라 모델이 맞힐 수 없다 —
    한 번의 오류에 **실제 이름 몇 개**를 얹으면 추측이 끝난다.

    도메인 단어는 쓰지 않는다. 금고 자신의 색인만 본다.
    """
    import difflib

    payload: Dict[str, Any] = {"error": f"Note not found: {filename}"}

    category = filename.split("/", 1)[0] if "/" in filename else None
    names = await _list_filenames(provider, category)
    scope = category
    if not names and category:
        # 그 카테고리가 통째로 없다 — 금고 전체에서 다시 본다.
        names = await _list_filenames(provider, None)
        scope = None

    if not names:
        payload["available"] = []
        payload["hint"] = (
            "This memory vault has no notes yet. Write one with memory_write before reading."
        )
        return payload

    near = difflib.get_close_matches(filename, names, n=_MISS_SUGGESTIONS, cutoff=0.5)
    if not near:
        near = names[:_MISS_SUGGESTIONS]
    payload["did_you_mean"] = near
    payload["available_count"] = len(names)
    if scope:
        payload["scanned_category"] = scope
    payload["hint"] = (
        "Filenames are assigned by the vault, not chosen by you — do not guess them. "
        "Use memory_search (by content) or memory_list (by category) to get a real "
        "filename, then read that."
    )
    return payload


def build_memory_tools(provider: Any) -> List[Any]:
    """provider 에 바인딩된 executor ``Tool`` 목록을 만든다.

    provider 가 None 이면 빈 리스트. 도구 실행은 전부 async — 파이프라인
    루프(스레드 소유)에서만 provider 를 만진다.
    """
    if provider is None:
        return []

    from xgen_agent_runtime.memory.provider import Importance, NoteDraft, NotePatch  # noqa: F401
    from xgen_agent_runtime.tools import ToolResult, build_tool
    from xgen_agent_runtime.tools.base import ToolCapabilities, with_origin

    def _importance(value: str, default: str = "medium") -> "Importance":
        v = (value or default).strip().lower()
        return Importance(v if v in _IMPORTANCE_VALUES else default)

    async def _write(tool_input: Dict[str, Any], _ctx: Any) -> ToolResult:
        title = str(tool_input.get("title") or "").strip()
        content = str(tool_input.get("content") or "").strip()
        if not title or not content:
            return ToolResult(content=_err("title and content are required"), is_error=True)
        category = str(tool_input.get("category") or "topics").strip() or "topics"
        draft = NoteDraft(
            title=title,
            body=content,
            category=category,
            tags=_split_tags(tool_input.get("tags")),
            importance=_importance(str(tool_input.get("importance") or "medium")),
        )
        meta = await provider.notes().write(draft)
        return ToolResult(
            content=_ok(
                {
                    "status": "created",
                    "filename": meta.ref.filename,
                    "title": meta.title,
                    "category": meta.category,
                    "tags": list(meta.tags or []),
                }
            )
        )

    async def _pin(tool_input: Dict[str, Any], _ctx: Any) -> ToolResult:
        title = str(tool_input.get("title") or "").strip()
        content = str(tool_input.get("content") or "").strip()
        if not title or not content:
            return ToolResult(content=_err("title and content are required"), is_error=True)
        tags = _split_tags(tool_input.get("tags"))
        if "pinned" not in tags:
            tags.append("pinned")
        draft = NoteDraft(
            title=title,
            body=content,
            category=PINNED_CATEGORY,
            tags=tags,
            importance=_importance("high"),
        )
        meta = await provider.notes().write(draft)
        return ToolResult(
            content=_ok(
                {
                    "status": "pinned",
                    "filename": meta.ref.filename,
                    "title": meta.title,
                    "category": PINNED_CATEGORY,
                    "tags": tags,
                }
            )
        )

    async def _read(tool_input: Dict[str, Any], _ctx: Any) -> ToolResult:
        filename = str(tool_input.get("filename") or "").strip()
        if not filename:
            return ToolResult(content=_err("filename is required"), is_error=True)
        note = await provider.notes().read(filename)
        if note is None:
            miss = await _read_miss_payload(provider, filename)
            return ToolResult(
                content=json.dumps(miss, ensure_ascii=False, indent=2, default=str),
                is_error=True,
            )
        return ToolResult(
            content=_ok(
                {
                    "filename": note.ref.filename,
                    "title": note.title,
                    "category": note.category,
                    "tags": list(note.tags or []),
                    "importance": note.importance.value
                    if hasattr(note.importance, "value")
                    else str(note.importance),
                    "body": note.body,
                    "links_out": list(getattr(note, "links_out", []) or []),
                }
            )
        )

    async def _list(tool_input: Dict[str, Any], _ctx: Any) -> ToolResult:
        category = str(tool_input.get("category") or "").strip() or None
        tag = str(tool_input.get("tag") or "").strip() or None
        summaries = await provider.index().list_notes(category=category, tag=tag, limit=100)
        return ToolResult(
            content=_ok(
                {
                    "total": len(summaries),
                    "filters": {"category": category, "tag": tag},
                    "notes": [_summary_dict(s) for s in summaries],
                }
            )
        )

    async def _search(tool_input: Dict[str, Any], _ctx: Any) -> ToolResult:
        query = str(tool_input.get("query") or "").strip()
        if not query:
            return ToolResult(content=_err("query is required"), is_error=True)
        limit = max(1, min(int(tool_input.get("max_results") or 10), 30))
        # 키워드(노트 본문) + 시맨틱(벡터 플레인, 임베딩 활성 시) 병합 —
        # retriever L3+L4 와 같은 개념. 벡터 실패는 키워드-only 로 조용히 폴백.
        chunks = list(await provider.notes().search(query, limit=limit))
        vector = provider.vector()
        if vector is not None:
            try:
                chunks.extend(await vector.search(query, top_k=limit))
            except Exception:  # noqa: BLE001 — 시맨틱은 결과를 보강할 뿐
                logger.debug("memory_search: vector search failed (keyword-only)", exc_info=True)
        # 두 목록을 **순위**로 합친다(RRF). 예전에는 키워드 점수(상한 없음)와 코사인(1 이하)을 한 줄로
        # 정렬해 의미 검색 결과가 늘 뒤로 밀렸다(2026-10-23 감사).
        keyword_hits = list(chunks)
        fused: Dict[str, Dict[str, Any]] = {}
        by_source: Dict[str, List[Any]] = {}
        for c in keyword_hits:
            by_source.setdefault(str(getattr(c, "source", "") or "keyword"), []).append(c)
        for hits in by_source.values():
            for rank, c in enumerate(hits):
                key = getattr(c, "key", "")
                if not key:
                    continue
                entry = fused.setdefault(key, {"score": 0.0, "chunk": c})
                entry["score"] += 1.0 / (60 + rank + 1)
        items = []
        for key, entry in sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:limit]:
            c = entry["chunk"]
            content = (getattr(c, "content", "") or "").strip()
            snippet = content[:400]
            if len(content) > 400:
                snippet += " …(excerpt; memory_read for the whole note)"
            items.append(
                {
                    "filename": key,
                    "snippet": snippet,
                    "score": round(entry["score"], 4),
                    "source": getattr(c, "source", ""),
                }
            )
        return ToolResult(content=_ok({"query": query, "total": len(items), "results": items}))

    async def _categories(_tool_input: Dict[str, Any], _ctx: Any) -> ToolResult:
        cats = await provider.index().list_categories()
        return ToolResult(
            content=_ok(
                {
                    "categories": cats,
                    "next_steps": [
                        "memory_list(category=<name>) — list files in a folder",
                        "memory_search(query=<text>) — keyword search",
                        "memory_read(filename=<path>) — open a specific note",
                    ],
                }
            )
        )

    async def _history(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        return await conversation_history(
            provider, tool_input, state=getattr(ctx, "state_view", None)
        )

    def _schema(props: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    read_caps = ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)
    write_caps = ToolCapabilities(concurrency_safe=False)

    tools = [
        build_tool(
            name="memory_write",
            description=(
                "Create a persistent memory note (survives across conversations). "
                "Use for durable knowledge, decisions, insights. Link related notes "
                "with [[filename]] wikilinks in the content."
            ),
            input_schema=_schema(
                {
                    "title": {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Markdown body"},
                    "category": {
                        "type": "string",
                        "description": "topics|projects|insights|daily|critical (default topics)",
                    },
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                    "importance": {
                        "type": "string",
                        "description": "low|medium|high|critical (default medium)",
                    },
                },
                ["title", "content"],
            ),
            execute=_write,
            capabilities=write_caps,
        ),
        build_tool(
            name="memory_pin",
            description=(
                "Pin a must-always-know fact (user preferences, binding decisions, "
                "standing goals). Pinned facts are injected into every future turn."
            ),
            input_schema=_schema(
                {
                    "title": {"type": "string", "description": "Short fact title"},
                    "content": {"type": "string", "description": "The fact (1-3 sentences)"},
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                },
                ["title", "content"],
            ),
            execute=_pin,
            capabilities=write_caps,
        ),
        build_tool(
            name="memory_read",
            description="Read one memory note's full body + metadata by filename.",
            input_schema=_schema(
                {
                    "filename": {
                        "type": "string",
                        "description": "Note filename, e.g. topics/design.md",
                    },
                },
                ["filename"],
            ),
            execute=_read,
            capabilities=read_caps,
        ),
        build_tool(
            name="memory_list",
            description=(
                "List memory notes (lightweight metadata). Optional category/tag "
                "filters. Use memory_categories first when unsure which folder."
            ),
            input_schema=_schema(
                {
                    "category": {"type": "string", "description": "Category filter"},
                    "tag": {"type": "string", "description": "Tag filter"},
                }
            ),
            execute=_list,
            capabilities=read_caps,
        ),
        build_tool(
            name="memory_search",
            description="Search across memory notes — keyword + semantic (when platform embeddings are configured). Returns filename + snippet hints; use memory_read for full bodies.",
            input_schema=_schema(
                {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 10)"},
                },
                ["query"],
            ),
            execute=_search,
            capabilities=read_caps,
        ),
        build_tool(
            name="ConversationHistory",
            description=(
                "Read THIS conversation's own record, word for word: earlier turns that are no longer in "
                "your context (only the most recent turns are), what tools returned then, and the full "
                "conversation summary. view='turns' lists the turns; view='turn' with turn=<n> shows one "
                "in full (user words, your answers, tool calls and results; long turns come in pages); "
                "view='search' with query finds where something was said; view='summary' returns the "
                "whole summary. Use it before relying on what you remember of earlier turns."
            ),
            input_schema=_schema(
                {
                    "view": {
                        "type": "string",
                        "enum": ["turns", "turn", "search", "summary"],
                        "description": "What to read (default 'turns').",
                    },
                    "turn": {
                        "type": "integer",
                        "description": "Turn number for view='turn' (1 = oldest).",
                    },
                    "query": {"type": "string", "description": "Text to find for view='search'."},
                    "offset": {
                        "type": "integer",
                        "description": "Character offset within a turn, or first turn number for view='turns'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Characters per page (view='turn') or turns per page (view='turns').",
                    },
                }
            ),
            execute=_history,
            capabilities=read_caps,
        ),
        build_tool(
            name="memory_categories",
            description="Vault overview — category map with counts. Use FIRST when you don't know where to look.",
            input_schema=_schema({}),
            execute=_categories,
            capabilities=read_caps,
        ),
    ]
    # TASK 층(memory.tasks)이 있는 provider 면 Task·TaskRecall 도 준다.
    from xgen_agent_runtime.host.task_tools import build_task_tools

    tools.extend(build_task_tools(provider))
    # 종류 표지 — 호스트의 결과 필터가 이름이 아니라 종류로 판정한다(tools.base.tool_origin).
    return [with_origin(t, "memory") for t in tools]


# ── ConversationHistory ────────────────────────────────────────────────────

#: 대화 기록에서 읽어 오는 행 상한. 도구 결과가 많은 턴은 행이 많다.
_HISTORY_SCAN_ROWS = 5000
#: view='turn' 한 쪽의 기본 글자 수.
_HISTORY_PAGE_CHARS = 40_000
#: view='turns' 한 쪽의 기본 턴 수.
_HISTORY_PAGE_TURNS = 30


def _render_block(block: Dict[str, Any]) -> str:
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text") or "")
    if kind == "tool_use":
        args = json.dumps(block.get("input") or {}, ensure_ascii=False, default=str)
        return f"[tool call {block.get('id') or ''}: {block.get('name') or ''}({args})]"
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text") or "")
                if isinstance(b, dict) and b.get("type") == "text"
                else "[image]"
                for b in content
            )
        flag = " (error)" if block.get("is_error") else ""
        return f"[tool result {block.get('tool_use_id') or ''}{flag}]\n{content}"
    if kind in ("image", "image_url"):
        return "[image]"
    if kind == "document":
        return "[document]"
    return ""


def _render_turn(turn: Any) -> str:
    from xgen_agent_runtime.memory.transcript import _blocks

    parts: List[str] = []
    for m in getattr(turn, "messages", []) or []:
        role = str(getattr(m, "role", "") or "")
        ts = getattr(m, "timestamp", None)
        stamp = ts.isoformat(timespec="seconds") if hasattr(ts, "isoformat") else ""
        body = "\n".join(
            t for t in (_render_block(b) for b in _blocks(getattr(m, "content", ""))) if t
        )
        if body:
            parts.append(f"### {role}{(' · ' + stamp) if stamp else ''}\n{body}")
    return "\n\n".join(parts)


def _first_line(text: str, n: int = 200) -> str:
    line = next((ln.strip() for ln in str(text or "").splitlines() if ln.strip()), "")
    return line if len(line) <= n else line[:n] + "…"


def _unrecorded_rows(state: Any) -> List[Any]:
    """This turn's messages that short-term memory has not recorded yet (it records at the turn's
    end), as they really were: those a mid-turn compaction summarized away first, then the rest."""
    if state is None:
        return []
    from types import SimpleNamespace

    from xgen_agent_runtime.core.compaction import RECORDER_STM, peek_pending_records
    from xgen_agent_runtime.core.originals import original_of
    from xgen_agent_runtime.memory.strategy import _recorded_upto
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
        _is_ack,
        _is_checkpoint,
    )

    try:
        messages = list(getattr(state, "messages", None) or [])
        fresh = peek_pending_records(state, RECORDER_STM) + messages[_recorded_upto(state) :]
    except Exception:  # noqa: BLE001 (the record still answers without this turn)
        return []
    rows = []
    for m in fresh:
        if not isinstance(m, dict):
            continue
        # The runtime's checkpoint (and its ack) stands in for messages it summarized: those are
        # listed themselves, from the pending records.
        if _is_checkpoint(m) or _is_ack(m):
            continue
        m = original_of(state, m)
        rows.append(
            SimpleNamespace(role=str(m.get("role") or ""), content=m.get("content"), timestamp=None)
        )
    return rows


async def conversation_history(provider: Any, tool_input: Dict[str, Any], state: Any = None) -> Any:
    """This conversation's own record, from the short-term store (every message, tool results included).

    The context replays only the most recent turns; older turns survive in the prompt as a short
    summary at most. Without a way to read the record the model reconstructs earlier work from
    memory and gets it wrong (2026-10-23 audit). Read-only.
    """
    from xgen_agent_runtime.memory.transcript import _blocks, group_logical_turns
    from xgen_agent_runtime.tools.base import ToolResult

    view = str(tool_input.get("view") or "turns").strip().lower()
    stm = provider.stm()
    if view == "summary":
        reader = getattr(stm, "read_summary", None)
        summary = (await reader()) if reader is not None else None
        if not summary:
            return ToolResult(
                content="This conversation has no summary yet (it is written after turns end)."
            )
        return ToolResult(content=f"[conversation summary, {len(summary)} chars]\n{summary}")

    rows = list(await stm.recent(_HISTORY_SCAN_ROWS)) + _unrecorded_rows(state)
    turns = group_logical_turns(rows, 1_000_000)
    total = len(turns)
    clipped_history = len(rows) >= _HISTORY_SCAN_ROWS
    older_note = (
        f" (only the last {_HISTORY_SCAN_ROWS} records are read; earlier turns are not listed)"
        if clipped_history
        else ""
    )
    if total == 0:
        return ToolResult(content="This conversation has no recorded turns yet.")

    def _int(name: str, default: int) -> int:
        try:
            return int(tool_input.get(name) if tool_input.get(name) is not None else default)
        except (TypeError, ValueError):
            return default

    if view == "turns":
        count = max(1, _int("limit", _HISTORY_PAGE_TURNS))
        start = _int("offset", max(1, total - count + 1))
        start = min(max(1, start), total)
        end = min(total, start + count - 1)
        lines = [
            f"[turns {start}-{end} of {total}{older_note}; read one with view='turn', turn=<n>]"
        ]
        for n in range(start, end + 1):
            t = turns[n - 1]
            msgs = getattr(t, "messages", []) or []
            user = next(
                (
                    " ".join(
                        b.get("text", "")
                        for b in _blocks(getattr(m, "content", ""))
                        if b.get("type") == "text"
                    )
                    for m in msgs
                    if str(getattr(m, "role", "")) == "user"
                ),
                "",
            )
            answer = ""
            tools = 0
            for m in msgs:
                for b in _blocks(getattr(m, "content", "")):
                    if b.get("type") == "tool_use":
                        tools += 1
                if str(getattr(m, "role", "")) == "assistant":
                    text = " ".join(
                        b.get("text", "")
                        for b in _blocks(getattr(m, "content", ""))
                        if b.get("type") == "text"
                    )
                    if text.strip():
                        answer = text
            ts = getattr(msgs[0], "timestamp", None) if msgs else None
            stamp = ts.isoformat(timespec="minutes") if hasattr(ts, "isoformat") else ""
            lines.append(
                f"{n}. {stamp} user: {_first_line(user)} | answer: {_first_line(answer)} | tool calls: {tools}"
            )
        return ToolResult(content="\n".join(lines))

    if view == "turn":
        n = _int("turn", total)
        if not 1 <= n <= total:
            return ToolResult(
                content=f"No turn {n}: this conversation has {total} turns{older_note}.",
                is_error=True,
            )
        body = _render_turn(turns[n - 1])
        offset = max(0, _int("offset", 0))
        limit = max(1, _int("limit", _HISTORY_PAGE_CHARS))
        end = min(len(body), offset + limit)
        tail = f"next offset={end}" if end < len(body) else "end of the turn"
        header = f"[turn {n} of {total}: chars {offset}-{end} of {len(body)}; {tail}]"
        return ToolResult(content=f"{header}\n{body[offset:end]}")

    if view == "search":
        query = str(tool_input.get("query") or "").strip()
        if not query:
            return ToolResult(content="query is required for view='search'", is_error=True)
        needle = query.lower()
        hits: List[str] = []
        for n, t in enumerate(turns, start=1):
            text = _render_turn(t)
            low = text.lower()
            at = low.find(needle)
            if at < 0:
                continue
            count = low.count(needle)
            lo = max(0, at - 150)
            snippet = text[lo : at + len(query) + 250].replace("\n", " ")
            hits.append(f"turn {n} ({count} match{'es' if count != 1 else ''}): …{snippet}…")
        if not hits:
            return ToolResult(
                content=f"'{query}' does not appear in this conversation's record{older_note}."
            )
        shown = hits[:30]
        more = (
            f"\n... ({len(hits) - 30} more turns match; narrow the query)" if len(hits) > 30 else ""
        )
        return ToolResult(
            content=f"[{len(hits)} turns mention '{query}'{older_note}]\n" + "\n".join(shown) + more
        )

    return ToolResult(
        content=f"unknown view {view!r}; use turns, turn, search or summary", is_error=True
    )
