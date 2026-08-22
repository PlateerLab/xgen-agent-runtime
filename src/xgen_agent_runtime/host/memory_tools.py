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


def build_memory_tools(provider: Any) -> List[Any]:
    """provider 에 바인딩된 executor ``Tool`` 목록을 만든다.

    provider 가 None 이면 빈 리스트. 도구 실행은 전부 async — 파이프라인
    루프(스레드 소유)에서만 provider 를 만진다.
    """
    if provider is None:
        return []

    from xgen_agent_runtime.memory.provider import Importance, NoteDraft, NotePatch  # noqa: F401
    from xgen_agent_runtime.tools import ToolResult, build_tool
    from xgen_agent_runtime.tools.base import ToolCapabilities

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
            return ToolResult(content=_err(f"Note not found: {filename}"), is_error=True)
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
        items = []
        seen = set()
        for c in chunks:
            key = getattr(c, "key", "")
            if key in seen:
                continue
            seen.add(key)
            content = (getattr(c, "content", "") or "").strip()
            first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
            items.append(
                {
                    "filename": key,
                    "snippet_first_line": first_line[:200],
                    "score": round(float(getattr(c, "relevance_score", 0.0)), 4),
                    "source": getattr(c, "source", ""),
                }
            )
        items.sort(key=lambda x: -x["score"])
        items = items[:limit]
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

    def _schema(props: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    read_caps = ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)
    write_caps = ToolCapabilities(concurrency_safe=False)

    return [
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
            name="memory_categories",
            description="Vault overview — category map with counts. Use FIRST when you don't know where to look.",
            input_schema=_schema({}),
            execute=_categories,
            capabilities=read_caps,
        ),
    ]
