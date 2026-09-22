"""WebSearch — pluggable web search for URLs + snippets.

Cycle 20260424 executor uplift — Phase 3 Week 5.
Cycle 20260620 — generalized from DuckDuckGo-only to **pluggable search
backends** (``ddg`` default, plus ``brave`` / ``tavily`` / ``searxng``).

Issues a text search query and returns a compact, LLM-friendly list of
result headlines, URLs, and snippets. Intended as a companion to
``WebFetch`` — the LLM uses ``WebSearch`` to discover candidate URLs and
then pulls the interesting ones through ``WebFetch`` for details.

Backends (see ``_web_search_backends.py``):

* ``ddg`` (default) — DuckDuckGo via the optional ``ddgs`` package
  (``[web]`` extra). No key required. The legacy WebSearch behaviour
  and output format are preserved byte-for-byte.
* ``brave`` — Brave Search API; key from
  ``ctx.extras['web_search']['brave_api_key']`` or ``BRAVE_SEARCH_API_KEY``.
* ``tavily`` — Tavily API; key from
  ``ctx.extras['web_search']['tavily_api_key']`` or ``TAVILY_API_KEY``.
* ``searxng`` — self-hosted SearXNG JSON API; base URL from
  ``ctx.extras['web_search']['searxng_url']`` or ``SEARXNG_URL``.

The non-ddg backends speak HTTP through ``httpx`` (already a hard
dependency, used by ``WebFetch``), so they add **no new required
packages**. ``ddgs`` stays the optional ``[web]`` extra.

Backend selection precedence (highest first): input param ``backend`` >
``ctx.extras['web_search']['backend']`` > ``GENY_WEBSEARCH_BACKEND`` env
> ``"ddg"``.

Capabilities: ``concurrency_safe=True`` + ``read_only=True`` +
``network_egress=True``. Searches are idempotent-ish on the scale of
a single turn, so the orchestrator is free to run WebSearch in parallel
with Read/Grep/Glob reads.

See ``executor_uplift/06_design_tool_system.md`` §7 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 4-5 Web 계열).
"""

from __future__ import annotations

import asyncio

import logging
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in import _web_search_backends as _backends
from xgen_agent_runtime.tools.built_in._web_search_backends import (
    WebSearchConfigError,
    build_backend,
    select_backend_name,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 10
_HARD_MAX_RESULTS = 30


#: Concurrent searches per process. Public search backends throttle bursts —
#: when the model fires three queries in one ToolBatch (parallel 8) the third
#: comes back "RequestError … yahoo" / brave 429 and, with ddgs' wikipedia
#: engine already down, the whole call fails (dev 2026-09-22: 22 of 50
#: WebSearch calls in one session). The cap lives in the tool, not the
#: executor, so it holds on every path — Stage 10 parallel batches, ToolBatch,
#: sub-agents sharing the process. Two keeps a pair of queries in flight
#: without tripping the free backends.
_MAX_CONCURRENT_SEARCHES = 2
_search_semaphores: Dict[int, asyncio.Semaphore] = {}


def _search_slot() -> asyncio.Semaphore:
    """One semaphore per running event loop (a semaphore is loop-bound)."""
    loop = asyncio.get_running_loop()
    sem = _search_semaphores.get(id(loop))
    if sem is None:
        sem = _search_semaphores[id(loop)] = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)
    return sem


def _load_ddgs() -> Optional[Any]:
    """Return the ``DDGS`` class or ``None`` if ``ddgs`` is not installed.

    Re-exported from the backends module for backward compatibility:
    existing hosts / tests monkey-patch
    ``web_search_tool._load_ddgs``. The ddg backend resolves ``DDGS``
    through this same indirection so the patch still takes effect.
    """
    return _backends._load_ddgs()


class WebSearchTool(Tool):
    """Search the web and return ranked hits with title / URL / snippet.

    Usage pattern: pair with ``WebFetch`` — call WebSearch once to
    surface candidate URLs, inspect the snippets, then issue WebFetch
    on the ones worth reading.

    The default backend is DuckDuckGo. Hosts may switch backends per
    call (``backend`` input), per session (``extras['web_search']
    ['backend']``), or globally (``GENY_WEBSEARCH_BACKEND`` env).
    """

    @property
    def name(self) -> str:
        return "WebSearch"

    @property
    def description(self) -> str:
        # 스키마는 매 호출에 실린다(4.33.0, 332 → ~150 토큰). dev 28일 96회 중 query 96 ·
        # max_results 39 · region 2, safesearch 0 · backend 0 — 뒤의 둘은 스키마에서 뺐다
        # (execute 는 여전히 받는다; 백엔드는 호스트가 extras/env 로 고른다).
        return (
            "Search the web; returns ranked results (title, URL, snippet). "
            "Pair with WebFetch to read a result."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                    "minLength": 1,
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Default {_DEFAULT_MAX_RESULTS}, max {_HARD_MAX_RESULTS}.",
                    "exclusiveMinimum": 0,
                },
                "region": {
                    "type": "string",
                    "description": "Region code, e.g. 'kr-kr', 'us-en'. Default worldwide.",
                },
            },
            "required": ["query"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=False,  # ranking may drift between calls
            network_egress=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        query = (input.get("query") or "").strip()
        if not query:
            return ToolResult(content="query must not be empty", is_error=True)

        max_results = int(input.get("max_results", _DEFAULT_MAX_RESULTS))
        max_results = max(1, min(_HARD_MAX_RESULTS, max_results))
        # ddgs 9.x fans the query out to several engines and builds the
        # wikipedia one's host from the region prefix — ``wt-wt`` (its own
        # "worldwide" code) becomes ``wt.wikipedia.org``, which does not exist,
        # so that engine fails on every call and when the others are throttled
        # the whole search fails (dev 2026-09-22: 22 of 50 WebSearch calls in
        # one session died with "DNSError … wt.wikipedia.org"). Pass a region
        # only when the caller asked for one; ddgs' own default works.
        region = input.get("region") or None
        safesearch = input.get("safesearch") or "moderate"

        backend_name = select_backend_name(input, context)
        try:
            backend = build_backend(
                backend_name,
                context,
                # Resolve through the module / class indirection at call
                # time so monkey-patches of ``web_search_tool._load_ddgs``
                # and ``WebSearchTool._search_sync`` keep working.
                ddg_load_ddgs=_load_ddgs,
                ddg_search_sync=type(self)._search_sync,
            )
        except WebSearchConfigError as exc:
            # Unknown backend name → "valid options" hint.
            return ToolResult(content=str(exc), is_error=True)

        try:
            async with _search_slot():
                hits = await backend.search(query, max_results, region, safesearch)
        except WebSearchConfigError as exc:
            # Missing key / url (or missing ddgs package) → config hint.
            return ToolResult(content=str(exc), is_error=True)
        except Exception as exc:
            logger.exception("WebSearch backend %r call failed", backend_name)
            return ToolResult(
                content=f"web search failed: {exc}",
                is_error=True,
            )

        if not hits:
            return ToolResult(
                content=f"No results for {query!r}.",
                metadata={"query": query, "results_count": 0},
            )

        hits = hits[:max_results]
        header = f"Search results for {query!r} ({len(hits)} of max {max_results}):"
        body = "\n\n".join(self._format_hit(h) for h in hits)
        return ToolResult(
            content=f"{header}\n\n{body}",
            metadata={
                "query": query,
                "results_count": len(hits),
                "results": hits,
            },
        )

    @staticmethod
    def _search_sync(
        ddgs_cls: Any,
        query: str,
        max_results: int,
        region: Optional[str],
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        """Blocking ddgs body — kept for backward compatibility.

        The ddg backend delegates here (via the class reference) so
        existing hosts / tests that monkey-patch
        ``WebSearchTool._search_sync`` keep working unchanged.
        """
        kwargs: Dict[str, Any] = {"safesearch": safesearch, "max_results": max_results}
        if region:
            kwargs["region"] = region
        with ddgs_cls() as client:
            return list(client.text(query, **kwargs))

    @staticmethod
    def _normalise_hit(index: int, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map a provider result dict to a stable shape.

        ddgs returns ``title`` / ``href`` / ``body`` — we rename ``href``
        to ``url`` and ``body`` to ``snippet`` to match the more common
        search-API conventions. ``rank`` is the zero-based position.
        """
        return {
            "rank": index,
            "title": str(raw.get("title") or "").strip(),
            "url": str(raw.get("href") or raw.get("url") or "").strip(),
            "snippet": str(raw.get("body") or raw.get("snippet") or "").strip(),
        }

    @staticmethod
    def _format_hit(hit: Dict[str, Any]) -> str:
        rank = hit.get("rank", 0) + 1
        title = hit.get("title") or "(no title)"
        url = hit.get("url") or "(no url)"
        snippet = hit.get("snippet") or ""
        if snippet:
            return f"{rank}. {title}\n   {url}\n   {snippet}"
        return f"{rank}. {title}\n   {url}"
