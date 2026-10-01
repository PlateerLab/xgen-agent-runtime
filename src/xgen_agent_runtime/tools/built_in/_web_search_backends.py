"""Pluggable search backends for the ``WebSearch`` built-in tool.

The default backend remains DuckDuckGo (``ddg``) via the optional
``ddgs`` package, so the legacy WebSearch path is byte-for-byte
unchanged. Hosts that want a different provider can switch backends
without touching the tool surface — selection happens in
:meth:`WebSearchTool.execute` (input param ``backend`` > extras >
``GENY_WEBSEARCH_BACKEND`` env > ``"ddg"``).

Every backend implements the async :class:`WebSearchBackend` protocol
and returns *normalized hits* — the exact shape
:meth:`WebSearchTool._normalise_hit` produces (``rank`` / ``title`` /
``url`` / ``snippet``) — so the tool's formatting + metadata stay
identical regardless of provider.

Non-ddg backends speak HTTP through ``httpx`` (already a hard
dependency, used by ``WebFetch``), so adding them introduces **no new
required packages**. ``ddgs`` stays the optional ``[web]`` extra.

Config / credentials are read from ``ToolContext.extras["web_search"]``
first, then environment variables:

* ``brave``   — ``brave_api_key`` / ``BRAVE_SEARCH_API_KEY``
* ``tavily``  — ``tavily_api_key`` / ``TAVILY_API_KEY``
* ``searxng`` — ``searxng_url``   / ``SEARXNG_URL``
* ``ddg``     — ``ddg_extra_engines`` / ``GENY_WEBSEARCH_DDG_EXTRA_ENGINES``:
  engines ddgs ships switched off that we run anyway (comma list, default
  ``yandex``; empty turns it off)

When a backend is missing its key/url, it raises
:class:`WebSearchConfigError` with a clear config hint; the tool turns
that into a ``ToolResult(is_error=True)``.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import httpx

from xgen_agent_runtime.tools.base import ToolContext

logger = logging.getLogger(__name__)

# Shared HTTP timeout for the API-backed backends (seconds).
_HTTP_TIMEOUT = 15.0


class WebSearchBackendError(Exception):
    """Base class for backend-level failures surfaced to the tool."""


class WebSearchConfigError(WebSearchBackendError):
    """Raised when a backend is missing required credentials / config.

    The message is a user-facing hint (which extras key / env var to
    set) and is rendered verbatim into ``ToolResult.content``.
    """


class WebSearchBlockedError(WebSearchBackendError):
    """Raised when the search engines turned the request away.

    Distinct from "nothing matched": the engines refused this server
    (rate limiting) or answered with a page that yielded no result. The
    message tells the model that rewording the query will not help and
    is rendered verbatim into ``ToolResult.content``; ``engines`` is the
    per-engine outcome for metadata / logs.
    """

    def __init__(self, message: str, engines: Dict[str, Any]) -> None:
        super().__init__(message)
        self.engines = engines


def _extras_web_search(context: ToolContext) -> Dict[str, Any]:
    """Return the ``web_search`` sub-dict from ``ctx.extras`` (or empty)."""
    raw = (context.extras or {}).get("web_search")
    return raw if isinstance(raw, dict) else {}


def _normalise_hit(index: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map a provider result dict to the stable WebSearch hit shape.

    Mirrors :meth:`WebSearchTool._normalise_hit`: ``rank`` is the
    zero-based position; ``href``/``url`` and ``body``/``snippet`` are
    accepted interchangeably so each backend can hand back whichever
    key its API uses.
    """
    return {
        "rank": index,
        "title": str(raw.get("title") or "").strip(),
        "url": str(raw.get("href") or raw.get("url") or "").strip(),
        "snippet": str(raw.get("body") or raw.get("snippet") or "").strip(),
    }


@runtime_checkable
class WebSearchBackend(Protocol):
    """Async search backend contract.

    Implementations return a list of normalized hits (the
    :func:`_normalise_hit` shape). ``name`` is the registry key used in
    the ``backend`` input enum + selection precedence.
    """

    name: str

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        """Run ``query`` and return up to ``max_results`` normalized hits."""
        ...


# ─────────────────────────────────────────────────────────────────
# ddg — default backend, wraps the existing ddgs logic
# ─────────────────────────────────────────────────────────────────


def _load_ddgs() -> Optional[Any]:
    """Return the ``DDGS`` class or ``None`` if ``ddgs`` is not installed.

    Imported lazily so core hosts don't pay the startup cost of
    ``ddgs`` + ``primp`` + ``lxml`` unless WebSearch is actually used.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return None
    return DDGS


# ddgs cannot tell "the engines turned us away" from "nothing matched":
# ``BaseSearchEngine.request`` maps every non-200 reply to ``None`` — the same
# as an empty page — and when no engine yields a hit ``DDGS.text`` raises
# "No results found.". From our servers most engines refuse: google 429,
# brave 429, mojeek 403, duckduckgo 202 on dev, stage and the home box alike
# (2026-10-01), and yahoo — often the only one left — answers half the time
# with a layout ddgs 9.16 parses to nothing. On dev 27 of 101 searches
# (09-29~10-01) ended "No results found." and the model took it for a bad
# query: one turn searched 14 times, 7 of them failing. The watch records what
# each engine actually got back, so the backend can retry what is worth
# retrying and say what happened.

#: Title lookups ddgs always runs first for text search (``DDGS._get_engines``).
#: They find nothing for most queries by design, so their answer says nothing
#: about whether the web engines could be reached.
_LOOKUP_ENGINES = frozenset({"wikipedia", "grokipedia"})

#: Pause before the single retry when the web engines refused us. A second
#: search two seconds later got through for 18 of 24 such failures on dev,
#: stage and the home box (2026-10-01, three runs): which engines answer varies
#: per call (duckduckgo, google, yahoo's layout) — the refusals are not a ban.
_DDG_RETRY_PAUSE_S = 2.0

#: Engines ddgs ships but has switched off, run anyway for our searches.
#: ddgs 9.15.0 (2026-08-16) disabled yandex without a stated reason; from dev,
#: stage and the home box it answered 36 of 36 queries (about 1.4 s each,
#: Korean ones included) while google, brave, mojeek and duckduckgo mostly
#: refuse us — with it the first try got results 24 of 24 times on dev and
#: stage. Yandex is a Russian service: a host whose policy forbids sending
#: queries there sets ``ddg_extra_engines`` / ``GENY_WEBSEARCH_DDG_EXTRA_ENGINES``
#: to "" (or to another comma list).
_DEFAULT_DDG_EXTRA_ENGINES: Tuple[str, ...] = ("yandex",)


def _ddg_extra_engines(context: Optional[ToolContext]) -> Tuple[str, ...]:
    """Extra ddgs engine names: extras > ``GENY_WEBSEARCH_DDG_EXTRA_ENGINES`` > default."""
    raw: Any = _extras_web_search(context).get("ddg_extra_engines") if context else None
    if raw is None:
        raw = os.environ.get("GENY_WEBSEARCH_DDG_EXTRA_ENGINES")
    if raw is None:
        return _DEFAULT_DDG_EXTRA_ENGINES
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    return tuple(name for name in (str(item).strip().lower() for item in items) if name)


def _ddg_engine_class(name: str) -> Optional[type]:
    """ddgs' text engine class called ``name``, even when ddgs switched it off."""
    if not name.isidentifier():
        return None
    try:
        module = importlib.import_module(f"ddgs.engines.{name}")
    except ImportError:
        return None
    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and getattr(obj, "name", None) == name
            and getattr(obj, "category", None) == "text"
        ):
            return obj
    return None


def _add_ddg_engines(client: Any, names: Tuple[str, ...]) -> None:
    """Make ``client`` also run the text engines ``names`` that ddgs leaves out.

    Wraps the instance's ``_get_engines`` — ddgs' global registry stays as
    it is. The engines are ordered the way ddgs orders its own (shuffled,
    then by priority), as if ddgs had never switched them off; an engine
    ddgs already runs is not added twice.
    """
    get_engines = getattr(client, "_get_engines", None)
    if not names or not callable(get_engines):
        return

    def _get_engines(category: Any, backend: Any, *args: Any, **kwargs: Any) -> Any:
        engines = get_engines(category, backend, *args, **kwargs)
        asked = backend if isinstance(backend, (list, tuple)) else str(backend).split(",")
        if category != "text" or not {"auto", "all"} & {str(b).strip() for b in asked}:
            return engines
        present = {getattr(engine, "name", None) for engine in engines or ()}
        added = []
        for name in names:
            cls = None if name in present else _ddg_engine_class(name)
            if cls is None:
                continue
            try:
                added.append(
                    cls(
                        proxy=getattr(client, "_proxy", None),
                        timeout=getattr(client, "_timeout", None),
                        verify=getattr(client, "_verify", True),
                    )
                )
            except Exception:
                logger.debug("WebSearch: could not start ddgs engine %r", name, exc_info=True)
        if not added:
            return engines
        merged = list(engines or ()) + added
        random.shuffle(merged)
        merged.sort(key=lambda engine: getattr(engine, "priority", 1), reverse=True)
        return merged

    try:
        client._get_engines = _get_engines
    except (AttributeError, TypeError):
        return


class _EngineWatch:
    """What each ddgs engine got back during one search.

    :meth:`attach` wraps the ``DDGS`` instance's ``_get_engines`` so every
    engine it hands out gets its HTTP client's ``request`` and its
    ``search`` wrapped — on the instance only. ddgs caches engines per
    ``DDGS`` instance and each search builds a new one, so nothing leaks
    between calls. When the hooks are missing (another ddgs version, a
    test double) the watch stays empty and the caller keeps ddgs' own
    behaviour.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engines: Dict[str, Dict[str, Any]] = {}

    @property
    def seen(self) -> bool:
        return bool(self._engines)

    def attach(self, client: Any) -> None:
        get_engines = getattr(client, "_get_engines", None)
        if not callable(get_engines):
            return

        def _get_engines(*args: Any, **kwargs: Any) -> Any:
            engines = get_engines(*args, **kwargs)
            for engine in engines or ():
                self._watch(engine)
            return engines

        try:
            client._get_engines = _get_engines
        except (AttributeError, TypeError):
            return

    def _watch(self, engine: Any) -> None:
        if getattr(engine, "_xgen_watched", False):
            return
        name = str(getattr(engine, "name", "") or type(engine).__name__)
        http = getattr(engine, "http_client", None)
        request = getattr(http, "request", None)
        search = getattr(engine, "search", None)
        if http is None or not callable(request) or not callable(search):
            return

        def _request(*args: Any, **kwargs: Any) -> Any:
            try:
                resp = request(*args, **kwargs)
            except Exception as exc:
                self._note(name, error=exc)
                raise
            self._note(name, status=getattr(resp, "status_code", None))
            return resp

        def _search(*args: Any, **kwargs: Any) -> Any:
            try:
                found = search(*args, **kwargs)
            except Exception as exc:
                self._note(name, error=exc)
                raise
            self._note(
                name, hits=len(found) if isinstance(found, (list, tuple)) else int(bool(found))
            )
            return found

        try:
            http.request = _request
            engine.search = _search
            engine._xgen_watched = True
        except (AttributeError, TypeError):
            return

    def _note(
        self,
        name: str,
        *,
        status: Optional[int] = None,
        error: Optional[BaseException] = None,
        hits: int = 0,
    ) -> None:
        with self._lock:
            rec = self._engines.setdefault(name, {"statuses": [], "error": None, "hits": 0})
            if status is not None:
                rec["statuses"].append(status)
            if error is not None and rec["error"] is None:
                rec["error"] = error
            rec["hits"] += hits

    @staticmethod
    def _refusal(rec: Dict[str, Any]) -> Optional[str]:
        """Why an engine without hits gave us nothing, or ``None`` if it answered."""
        for status in rec["statuses"]:
            if status != 200:
                return f"HTTP {status}"
        if rec["statuses"]:
            # It answered; a page we could not parse is not a refusal.
            return None
        error = rec["error"]
        if error is None:
            return None
        if "timeout" in type(error).__name__.lower() or "timed out" in str(error).lower():
            return "timeout"
        return "unreachable"

    def outcome(self) -> Dict[str, Any]:
        """Split the engines into found (name → hits), refused (name → reason), empty."""
        found: Dict[str, int] = {}
        refused: Dict[str, str] = {}
        empty: List[str] = []
        with self._lock:
            for name, rec in sorted(self._engines.items()):
                if rec["hits"]:
                    found[name] = rec["hits"]
                elif (reason := self._refusal(rec)) is not None:
                    refused[name] = reason
                else:
                    empty.append(name)
        return {"found": found, "refused": refused, "empty": empty}

    def web_turned_away(self) -> bool:
        """No web engine found anything, and more of them refused us than answered.

        A refusal or two next to engines that answered with nothing is still
        "nothing matched" — brave and mojeek refuse us on every call. Every
        failure seen on dev, stage and the home box had four refusals to two
        empty answers.
        """
        out = self.outcome()
        if any(name not in _LOOKUP_ENGINES for name in out["found"]):
            return False
        refused = sum(name not in _LOOKUP_ENGINES for name in out["refused"])
        answered = sum(name not in _LOOKUP_ENGINES for name in out["empty"])
        return refused > 0 and refused >= answered


def _refusal_list(outcome: Dict[str, Any]) -> str:
    return ", ".join(
        f"{name} {reason}"
        for name, reason in outcome["refused"].items()
        if name not in _LOOKUP_ENGINES
    )


def _blocked_message(outcome: Dict[str, Any]) -> str:
    refused = {n: r for n, r in outcome["refused"].items() if n not in _LOOKUP_ENGINES}
    if all(r.startswith("HTTP") for r in refused.values()):
        what, cause = (
            "refused this server's requests",
            "the engines blocking or rate-limiting this server",
        )
    else:
        what, cause = "refused or did not answer this server", "on this server's side"
    message = f"web search failed twice: the search engines {what} ({_refusal_list(outcome)})"
    empty = [n for n in outcome["empty"] if n not in _LOOKUP_ENGINES]
    if empty:
        message += f"; {', '.join(empty)} answered without a usable result"
    return message + (
        f". This is {cause}, not a problem with your query — rewording it will "
        "not help. WebFetch a page whose URL you already know, or tell the user "
        "web search is unavailable right now."
    )


def _lookup_only_notice(outcome: Dict[str, Any]) -> str:
    return (
        "Note: only encyclopedia lookups returned results — the web search engines "
        f"refused this server or returned nothing usable ({_refusal_list(outcome)}), "
        "so these results are incomplete. Rewording the query will not help."
    )


class DdgBackend:
    """DuckDuckGo backend via the optional ``ddgs`` package.

    This is the default and preserves the legacy WebSearch output.
    ``ddgs`` is blocking, so the blocking body is pushed to a worker
    thread via :func:`asyncio.to_thread`.

    The DDGS-class loader and the blocking search body are injected by
    the tool (``load_ddgs`` / ``search_sync``) rather than referenced
    directly here, so existing hosts / tests that monkey-patch
    ``web_search_tool._load_ddgs`` or ``WebSearchTool._search_sync``
    continue to take effect through the indirection. The body receives
    a factory that builds the ``DDGS`` client with the extra engines added
    (:func:`_add_ddg_engines`) and an :class:`_EngineWatch` attached; when
    the web engines turned the search away it is run once more, and if that
    fails too :class:`WebSearchBlockedError` says so instead of ddgs'
    "No results found.".
    """

    name = "ddg"

    def __init__(
        self,
        load_ddgs: Optional[Callable[[], Optional[Any]]] = None,
        search_sync: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        extra_engines: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._load_ddgs = load_ddgs or _load_ddgs
        self._search_sync = search_sync or _default_ddg_search_sync
        self._extra_engines = (
            _ddg_extra_engines(None) if extra_engines is None else tuple(extra_engines)
        )
        #: Caveat about the last search's results for the caller, or ``None``.
        self.notice: Optional[str] = None
        #: Per-engine outcome of the last search (empty when not observed).
        self.engines: Dict[str, Any] = {}

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        ddgs_cls = self._load_ddgs()
        if ddgs_cls is None:
            raise WebSearchConfigError(
                "WebSearch requires the 'ddgs' package. Install the "
                "executor's [web] extra:\n"
                "    pip install 'xgen-agent-runtime[web]'\n"
                "or pin ddgs directly:\n"
                "    pip install 'ddgs>=9.11'"
            )
        self.notice = None
        self.engines = {}
        args = (ddgs_cls, query, max_results, region, safesearch)
        raw, watch = await self._attempt(*args)
        first_raw: List[Dict[str, Any]] = []
        attempts = 1
        if watch.web_turned_away():
            logger.info(
                "WebSearch: engines turned the search away, retrying once: %s", watch.outcome()
            )
            first_raw = raw
            await asyncio.sleep(_DDG_RETRY_PAUSE_S)
            raw, watch = await self._attempt(*args)
            attempts = 2
        if watch.seen:
            self.engines = {**watch.outcome(), "attempts": attempts}
        if attempts == 2:
            raw = raw or first_raw  # encyclopedia hits, if only the first try had them
            if watch.web_turned_away():
                if not raw:
                    raise WebSearchBlockedError(_blocked_message(self.engines), self.engines)
                self.notice = _lookup_only_notice(self.engines)
        return [_normalise_hit(i, r) for i, r in enumerate(raw[:max_results])]

    async def _attempt(
        self,
        ddgs_cls: Any,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> Tuple[List[Dict[str, Any]], _EngineWatch]:
        watch = _EngineWatch()

        def _watched_ddgs(*args: Any, **kwargs: Any) -> Any:
            client = ddgs_cls(*args, **kwargs)
            _add_ddg_engines(client, self._extra_engines)
            watch.attach(client)
            return client

        try:
            raw = await asyncio.to_thread(
                self._search_sync,
                _watched_ddgs,
                query,
                max_results,
                region,
                safesearch,
            )
        except Exception:
            # ddgs raises "No results found." (or the last engine error) when
            # nothing came back. Once the engines were watched we know more
            # than that message says; otherwise keep ddgs' error as before.
            if not watch.seen:
                raise
            raw = []
        return list(raw or []), watch


def _default_ddg_search_sync(
    ddgs_cls: Any,
    query: str,
    max_results: int,
    region: str,
    safesearch: str,
) -> List[Dict[str, Any]]:
    """Blocking ddgs body — runs inside ``asyncio.to_thread``.

    Used only when the tool does not inject its own (back-compat)
    ``_search_sync``.
    """
    kwargs: Dict[str, Any] = {"safesearch": safesearch, "max_results": max_results}
    if region:
        kwargs["region"] = (
            region  # ``wt-wt`` would break ddgs' wikipedia engine — see WebSearchTool
        )
    with ddgs_cls() as client:
        return list(client.text(query, **kwargs))


# ─────────────────────────────────────────────────────────────────
# brave — Brave Search API
# ─────────────────────────────────────────────────────────────────


class BraveBackend:
    """Brave Search API backend (``api.search.brave.com``).

    Key resolution: ``ctx.extras["web_search"]["brave_api_key"]`` then
    the ``BRAVE_SEARCH_API_KEY`` environment variable.
    """

    name = "brave"
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, context: ToolContext) -> None:
        self._api_key = _extras_web_search(context).get("brave_api_key") or os.environ.get(
            "BRAVE_SEARCH_API_KEY"
        )

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        if not self._api_key:
            raise WebSearchConfigError(
                "Brave backend requires an API key. Set "
                "extras['web_search']['brave_api_key'] or the "
                "BRAVE_SEARCH_API_KEY environment variable. Get a key at "
                "https://brave.com/search/api/."
            )
        # Brave caps web results at 20 per request.
        count = max(1, min(20, max_results))
        params: Dict[str, Any] = {"q": query, "count": count}
        # Brave safesearch vocabulary: off | moderate | strict.
        params["safesearch"] = "strict" if safesearch == "on" else safesearch
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(self._ENDPOINT, params=params, headers=headers)
        if resp.status_code == 401 or resp.status_code == 403:
            raise WebSearchConfigError(
                f"Brave rejected the API key (HTTP {resp.status_code}). "
                "Check extras['web_search']['brave_api_key'] / "
                "BRAVE_SEARCH_API_KEY."
            )
        resp.raise_for_status()
        data = resp.json()
        results = ((data or {}).get("web") or {}).get("results") or []
        hits: List[Dict[str, Any]] = []
        for i, r in enumerate(results[:max_results]):
            hits.append(
                _normalise_hit(
                    i,
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("description"),
                    },
                )
            )
        return hits


# ─────────────────────────────────────────────────────────────────
# tavily — Tavily Search API
# ─────────────────────────────────────────────────────────────────


class TavilyBackend:
    """Tavily Search API backend (``api.tavily.com``).

    Key resolution: ``ctx.extras["web_search"]["tavily_api_key"]`` then
    the ``TAVILY_API_KEY`` environment variable.
    """

    name = "tavily"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, context: ToolContext) -> None:
        self._api_key = _extras_web_search(context).get("tavily_api_key") or os.environ.get(
            "TAVILY_API_KEY"
        )

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        if not self._api_key:
            raise WebSearchConfigError(
                "Tavily backend requires an API key. Set "
                "extras['web_search']['tavily_api_key'] or the "
                "TAVILY_API_KEY environment variable. Get a key at "
                "https://tavily.com/."
            )
        payload: Dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max(1, min(20, max_results)),
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(self._ENDPOINT, json=payload)
        if resp.status_code == 401 or resp.status_code == 403:
            raise WebSearchConfigError(
                f"Tavily rejected the API key (HTTP {resp.status_code}). "
                "Check extras['web_search']['tavily_api_key'] / "
                "TAVILY_API_KEY."
            )
        resp.raise_for_status()
        data = resp.json()
        results = (data or {}).get("results") or []
        hits: List[Dict[str, Any]] = []
        for i, r in enumerate(results[:max_results]):
            hits.append(
                _normalise_hit(
                    i,
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("content"),
                    },
                )
            )
        return hits


# ─────────────────────────────────────────────────────────────────
# searxng — self-hosted SearXNG JSON API
# ─────────────────────────────────────────────────────────────────


class SearxngBackend:
    """SearXNG JSON API backend (self-hosted meta-search).

    Base URL resolution: ``ctx.extras["web_search"]["searxng_url"]``
    then the ``SEARXNG_URL`` environment variable. The instance must
    have the JSON output format enabled.
    """

    name = "searxng"

    def __init__(self, context: ToolContext) -> None:
        base = _extras_web_search(context).get("searxng_url") or os.environ.get("SEARXNG_URL")
        self._base = base.rstrip("/") if isinstance(base, str) and base else None

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        if not self._base:
            raise WebSearchConfigError(
                "SearXNG backend requires a base URL. Set "
                "extras['web_search']['searxng_url'] or the SEARXNG_URL "
                "environment variable (e.g. 'https://searx.example.org')."
            )
        # SearXNG safesearch is numeric: 0 off / 1 moderate / 2 strict.
        safe_map = {"off": 0, "moderate": 1, "on": 2}
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "safesearch": safe_map.get(safesearch, 1),
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(f"{self._base}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = (data or {}).get("results") or []
        hits: List[Dict[str, Any]] = []
        for i, r in enumerate(results[:max_results]):
            hits.append(
                _normalise_hit(
                    i,
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("content"),
                    },
                )
            )
        return hits


# ─────────────────────────────────────────────────────────────────
# Registry + factory
# ─────────────────────────────────────────────────────────────────

#: Valid backend names, in a stable order (also used for the input enum
#: and the "valid options" hint in error messages).
BACKEND_NAMES: tuple[str, ...] = ("ddg", "brave", "tavily", "searxng")

DEFAULT_BACKEND = "ddg"


def build_backend(
    name: str,
    context: ToolContext,
    *,
    ddg_load_ddgs: Optional[Callable[[], Optional[Any]]] = None,
    ddg_search_sync: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> WebSearchBackend:
    """Instantiate the backend ``name`` bound to ``context``.

    Raises :class:`WebSearchConfigError` for an unknown name so the
    tool surfaces a clear "valid options" hint. The API backends
    capture their credentials at construction time. ``ddg_load_ddgs`` /
    ``ddg_search_sync`` let the tool inject its own (monkey-patchable)
    DDGS hooks for backward compatibility.
    """
    if name == "ddg":
        return DdgBackend(
            load_ddgs=ddg_load_ddgs,
            search_sync=ddg_search_sync,
            extra_engines=_ddg_extra_engines(context),
        )
    if name == "brave":
        return BraveBackend(context)
    if name == "tavily":
        return TavilyBackend(context)
    if name == "searxng":
        return SearxngBackend(context)
    raise WebSearchConfigError(
        f"Unknown WebSearch backend {name!r}. Valid options: {', '.join(BACKEND_NAMES)}."
    )


def select_backend_name(input: Dict[str, Any], context: ToolContext) -> str:
    """Resolve the backend name by precedence.

    Precedence (highest first):
    ``input['backend']`` > ``ctx.extras['web_search']['backend']`` >
    ``GENY_WEBSEARCH_BACKEND`` env > ``"ddg"``.
    """
    candidate = (
        input.get("backend")
        or _extras_web_search(context).get("backend")
        or os.environ.get("GENY_WEBSEARCH_BACKEND")
        or DEFAULT_BACKEND
    )
    return str(candidate).strip().lower()
