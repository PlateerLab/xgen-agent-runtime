"""Phase 3 Week 5 — WebSearch tests.

Tests stub the ``ddgs`` client so we don't hit the live network.
``_search_sync`` is monkey-patched to return canned results, so the
test suite stays deterministic and works even when the ``[web]`` extra
is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.web_search_tool import (
    _DEFAULT_MAX_RESULTS,
    _HARD_MAX_RESULTS,
    WebSearchTool,
)


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", working_dir="")


def _stub_search(results: List[Dict[str, Any]]):
    """Return a replacement for ``_search_sync`` that yields ``results``.

    Signature matches the real static method so monkey-patching is a
    drop-in.
    """

    def _fake(ddgs_cls, query, max_results, region, safesearch):
        return list(results[:max_results])

    return _fake


@pytest.fixture
def ddgs_available(monkeypatch):
    """Pretend ddgs is installed so tests don't depend on the [web] extra.

    The real ``_load_ddgs`` returns the actual DDGS class when the
    package is installed. Our tests monkey-patch ``_search_sync`` to
    return canned data, so we only need a non-None sentinel here —
    the sentinel never has any methods called on it.
    """
    monkeypatch.setattr(
        "xgen_agent_runtime.tools.built_in.web_search_tool._load_ddgs",
        lambda: object,
    )


class TestSchemaAndCapabilities:
    def test_capabilities(self):
        caps = WebSearchTool().capabilities({"query": "x"})
        assert caps.concurrency_safe is True
        assert caps.read_only is True
        assert caps.network_egress is True
        assert caps.destructive is False

    def test_schema_has_query(self):
        schema = WebSearchTool().input_schema
        assert schema["properties"]["query"]["minLength"] == 1
        assert "query" in schema["required"]
        # 4.33.0: safesearch/backend 는 스키마에서 뺐다(dev 28일 0회) — execute 는 여전히 받는다.
        assert "safesearch" not in schema["properties"] and "backend" not in schema["properties"]


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_hits(self, monkeypatch, ddgs_available):
        monkeypatch.setattr(
            WebSearchTool,
            "_search_sync",
            staticmethod(
                _stub_search(
                    [
                        {
                            "title": "Python docs",
                            "href": "https://docs.python.org/",
                            "body": "Official documentation",
                        },
                        {
                            "title": "PEP 8",
                            "href": "https://peps.python.org/pep-0008/",
                            "body": "Style guide",
                        },
                    ]
                )
            ),
        )
        result = await WebSearchTool().execute({"query": "python"}, _ctx())
        assert not result.is_error
        assert "Python docs" in result.content
        assert "https://peps.python.org/pep-0008/" in result.content
        assert result.metadata["results_count"] == 2
        assert result.metadata["results"][0]["url"] == "https://docs.python.org/"
        # Rank is zero-based in metadata, one-based in rendered content
        assert result.metadata["results"][0]["rank"] == 0
        assert "1. Python docs" in result.content
        assert "2. PEP 8" in result.content

    @pytest.mark.asyncio
    async def test_respects_max_results(self, monkeypatch, ddgs_available):
        many = [
            {"title": f"T{i}", "href": f"https://x/{i}", "body": f"B{i}"}
            for i in range(20)
        ]
        monkeypatch.setattr(
            WebSearchTool, "_search_sync", staticmethod(_stub_search(many))
        )
        result = await WebSearchTool().execute(
            {"query": "x", "max_results": 3}, _ctx()
        )
        assert result.metadata["results_count"] == 3
        assert "1. T0" in result.content
        assert "3. T2" in result.content
        assert "T3" not in result.content

    @pytest.mark.asyncio
    async def test_hard_cap_enforced(self, monkeypatch, ddgs_available):
        # Ask for more than the hard cap — should silently clamp.
        many = [
            {"title": f"T{i}", "href": f"https://x/{i}", "body": ""}
            for i in range(_HARD_MAX_RESULTS + 20)
        ]
        captured: Dict[str, int] = {}

        def _spy(ddgs_cls, query, max_results, region, safesearch):
            captured["max_results"] = max_results
            return many[:max_results]

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_spy))
        result = await WebSearchTool().execute(
            {"query": "x", "max_results": 9999}, _ctx()
        )
        assert captured["max_results"] == _HARD_MAX_RESULTS
        assert result.metadata["results_count"] == _HARD_MAX_RESULTS

    @pytest.mark.asyncio
    async def test_empty_results_produces_no_results_message(
        self, monkeypatch, ddgs_available
    ):
        monkeypatch.setattr(
            WebSearchTool, "_search_sync", staticmethod(_stub_search([]))
        )
        result = await WebSearchTool().execute({"query": "nothing"}, _ctx())
        assert not result.is_error
        assert "No results" in result.content
        assert result.metadata["results_count"] == 0

    @pytest.mark.asyncio
    async def test_default_max_results_applied(self, monkeypatch, ddgs_available):
        captured: Dict[str, int] = {}

        def _spy(ddgs_cls, query, max_results, region, safesearch):
            captured["max_results"] = max_results
            return []

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_spy))
        await WebSearchTool().execute({"query": "foo"}, _ctx())
        assert captured["max_results"] == _DEFAULT_MAX_RESULTS

    @pytest.mark.asyncio
    async def test_region_and_safesearch_forwarded(self, monkeypatch, ddgs_available):
        captured: Dict[str, Any] = {}

        def _spy(ddgs_cls, query, max_results, region, safesearch):
            captured["region"] = region
            captured["safesearch"] = safesearch
            return []

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_spy))
        await WebSearchTool().execute(
            {"query": "foo", "region": "kr-kr", "safesearch": "off"}, _ctx()
        )
        assert captured == {"region": "kr-kr", "safesearch": "off"}


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_empty_query_error(self):
        result = await WebSearchTool().execute({"query": ""}, _ctx())
        assert result.is_error
        assert "must not be empty" in result.content

    @pytest.mark.asyncio
    async def test_whitespace_query_error(self):
        result = await WebSearchTool().execute({"query": "   "}, _ctx())
        assert result.is_error

    @pytest.mark.asyncio
    async def test_missing_ddgs_gives_install_hint(self, monkeypatch):
        monkeypatch.setattr(
            "xgen_agent_runtime.tools.built_in.web_search_tool._load_ddgs",
            lambda: None,
        )
        result = await WebSearchTool().execute({"query": "x"}, _ctx())
        assert result.is_error
        assert "pip install" in result.content
        assert "xgen-agent-runtime[web]" in result.content

    @pytest.mark.asyncio
    async def test_ddg_exception_surfaces_as_error_result(
        self, monkeypatch, ddgs_available
    ):
        def _boom(ddgs_cls, query, max_results, region, safesearch):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_boom))
        result = await WebSearchTool().execute({"query": "x"}, _ctx())
        assert result.is_error
        assert "web search failed" in result.content


class TestRegistry:
    def test_registered_in_built_in_tool_classes(self):
        from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_CLASSES, WebSearchTool

        assert "WebSearch" in BUILT_IN_TOOL_CLASSES
        assert BUILT_IN_TOOL_CLASSES["WebSearch"] is WebSearchTool


# ── region 기본값 — ddgs 에 'wt-wt' 를 넘기지 않는다 (4.42.0) ─────────────
#
# dev 2026-09-22: 한 세션 WebSearch 50회 중 22회가 "DNSError … wt.wikipedia.org".
# ddgs 9.x 는 region 접두사로 wikipedia 엔진 호스트를 만든다 — 우리 기본값
# 'wt-wt' 가 존재하지 않는 wt.wikipedia.org 가 되어 매번 실패하고, 다른 엔진이
# 막히면 검색 전체가 죽는다. region 은 호출자가 준 것만 넘긴다.


class TestRegionDefault:
    @pytest.mark.asyncio
    async def test_default_region_is_not_sent_to_ddgs(self, monkeypatch, ddgs_available):
        seen: Dict[str, Any] = {}

        def _capture(ddgs_cls, query, max_results, region, safesearch):
            seen["region"] = region
            return []

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_capture))
        await WebSearchTool().execute({"query": "제주은행 입찰공고"}, ToolContext(working_dir="/tmp"))
        assert seen["region"] is None  # 'wt-wt' 가 아니다

    @pytest.mark.asyncio
    async def test_explicit_region_is_passed_through(self, monkeypatch, ddgs_available):
        seen: Dict[str, Any] = {}

        def _capture(ddgs_cls, query, max_results, region, safesearch):
            seen["region"] = region
            return []

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_capture))
        await WebSearchTool().execute({"query": "q", "region": "kr-kr"}, ToolContext(working_dir="/tmp"))
        assert seen["region"] == "kr-kr"

    def test_real_search_sync_omits_region_kwarg_when_none(self):
        calls: List[Dict[str, Any]] = []

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def text(self, query, **kwargs):
                calls.append(kwargs)
                return []

        WebSearchTool._search_sync(_Client, "q", 5, None, "moderate")
        WebSearchTool._search_sync(_Client, "q", 5, "us-en", "moderate")
        assert "region" not in calls[0] and calls[1]["region"] == "us-en"


# ── 동시성 상한 — 프로세스당 2 (4.44.0) ────────────────────────────────
#
# dev 2026-09-22: 모델이 ToolBatch(parallel 8)로 검색 3개를 동시에 쏘자 yahoo RequestError·
# brave 429. 상한은 실행기가 아니라 도구 안에 두어 Stage 10 병렬·ToolBatch·서브에이전트
# 어느 경로에서든 같이 걸린다.


class TestConcurrencyCap:
    @pytest.mark.asyncio
    async def test_at_most_two_searches_run_at_once(self, monkeypatch, ddgs_available):
        import asyncio

        from xgen_agent_runtime.tools.built_in import web_search_tool as m

        state = {"active": 0, "peak": 0}

        class _SlowBackend:
            async def search(self, query, max_results, region, safesearch):
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                await asyncio.sleep(0.05)
                state["active"] -= 1
                return []

        monkeypatch.setattr(m, "build_backend", lambda *a, **k: _SlowBackend())
        tool = WebSearchTool()
        ctx = ToolContext(working_dir="/tmp")
        await asyncio.gather(*(tool.execute({"query": f"q{i}"}, ctx) for i in range(6)))
        assert state["peak"] == m._MAX_CONCURRENT_SEARCHES == 2


# ── 엔진이 막은 것 vs 진짜 결과 없음 — 한 번 더, 그래도 막히면 그렇다고 (4.81.0) ──────
#
# 2026-10-01 dev·stage·홈서버: google 429·brave 429·mojeek 403·duckduckgo 202 로 막히고,
# 남은 yahoo 는 절반이 ddgs 가 못 읽는 레이아웃. ddgs 는 non-200 을 빈 페이지와 똑같이 버려
# "No results found." 라고만 한다 — dev 실사용 101회 중 27회(09-29~10-01), 한 턴은 14번 검색에
# 7번 실패. 2초 뒤 한 번 더 하자 24건 중 18건이 결과를 받았다(세 번 측정).

_BLOCKED = [
    {"name": "wikipedia"},
    {"name": "google", "status": 429},
    {"name": "brave", "status": 429},
    {"name": "mojeek", "status": 403},
    {"name": "duckduckgo", "status": 202},
    {"name": "yahoo"},
]
_WEB_HITS = [{"name": "wikipedia"}, {"name": "google", "status": 429}, {"name": "yahoo", "hits": 7}]
_NOTHING_MATCHED = [{"name": "wikipedia"}, {"name": "google"}, {"name": "yahoo"}]
_LOOKUP_ONLY = [{"name": "wikipedia", "hits": 1}, {"name": "google", "status": 429}, {"name": "yahoo"}]
# brave·mojeek refuse every call from our servers; the rest answered with nothing.
_FEW_REFUSED = [
    {"name": "wikipedia"},
    {"name": "brave", "status": 429},
    {"name": "mojeek", "status": 403},
    {"name": "google"},
    {"name": "duckduckgo"},
    {"name": "yahoo"},
]
_UNREACHABLE = [
    {"name": "google", "error": TimeoutError("request timed out")},
    {"name": "yahoo", "error": ConnectionError("dns error")},
]


class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status_code = status


class _FakeHttp:
    def __init__(self, status: int, error: Exception | None) -> None:
        self._status, self._error = status, error

    def request(self, *args, **kwargs):
        if self._error is not None:
            raise self._error
        return _FakeResp(self._status)


class _FakeEngine:
    """Stands in for a ddgs engine: non-200 → ``None``, like ``BaseSearchEngine``."""

    def __init__(self, name: str, status: int = 200, hits: int = 0, error: Exception | None = None):
        self.name = name
        self.http_client = _FakeHttp(status, error)
        self._hits = hits

    def search(self, query, **kwargs):
        if self.http_client.request("GET", "https://engine.example").status_code != 200:
            return None
        return [
            {"title": f"{self.name} {i}", "href": f"https://{self.name}.example/{i}", "body": "b"}
            for i in range(self._hits)
        ]


def _fake_ddgs(*plans):
    """A DDGS stand-in; the n-th client runs the n-th engine plan (last one repeats)."""
    made: List[Any] = []

    class _DDGS:
        def __init__(self):
            self._plan = plans[min(len(made), len(plans) - 1)]
            made.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def _get_engines(self, category, backend):
            return [_FakeEngine(**spec) for spec in self._plan]

        def text(self, query, **kwargs):
            results: List[Dict[str, Any]] = []
            err = None
            for engine in self._get_engines("text", "auto"):
                try:
                    found = engine.search(query)
                except Exception as ex:  # ddgs logs and moves on
                    err = ex
                    continue
                results.extend(found or [])
            if results:
                return results
            raise RuntimeError(err or "No results found.")

    _DDGS.made = made
    return _DDGS


@pytest.fixture
def fake_ddgs(monkeypatch):
    from xgen_agent_runtime.tools.built_in import _web_search_backends as backends

    monkeypatch.setattr(backends, "_DDG_RETRY_PAUSE_S", 0)

    def _install(*plans):
        cls = _fake_ddgs(*plans)
        monkeypatch.setattr(
            "xgen_agent_runtime.tools.built_in.web_search_tool._load_ddgs", lambda: cls
        )
        return cls

    return _install


class TestTurnedAway:
    @pytest.mark.asyncio
    async def test_refused_twice_says_so_instead_of_no_results(self, fake_ddgs):
        cls = fake_ddgs(_BLOCKED)
        result = await WebSearchTool().execute({"query": "2026년 최저임금"}, _ctx())
        assert result.is_error
        assert len(cls.made) == 2
        assert "No results" not in result.content
        assert "refused this server's requests" in result.content
        for part in ("google HTTP 429", "brave HTTP 429", "mojeek HTTP 403", "duckduckgo HTTP 202"):
            assert part in result.content
        assert "yahoo answered without a usable result" in result.content
        assert "rewording it will not help" in result.content and "WebFetch" in result.content
        assert "wikipedia" not in result.content  # title lookups say nothing about blocking
        assert result.metadata["engines"]["attempts"] == 2

    @pytest.mark.asyncio
    async def test_retry_gets_through(self, fake_ddgs):
        cls = fake_ddgs(_BLOCKED, _WEB_HITS)
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert not result.is_error
        assert len(cls.made) == 2
        assert result.metadata["results_count"] == 7
        assert "Note:" not in result.content
        assert result.metadata["engines"]["found"] == {"yahoo": 7}

    @pytest.mark.asyncio
    async def test_nothing_matched_is_plain_no_results_without_retry(self, fake_ddgs):
        cls = fake_ddgs(_NOTHING_MATCHED)
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert not result.is_error
        assert result.content == "No results for 'q'."
        assert len(cls.made) == 1

    @pytest.mark.asyncio
    async def test_a_few_refusals_next_to_empty_answers_is_still_no_results(self, fake_ddgs):
        cls = fake_ddgs(_FEW_REFUSED)
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert not result.is_error
        assert result.content == "No results for 'q'."
        assert len(cls.made) == 1
        assert result.metadata["engines"]["refused"] == {"brave": "HTTP 429", "mojeek": "HTTP 403"}

    @pytest.mark.asyncio
    async def test_web_hits_first_time_no_retry_no_note(self, fake_ddgs):
        cls = fake_ddgs(_WEB_HITS)
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert len(cls.made) == 1
        assert result.metadata["results_count"] == 7
        assert "Note:" not in result.content
        assert result.metadata["engines"]["refused"] == {"google": "HTTP 429"}

    @pytest.mark.asyncio
    async def test_lookup_only_hits_come_with_a_note(self, fake_ddgs):
        cls = fake_ddgs(_LOOKUP_ONLY)
        result = await WebSearchTool().execute({"query": "삼성전자 주가"}, _ctx())
        assert not result.is_error
        assert len(cls.made) == 2
        assert result.metadata["results_count"] == 1
        assert "Note: only encyclopedia lookups returned results" in result.content
        assert "google HTTP 429" in result.content

    @pytest.mark.asyncio
    async def test_lookup_hit_then_web_hits_on_retry(self, fake_ddgs):
        fake_ddgs(_LOOKUP_ONLY, _WEB_HITS)
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert result.metadata["results_count"] == 7
        assert "Note:" not in result.content

    @pytest.mark.asyncio
    async def test_unreachable_engines(self, fake_ddgs):
        fake_ddgs(_UNREACHABLE)
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert result.is_error
        assert "refused or did not answer" in result.content
        assert "on this server's side" in result.content
        assert "google timeout" in result.content and "yahoo unreachable" in result.content

    @pytest.mark.asyncio
    async def test_client_without_engine_hooks_keeps_ddgs_error(self, monkeypatch):
        class _Opaque:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def text(self, query, **kwargs):
                raise RuntimeError("No results found.")

        monkeypatch.setattr(
            "xgen_agent_runtime.tools.built_in.web_search_tool._load_ddgs", lambda: _Opaque
        )
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert result.is_error
        assert result.content == "web search failed: No results found."

    def test_answered_page_that_fails_to_parse_is_not_a_refusal(self):
        from xgen_agent_runtime.tools.built_in._web_search_backends import _EngineWatch

        watch = _EngineWatch()
        watch._note("yahoo", status=200)
        watch._note("yahoo", error=ValueError("bad markup"))
        watch._note("brave", status=429)
        out = watch.outcome()
        assert out["empty"] == ["yahoo"] and out["refused"] == {"brave": "HTTP 429"}
