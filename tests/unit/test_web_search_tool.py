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
    # No real ddgs engine gets added to the fake client unless a test asks for one.
    monkeypatch.setenv("GENY_WEBSEARCH_DDG_EXTRA_ENGINES", "")

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


# ── ddgs 가 꺼 둔 yandex 를 우리 검색에서만 다시 켠다 (4.81.0) ────────────────────
#
# ddgs 9.15.0(2026-08-16)이 yandex 엔진을 이유 없이 껐다. dev·stage·홈서버에서 yandex 는
# 36개 질의 모두 결과(약 1.4초)였고, 켜고 돌리면 dev·stage 첫 시도 24/24 성공.
# ddgs 전역 레지스트리는 건드리지 않고 이번 검색의 DDGS 인스턴스에만 넣는다. 러시아 서비스라
# 정책상 안 되는 호스트는 GENY_WEBSEARCH_DDG_EXTRA_ENGINES="" 로 끈다.


class _FakeYandex(_FakeEngine):
    started = 0

    def __init__(self, proxy=None, timeout=None, verify=True):
        type(self).started += 1
        super().__init__("yandex", hits=10)


class _Ranked:
    def __init__(self, name: str, priority: float) -> None:
        self.name, self.priority = name, priority


class _EnginesClient:
    def __init__(self, engines):
        self._engines = engines

    def _get_engines(self, category, backend):
        return list(self._engines)


class TestExtraEngines:
    def test_default_is_yandex(self, monkeypatch):
        from xgen_agent_runtime.tools.built_in._web_search_backends import _ddg_extra_engines

        monkeypatch.delenv("GENY_WEBSEARCH_DDG_EXTRA_ENGINES", raising=False)
        assert _ddg_extra_engines(_ctx()) == ("yandex",)

    def test_env_empty_turns_it_off_and_extras_win(self, monkeypatch):
        from xgen_agent_runtime.tools.built_in._web_search_backends import _ddg_extra_engines

        monkeypatch.setenv("GENY_WEBSEARCH_DDG_EXTRA_ENGINES", "")
        assert _ddg_extra_engines(_ctx()) == ()
        monkeypatch.setenv("GENY_WEBSEARCH_DDG_EXTRA_ENGINES", " Yandex , foo ")
        assert _ddg_extra_engines(_ctx()) == ("yandex", "foo")
        ctx = ToolContext(working_dir="", extras={"web_search": {"ddg_extra_engines": []}})
        assert _ddg_extra_engines(ctx) == ()

    @pytest.mark.asyncio
    async def test_added_engine_answers_when_the_others_refuse(self, monkeypatch, fake_ddgs):
        from xgen_agent_runtime.tools.built_in import _web_search_backends as backends

        monkeypatch.setenv("GENY_WEBSEARCH_DDG_EXTRA_ENGINES", "yandex")
        monkeypatch.setattr(
            backends, "_ddg_engine_class", lambda name: _FakeYandex if name == "yandex" else None
        )
        cls = fake_ddgs(_BLOCKED)
        result = await WebSearchTool().execute({"query": "2026년 최저임금"}, _ctx())
        assert not result.is_error
        assert len(cls.made) == 1  # no retry needed
        assert result.metadata["engines"]["found"] == {"yandex": 10}

    def test_engine_ddgs_already_runs_is_not_added_twice(self, monkeypatch):
        from xgen_agent_runtime.tools.built_in import _web_search_backends as backends

        monkeypatch.setattr(backends, "_ddg_engine_class", lambda name: _FakeYandex)
        _FakeYandex.started = 0
        client = _EnginesClient([_Ranked("wikipedia", 2), _Ranked("yandex", 1)])
        backends._add_ddg_engines(client, ("yandex",))
        assert [e.name for e in client._get_engines("text", "auto")] == ["wikipedia", "yandex"]
        assert _FakeYandex.started == 0

    def test_lookups_stay_first_and_other_categories_untouched(self, monkeypatch):
        from xgen_agent_runtime.tools.built_in import _web_search_backends as backends

        monkeypatch.setattr(backends, "_ddg_engine_class", lambda name: _FakeYandex)
        engines = [_Ranked("wikipedia", 2), _Ranked("grokipedia", 1.9), _Ranked("brave", 1), _Ranked("yahoo", 1)]
        client = _EnginesClient(engines)
        backends._add_ddg_engines(client, ("yandex",))
        for _ in range(20):
            names = [e.name for e in client._get_engines("text", "auto")]
            assert names[:2] == ["wikipedia", "grokipedia"]
            assert sorted(names[2:]) == ["brave", "yahoo", "yandex"]
        assert [e.name for e in client._get_engines("news", "auto")] == [e.name for e in engines]
        assert [e.name for e in client._get_engines("text", "brave")] == [e.name for e in engines]


# ---------------------------------------------------------------------------
# days → dated news search
# ---------------------------------------------------------------------------

from datetime import date, datetime, timedelta, timezone  # noqa: E402

from xgen_agent_runtime.tools.built_in import _web_search_backends as wsb  # noqa: E402

_NOW = datetime(2026, 10, 3, 22, 0, tzinfo=timezone(timedelta(hours=9)))


class TestNewsDate:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2026. 9. 28.", date(2026, 9, 28)),  # bing kr-kr — ddgs reads 2026 days ago
            ("2026년 9월 28일", date(2026, 9, 28)),
            ("4 hours ago", date(2026, 10, 3)),  # ddgs reads four days ago
            ("1 day ago", date(2026, 10, 2)),
            ("Opinion12 days ago", date(2026, 9, 21)),  # ddgs leaves it as is
            ("an hour ago", date(2026, 10, 3)),
            ("3시간 전", date(2026, 10, 3)),
            ("2일 전", date(2026, 10, 1)),
            ("8/30/2026", date(2026, 8, 30)),
            ("Opinion9/13/2026", date(2026, 9, 13)),
            ("Sep 28, 2026", date(2026, 9, 28)),
            ("2026-09-28T13:55:33+00:00", date(2026, 9, 28)),
            ("2026-09-28", date(2026, 9, 28)),
            ("yesterday", date(2026, 10, 2)),
            ("", None),
            ("Opinion", None),
            ("2026. 13. 40.", None),
        ],
    )
    def test_reads_what_the_engines_show(self, raw, expected):
        assert wsb._news_date(raw, now=_NOW) == expected

    def test_epoch_seconds(self):
        ts = int(datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc).timestamp())
        assert wsb._news_date(ts, now=_NOW) == date(2026, 9, 28)
        assert wsb._news_date(str(ts), now=_NOW) == date(2026, 9, 28)


class _R:
    def __init__(self, raw: str) -> None:
        self.date = raw


class _Bing:
    """bing news as ddgs builds it: d → interval 4 (an hour), dates read the ddgs way."""

    name = "bing"

    def build_payload(self, **kwargs):
        return {"q": kwargs.get("query"), "qft": 'interval="4"'}

    def post_extract_results(self, results):
        for r in results:
            r.date = "2021-03-17T13:55:33+00:00"  # what ddgs makes of "2026. 9. 28."
        return results


class _OneEngineClient:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _get_engines(self, category, backend):
        return [self.engine]


class TestNewsEngines:
    @pytest.mark.parametrize("days, tl", [(1, "d"), (7, "w"), (30, "m"), (31, "m"), (90, None)])
    def test_timelimit(self, days, tl):
        assert wsb._news_timelimit(days) == tl

    @pytest.mark.parametrize(
        "days, qft",
        [(1, 'interval="7"'), (7, 'interval="8"'), (30, 'interval="9"'), (90, None)],
    )
    def test_bing_gets_the_window_it_understands(self, days, qft):
        client = _OneEngineClient(_Bing())
        wsb._date_news_engines(client, days)
        engine = client._get_engines("news", "auto")[0]
        assert engine.build_payload(query="q").get("qft") == qft

    def test_dates_are_read_from_the_engine_text(self):
        client = _OneEngineClient(_Bing())
        wsb._date_news_engines(client, 30)
        engine = client._get_engines("news", "auto")[0]
        out = engine.post_extract_results([_R("2026. 9. 28."), _R("no date")])
        assert [r.date for r in out] == ["2026-09-28", ""]

    def test_text_engines_are_left_alone(self):
        engine = _Bing()
        client = _OneEngineClient(engine)
        wsb._date_news_engines(client, 30)
        client._get_engines("text", "auto")
        assert engine.build_payload(query="q")["qft"] == 'interval="4"'


def _ago(days: int) -> str:
    return (datetime.now().astimezone() - timedelta(days=days)).date().isoformat()


def _news_ddgs(news_items=(), text_items=(), news_error=None):
    calls: List[Any] = []

    class _DDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def _get_engines(self, category, backend):
            return []

        def news(self, query, **kwargs):
            calls.append(("news", kwargs))
            if news_error is not None:
                raise news_error
            return [dict(item) for item in news_items]

        def text(self, query, **kwargs):
            calls.append(("text", kwargs))
            return [dict(item) for item in text_items]

    _DDGS.calls = calls
    return _DDGS


@pytest.fixture
def news_ddgs(monkeypatch):
    monkeypatch.setenv("GENY_WEBSEARCH_DDG_EXTRA_ENGINES", "")

    def _install(**kwargs):
        cls = _news_ddgs(**kwargs)
        monkeypatch.setattr(
            "xgen_agent_runtime.tools.built_in.web_search_tool._load_ddgs", lambda: cls
        )
        return cls

    return _install


def _news(title: str, published: str, source: str = "연합뉴스") -> Dict[str, Any]:
    return {
        "title": title,
        "url": f"https://news.example/{title}",
        "body": "본문",
        "date": published,
        "source": source,
    }


class TestDaysInput:
    def test_schema_has_days(self):
        prop = WebSearchTool().input_schema["properties"]["days"]
        assert prop["type"] == "integer" and prop["exclusiveMinimum"] == 0

    @pytest.mark.asyncio
    async def test_days_searches_dated_news_in_the_window(self, news_ddgs):
        cls = news_ddgs(
            news_items=[
                _news("fresh", _ago(3)),
                _news("old", _ago(100)),
                _news("undated", ""),
            ]
        )
        result = await WebSearchTool().execute({"query": "신제품 출시", "days": 30}, _ctx())
        assert not result.is_error
        assert result.content.startswith("News from the last 30 days for '신제품 출시'")
        assert f"   {_ago(3)} · 연합뉴스" in result.content
        assert "old" not in result.content and "undated" in result.content
        assert result.metadata["mode"] == "news" and result.metadata["days"] == 30
        assert cls.calls == [
            ("news", {"safesearch": "moderate", "max_results": 20, "timelimit": "m"})
        ]

    @pytest.mark.asyncio
    async def test_longer_than_a_month_filters_by_date_only(self, news_ddgs):
        cls = news_ddgs(news_items=[_news("q2", _ago(60)), _news("old", _ago(120)), _news("undated", "")])
        result = await WebSearchTool().execute({"query": "실적", "days": 90}, _ctx())
        assert "q2" in result.content
        assert "old" not in result.content and "undated" not in result.content
        assert "timelimit" not in cls.calls[0][1]

    @pytest.mark.asyncio
    async def test_no_news_falls_back_to_web_and_says_so(self, news_ddgs):
        cls = news_ddgs(
            news_items=[_news("old", _ago(400))],
            text_items=[{"title": "page", "href": "https://w.example", "body": "b"}],
        )
        result = await WebSearchTool().execute({"query": "site:mfds.go.kr 공지", "days": 30}, _ctx())
        assert result.content.startswith(
            "No news from the last 30 days for 'site:mfds.go.kr 공지'; web results instead"
        )
        assert result.metadata["mode"] == "web"
        assert [c[0] for c in cls.calls] == ["news", "text"]

    @pytest.mark.asyncio
    async def test_news_failure_falls_back_to_web(self, news_ddgs):
        news_ddgs(
            news_error=RuntimeError("No results found."),
            text_items=[{"title": "page", "href": "https://w.example", "body": "b"}],
        )
        result = await WebSearchTool().execute({"query": "x", "days": 7}, _ctx())
        assert not result.is_error and "web results instead" in result.content

    @pytest.mark.asyncio
    async def test_without_days_nothing_changes(self, news_ddgs):
        cls = news_ddgs(text_items=[{"title": "page", "href": "https://w.example", "body": "b"}])
        result = await WebSearchTool().execute({"query": "x"}, _ctx())
        assert result.content.startswith("Search results for 'x'")
        assert [c[0] for c in cls.calls] == ["text"] and "mode" not in result.metadata

    @pytest.mark.asyncio
    async def test_days_must_be_a_number(self, news_ddgs):
        news_ddgs()
        result = await WebSearchTool().execute({"query": "x", "days": "recent"}, _ctx())
        assert result.is_error and "whole number of days" in result.content

    @pytest.mark.asyncio
    async def test_backend_without_dates_says_days_was_ignored(self, monkeypatch):
        class _Plain:
            name = "searxng"

            async def search(self, query, max_results, region, safesearch):
                return [{"rank": 0, "title": "t", "url": "https://u.example", "snippet": "s"}]

        monkeypatch.setattr(
            "xgen_agent_runtime.tools.built_in.web_search_tool.build_backend",
            lambda *a, **k: _Plain(),
        )
        result = await WebSearchTool().execute(
            {"query": "x", "days": 7, "backend": "searxng"}, _ctx()
        )
        assert result.content.startswith("Search results for 'x'")
        assert "cannot filter by date" in result.content
