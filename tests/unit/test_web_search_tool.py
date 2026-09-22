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
