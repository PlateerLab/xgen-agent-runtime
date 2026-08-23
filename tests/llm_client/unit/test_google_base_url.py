"""GoogleClient / VertexClient honour ``base_url`` + ``default_headers``
via ``genai.Client(http_options=...)`` — and warn ONCE when the SDK
resolves a different endpoint instead of silently falling back."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from xgen_agent_runtime.llm_client.google import GoogleClient
from xgen_agent_runtime.llm_client.vertex import VertexClient


def _stub_genai(monkeypatch, captured, *, effective_base_url=None):
    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            http = (kwargs.get("http_options") or {})
            base = http.get("base_url") if effective_base_url is None else effective_base_url
            self._api_client = types.SimpleNamespace(
                _http_options=types.SimpleNamespace(base_url=base)
            )

    fake = types.SimpleNamespace(Client=_FakeClient)
    import google

    monkeypatch.setattr(google, "genai", fake, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake)


def test_google_without_base_url_builds_bare_client(monkeypatch):
    captured = {}
    _stub_genai(monkeypatch, captured)
    GoogleClient(api_key="k")._get_client()
    assert captured == {"api_key": "k"}


def test_google_base_url_and_headers_become_http_options(monkeypatch):
    captured = {}
    _stub_genai(monkeypatch, captured)
    GoogleClient(
        api_key="k", base_url="https://gw.example/gemini", default_headers={"X-A": "1"}
    )._get_client()
    assert captured["api_key"] == "k"
    assert captured["http_options"] == {
        "base_url": "https://gw.example/gemini",
        "headers": {"X-A": "1"},
    }


def test_google_configure_base_url_rebuilds_client(monkeypatch):
    captured = {}
    _stub_genai(monkeypatch, captured)
    client = GoogleClient(api_key="k")
    client._get_client()
    assert "http_options" not in captured
    client.configure(base_url="https://gw.example/v2")
    client._get_client()
    assert captured["http_options"]["base_url"] == "https://gw.example/v2"


def test_vertex_adc_channel_passes_http_options(monkeypatch):
    captured = {}
    _stub_genai(monkeypatch, captured)
    VertexClient(
        project="p-1", location="asia-northeast3", base_url="https://gw.example/vertex/"
    )._get_client()
    assert captured == {
        "vertexai": True,
        "project": "p-1",
        "location": "asia-northeast3",
        "http_options": {"base_url": "https://gw.example/vertex/"},
    }


def test_vertex_express_channel_passes_http_options(monkeypatch):
    captured = {}
    _stub_genai(monkeypatch, captured)
    VertexClient(api_key="express", base_url="https://gw.example/vertex")._get_client()
    assert captured == {
        "vertexai": True,
        "api_key": "express",
        "http_options": {"base_url": "https://gw.example/vertex"},
    }


def test_vertex_without_base_url_unchanged(monkeypatch):
    captured = {}
    _stub_genai(monkeypatch, captured)
    VertexClient(project="p-1")._get_client()
    assert captured == {"vertexai": True, "project": "p-1", "location": "us-central1"}


def test_unhonoured_base_url_warns_once(monkeypatch, caplog):
    """An SDK that ignores the override (older google-genai on Vertex)
    must produce a loud WARNING — exactly one per client, not per call."""
    captured = {}
    _stub_genai(
        monkeypatch, captured, effective_base_url="https://us-central1-aiplatform.googleapis.com/"
    )
    client = VertexClient(project="p-1", base_url="https://gw.example/vertex")
    with caplog.at_level(logging.WARNING, logger="xgen_agent_runtime.llm_client.google"):
        client._get_client()
        client._client = None  # force a rebuild — the warning must not repeat
        client._get_client()
    warnings = [r for r in caplog.records if "NOT honoured" in r.getMessage()]
    assert len(warnings) == 1
    assert "https://gw.example/vertex" in warnings[0].getMessage()


def test_honoured_base_url_does_not_warn(monkeypatch, caplog):
    captured = {}
    _stub_genai(monkeypatch, captured)
    client = GoogleClient(api_key="k", base_url="https://gw.example/gemini/")
    with caplog.at_level(logging.WARNING, logger="xgen_agent_runtime.llm_client.google"):
        client._get_client()
    assert not [r for r in caplog.records if "NOT honoured" in r.getMessage()]


def test_real_sdk_honours_base_url_for_both_apis():
    """Pin the SDK behaviour the warning guards: the installed google-genai
    resolves the custom base_url on both the Gemini and Vertex paths."""
    pytest.importorskip("google.genai")
    gem = GoogleClient(api_key="k", base_url="https://gw.example/gemini/")._get_client()
    assert gem._api_client._http_options.base_url == "https://gw.example/gemini/"
    vtx = VertexClient(
        project="p", location="us-central1", base_url="https://gw.example/vertex/"
    )._get_client()
    assert vtx._api_client._http_options.base_url == "https://gw.example/vertex/"
