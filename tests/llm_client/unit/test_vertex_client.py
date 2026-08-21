"""VertexClient — construction validation + auth-channel selection."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from xgen_agent_runtime.llm_client.vertex import VertexClient


def _stub_genai(monkeypatch, captured):
    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.SimpleNamespace(Client=_FakeClient)
    import google

    monkeypatch.setattr(google, "genai", fake, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake)


def test_requires_project_or_express_key():
    with pytest.raises(ValueError):
        VertexClient()
    VertexClient(project="p-1")           # ADC channel — fine
    VertexClient(api_key="express-key")   # express mode — fine


def test_adc_channel_passes_project_and_location(monkeypatch):
    captured = {}
    _stub_genai(monkeypatch, captured)
    VertexClient(project="p-1", location="asia-northeast3")._get_client()
    assert captured == {
        "vertexai": True,
        "project": "p-1",
        "location": "asia-northeast3",
    }


def test_express_key_channel_omits_project(monkeypatch):
    """Express-mode keys bind their own project — passing project/location
    alongside is rejected by the SDK, so we must not."""
    captured = {}
    _stub_genai(monkeypatch, captured)
    VertexClient(project="p-1", api_key="express")._get_client()
    assert captured == {"vertexai": True, "api_key": "express"}


def test_service_account_json_is_validated():
    client = VertexClient(project="p-1", credentials_json="{not json")
    with pytest.raises(ValueError):
        client._load_sa_credentials()
