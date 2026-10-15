"""Gemini/Vertex 가 구조화 출력을 실제로 요청에 싣는다.

``GoogleClient`` 는 ``supports_structured_output=True`` 를 선언해 왔지만 ``response_format`` 을
``GenerateContentConfig`` 에 옮기지 않았다 — 부르는 쪽은 JSON 이 강제된다고 믿고 산문을 받았다
(2026-09-30, 앱 LLM provider 감사). Vertex 는 같은 요청 조립을 물려받는다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai")

from xgen_agent_runtime.core.config import ModelConfig  # noqa: E402
from xgen_agent_runtime.llm_client.google import (  # noqa: E402
    GoogleClient,
    _json_schema_field,
    _structured_output_config,
)

SCHEMA = {"type": "object", "properties": {"colors": {"type": "array", "items": {"type": "string"}}}}


def _kwargs(client, response_format):
    request = client._build_request(
        model_config=ModelConfig(model="gemini-2.5-flash", max_tokens=64),
        messages=[{"role": "user", "content": "colors"}],
        system="s",
        tools=None,
        tool_choice=None,
        stream=False,
        response_format=response_format,
    )
    return client._build_kwargs(request)


def test_json_object_asks_for_a_json_reply():
    config = _kwargs(GoogleClient(api_key="k"), {"type": "json_object"})["config"]
    assert config["response_mime_type"] == "application/json"
    assert "response_json_schema" not in config and "response_schema" not in config


def test_json_schema_is_enforced_with_the_schema():
    config = _kwargs(GoogleClient(api_key="k"), {"type": "json_schema", "json_schema": SCHEMA})["config"]
    assert config["response_mime_type"] == "application/json"
    assert config[_json_schema_field()] == SCHEMA


def test_openai_style_nesting_is_accepted():
    nested = {"type": "json_schema", "json_schema": {"name": "out", "schema": SCHEMA}}
    assert _structured_output_config(nested)[_json_schema_field()] == SCHEMA


def test_no_format_no_change():
    assert _structured_output_config(None) == {}
    assert _structured_output_config({"type": "text"}) == {}
    assert "response_mime_type" not in _kwargs(GoogleClient(api_key="k"), None).get("config", {})


def test_the_config_is_valid_for_this_sdk():
    from google.genai import types

    config = _kwargs(GoogleClient(api_key="k"), {"type": "json_schema", "json_schema": SCHEMA})["config"]
    types.GenerateContentConfig(**config)  # the SDK accepts every field we send


def test_vertex_inherits_it():
    from xgen_agent_runtime.llm_client.vertex import VertexClient

    config = _kwargs(VertexClient(project="p"), {"type": "json_object"})["config"]
    assert config["response_mime_type"] == "application/json"
