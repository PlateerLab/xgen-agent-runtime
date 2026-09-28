"""구조화 출력 전달 — 표준 response_format 을 OpenAI 전송 형식으로.

예전엔 OpenAI 계열 요청 조립이 response_format 을 전달하지 않아 스키마가 지시문에만 있었다.
Qwen(vLLM) 은 키를 지어내 메모리 사실 추출이 매번 schema_mismatch 로 버려졌다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from xgen_agent_runtime.llm_client.openai import OpenAIClient, _response_format_to_openai
from xgen_agent_runtime.llm_client.openai_compatible import CustomOpenAIClient
from xgen_agent_runtime.llm_client.types import APIRequest

SCHEMA = {
    "type": "object",
    "required": ["upserts", "supersedes"],
    "properties": {"upserts": {"type": "array"}, "supersedes": {"type": "array"}},
}


def _req(rf):
    return APIRequest(model="qwen3.8-27b", messages=[{"role": "user", "content": "x"}], max_tokens=100, response_format=rf)


def test_canonical_schema_is_wrapped_with_a_name():
    assert _response_format_to_openai({"type": "json_schema", "json_schema": SCHEMA}) == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": SCHEMA},
    }


def test_already_wrapped_and_json_object_pass_through():
    wrapped = {"type": "json_schema", "json_schema": {"name": "facts", "schema": SCHEMA, "strict": True}}
    assert _response_format_to_openai(wrapped) == wrapped
    assert _response_format_to_openai({"type": "json_object"}) == {"type": "json_object"}
    assert _response_format_to_openai({"type": "text"}) is None


def test_compatible_client_sends_response_format():
    client = CustomOpenAIClient(api_key="EMPTY", base_url="http://localhost:1/v1")
    kwargs = client._build_kwargs(_req({"type": "json_schema", "json_schema": SCHEMA}))
    assert kwargs["response_format"]["json_schema"]["schema"] == SCHEMA


def test_client_without_structured_output_support_does_not_send_it():
    client = OpenAIClient(api_key="k")
    client.capabilities = client.capabilities.__class__(
        **{**client.capabilities.__dict__, "supports_structured_output": False}
    )
    assert "response_format" not in client._build_kwargs(_req({"type": "json_object"}))
