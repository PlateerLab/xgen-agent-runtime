"""Bedrock 류 응답의 usage None 필드 흡수 계약.

anthropic SDK 의 Usage 모델은 응답에 없는 캐시 필드를 **None** 으로 준다
(getattr 기본값은 속성이 존재하면 무력). Bedrock 응답은 캐시 필드를 생략하므로
None 이 TokenUsage 로 흐르면 토큰 회계가 TypeError 로 죽는다 — 추출 지점에서
0 으로 강제한다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from anthropic.types import Message, Usage

from xgen_agent_runtime.llm_client.anthropic import AnthropicClient


def test_usage_none_cache_fields_coerced_to_zero():
    raw = Message(
        id="msg_1", type="message", role="assistant",
        content=[{"type": "text", "text": "pong"}],
        model="claude-sonnet-4-5-20250929", stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=4),
    )
    assert raw.usage.cache_read_input_tokens is None  # SDK 계약 전제 확인

    client = AnthropicClient(api_key="k")
    response = client._parse_response(raw)
    assert response.usage.cache_read_input_tokens == 0
    assert response.usage.cache_creation_input_tokens == 0
    assert response.usage.input_tokens == 10
    # 회계가 실제로 산술을 돌릴 수 있는지 (None 이면 여기서 TypeError)
    assert response.usage.cache_read_input_tokens + response.usage.input_tokens == 10
