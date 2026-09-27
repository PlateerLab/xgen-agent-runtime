"""thinking 블록 되돌려 보내기 계약 — 서명까지 그대로.

Anthropic 은 도구 루프에서 직전 assistant 의 thinking 블록을 수정 없이(signature 포함)
요구한다. 응답을 읽을 때 signature 를 버리면 다음 호출이
400 ``messages.1.content.0.thinking.signature: Field required`` 로 거절돼 턴이 첫 도구 호출
뒤에 끝났다(claude-sonnet-5·opus-5 — 기본으로 thinking 을 돌려준다).
redacted_thinking 도 같은 이유로 그대로 되돌려 보낸다.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from anthropic.types import Message, Usage

from xgen_agent_runtime.llm_client.anthropic import AnthropicClient
from xgen_agent_runtime.llm_client.translators._canonical import canonical_messages_to_anthropic
from xgen_agent_runtime.stages.s06_api.artifact.default.tool_loop import assistant_content_blocks


def _message(content):
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=content,
        model="claude-sonnet-5",
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=4),
    )


def test_thinking_signature_and_redacted_blocks_survive_the_round_trip():
    raw = _message(
        [
            {"type": "thinking", "thinking": "list the folder first", "signature": "sig-abc"},
            {"type": "redacted_thinking", "data": "opaque-blob"},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ]
    )
    response = AnthropicClient(api_key="k")._parse_response(raw)

    replay = assistant_content_blocks(response)
    assert replay[0] == {
        "type": "thinking",
        "thinking": "list the folder first",
        "signature": "sig-abc",
    }
    assert replay[1] == {"type": "redacted_thinking", "data": "opaque-blob"}
    assert replay[2]["type"] == "tool_use"

    # 요청 조립(sanitize)도 서명을 지우지 않는다.
    wire = canonical_messages_to_anthropic([{"role": "assistant", "content": replay}])
    assert wire[0]["content"][0]["signature"] == "sig-abc"
    assert wire[0]["content"][1] == {"type": "redacted_thinking", "data": "opaque-blob"}


def test_thinking_text_is_still_exposed_for_display():
    raw = _message([{"type": "thinking", "thinking": "plan", "signature": "s"}, {"type": "text", "text": "ok"}])
    response = AnthropicClient(api_key="k")._parse_response(raw)
    assert [b.thinking_text for b in response.thinking_blocks] == ["plan"]
