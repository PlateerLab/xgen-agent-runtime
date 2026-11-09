"""같은 tool_use id 는 응답에서 한 번만 받는다.

두 번 실리면 도구가 두 번 돌고, 그 응답을 담은 다음 요청이
400 ``messages.N.content.M: tool_use ids must be unique`` 로 거절된다
(2026-11-09 Agent-Geny 에서 claude-sonnet-5-5 를 쓰면 났다).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from anthropic.lib.streaming._messages import accumulate_event
from anthropic.types import Message, RawMessageStreamEvent, Usage
from anthropic._models import construct_type

from xgen_agent_runtime.llm_client.anthropic import AnthropicClient
from xgen_agent_runtime.stages.s06_api.artifact.default.providers import AnthropicProvider
from xgen_agent_runtime.stages.s06_api.artifact.default.tool_loop import assistant_content_blocks


def _message(content, model="claude-sonnet-5-5"):
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=content,
        model=model,
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=4),
    )


def _tool_ids(blocks):
    return [b["id"] for b in blocks if b.get("type") == "tool_use"]


def test_a_repeated_tool_use_is_received_once():
    raw = _message(
        [
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"path": "a"}},
        ]
    )
    for parser in (AnthropicClient(api_key="k"), AnthropicProvider(api_key="k")):
        response = parser._parse_response(raw)
        blocks = assistant_content_blocks(response)
        assert _tool_ids(blocks) == ["toolu_1", "toolu_2"]
        assert [c.tool_use_id for c in response.content if c.type == "tool_use"] == ["toolu_1", "toolu_2"]
        assert blocks[0]["signature"] == "sig", "thinking 은 그대로"


def test_the_copy_with_input_wins_over_an_empty_one():
    raw = _message(
        [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ]
    )
    response = AnthropicClient(api_key="k")._parse_response(raw)
    (call,) = [c for c in response.content if c.type == "tool_use"]
    assert call.tool_input == {"command": "ls"}
    assert assistant_content_blocks(response)[0]["input"] == {"command": "ls"}


def test_distinct_tool_uses_are_untouched():
    raw = _message(
        [
            {"type": "text", "text": "two at once"},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {"command": "pwd"}},
        ]
    )
    blocks = assistant_content_blocks(AnthropicClient(api_key="k")._parse_response(raw))
    assert [b["type"] for b in blocks] == ["text", "tool_use", "tool_use"]
    assert _tool_ids(blocks) == ["toolu_1", "toolu_2"]


def _event(value):
    return construct_type(type_=RawMessageStreamEvent, value=value)


def test_a_block_start_sent_twice_in_a_stream_still_gives_one_call():
    """SDK 의 스트림 조립은 content_block_start 마다 블록을 덧붙인다(인덱스를 보지 않는다).

    같은 블록의 시작이 두 번 오면 같은 id 의 tool_use 가 둘이 되고, 입력은 처음 것에만 채워진다.
    """
    start = {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}}
    events = [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "claude-sonnet-5-5",
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}},
        start,
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'}},
        start,
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 5}},
    ]
    snapshot = None
    for value in events:
        snapshot = accumulate_event(event=_event(value), current_snapshot=snapshot)
    assert [b.id for b in snapshot.content] == ["toolu_1", "toolu_1"], "SDK 는 둘로 만든다"

    blocks = assistant_content_blocks(AnthropicClient(api_key="k")._parse_response(snapshot))
    assert blocks == [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]
