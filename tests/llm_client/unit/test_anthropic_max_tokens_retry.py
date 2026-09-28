"""출력 한도에 걸려 쓸 수 있는 출력이 없을 때 한 번 더 — Claude 5 adaptive thinking.

Claude 5 계열은 기본으로 thinking 을 하고 그 토큰도 max_tokens 안에서 쓴다. 노드 기본값
8192 로는 생각만 하다 한도에 닿아 텍스트도 도구 호출도 없이 끝났다(로컬 벤치 70과제 중 8).
그 경우에만 한도를 올려 한 번 더 부른다. 텍스트가 이미 나갔으면 다시 부르지 않는다.
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from xgen_agent_runtime.llm_client import anthropic as mod
from xgen_agent_runtime.llm_client.anthropic import AnthropicClient


def _msg(stop: str, blocks: List[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0),
        stop_reason=stop,
        model="claude-sonnet-5",
        id="msg_1",
    )


THINKING_ONLY = _msg("max_tokens", [SimpleNamespace(type="thinking", thinking="", signature="s")])
ANSWER = _msg("end_turn", [SimpleNamespace(type="text", text="done")])


class _Stream:
    def __init__(self, events, final):
        self._events, self._final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def _gen():
            for e in self._events:
                yield e

        return _gen()

    async def get_final_message(self):
        return self._final


class _StreamClient:
    def __init__(self, scripted):
        self.calls: List[dict] = []
        self._scripted = list(scripted)

        def _stream(**kw):
            self.calls.append(kw)
            events, final = self._scripted.pop(0)
            return _Stream(events, final)

        self.messages = SimpleNamespace(stream=_stream)


def _cfg(max_tokens: int = 8192) -> SimpleNamespace:
    return SimpleNamespace(
        model="claude-sonnet-5", max_tokens=max_tokens, temperature=None, top_p=None, top_k=None,
        stop_sequences=None, thinking_enabled=False, thinking_type="enabled",
        thinking_budget_tokens=0, thinking_display=None,
    )


async def _run(client, cfg):
    return [c async for c in client.create_message_stream(model_config=cfg, messages=[{"role": "user", "content": "q"}])]


def _text(s: str) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=s))


@pytest.mark.asyncio
async def test_stream_retries_once_with_more_room_when_nothing_usable_came_back():
    client = AnthropicClient(api_key="k")
    fake = _StreamClient([([], THINKING_ONLY), ([_text("done")], ANSWER)])
    client._client = fake
    chunks = await _run(client, _cfg())
    assert [c["max_tokens"] for c in fake.calls] == [8192, mod._TRUNCATION_RETRY_MAX_TOKENS_STREAM]
    assert chunks[-1]["type"] == "message_complete"
    assert chunks[-1]["response"].text == "done"


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_text_reached_the_user():
    partial = _msg("max_tokens", [SimpleNamespace(type="text", text="half an ans")])
    client = AnthropicClient(api_key="k")
    fake = _StreamClient([([_text("half an ans")], partial)])
    client._client = fake
    chunks = await _run(client, _cfg())
    assert len(fake.calls) == 1
    assert chunks[-1]["response"].stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_stream_does_not_retry_when_already_at_the_ceiling():
    client = AnthropicClient(api_key="k")
    fake = _StreamClient([([], THINKING_ONLY)])
    client._client = fake
    await _run(client, _cfg(max_tokens=mod._TRUNCATION_RETRY_MAX_TOKENS_STREAM))
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_truncated_tool_use_is_not_treated_as_usable_output():
    truncated = _msg("max_tokens", [SimpleNamespace(type="tool_use", id="t", name="Bash", input={})])
    assert mod._truncated_without_output(truncated) is True
    assert mod._truncated_without_output(ANSWER) is False


@pytest.mark.asyncio
async def test_send_retries_once_and_keeps_the_original_if_the_retry_fails():
    client = AnthropicClient(api_key="k")
    seen: List[int] = []

    async def create(**kw):
        seen.append(kw["max_tokens"])
        if len(seen) == 1:
            return THINKING_ONLY
        raise RuntimeError("boom")

    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    request = client._build_request(model_config=_cfg(), messages=[{"role": "user", "content": "q"}], system="", tools=None, tool_choice=None, stream=False)
    response = await client._send(request)
    assert seen == [8192, mod._TRUNCATION_RETRY_MAX_TOKENS_SEND]
    assert response.stop_reason == "max_tokens"
