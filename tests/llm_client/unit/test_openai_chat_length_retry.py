"""Chat Completions 가 생각만 하다 출력 상한에 닿았을 때 한 번 더 (OpenRouter·vLLM·OpenAI 호환).

OpenRouter qwen3.8-flash(max_tokens 4096, thinking auto)가 긴 화면 결과 뒤에 53초 동안 생각만 하다
상한에 닿아 글도 도구 호출도 없이 끝났고, 루프는 그 빈 답으로 턴을 "완료" 로 닫았다(dev 10/26).
Anthropic·Responses 경로에는 있던 "상한을 넓혀 한 번 더" 가 Chat Completions 경로에만 없었다.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

openai_sdk = pytest.importorskip("openai")

from xgen_agent_runtime.core.config import ModelConfig  # noqa: E402
from xgen_agent_runtime.llm_client.openai import (  # noqa: E402
    _chat_truncated_without_output,
    _chat_with_more_room,
)
from xgen_agent_runtime.llm_client.openrouter import OpenRouterClient  # noqa: E402

MODEL = "qwen/qwen3.8-flash"


def _chunk(*, content: Any = None, tool_calls: Any = None, finish: Any = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(model=MODEL, choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


THINKING_ONLY = [_chunk(), _chunk(finish="length")]
ANSWER = [_chunk(content="done"), _chunk(finish="stop")]


def _stream_client(scripted: List[Any], calls: List[Dict[str, Any]]) -> OpenRouterClient:
    async def create(**kwargs: Any):
        calls.append(kwargs)
        step = scripted.pop(0)
        if isinstance(step, Exception):
            raise step

        async def _gen():
            for chunk in step:
                yield chunk

        return _gen()

    client = OpenRouterClient(api_key="sk-or-mock")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


async def _drain(client: OpenRouterClient, max_tokens: int = 4096) -> List[Dict[str, Any]]:
    return [
        evt
        async for evt in client.create_message_stream(
            model_config=ModelConfig(model=MODEL, max_tokens=max_tokens),
            messages=[{"role": "user", "content": "x"}],
        )
    ]


def _final(events: List[Dict[str, Any]]):
    done = [e for e in events if e["type"] == "message_complete"]
    assert len(done) == 1
    return done[0]["response"]


@pytest.mark.asyncio
async def test_stream_retries_once_with_double_room_when_thinking_hit_the_limit():
    calls: List[Dict[str, Any]] = []
    client = _stream_client([THINKING_ONLY, ANSWER], calls)
    response = _final(await _drain(client))
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 4096 and calls[1]["max_tokens"] == 8192
    assert [b.text for b in response.content if b.type == "text"] == ["done"]
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_stream_does_not_retry_when_text_already_went_out():
    calls: List[Dict[str, Any]] = []
    client = _stream_client([[_chunk(content="partial"), _chunk(finish="length")]], calls)
    response = _final(await _drain(client))
    assert len(calls) == 1
    assert response.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_stream_keeps_the_truncated_response_when_the_retry_fails():
    calls: List[Dict[str, Any]] = []
    client = _stream_client([THINKING_ONLY, RuntimeError("upstream 400")], calls)
    response = _final(await _drain(client))
    assert len(calls) == 2
    assert response.stop_reason == "max_tokens"
    assert not [b for b in response.content if b.type == "text"]


@pytest.mark.asyncio
async def test_send_retries_once_with_double_room():
    calls: List[Dict[str, Any]] = []
    replies = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None), finish_reason="length")], usage=None, model=MODEL, id="1"),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop")], usage=None, model=MODEL, id="2"),
    ]

    async def create(**kwargs: Any):
        calls.append(kwargs)
        return replies.pop(0)

    client = OpenRouterClient(api_key="sk-or-mock")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = await client.create_message(
        model_config=ModelConfig(model=MODEL, max_tokens=4096), messages=[{"role": "user", "content": "x"}]
    )
    assert [c["max_tokens"] for c in calls] == [4096, 8192]
    assert [b.text for b in response.content if b.type == "text"] == ["done"]


def test_room_doubles_up_to_the_ceiling_and_keeps_the_kwarg_name():
    assert _chat_with_more_room({"max_tokens": 4096}) == {"max_tokens": 8192}
    assert _chat_with_more_room({"max_completion_tokens": 20_000}) == {"max_completion_tokens": 32_000}
    assert _chat_with_more_room({"max_tokens": 32_000}) is None
    assert _chat_with_more_room({}) is None


def test_only_an_empty_answer_cut_at_the_limit_counts():
    assert _chat_truncated_without_output("length", "")
    assert _chat_truncated_without_output("length", None)
    assert not _chat_truncated_without_output("length", "답")
    assert not _chat_truncated_without_output("stop", "")
