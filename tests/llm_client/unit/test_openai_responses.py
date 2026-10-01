"""OpenAI 추론 모델은 Responses API 로 — 도구와 생각을 함께 (2026-10-01).

dev 실측: GPT-5.4 부터 Chat Completions 는 함수 도구와 생각을 함께 받지 않는다(400 "Function tools with
reasoning_effort are not supported … use /v1/responses"). gpt-6-sol 은 기본 강도가 medium 이라 생각을
고르지 않아도 도구만 있으면 실패했고, gpt-6-astra·6.1-sol 은 끌 수도 없어 도구를 쓸 수 없었다. Responses 는
10개 모델의 모든 강도에서 도구 호출 → 결과 → 답까지 받았다.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

pytest.importorskip("openai")

from xgen_agent_runtime.core.config import ModelConfig  # noqa: E402
from xgen_agent_runtime.llm_client import openai as openai_mod  # noqa: E402
from xgen_agent_runtime.llm_client.openai import OpenAIClient  # noqa: E402
from xgen_agent_runtime.llm_client.translators._canonical import (  # noqa: E402
    canonical_messages_to_anthropic,
    canonical_messages_to_openai,
)
from xgen_agent_runtime.llm_client.translators._responses import (  # noqa: E402
    canonical_to_responses_input,
)
from xgen_agent_runtime.llm_client.types import APIRequest  # noqa: E402

TOOLS = [
    {
        "name": "get_weather",
        "description": "Current weather.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]


def _req(**kw: Any) -> APIRequest:
    base: Dict[str, Any] = {
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "weather in Seoul?"}],
        "max_tokens": 8192,
        "system": "You are helpful.",
    }
    base.update(kw)
    return APIRequest(**base)


def _reasoning_block(model: str = "gpt-6-sol") -> Dict[str, Any]:
    return {
        "type": "reasoning",
        "provider": "openai",
        "model": model,
        "summary": "Need the tool.",
        "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"},
    }


def _history(model: str = "gpt-6-sol") -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "weather in Seoul?"},
        {
            "role": "assistant",
            "content": [
                _reasoning_block(model),
                {"type": "text", "text": "Checking.", "_meta": {"openai_phase": "commentary"}},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Seoul"},
                    "_meta": {"openai_item_id": "fc_1"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "18C clear"}],
        },
    ]


# ── 어느 표면으로 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,base_url,expected",
    [
        ("gpt-6-sol", None, True),
        ("gpt-6-astra", None, True),
        ("gpt-5.4", "https://api.openai.com/v1", True),
        ("gpt-5", None, True),
        ("o3-mini", None, True),
        ("gpt-4.1", None, False),
        ("gpt-4o-mini", None, False),
        ("gpt-6-sol", "https://llm-gateway.corp.local/v1", False),
    ],
)
def test_reasoning_models_on_the_official_endpoint_use_responses(model, base_url, expected):
    assert OpenAIClient(api_key="sk", base_url=base_url)._uses_responses(model) is expected


def test_surface_switch_and_subclasses(monkeypatch):
    from xgen_agent_runtime.llm_client.vllm import VLLMClient

    monkeypatch.setenv("XGEN_OPENAI_API_SURFACE", "chat")
    assert OpenAIClient(api_key="sk")._uses_responses("gpt-6-sol") is False
    monkeypatch.setenv("XGEN_OPENAI_API_SURFACE", "responses")
    assert OpenAIClient(api_key="sk", base_url="https://gw.local/v1")._uses_responses("gpt-4.1") is True
    # vLLM·호환 서버·Azure 는 언제나 Chat Completions
    assert VLLMClient(api_key="sk", base_url="http://vllm:8000/v1")._uses_responses("gpt-6-sol") is False


# ── 요청 모양 ─────────────────────────────────────────────────────────


def test_request_shape_tools_reasoning_and_budget():
    kw = OpenAIClient(api_key="sk")._build_responses_kwargs(
        _req(tools=TOOLS, tool_choice={"type": "any"}, thinking_level="high", temperature=0.3)
    )
    assert kw["store"] is False
    assert kw["instructions"] == "You are helpful."
    assert kw["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Current weather.",
            "parameters": TOOLS[0]["input_schema"],
            "strict": False,
        }
    ]
    assert kw["tool_choice"] == "required"
    assert kw["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kw["include"] == ["reasoning.encrypted_content"]
    # 생각할 자리(high 32768)를 답(8192) 위에 더 둔다
    assert kw["max_output_tokens"] == 8192 + 32768
    # 생각하는 동안 temperature 는 받지 않는다
    assert "temperature" not in kw


def test_default_thinking_still_reserves_room_and_asks_for_summary():
    """gpt-6-sol 은 아무것도 보내지 않아도 medium 으로 생각한다 — 그 자리를 두고 요약을 받는다."""
    kw = OpenAIClient(api_key="sk")._build_responses_kwargs(_req(tools=TOOLS))
    assert kw["reasoning"] == {"summary": "auto"}
    assert kw["max_output_tokens"] == 8192 + 16384


def test_thinking_off_sends_none_without_summary():
    kw = OpenAIClient(api_key="sk")._build_responses_kwargs(_req(model="gpt-5.4", thinking_level="off"))
    assert kw["reasoning"] == {"effort": "none"}
    assert kw["max_output_tokens"] == 8192


def test_structured_output_goes_to_text_format():
    kw = OpenAIClient(api_key="sk")._build_responses_kwargs(
        _req(response_format={"type": "json_schema", "json_schema": {"title": "Facts", "type": "object"}})
    )
    assert kw["text"]["format"]["type"] == "json_schema"
    assert kw["text"]["format"]["name"] == "Facts"
    assert kw["text"]["format"]["strict"] is False


# ── 대화 → 입력 항목 ───────────────────────────────────────────────────


def test_history_keeps_reasoning_before_its_call_and_pairs_ids():
    instructions, items = canonical_to_responses_input(_history(), "sys", model="gpt-6-sol")
    assert instructions == "sys"
    assert [i.get("type", "message") for i in items] == [
        "message",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
    ]
    assert items[1] == {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}
    assert items[2]["phase"] == "commentary"
    assert items[3] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": json.dumps({"city": "Seoul"}),
        "id": "fc_1",
    }
    assert items[4] == {"type": "function_call_output", "call_id": "call_1", "output": "18C clear"}


def test_reasoning_from_another_model_is_not_replayed_and_call_loses_its_item_id():
    """다른 모델의 암호화된 생각은 풀 수 없다 — 돌려주지 않고, 짝이 없는 호출에는 id 를 달지 않는다."""
    _, items = canonical_to_responses_input(_history(model="gpt-5.4"), "", model="gpt-6-sol")
    assert all(i.get("type") != "reasoning" for i in items)
    call = next(i for i in items if i.get("type") == "function_call")
    assert "id" not in call


def test_other_providers_drop_the_reasoning_block():
    """대화 중에 모델을 Anthropic·Chat Completions 로 바꿔도 OpenAI 의 생각 블록 때문에 깨지지 않는다."""
    anth = canonical_messages_to_anthropic(_history())
    assert all(b.get("type") != "reasoning" for m in anth if isinstance(m["content"], list) for b in m["content"])
    assert canonical_messages_to_anthropic([{"role": "assistant", "content": [_reasoning_block()]}]) == []
    cc = canonical_messages_to_openai([{"role": "assistant", "content": [_reasoning_block()]}])
    assert cc == []


# ── 응답 → 표준 ───────────────────────────────────────────────────────


def _output_response(status="completed", incomplete=None):
    return SimpleNamespace(
        id="resp_1",
        model="gpt-6-sol",
        status=status,
        incomplete_details=incomplete,
        output=[
            SimpleNamespace(
                type="reasoning",
                model_dump=lambda exclude_none=True: {
                    "type": "reasoning",
                    "id": "rs_9",
                    "summary": [{"type": "summary_text", "text": "Think."}],
                    "encrypted_content": "E9",
                    "status": None,
                },
                summary=[SimpleNamespace(text="Think.")],
            ),
            SimpleNamespace(
                type="message",
                phase="commentary",
                content=[SimpleNamespace(type="output_text", text="Let me check.")],
            ),
            SimpleNamespace(
                type="function_call", id="fc_9", call_id="call_9", name="get_weather", arguments='{"city":"Seoul"}'
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=40,
            input_tokens_details=SimpleNamespace(cached_tokens=64),
            output_tokens_details=SimpleNamespace(reasoning_tokens=20),
        ),
    )


def test_parse_keeps_order_reasoning_text_call():
    resp = OpenAIClient(api_key="sk")._parse_responses_response(_output_response(), "gpt-6-sol")
    assert [b.type for b in resp.content] == ["reasoning", "text", "tool_use"]
    assert resp.stop_reason == "tool_use"
    assert resp.content[0].raw["item"] == {
        "type": "reasoning",
        "id": "rs_9",
        "summary": [{"type": "summary_text", "text": "Think."}],
        "encrypted_content": "E9",
    }
    assert resp.content[0].raw["model"] == "gpt-6-sol"
    assert resp.content[1].raw == {"type": "text", "text": "Let me check.", "_meta": {"openai_phase": "commentary"}}
    assert resp.content[2].tool_input == {"city": "Seoul"}
    assert resp.content[2].raw["_meta"] == {"openai_item_id": "fc_9"}
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cache_read_input_tokens) == (100, 40, 64)


def test_incomplete_for_output_limit_is_max_tokens():
    raw = _output_response(status="incomplete", incomplete=SimpleNamespace(reason="max_output_tokens"))
    raw.output = raw.output[:1]
    resp = OpenAIClient(api_key="sk")._parse_responses_response(raw, "gpt-6-sol")
    assert resp.stop_reason == "max_tokens"


# ── 스트림 ────────────────────────────────────────────────────────────


class _Stream:
    def __init__(self, events, delays=None):
        self._events = list(events)
        self._delays = delays or {}
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i, ev in enumerate(self._events):
            if i in self._delays:
                await asyncio.sleep(self._delays[i])
            yield ev

    async def close(self):
        self.closed = True


class _Responses:
    def __init__(self, stream=None, response=None, fail_first=None):
        self.calls: List[Dict[str, Any]] = []
        self._stream = stream
        self._response = response
        self._fail_first = fail_first

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_first is not None and len(self.calls) == 1:
            raise self._fail_first
        return self._stream if kwargs.get("stream") else self._response


def _client_with(responses):
    client = OpenAIClient(api_key="sk")
    client._client = SimpleNamespace(responses=responses)
    return client


async def _collect(client, **kw):
    out = []
    async for chunk in client.create_message_stream(
        model_config=ModelConfig(model="gpt-6-sol", max_tokens=4096, thinking_level=kw.pop("level", None)),
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        **kw,
    ):
        out.append(chunk)
    return out


def test_stream_turns_events_into_chunks():
    final = _output_response()
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(type="reasoning")),
        SimpleNamespace(type="response.reasoning_summary_part.added"),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Think"),
        SimpleNamespace(type="response.reasoning_summary_part.added"),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="More"),
        SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(type="message")),
        SimpleNamespace(type="response.output_text.delta", delta="Let me check."),
        SimpleNamespace(type="response.completed", response=final),
    ]
    stream = _Stream(events)
    responses = _Responses(stream=stream)
    chunks = asyncio.run(_collect(_client_with(responses), level="high"))
    kinds = [(c["type"], c.get("text")) for c in chunks if c["type"] != "message_complete"]
    assert kinds == [
        ("thinking_delta", ""),
        ("thinking_delta", "Think"),
        ("thinking_delta", "\n\n"),
        ("thinking_delta", "More"),
        ("text_delta", "Let me check."),
    ]
    done = chunks[-1]["response"]
    assert [b.type for b in done.content] == ["reasoning", "text", "tool_use"]
    assert responses.calls[0]["stream"] is True
    assert responses.calls[0]["reasoning"] == {"effort": "high", "summary": "auto"}
    assert stream.closed is True


def test_stream_sends_heartbeats_while_the_model_thinks(monkeypatch):
    """생각하는 동안 서버는 아무것도 보내지 않는다 — 스트림 감시가 무응답으로 끊지 않게 신호를 보낸다."""
    monkeypatch.setattr(openai_mod, "_HEARTBEAT_S", 0.02)
    events = [
        SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(type="reasoning")),
        SimpleNamespace(type="response.output_text.delta", delta="ok"),
        SimpleNamespace(type="response.completed", response=_output_response()),
    ]
    chunks = asyncio.run(_collect(_client_with(_Responses(stream=_Stream(events, delays={1: 0.15})))))
    assert any(c["type"] == "heartbeat" for c in chunks)
    first_text = next(i for i, c in enumerate(chunks) if c["type"] == "text_delta")
    assert all(c["type"] != "heartbeat" for c in chunks[first_text:])


def test_stream_without_final_response_is_an_error():
    from xgen_agent_runtime.core.errors import APIError

    events = [SimpleNamespace(type="response.output_text.delta", delta="partial")]
    with pytest.raises(APIError):
        asyncio.run(_collect(_client_with(_Responses(stream=_Stream(events)))))


def test_failed_response_raises():
    from xgen_agent_runtime.core.errors import APIError

    failed = SimpleNamespace(error=SimpleNamespace(code="server_error", message="boom"))
    events = [SimpleNamespace(type="response.failed", response=failed)]
    with pytest.raises(APIError, match="boom"):
        asyncio.run(_collect(_client_with(_Responses(stream=_Stream(events)))))


# ── 한 번 고쳐 다시 보내기 ─────────────────────────────────────────────


def test_chat_completions_drops_reasoning_when_tools_are_refused():
    """프록시·Azure 처럼 Chat Completions 로 가는 곳 — 같은 400 이면 생각을 끄고 다시 보낸다."""
    client = OpenAIClient(api_key="sk", base_url="https://gw.local/v1")
    kwargs = {"model": "gpt-6-sol", "messages": [], "tools": [{}], "reasoning_effort": "high", "temperature": 1}
    exc = Exception(
        "Function tools with reasoning_effort are not supported for gpt-6-sol in /v1/chat/completions. "
        "To use function tools, use /v1/responses or set reasoning_effort to 'none'."
    )
    retry = client._heal_request_kwargs(kwargs, exc)
    assert retry["reasoning_effort"] == "none" and "temperature" not in retry
    # 기본 강도(아무것도 안 보냄)로 실패한 경우도
    retry2 = client._heal_request_kwargs({"model": "gpt-6-sol", "messages": [], "tools": [{}]}, exc)
    assert retry2["reasoning_effort"] == "none"


def test_unsupported_effort_moves_to_the_nearest_supported_value():
    client = OpenAIClient(api_key="sk")
    msg = (
        "Unsupported value: 'none' is not supported with the 'gpt-6-astra' model. "
        "Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'."
    )
    retry = client._heal_request_kwargs(
        {"model": "gpt-6-astra", "input": [], "reasoning": {"effort": "none"}}, Exception(msg)
    )
    assert retry["reasoning"] == {"effort": "low"}
    cc = client._heal_request_kwargs(
        {"model": "gpt-5.4", "messages": [], "reasoning_effort": "max"},
        Exception(
            "Unsupported value: 'max' is not supported with the 'gpt-5.4' model. "
            "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."
        ),
    )
    assert cc["reasoning_effort"] == "xhigh"


def test_rejected_reasoning_replay_is_sent_again_without_it():
    client = OpenAIClient(api_key="sk")
    _, items = canonical_to_responses_input(_history(), "", model="gpt-6-sol")
    retry = client._heal_request_kwargs(
        {"model": "gpt-6-sol", "input": items},
        Exception("Item 'rs_1' of type 'reasoning' was provided without its required following item."),
    )
    assert all(i.get("type") != "reasoning" for i in retry["input"])
    assert all("id" not in i for i in retry["input"] if i.get("type") == "function_call")


def test_send_heals_once_and_reports(monkeypatch):
    """표가 낡아(새 모델이 none 을 안 받게 됨) 400 이 나면 받는 값으로 한 번 다시 보내고 알린다."""
    from xgen_agent_runtime.llm_client import thinking

    stale = thinking._levels(("low", "medium", "high"), default="medium", via="openai_effort")
    monkeypatch.setattr(thinking, "_OPENAI", (("gpt-6-astra", stale),))
    events: List[Dict[str, Any]] = []
    client = OpenAIClient(api_key="sk", event_sink=events.append)
    responses = _Responses(
        response=_output_response(),
        fail_first=Exception(
            "Unsupported value: 'none' is not supported with the 'gpt-6-astra' model. "
            "Supported values are: 'low', 'medium', 'high', 'xhigh', and 'max'."
        ),
    )
    client._client = SimpleNamespace(responses=responses)
    resp = asyncio.run(
        client.create_message(
            model_config=ModelConfig(model="gpt-6-astra", max_tokens=1024, thinking_level="off"),
            messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS,
        )
    )
    assert [c["reasoning"]["effort"] for c in responses.calls] == ["none", "low"]
    assert resp.stop_reason == "tool_use"
    assert any(e["type"] == "llm_client.drift_healed" for e in events)


# ── 저장·창·토큰 ─────────────────────────────────────────────────────


def test_memory_does_not_store_or_replay_reasoning():
    from xgen_agent_runtime.core.token_estimate import _estimate_block
    from xgen_agent_runtime.stages.s18_memory._dehydrate import dehydrate_message

    msg = _history()[1]
    stored = dehydrate_message(msg)
    assert [b["type"] for b in stored["content"]] == ["text", "tool_use"]
    big = dict(_reasoning_block())
    big["item"] = {**big["item"], "encrypted_content": "x" * 40_000}
    assert _estimate_block(big) < 200
