"""OpenRouter client (4.92.0).

OpenAI Chat Completions 모양에 OpenRouter 만의 것 셋을 더한 클라이언트:

  * 생각 이어 가기 — ``reasoning_details`` 를 생각 블록으로 받아 같은 모델의 다음 호출에 그대로 돌려준다.
  * 200 으로 오는 실패 — ``choices`` 없이 ``error`` 만 있는 본문, 스트림의 ``error`` 청크.
  * 비용 — ``usage.cost`` → ``TokenUsage.cost_usd``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

openai_sdk = pytest.importorskip("openai")

from xgen_agent_runtime.core.config import ModelConfig  # noqa: E402
from xgen_agent_runtime.core.errors import APIError, ErrorCategory  # noqa: E402
from xgen_agent_runtime.llm_client.openai import OpenAIClient  # noqa: E402
from xgen_agent_runtime.llm_client.openrouter import (  # noqa: E402
    OPENROUTER_BASE_URL,
    OpenRouterClient,
    assistant_reasoning_fields,
    error_from_body,
    merge_reasoning_details,
)
from xgen_agent_runtime.llm_client.registry import ClientRegistry  # noqa: E402
from xgen_agent_runtime.llm_client.thinking import thinking_spec, vllm_request  # noqa: E402
from xgen_agent_runtime.llm_client.types import APIRequest  # noqa: E402

MODEL = "anthropic/claude-sonnet-5.5"


def _message(**fields: Any) -> SimpleNamespace:
    base = {"content": None, "tool_calls": None}
    base.update(fields)
    return SimpleNamespace(**base)


def _raw(message: SimpleNamespace, *, usage: Any = None, finish: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=usage,
        model=MODEL,
        id="gen-1",
    )


def _stream_client(chunks: List[Any], captured: Dict[str, Any]) -> OpenRouterClient:
    async def create(**kwargs: Any):
        captured.clear()
        captured.update(kwargs)

        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    client = OpenRouterClient(api_key="sk-or-mock")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


async def _drain(client: OpenAIClient) -> List[Dict[str, Any]]:
    events = []
    async for evt in client.create_message_stream(
        model_config=ModelConfig(model=MODEL, max_tokens=64),
        messages=[{"role": "user", "content": "x"}],
    ):
        events.append(evt)
    return events


def _chunk(delta: SimpleNamespace, *, finish: Any = None, usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        model=MODEL,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
        usage=usage,
    )


def _delta(**fields: Any) -> SimpleNamespace:
    base = {"content": None, "tool_calls": None}
    base.update(fields)
    return SimpleNamespace(**base)


# ── 등록·기본값 ──────────────────────────────────────────────────────


def test_registered_under_openrouter() -> None:
    assert ClientRegistry.get("openrouter") is OpenRouterClient


def test_defaults_to_public_endpoint_and_system_role() -> None:
    client = OpenRouterClient(api_key="sk-or")
    assert client._base_url == OPENROUTER_BASE_URL
    assert client.system_role == "system"
    assert client.capabilities.supports_tools is True
    assert client.capabilities.supports_cost_usage is True


def test_base_url_override_kept() -> None:
    client = OpenRouterClient(api_key="sk-or", base_url="https://eu.openrouter.ai/api/v1")
    assert client._base_url == "https://eu.openrouter.ai/api/v1"


def test_never_routes_to_responses_api() -> None:
    client = OpenRouterClient(api_key="sk-or")
    assert client._uses_responses("openai/gpt-5.5") is False


# ── 비스트림 응답 ────────────────────────────────────────────────────


def test_reasoning_details_become_leading_reasoning_block() -> None:
    client = OpenRouterClient(api_key="sk-or")
    details = [
        {"type": "reasoning.text", "text": "think", "signature": "sig", "format": "anthropic-claude-v1", "index": 0}
    ]
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="Read", arguments='{"path": "a.txt"}')
    )
    resp = client._parse_response(
        _raw(_message(reasoning="think", reasoning_details=details, tool_calls=[tool_call]), finish="tool_calls")
    )
    assert [b.type for b in resp.content] == ["reasoning", "tool_use"]
    block = resp.content[0]
    assert block.thinking_text == "think"
    assert block.raw["provider"] == "openrouter"
    assert block.raw["model"] == MODEL
    assert block.raw["details"] == details
    assert resp.stop_reason == "tool_use"


def test_reasoning_details_from_model_extra() -> None:
    """SDK 가 모르는 필드는 ``model_extra`` 에 산다 — 실제 SDK 객체로 확인."""
    from openai.types.chat import ChatCompletion

    raw = ChatCompletion.model_validate(
        {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_details": [{"type": "reasoning.summary", "summary": "s", "index": 0}],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "cost": 0.00042},
        }
    )
    resp = OpenRouterClient(api_key="sk-or")._parse_response(raw)
    assert [b.type for b in resp.content] == ["reasoning", "text"]
    assert resp.content[0].thinking_text == "s"
    assert resp.usage.cost_usd == pytest.approx(0.00042)
    assert resp.usage.input_tokens == 10


def test_no_reasoning_means_no_block() -> None:
    resp = OpenRouterClient(api_key="sk-or")._parse_response(_raw(_message(content="hi")))
    assert [b.type for b in resp.content] == ["text"]


@pytest.mark.parametrize(
    "code,category",
    [
        (402, ErrorCategory.TERMINAL),
        (429, ErrorCategory.RATE_LIMITED),
        (401, ErrorCategory.AUTH),
        (502, ErrorCategory.SERVER_ERROR),
        (400, ErrorCategory.BAD_REQUEST),
    ],
)
def test_error_body_with_200_is_raised(code: int, category: ErrorCategory) -> None:
    raw = SimpleNamespace(
        choices=None,
        error={"code": code, "message": "boom", "metadata": {"provider_name": "Anthropic"}},
        model=MODEL,
        id="gen-1",
        usage=None,
    )
    with pytest.raises(APIError) as info:
        OpenRouterClient(api_key="sk-or")._parse_response(raw)
    assert info.value.category is category
    assert "boom" in str(info.value)
    assert "Anthropic" in str(info.value)


def test_empty_choices_without_error_is_raised() -> None:
    raw = SimpleNamespace(choices=[], model=MODEL, id="gen-1", usage=None)
    with pytest.raises(APIError):
        OpenRouterClient(api_key="sk-or")._parse_response(raw)


def test_context_overflow_message_is_token_limit() -> None:
    err = error_from_body({"message": "This endpoint's maximum context length is 8192 tokens"})
    assert err.category is ErrorCategory.TOKEN_LIMIT


# ── 스트림 ───────────────────────────────────────────────────────────


async def test_stream_merges_reasoning_and_emits_thinking() -> None:
    chunks = [
        _chunk(_delta(reasoning_details=[{"type": "reasoning.text", "text": "Let ", "index": 0}])),
        _chunk(_delta(reasoning_details=[{"type": "reasoning.text", "text": "me", "index": 0, "signature": "sig"}])),
        _chunk(_delta(content="done")),
        # OpenRouter 의 사용량 청크는 choices 가 비어 있지 않다.
        _chunk(
            _delta(),
            finish="stop",
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, prompt_tokens_details=None, cost=0.01),
        ),
    ]
    captured: Dict[str, Any] = {}
    events = await _drain(_stream_client(chunks, captured))
    thinking = [e["text"] for e in events if e["type"] == "thinking_delta"]
    assert thinking == ["Let ", "me"]
    resp = [e for e in events if e["type"] == "message_complete"][0]["response"]
    assert [b.type for b in resp.content] == ["reasoning", "text"]
    assert resp.content[0].raw["details"] == [
        {"type": "reasoning.text", "text": "Let me", "index": 0, "signature": "sig"}
    ]
    assert resp.usage.input_tokens == 7
    assert resp.usage.cost_usd == pytest.approx(0.01)


async def test_stream_plain_reasoning_string() -> None:
    chunks = [_chunk(_delta(reasoning="hmm")), _chunk(_delta(content="ok"), finish="stop")]
    events = await _drain(_stream_client(chunks, {}))
    resp = [e for e in events if e["type"] == "message_complete"][0]["response"]
    assert resp.content[0].type == "reasoning"
    assert resp.content[0].raw["summary"] == "hmm"
    assert resp.content[0].raw["details"] == []


async def test_stream_error_chunk_classified_by_code() -> None:
    async def create(**kwargs: Any):
        async def _gen():
            yield _chunk(_delta(content="par"))
            raise openai_sdk.APIError(
                "Insufficient credits",
                request=None,  # type: ignore[arg-type]
                body={"code": 402, "message": "Insufficient credits"},
            )

        return _gen()

    client = OpenRouterClient(api_key="sk-or")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(APIError) as info:
        await _drain(client)
    assert info.value.category is ErrorCategory.TERMINAL


async def test_openai_stream_takes_usage_from_chunk_with_choices() -> None:
    """사용량 청크에 choices 를 함께 싣는 서버 — 일반 OpenAI 경로도 사용량을 잃지 않는다."""
    chunks = [
        _chunk(_delta(content="hi")),
        _chunk(_delta(), finish="stop", usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1)),
    ]

    async def create(**kwargs: Any):
        async def _gen():
            for c in chunks:
                yield c

        return _gen()

    client = OpenAIClient(api_key="sk")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    events = await _drain(client)
    resp = [e for e in events if e["type"] == "message_complete"][0]["response"]
    assert resp.usage.input_tokens == 3
    assert [b.type for b in resp.content] == ["text"]


# ── 생각 되돌려 주기 ─────────────────────────────────────────────────


def _history(block_model: str, provider: str = "openrouter") -> List[Dict[str, Any]]:
    details = [{"type": "reasoning.encrypted", "data": "enc", "index": 0}]
    return [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "provider": provider, "model": block_model, "summary": "", "details": details},
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "a"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "data"}]},
    ]


def _assistant_wire(client: OpenRouterClient, messages: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    kwargs = client._build_kwargs(APIRequest(model=model, messages=messages, max_tokens=64))
    return [m for m in kwargs["messages"] if m["role"] == "assistant"][0]


def test_reasoning_details_replayed_to_same_model() -> None:
    wire = _assistant_wire(OpenRouterClient(api_key="sk-or"), _history(MODEL), MODEL)
    assert wire["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "enc", "index": 0}]
    assert wire["tool_calls"][0]["id"] == "call_1"


def test_reasoning_not_replayed_to_other_model() -> None:
    wire = _assistant_wire(OpenRouterClient(api_key="sk-or"), _history(MODEL), "openai/gpt-5.5")
    assert "reasoning_details" not in wire


def test_foreign_reasoning_blocks_not_replayed() -> None:
    wire = _assistant_wire(OpenRouterClient(api_key="sk-or"), _history(MODEL, provider="openai"), MODEL)
    assert "reasoning_details" not in wire
    assert "reasoning" not in wire


def test_openai_client_ignores_openrouter_reasoning() -> None:
    kwargs = OpenAIClient(api_key="sk")._build_kwargs(
        APIRequest(model="gpt-4o", messages=_history(MODEL), max_tokens=64)
    )
    assistant = [m for m in kwargs["messages"] if m["role"] == "assistant"][0]
    assert "reasoning_details" not in assistant


def test_assistant_fields_fall_back_to_reasoning_text() -> None:
    content = [{"type": "reasoning", "provider": "openrouter", "model": MODEL, "summary": "why", "details": None}]
    assert assistant_reasoning_fields(content, MODEL) == {"reasoning": "why"}


def test_merge_keeps_separate_indexes_and_types() -> None:
    merged = merge_reasoning_details(
        [
            {"type": "reasoning.summary", "summary": "a", "index": 0},
            {"type": "reasoning.encrypted", "data": "x", "index": 1},
            {"type": "reasoning.summary", "summary": "b", "index": 0},
            {"type": "reasoning.encrypted", "data": "y", "index": 1, "id": "r1"},
        ]
    )
    assert merged == [
        {"type": "reasoning.summary", "summary": "ab", "index": 0},
        {"type": "reasoning.encrypted", "data": "xy", "index": 1, "id": "r1"},
    ]


# ── 생각 강도 ────────────────────────────────────────────────────────


def test_thinking_spec_follows_underlying_vendor_table() -> None:
    assert thinking_spec("openrouter", "anthropic/claude-opus-5.5").can_disable is False
    assert thinking_spec("openrouter", "openai/gpt-5.5").levels == ("low", "medium", "high", "xhigh")
    generic = thinking_spec("openrouter", "qwen/qwen3.6-27b:free")
    assert generic.levels == ("low", "medium", "high")
    assert generic.via == "openrouter_effort"


@pytest.mark.parametrize(
    "level,reasoning",
    [("high", {"effort": "high"}), ("off", {"enabled": False}), ("on", {"enabled": True})],
)
def test_openrouter_effort_wire(level: str, reasoning: Dict[str, Any]) -> None:
    spec = thinking_spec("openrouter", "qwen/qwen3.6-27b")
    assert vllm_request(spec, level) == {"extra_body": {"reasoning": reasoning}}


def test_thinking_level_lands_in_extra_body() -> None:
    kwargs = OpenRouterClient(api_key="sk-or")._build_kwargs(
        APIRequest(model="qwen/qwen3.6-27b", messages=[{"role": "user", "content": "q"}], max_tokens=64, thinking_level="high")
    )
    assert kwargs["extra_body"] == {"reasoning": {"effort": "high"}}
    assert "reasoning_effort" not in kwargs


def test_heal_drops_refused_reasoning_setting() -> None:
    client = OpenRouterClient(api_key="sk-or")
    kwargs = {"model": MODEL, "messages": [], "extra_body": {"reasoning": {"enabled": False}}}
    healed = client._heal_request_kwargs(kwargs, Exception("Reasoning is mandatory for this endpoint and cannot be disabled."))
    assert healed is not None
    assert "extra_body" not in healed


def test_heal_leaves_unrelated_errors() -> None:
    client = OpenRouterClient(api_key="sk-or")
    kwargs = {"model": MODEL, "messages": [], "extra_body": {"reasoning": {"effort": "high"}}}
    assert client._heal_request_kwargs(kwargs, Exception("No endpoints found for model")) is None
