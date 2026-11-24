"""thinking 블록의 묶임(binding) — Claude 5.5 세대가 되돌려 받은 생각 블록을 검사하는 것에 맞춘다.

생각 블록의 서명은 그것을 쓸 때 앞에 있던 것(system · 도구 목록 · 앞선 메시지)을 기억하고, API 는 되돌려
받을 때 그대로인지 본다. 다르면 400 "Invalid `signature` in `thinking` block. The block is bound to a
different conversation" 이다(2026-11-20 홈 XGEN, Sonnet 5.5 · Haiku 5.5 — 문 도구가 가족을 연 뒤의 요청).

- 생각 블록마다 쓴 때의 system · 도구 목록을 적어 두고(보내지 않는 ``_meta``), 다른 목록 아래의 요청에서는 뺀다.
- adaptive 요청에는 API 의 안전망(``drop_block`` + 베타 헤더)을 싣고, 받지 않는 곳에서는 빼고 다시 보낸다.
- 안전망을 못 싣는 thinking 형태(between_tools 등)는 거절되면 생각 블록 없이 한 번 다시 보낸다.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from anthropic.types import Message, Usage

from xgen_agent_runtime.llm_client.anthropic import (
    AnthropicClient,
    _log_input_transformations,
    _retry_kwargs_after_deprecation,
    thinking_binding,
)
from xgen_agent_runtime.llm_client.bedrock import BedrockClient
from xgen_agent_runtime.llm_client.translators._canonical import canonical_messages_to_anthropic
from xgen_agent_runtime.llm_client.types import APIRequest
from xgen_agent_runtime.stages.s06_api.artifact.default.tool_loop import assistant_content_blocks

BETA = "thinking-binding-controls-2026-08-01"

TOOL_A = {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}
TOOL_B = {"name": "Glob", "description": "find files", "input_schema": {"type": "object"}}
TOOL_C = {"name": "AppCreate", "description": "make an app", "input_schema": {"type": "object"}}


def _req(**overrides) -> APIRequest:
    base: dict = {
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1024,
        "system": "You are an agent.",
        "tools": [TOOL_A, TOOL_B],
    }
    base.update(overrides)
    return APIRequest(**base)


def _thinking(sig: str, binding: str | None) -> dict:
    block = {"type": "thinking", "thinking": "", "signature": sig}
    if binding is not None:
        block["_meta"] = {"binding": binding}
    return block


def _history(binding_of_first: str | None, binding_of_second: str | None) -> list:
    return [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": [
                _thinking("s1", binding_of_first),
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        {
            "role": "assistant",
            "content": [
                _thinking("s2", binding_of_second),
                {"type": "tool_use", "id": "t2", "name": "AppCreate", "input": {}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]},
    ]


def _signatures(kwargs: dict) -> list:
    return [
        b.get("signature")
        for m in kwargs["messages"]
        if m["role"] == "assistant"
        for b in m["content"]
        if b.get("type") == "thinking"
    ]


# ── what the binding is ─────────────────────────────────────────────


def test_binding_ignores_cache_markers_tool_order_and_surrounding_whitespace() -> None:
    plain = thinking_binding("You are an agent.", [TOOL_A, TOOL_B])
    assert plain == thinking_binding("  You are an agent.\n", [TOOL_B, TOOL_A])
    cached_tools = [TOOL_A, {**TOOL_B, "cache_control": {"type": "ephemeral"}}]
    cached_system = [{"type": "text", "text": "You are an agent.", "cache_control": {"type": "ephemeral"}}]
    assert plain == thinking_binding(cached_system, cached_tools)


def test_binding_changes_with_the_tool_set_the_descriptions_and_the_system_prompt() -> None:
    plain = thinking_binding("You are an agent.", [TOOL_A, TOOL_B])
    assert plain != thinking_binding("You are an agent.", [TOOL_A, TOOL_B, TOOL_C])
    assert plain != thinking_binding("You are an agent.", [TOOL_A, {**TOOL_B, "description": "x"}])
    assert plain != thinking_binding("You are an agent. Catalogue: AppCreate", [TOOL_A, TOOL_B])


# ── blocks remember where they were written, and are left out elsewhere ──


def _message(content):
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=content,
        model="claude-sonnet-5-5",
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=4),
    )


def test_parsed_thinking_records_its_binding_and_the_wire_never_carries_it() -> None:
    client = AnthropicClient(api_key="k")
    raw = _message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "blob"},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        ]
    )
    response = client._parse_response(raw, "b1")
    replay = assistant_content_blocks(response)
    assert replay[0]["_meta"] == {"binding": "b1"}
    assert replay[1]["_meta"] == {"binding": "b1"}
    wire = canonical_messages_to_anthropic([{"role": "assistant", "content": replay}])
    assert wire[0]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert wire[0]["content"][1] == {"type": "redacted_thinking", "data": "blob"}


def test_thinking_written_under_another_tool_set_is_left_out() -> None:
    client = AnthropicClient(api_key="k")
    before = thinking_binding("You are an agent.", [TOOL_A, TOOL_B])
    after = thinking_binding("You are an agent.", [TOOL_A, TOOL_B, TOOL_C])
    # The gate opened AppCreate between the two calls: s1 was written before, s2 after.
    kwargs = client._build_kwargs(
        _req(tools=[TOOL_A, TOOL_B, TOOL_C], messages=_history(before, after))
    )
    assert _signatures(kwargs) == ["s2"]
    # Text and tool calls of the earlier turn stay.
    assert kwargs["messages"][1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}
    ]
    assert all("_meta" not in b for m in kwargs["messages"] for b in m["content"] if isinstance(b, dict))


def test_thinking_under_the_same_binding_or_without_a_record_is_kept() -> None:
    client = AnthropicClient(api_key="k")
    same = thinking_binding("You are an agent.", [TOOL_A, TOOL_B])
    assert _signatures(client._build_kwargs(_req(messages=_history(same, same)))) == ["s1", "s2"]
    assert _signatures(client._build_kwargs(_req(messages=_history(None, None)))) == ["s1", "s2"]


def test_an_assistant_message_of_only_stale_thinking_is_not_sent_empty() -> None:
    client = AnthropicClient(api_key="k")
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [_thinking("s0", "other")]},
        {"role": "user", "content": "go on"},
    ]
    kwargs = client._build_kwargs(_req(messages=messages))
    assert [m["role"] for m in kwargs["messages"]] == ["user", "user"]


# ── the API's own safety net ────────────────────────────────────────


def test_adaptive_requests_carry_drop_block_and_the_beta_header() -> None:
    kwargs = AnthropicClient(api_key="k")._build_kwargs(_req())
    assert kwargs["thinking"]["type"] == "adaptive"
    assert kwargs["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert kwargs["extra_headers"]["anthropic-beta"] == BETA


def test_the_beta_header_keeps_the_betas_the_client_already_sends() -> None:
    client = AnthropicClient(api_key="k", default_headers={"anthropic-beta": "context-1m-2025-08-07"})
    kwargs = client._build_kwargs(_req())
    assert kwargs["extra_headers"]["anthropic-beta"] == f"context-1m-2025-08-07,{BETA}"


def test_haiku_5_5_thinks_by_default_so_it_gets_the_safety_net_too() -> None:
    # Haiku 5.5 returns thinking blocks without a thinking parameter; leaving it out of the list sent
    # its requests without drop_block, and only Haiku turns ended in the 400 (2026-11-21 home XGEN).
    kwargs = AnthropicClient(api_key="k")._build_kwargs(_req(model="claude-haiku-5-5"))
    assert kwargs["thinking"]["type"] == "adaptive"
    assert kwargs["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert kwargs["extra_headers"]["anthropic-beta"] == BETA
    # Haiku 4.5 is not a 5.5-generation model: no thinking parameter is added.
    assert "thinking" not in AnthropicClient(api_key="k")._build_kwargs(_req(model="claude-haiku-4-5"))


def test_thinking_types_that_reject_block_binding_do_not_get_it() -> None:
    for thinking in ({"type": "between_tools"}, {"type": "disabled"}):
        kwargs = AnthropicClient(api_key="k")._build_kwargs(_req(thinking=thinking))
        assert "block_binding" not in kwargs["thinking"], thinking
        assert "extra_headers" not in kwargs, thinking


def test_bedrock_does_not_send_the_controls() -> None:
    # Bedrock InvokeModel reads betas from the body, not the header.
    assert BedrockClient._thinking_binding_controls is False
    assert AnthropicClient._thinking_binding_controls is True


def test_an_endpoint_without_the_beta_gets_the_request_without_the_controls() -> None:
    kwargs = AnthropicClient(api_key="k")._build_kwargs(_req())
    exc = Exception("thinking.block_binding: Extra inputs are not permitted")
    retry = _retry_kwargs_after_deprecation(kwargs, exc)
    assert retry is not None
    assert "block_binding" not in retry["thinking"]
    assert "extra_headers" not in retry
    # Other betas the request carried survive the retry.
    kwargs["extra_headers"] = {"anthropic-beta": f"context-1m-2025-08-07,{BETA}"}
    retry = _retry_kwargs_after_deprecation(kwargs, exc)
    assert retry["extra_headers"] == {"anthropic-beta": "context-1m-2025-08-07"}


def test_a_rejected_thinking_block_is_retried_once_without_thinking() -> None:
    kwargs = AnthropicClient(api_key="k")._build_kwargs(
        _req(thinking={"type": "between_tools"}, messages=_history(None, None))
    )
    exc = Exception(
        "messages.3.content.0: Invalid `signature` in `thinking` block. The block is bound to a "
        "different conversation. Remove the block, or set "
        "`thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\"."
    )
    retry = _retry_kwargs_after_deprecation(kwargs, exc)
    assert retry is not None
    assert _signatures(retry) == []
    assert [b["type"] for b in retry["messages"][1]["content"]] == ["tool_use"]
    # Nothing to strip → no retry (the same body would fail the same way).
    assert _retry_kwargs_after_deprecation(retry, exc) is None


def test_unrelated_400s_are_not_retried() -> None:
    kwargs = AnthropicClient(api_key="k")._build_kwargs(_req(messages=_history(None, None)))
    assert _retry_kwargs_after_deprecation(kwargs, Exception("messages: roles must alternate")) is None


def test_dropped_blocks_reported_by_the_api_are_logged(caplog) -> None:
    raw = SimpleNamespace(
        input_transformations=[
            {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch"},
            {"type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch"},
        ]
    )
    with caplog.at_level(logging.INFO, logger="xgen_agent_runtime.llm_client.anthropic"):
        _log_input_transformations(raw)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "messages.1.content.0" in warnings[0].getMessage()
    assert any("model_binding_mismatch" in r.getMessage() for r in caplog.records)
