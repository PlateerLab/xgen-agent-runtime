"""4.89.0 context-compaction overhaul: the behaviours the fix rests on.

Each test pins one decision of the new policy (core.compaction + core.token_estimate + the
LLM summary compactor + Stage 6 overflow recovery + the host's input budget). They are the
regression guard for the problems found in the 2026-10-11 review: Korean under-counting, no
reaction to a provider "prompt too long", summaries that dropped history on failure, compacted
messages lost from memory, a window-blind tool-result cap, and an input clamp that cut the
user's question instead of the history.
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.core.compaction import (
    COMPACTION_CONTEXT_KEY,
    LLM_COMPACTIONS_KEY,
    MAX_LLM_COMPACTIONS_PER_TURN,
    note_api_call,
    run_compaction,
)
from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.core.errors import (
    APIError,
    ErrorCategory,
    is_context_overflow_message,
    parse_overflow_limit,
    parse_overflow_tokens,
)
from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.core.token_estimate import (
    CALIBRATION_KEY,
    estimate_prompt_tokens,
    record_prompt_usage,
)
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
    CHECKPOINT_MARKER,
    LLMSummaryCompactor,
    TruncateCompactor,
)


def _conversation(n: int, text: str = "message") -> list:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{text} {i}"} for i in range(n)
    ]


def _events(state: PipelineState, kind: str) -> list:
    return [e["data"] for e in state.events if e["type"] == kind]


class _SummaryClient:
    provider = "fake"

    def __init__(self, text: str = "SUMMARY OF EARLIER WORK"):
        self.text = text
        self.calls = 0

    async def create_message(self, *, model_config, messages, purpose: str = ""):
        self.calls += 1
        return APIResponse(content=[ContentBlock(type="text", text=self.text)], stop_reason="end_turn")


class _FailingClient(_SummaryClient):
    async def create_message(self, **kwargs):
        self.calls += 1
        raise RuntimeError("summary model down")


def _llm_compactor(keep_recent: int = 4) -> LLMSummaryCompactor:
    return LLMSummaryCompactor(
        keep_recent=keep_recent,
        resolve_cfg=lambda s: ModelConfig(model="fake", max_tokens=512),
        has_override=lambda: True,
    )


# ── estimate ────────────────────────────────────────────────────────


def test_korean_is_not_counted_at_four_chars_per_token():
    state = PipelineState()
    state.messages = [{"role": "user", "content": "한국어 문장은 글자마다 토큰이 거의 하나씩 든다. " * 200}]
    chars = len(state.messages[0]["content"])
    # The old estimate was chars / 4; real tokenizers spend close to a token per Hangul syllable.
    assert estimate_prompt_tokens(state) > chars / 2


def test_provider_count_anchors_the_estimate():
    state = PipelineState()
    state.messages = _conversation(6)
    record_prompt_usage(state, 50_000)  # the provider counted far more than our estimate
    assert estimate_prompt_tokens(state) >= 50_000
    state.messages.append({"role": "user", "content": "one more"})
    anchored = estimate_prompt_tokens(state)
    assert 50_000 < anchored < 50_100  # reported + a small delta for the appended message
    # The ratio corrects later unanchored estimates too, but within its clamp.
    assert 0.9 <= state.shared[CALIBRATION_KEY] <= 2.5


# ── provider overflow ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "message,count,limit",
    [
        ("prompt is too long: 210000 tokens > 200000 maximum", 210_000, 200_000),
        (
            "This model's maximum context length is 128000 tokens. However, your messages "
            "resulted in 140000 tokens. Please reduce the length of the messages.",
            140_000,
            128_000,
        ),
        (
            "This model's maximum context length is 32768 tokens. However, you requested 40000 "
            "tokens (32000 in the messages, 8000 in the completion).",
            32_000,
            32_768,
        ),
        (
            "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
            1_200_000,
            1_048_576,
        ),
        (
            "Unable to submit request because the input token count is 1200000 but model only "
            "supports up to 1048576. Reduce the input token count and try again.",
            1_200_000,
            1_048_576,
        ),
    ],
)
def test_overflow_messages_are_recognised_and_parsed(message, count, limit):
    assert is_context_overflow_message(message)
    assert parse_overflow_tokens(message) == count
    assert parse_overflow_limit(message) == limit


def test_unrelated_bad_request_is_not_an_overflow():
    assert not is_context_overflow_message("max_tokens: Field required")


class _OverflowOnce:
    """APIProvider double: the first call is rejected as too long, later calls succeed."""

    name = "overflow-once"

    def __init__(self, limit_message: str):
        self._message = limit_message
        self.requests: list = []

    async def create_message(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            raise APIError(self._message, category=ErrorCategory.TOKEN_LIMIT, status_code=400)
        return APIResponse(
            content=[ContentBlock(type="text", text="done")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=100, output_tokens=5),
        )


@pytest.mark.asyncio
async def test_stage6_compacts_and_retries_once_on_overflow():
    from xgen_agent_runtime.stages.s06_api.artifact.default.stage import APIStage

    provider = _OverflowOnce("prompt is too long: 250000 tokens > 200000 maximum")
    state = PipelineState()
    state.stream = False
    state.context_window_budget = 1_000_000  # what the table believed
    state.messages = _conversation(30)
    state._context_compactor = TruncateCompactor(keep_last=4)

    response = await APIStage(provider=provider).execute(None, state)

    assert response.text == "done"
    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) < len(provider.requests[0].messages)
    # The provider's own limit replaces the wrong table value for the rest of the turn.
    assert state.context_window_budget == 200_000
    overflow = _events(state, "context.overflow")[0]
    assert overflow["reported_tokens"] == 250_000 and overflow["provider_limit"] == 200_000


@pytest.mark.asyncio
async def test_stage6_without_a_compactor_surfaces_the_overflow():
    from xgen_agent_runtime.stages.s06_api.artifact.default.stage import APIStage

    provider = _OverflowOnce("prompt is too long: 250000 tokens > 200000 maximum")
    state = PipelineState()
    state.stream = False
    state.messages = _conversation(30)

    with pytest.raises(APIError) as exc:
        await APIStage(provider=provider).execute(None, state)
    assert exc.value.category == ErrorCategory.TOKEN_LIMIT
    assert len(provider.requests) == 1


# ── the summary compactor ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_keeps_user_messages_verbatim():
    state = PipelineState()
    state.model = "fake"
    state.llm_client = _SummaryClient()
    state.context_window_budget = 200_000
    state.messages = _conversation(20, "ask")
    state.messages[0]["content"] = "Please migrate the billing tables, keep the audit columns."

    await _llm_compactor().compact(state)

    head = state.messages[0]["content"]
    assert head.startswith(CHECKPOINT_MARKER)
    assert "SUMMARY OF EARLIER WORK" in head
    assert "Please migrate the billing tables, keep the audit columns." in head
    assert state.messages[-1]["content"] == "ask 19"


@pytest.mark.asyncio
async def test_kept_tail_never_opens_on_an_orphaned_tool_result():
    state = PipelineState()
    state.model = "fake"
    state.llm_client = _SummaryClient()
    state.messages = _conversation(10)
    state.messages += [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
    ]

    await _llm_compactor(keep_recent=1).compact(state)

    # Every tool_result left in the history still has its tool_use (providers reject orphans).
    calls = {
        b["id"]
        for m in state.messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_use"
    }
    results = [
        b["tool_use_id"]
        for m in state.messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    assert results == ["t1"] and set(results) <= calls


@pytest.mark.asyncio
async def test_soft_pressure_without_a_summary_leaves_history_alone():
    state = PipelineState()
    state.model = "fake"
    state.llm_client = _FailingClient()
    state.messages = _conversation(20)
    before = list(state.messages)
    state.shared[COMPACTION_CONTEXT_KEY] = {"trigger": "proactive", "hard": False}

    await _llm_compactor().compact(state)

    assert state.messages == before
    assert _events(state, "context.compaction_deferred")


@pytest.mark.asyncio
async def test_hard_pressure_without_a_summary_falls_back_to_a_digest():
    state = PipelineState()
    state.model = "fake"
    state.llm_client = _FailingClient()
    state.messages = _conversation(20)
    state.shared[COMPACTION_CONTEXT_KEY] = {"trigger": "guard", "hard": True}

    await _llm_compactor().compact(state)

    assert len(state.messages) < 20
    assert "[Summary of" in state.messages[0]["content"]


# ── policy gates ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_soft_summary_per_request():
    state = PipelineState()
    state.model = "fake"
    client = _SummaryClient()
    state.llm_client = client
    state.context_window_budget = 4_000
    state.messages = _conversation(60, "long message " * 30)
    compactor = _llm_compactor()

    first = await run_compaction(state, compactor, trigger="proactive", target_tokens=10)
    state.messages += _conversation(40, "long message " * 30)
    second = await run_compaction(state, compactor, trigger="proactive", target_tokens=10)

    assert first["summarized"] and client.calls == 1
    assert second.get("skipped") == "already_summarized_for_this_request"
    note_api_call(state)  # the request went out: the next one may summarize again
    third = await run_compaction(state, compactor, trigger="proactive", target_tokens=10)
    assert third["summarized"] and client.calls == 2


@pytest.mark.asyncio
async def test_hard_pressure_past_the_turn_limit_uses_the_digest():
    state = PipelineState()
    state.model = "fake"
    client = _SummaryClient()
    state.llm_client = client
    state.messages = _conversation(30, "long message " * 30)
    state.shared[LLM_COMPACTIONS_KEY] = MAX_LLM_COMPACTIONS_PER_TURN

    out = await run_compaction(state, _llm_compactor(), trigger="guard", target_tokens=10, hard=True)

    assert out["summarized"]
    assert client.calls == 0  # no model call past the limit
    assert "[Summary of" in state.messages[0]["content"]


def test_turn_counters_reset_on_a_new_turn():
    state = PipelineState()
    state.shared[LLM_COMPACTIONS_KEY] = MAX_LLM_COMPACTIONS_PER_TURN
    state.begin_turn()
    assert LLM_COMPACTIONS_KEY not in state.shared


# ── memory recording across a compaction ────────────────────────────


@pytest.mark.asyncio
async def test_messages_summarized_before_recording_still_reach_stm():
    from xgen_agent_runtime.memory.strategy import ProviderDrivenStrategy

    class _Recorder:
        def __init__(self):
            self.turns = []

        async def record_turn(self, turn):
            self.turns.append(turn)

    state = PipelineState()
    state.messages = _conversation(12)
    originals = [m["content"] for m in state.messages]
    # A compaction inside the turn, before Stage 18 recorded anything.
    await run_compaction(state, TruncateCompactor(keep_last=4), trigger="guard")
    recorder = _Recorder()

    await ProviderDrivenStrategy(recorder).update(state)

    recorded = [str(t.content) for t in recorder.turns]
    assert all(any(text in r for r in recorded) for text in originals)
    assert not any("previous messages" in r for r in recorded)  # the synthetic head is not a turn


# ── tool results and prune ──────────────────────────────────────────


def _persist(content, window, storage_path=None):
    from xgen_agent_runtime.stages.s10_tool.persistence import maybe_persist_large_result
    from xgen_agent_runtime.tools.base import ToolCapabilities, ToolContext, ToolResult

    class _View:
        context_window_budget = window

    return maybe_persist_large_result(
        ToolResult(content=content),
        tool_use_id="t1",
        tool_name="Read",
        capabilities=ToolCapabilities(max_result_chars=100_000),
        context=ToolContext(storage_path=storage_path, state_view=_View()),
    )


def test_tool_result_cap_follows_the_window_in_tokens():
    korean = "검색 결과 문단입니다. " * 5_000  # ~65k chars, about as many tokens
    out = _persist(korean, 8_192)
    assert out.display_text and "clipped" in out.display_text
    assert len(out.display_text) < 4_000  # a quarter of 8k tokens, at about a token per syllable


def test_a_medium_english_result_is_not_cut_on_a_large_window(tmp_path):
    # 63k chars of English is ~13k tokens: far under a quarter of a 200k window. The first 4.89.0
    # draft counted a char as a token and replaced it with a 480-char preview.
    english = "The quick brown fox jumps over the lazy dog near the river bank. " * 1_000
    out = _persist(english, 200_000, storage_path=str(tmp_path))
    assert out.display_text is None and out.persist_full is None


def test_stale_tool_call_arguments_are_trimmed_without_touching_the_original():
    from xgen_agent_runtime.core.context_prune import prune_messages

    original_input = {"path": "a.py", "content": "x" * 10_000}
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "w", "name": "Write", "input": original_input}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "w", "content": "ok"}]},
    ] + _conversation(8)

    metrics = prune_messages(messages)

    assert metrics["inputs_trimmed"] == 1
    trimmed = messages[0]["content"][0]["input"]
    assert trimmed["path"] == "a.py" and len(trimmed["content"]) < 1_000
    assert len(original_input["content"]) == 10_000  # the tool-call record keeps what was sent


# ── host input budget ───────────────────────────────────────────────


def test_api_history_is_left_to_compaction_not_the_question():
    from xgen_agent_runtime.host.context_budget import fit_input_to_budget

    history = _conversation(40, "지난 대화 내용입니다. " * 200)
    question = "이번 질문은 잘리면 안 된다. " * 50
    fit = fit_input_to_budget(
        text=question,
        rag_block="",
        system_prompt="system",
        history=history,
        registry=None,
        provider="openai",
        model="gpt-4o",
        max_tokens=4_096,
        window=32_000,
    )
    assert fit.text == question
    assert fit.history is None


def test_cli_drops_the_oldest_history_first():
    from xgen_agent_runtime.host.context_budget import fit_input_to_budget

    history = _conversation(40, "지난 대화 내용입니다. " * 200)
    question = "이번 질문은 잘리면 안 된다."
    fit = fit_input_to_budget(
        text=question,
        rag_block="",
        system_prompt="system",
        history=history,
        registry=None,
        provider="claude_code",
        model="sonnet",
        max_tokens=4_096,
        window=32_000,
        history_compressible=False,
    )
    assert fit.text == question
    assert fit.history is not None and 2 <= len(fit.history) < len(history)
    assert fit.history[-1] is history[-1]
    assert fit.history[0]["role"] == "user"


def test_memory_node_context_string_is_kept_and_fitted():
    from xgen_agent_runtime.host.context_budget import fit_history_context
    from xgen_agent_runtime.host.memory import history_context, history_messages

    value = ([{"role": "user", "content": "recent"}], "[이전 대화 기록]\n" + "오래된 대화 " * 20_000 + "가장 최근 대화")
    assert history_messages(value) == [{"role": "user", "content": "recent"}]
    context = history_context(value)
    fitted = fit_history_context(context, 32_000, "openai", "gpt-4o")
    assert len(fitted) < len(context)
    assert fitted.startswith("[이전 대화 기록]") and fitted.endswith("가장 최근 대화")


@pytest.mark.parametrize(
    "provider,model,window",
    [
        ("anthropic", "claude-opus-5-5", 1_000_000),
        ("anthropic", "claude-sonnet-4-5-20250929", 200_000),
        ("bedrock", "global.anthropic.claude-sonnet-4-6", 1_000_000),
        ("bedrock", "sonnet", 200_000),  # BedrockClient only calls Claude; aliases carry no name
        ("claude_code", "claude-opus-5-5", 200_000),  # the CLI's window, not the API table
        ("claude_code", "opus[1m]", 1_000_000),
        ("openai", "gpt-5.50", 272_000),  # not 'gpt-5.5' (digit boundary)
        ("google", "gemma-4-31b-it", 131_072),
        ("openai", "gpt-5", 272_000),
        ("codex", "gpt-5.5", 272_000),
        ("azure", "my-deployment", 128_000),
        ("google", "gemma-3-27b-it", 131_072),
    ],
)
def test_window_table(provider, model, window):
    from xgen_agent_runtime.host.token_budget import resolve_context_window

    assert resolve_context_window(provider, model) == window


# ── Claude Code argv ────────────────────────────────────────────────


def test_long_claude_code_system_prompt_travels_as_a_file():
    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient
    from xgen_agent_runtime.llm_client.types import APIRequest

    client = ClaudeCodeCLIClient(binary_path="/bin/true")
    system = "시스템 프롬프트 " * 20_000  # > 128 KiB in UTF-8
    argv = client._build_argv(
        APIRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}], system=system, stream=True)
    )
    assert "--system-prompt" not in argv
    path = argv[argv.index("--system-prompt-file") + 1]
    with open(path, encoding="utf-8") as fp:
        assert fp.read() == system


# ── review round 1 (2026-10-12) ─────────────────────────────────────


def test_turn_context_is_part_of_the_estimate_and_does_not_inflate_calibration():
    state = PipelineState()
    state.messages = _conversation(4)
    without = estimate_prompt_tokens(state)
    state.shared["turn_context_text"] = "오늘 날짜와 이번 질문에 고른 기억. " * 200
    with_context = estimate_prompt_tokens(state)
    assert with_context > without + 1_000
    # The provider counts the turn context too: a report equal to our estimate keeps the ratio ~1.
    record_prompt_usage(state, with_context)
    assert state.shared[CALIBRATION_KEY] < 1.1


def test_repeated_estimates_reuse_per_message_counts(monkeypatch):
    from xgen_agent_runtime.core import token_estimate as te

    messages = _conversation(300, "a long enough message body " * 5)
    te.estimate_message_tokens(messages)
    calls = []
    real = te._raw_message_uncached
    monkeypatch.setattr(te, "_raw_message_uncached", lambda m: calls.append(1) or real(m))
    te.estimate_message_tokens(messages)
    assert calls == []
    messages[0]["content"] = "changed in place"
    te.estimate_message_tokens(messages)
    assert len(calls) == 1


def test_anchor_never_collapses_after_a_large_removal():
    state = PipelineState()
    pdf = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0=" * 10}}
    state.messages = [{"role": "user", "content": [pdf, {"type": "text", "text": "summarize"}]}] + _conversation(3)
    plain_before = estimate_prompt_tokens(state)
    record_prompt_usage(state, max(1, plain_before // 3))  # the provider counted far less
    state.messages[0]["content"] = [{"type": "text", "text": "[document removed]"}]
    state.shared.pop("_prompt_tokens_memo", None)
    assert estimate_prompt_tokens(state) > 0


@pytest.mark.asyncio
async def test_failed_summaries_are_capped_and_hard_pressure_uses_the_digest():
    from xgen_agent_runtime.core.compaction import LLM_FAILURES_KEY, MAX_SUMMARY_FAILURES

    state = PipelineState()
    state.model = "fake"
    client = _FailingClient()
    state.llm_client = client
    state.context_window_budget = 8_000
    state.messages = _conversation(40, "long message " * 30)
    compactor = _llm_compactor()

    first = await run_compaction(state, compactor, trigger="proactive", target_tokens=10)
    same_request = await run_compaction(state, compactor, trigger="proactive", target_tokens=10)
    assert first.get("skipped") == "no_change" and client.calls == 2  # one attempt, retried once
    assert same_request.get("skipped") == "already_summarized_for_this_request"
    for _ in range(MAX_SUMMARY_FAILURES):
        note_api_call(state)
        await run_compaction(state, compactor, trigger="proactive", target_tokens=10)
    assert state.shared[LLM_FAILURES_KEY] >= MAX_SUMMARY_FAILURES
    calls = client.calls
    note_api_call(state)
    soft = await run_compaction(state, compactor, trigger="proactive", target_tokens=10)
    assert soft.get("skipped") == "summary_model_failing" and client.calls == calls
    hard = await run_compaction(state, compactor, trigger="guard", target_tokens=10, hard=True)
    assert hard["summarized"] and client.calls == calls
    assert "[Summary of" in state.messages[0]["content"]


@pytest.mark.asyncio
async def test_a_summary_request_that_is_too_long_is_retried_shorter():
    class _TooLongOnce(_SummaryClient):
        def __init__(self):
            super().__init__()
            self.prompts = []

        async def create_message(self, *, model_config, messages, purpose: str = ""):
            self.prompts.append(messages[0]["content"])
            if len(self.prompts) == 1:
                raise APIError(
                    "prompt is too long: 60000 tokens > 32000 maximum",
                    category=ErrorCategory.TOKEN_LIMIT,
                )
            return await super().create_message(model_config=model_config, messages=messages)

    state = PipelineState()
    state.model = "fake"
    client = _TooLongOnce()
    state.llm_client = client
    state.context_window_budget = 32_000
    state.messages = _conversation(120, "이전 작업 내용을 길게 적은 메시지입니다. " * 40)
    state.shared[COMPACTION_CONTEXT_KEY] = {"trigger": "guard", "hard": True}

    await _llm_compactor().compact(state)

    assert len(client.prompts) == 2 and len(client.prompts[1]) < len(client.prompts[0]) * 0.8
    assert "SUMMARY OF EARLIER WORK" in state.messages[0]["content"]
    assert state.shared[CALIBRATION_KEY] > 1.0


def test_proactive_threshold_comes_before_the_guard_on_a_small_window():
    from xgen_agent_runtime.core.compaction import PROACTIVE_RATIO, input_budget_of, output_reserve

    state = PipelineState()
    state.context_window_budget = 32_000
    state.max_tokens = 8_192
    guard_limit = 32_000 - output_reserve(32_000, 8_192)  # what the host's TokenBudgetGuard allows
    assert input_budget_of(state) == guard_limit
    assert input_budget_of(state) * PROACTIVE_RATIO < guard_limit


def test_host_budget_leaves_the_guards_headroom():
    from xgen_agent_runtime.core.compaction import output_reserve
    from xgen_agent_runtime.host.context_budget import fit_input_to_budget

    fit = fit_input_to_budget(
        text="질문",
        rag_block="참조 문서 " * 40_000,
        system_prompt="system",
        history=None,
        registry=None,
        provider="vllm",
        model="m",
        max_tokens=8_192,
        window=32_000,
    )
    assert fit.input_clamped and fit.budget <= 32_000 - output_reserve(32_000, 8_192)


def test_memory_node_context_is_trimmed_before_the_question():
    from xgen_agent_runtime.host.context_budget import fit_input_to_budget

    question = "이번 질문은 그대로 가야 한다. " * 300
    context = "[이전 대화 기록]\n" + "오래된 대화 " * 30_000 + "가장 최근 대화"
    fit = fit_input_to_budget(
        text=question,
        rag_block="",
        system_prompt="system",
        history=None,
        registry=None,
        provider="vllm",
        model="m",
        max_tokens=4_096,
        window=32_000,
        history_context=context,
    )
    assert fit.text == question and not fit.input_clamped
    assert len(fit.history_context) < len(context) and fit.history_context.endswith("가장 최근 대화")


@pytest.mark.asyncio
async def test_overflow_caused_by_max_tokens_is_not_compacted():
    from xgen_agent_runtime.stages.s06_api.artifact.default.stage import APIStage

    provider = _OverflowOnce(
        "This model's maximum context length is 8192 tokens. However, you requested 8292 tokens "
        "(100 in the messages, 8192 in the completion)."
    )
    state = PipelineState()
    state.stream = False
    state.context_window_budget = 8_192
    state.messages = _conversation(30)
    state._context_compactor = TruncateCompactor(keep_last=4)

    with pytest.raises(APIError):
        await APIStage(provider=provider).execute(None, state)
    assert len(provider.requests) == 1 and len(state.messages) == 30


def test_cost_prune_keeps_attached_documents_and_hard_pressure_removes_them():
    from xgen_agent_runtime.core.context_prune import prune_messages

    def _messages():
        doc = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}}
        return [{"role": "user", "content": [doc, {"type": "text", "text": "read this"}]}] + _conversation(10)

    soft = _messages()
    prune_messages(soft)
    assert soft[0]["content"][0]["type"] == "document"
    hard = _messages()
    prune_messages(hard, strip_documents=True)
    assert hard[0]["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_the_current_request_stays_labelled_across_two_compactions():
    state = PipelineState()
    state.model = "fake"
    state.llm_client = _SummaryClient()
    state.messages = [{"role": "user", "content": "Refactor the billing module."}]
    for i in range(12):
        state.messages += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 50}]},
        ]
    compactor = _llm_compactor(keep_recent=2)
    await compactor.compact(state)
    for i in range(12, 24):
        state.messages += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 50}]},
        ]
    await compactor.compact(state)

    head = state.messages[0]["content"]
    assert "The request being worked on, verbatim:\nRefactor the billing module." in head
    assert "Earlier user messages" not in head


@pytest.mark.asyncio
async def test_a_deferred_background_summary_is_not_repeated_by_stage_4_and_keeps_its_pins():
    import asyncio

    from xgen_agent_runtime.core.compaction import input_budget_of
    from xgen_agent_runtime.stages.s02_context import ContextStage
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import PINNED_USERS_KEY
    from xgen_agent_runtime.stages.s04_guard.artifact.default.guards import TokenBudgetGuard
    from xgen_agent_runtime.stages.s04_guard.artifact.default.stage import GuardStage

    client = _SummaryClient()
    compactor = _llm_compactor(keep_recent=2)
    state = PipelineState()
    state.model = "fake"
    state.llm_client = client
    state.context_window_budget = 4_000
    state.max_tokens = 0
    band = input_budget_of(state)
    for size in range(1, 4_000):
        state.messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"REQ{i} " + "글" * size} for i in range(12)]
        if estimate_prompt_tokens(state) > band * 0.8:
            break
    assert estimate_prompt_tokens(state) <= band * 0.9

    stage2 = ContextStage(compactor=compactor)
    guard = GuardStage(guards=[TokenBudgetGuard(min_remaining_tokens=10)])
    guard.attach_budget_recovery(compactor)
    await stage2.execute("in", state)
    await guard.execute("in", state)
    await asyncio.sleep(0)
    assert stage2._bg_compaction is not None and len(state.messages) == 12  # nothing synchronous

    state.iteration = 0
    await stage2.execute("in", state)
    assert client.calls == 1
    assert state.messages[0]["content"].startswith(CHECKPOINT_MARKER)
    assert PINNED_USERS_KEY in state.shared


def test_claude_code_prompt_file_is_private(tmp_path, monkeypatch):
    import os
    import stat
    import tempfile

    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient
    from xgen_agent_runtime.llm_client.types import APIRequest

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = ClaudeCodeCLIClient(binary_path="/bin/true")
    argv = client._build_argv(
        APIRequest(model="sonnet", messages=[{"role": "user", "content": "hi"}], system="가" * 60_000, stream=True)
    )
    path = argv[argv.index("--system-prompt-file") + 1]
    assert stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
