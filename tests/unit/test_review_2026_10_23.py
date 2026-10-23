"""2026-10-23 전체 흐름 검토에서 나온 것들 (4.101.0).

창(최대 길이), assignment 메모리, 정보 유실 방지의 실패 장면을 하나씩 고정한다.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from xgen_agent_runtime.core.state import PipelineState

# ── 창 ────────────────────────────────────────────────────────────────────


def test_a_window_learned_under_the_clients_name_is_used_on_the_next_turn():
    """vllm 노드는 "custom" 클라이언트로, Azure 는 "azure_foundry" 로 돈다. 예전에는 다음 턴에 못 찾았다."""
    from xgen_agent_runtime.host import token_budget as tb

    base = "http://xgen-model:8000/api/inference/v1"
    tb.learn_context_window("custom", "qwen-review", base, 16_384)
    probe = lambda _u, _m: 131_072  # noqa: E731
    assert tb.resolve_context_window("vllm", "qwen-review", 0, base_url=base, vllm_probe=probe) == 16_384


def test_catalog_window_values_given_as_text_count_and_bool_does_not():
    from xgen_agent_runtime.host.token_budget import _PROVIDER_TO_CATALOG_ID, _positive_int

    assert _positive_int("131072") == 131_072 and _positive_int("131,072") == 131_072
    assert _positive_int(True) is None and _positive_int("0") is None and _positive_int(None) is None
    assert _PROVIDER_TO_CATALOG_ID["azure"] == "azure_foundry"


@pytest.mark.parametrize(
    "message,overflow,count,limit",
    [
        # vLLM 0.20 serving (177 실측)
        (
            "This model's maximum context length is 2048 tokens. However, you requested 256 output "
            "tokens and your prompt contains at least 1793 input tokens, for a total of at least 2049 tokens.",
            True,
            1793,
            2048,
        ),
        # vLLM V1 engine
        (
            "The decoder prompt (length 41234) is longer than the maximum model length of 32768. "
            "Make sure that `max_model_len` is no smaller than the number of text tokens.",
            True,
            41234,
            32768,
        ),
        # 인코더 한도는 창이 아니다
        ("The encoder prompt (length 5000) is longer than the maximum model length of 4096.", False, 0, 0),
        # llama.cpp (llama-server, GGUF)
        (
            "Error code: 400 - {'error': {'code': 400, 'message': 'the request exceeds the available "
            "context size, try increasing it', 'type': 'exceed_context_size_error', "
            "'n_prompt_tokens': 9000, 'n_ctx': 8192}}",
            True,
            9000,
            8192,
        ),
    ],
)
def test_self_hosted_overflow_messages_are_read(message, overflow, count, limit):
    from xgen_agent_runtime.core.errors import (
        is_context_overflow_message,
        parse_overflow_limit,
        parse_overflow_tokens,
    )

    assert is_context_overflow_message(message) is overflow
    if overflow:
        assert parse_overflow_tokens(message) == count
        assert parse_overflow_limit(message) == limit


def test_the_budget_keeps_room_for_the_output_the_client_really_sends():
    from xgen_agent_runtime.core.compaction import ANSWER_ROOM_KEY, input_budget_of
    from xgen_agent_runtime.llm_client.thinking import output_tokens_sent

    sent = output_tokens_sent("anthropic", "claude-sonnet-4-5-20250929", "high", 8192)
    assert sent > 8192, "thinking raises max_tokens on budget models"
    assert output_tokens_sent("vllm", "qwen3-32b", "high", 8192) == 8192

    state = PipelineState()
    state.context_window_budget, state.max_tokens = 200_000, 8192
    before = input_budget_of(state)
    state.metadata[ANSWER_ROOM_KEY] = sent
    assert input_budget_of(state) == before - (sent - 8192)


# ── assignment 메모리 ────────────────────────────────────────────────────


def _tools(provider):
    from xgen_agent_runtime.host.memory_tools import build_memory_tools

    return {t.name: t for t in build_memory_tools(provider)}


def test_a_rewrite_keeps_what_it_does_not_give_and_an_assignment_stays_one():
    from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider

    provider = EphemeralMemoryProvider()
    tools = _tools(provider)

    async def go():
        made = json.loads(
            (
                await tools["memory_write"].execute(
                    {"title": "표 편집", "content": "v1", "tags": "assignment", "category": "projects"},
                    SimpleNamespace(session_id="A"),
                )
            ).content
        )
        done = json.loads(
            (
                await tools["memory_write"].execute(
                    {"title": "표 편집", "content": "끝남", "tags": "#done", "filename": made["filename"]},
                    SimpleNamespace(session_id="A"),
                )
            ).content
        )
        assert done["category"] == "projects"
        assert set(done["tags"]) == {"done", "assignment"}
        assert len(await provider.notes().list(tag="assignment")) == 1

    asyncio.run(go())


def test_a_rewrite_by_a_loose_name_writes_the_stored_note_not_a_copy():
    from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider

    provider = EphemeralMemoryProvider()
    notes = provider.notes()
    exact_read = notes.read

    async def loose_read(name):  # 호스트 저장소처럼 카테고리 접두를 풀어 준다
        return await exact_read(name.split("/", 1)[-1])

    notes.read = loose_read
    provider.notes = lambda: notes
    tools = _tools(provider)

    async def go():
        made = json.loads(
            (
                await tools["memory_write"].execute(
                    {"title": "보고서", "content": "v1", "tags": "assignment"}, SimpleNamespace(session_id="A")
                )
            ).content
        )
        await tools["memory_write"].execute(
            {"title": "보고서", "content": "v2", "filename": "projects/" + made["filename"]},
            SimpleNamespace(session_id="B"),
        )
        listed = await notes.list(tag="assignment")
        assert [m.ref.filename for m in listed] == [made["filename"]]
        assert (await exact_read(made["filename"])).body == "v2"

    asyncio.run(go())


def test_memory_read_shows_only_the_readers_own_conversations():
    from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider

    provider = EphemeralMemoryProvider()
    tools = _tools(provider)

    async def go():
        made = json.loads(
            (
                await tools["memory_write"].execute(
                    {"title": "공유 지시", "content": "v1", "tags": "assignment"},
                    SimpleNamespace(session_id="mine"),
                )
            ).content
        )
        await tools["memory_write"].execute(
            {"title": "공유 지시", "content": "v2", "filename": made["filename"]},
            SimpleNamespace(session_id="someone-else"),
        )
        provider.visible_conversations = lambda ids: {i for i in ids if i == "mine"}
        read = json.loads((await tools["memory_read"].execute({"filename": made["filename"]}, None)).content)
        assert [w["conversation"] for w in read["written_in"]] == ["mine"]
        assert "someone-else" not in json.dumps(read, ensure_ascii=False)
        assert "1 conversation" in read["written_in_by_others"]

    asyncio.run(go())


def test_the_file_backend_keeps_written_in():
    from xgen_agent_runtime.memory.providers.file import frontmatter as fm

    meta = {"title": "t", "written_in": [{"conversation": "A", "at": "2026-10-23T10:00:00+09:00"}]}
    parsed, _ = fm.split(fm.dump(meta) + "body\n")
    assert parsed["written_in"] == meta["written_in"]


# ── 정보 유실 방지 ───────────────────────────────────────────────────────


def _clip_context(tmp_path, state):
    from xgen_agent_runtime.tools.base import ToolContext

    return ToolContext(session_id="s", storage_path=str(tmp_path), state_view=state)


def test_a_result_clipped_to_the_window_is_recorded_whole_and_a_clipped_read_is_not_a_read(tmp_path):
    from xgen_agent_runtime.core.originals import keep_clipped_originals, original_of
    from xgen_agent_runtime.stages.s10_tool.persistence import _fit_to_window
    from xgen_agent_runtime.tools.base import ToolResult
    from xgen_agent_runtime.tools.built_in._file_witness import WITNESSED_KEY

    state = PipelineState()
    state.context_window_budget, state.max_tokens = 8_192, 1_024
    body = "".join(f"line {i} the middle has MIDDLE-{i}\n" for i in range(4_000))
    result = ToolResult(content=body, state_mutations={WITNESSED_KEY: ["/w/big.py"]})

    fitted = _fit_to_window(result, tool_name="Read", context=_clip_context(tmp_path, state), tool_use_id="t1")

    assert len(fitted.display_text) < len(body) and "does not count as reading the file" in fitted.display_text
    assert WITNESSED_KEY not in fitted.state_mutations
    message = {"role": "user", "content": [fitted.to_api_format("t1")]}
    state.messages.append(message)
    keep_clipped_originals(state, message)
    assert message["content"][0]["content"] != body, "the request keeps the clip"
    assert original_of(state, message)["content"][0]["content"] == body, "memory records it whole"


def test_older_turns_say_their_tools_are_left_out_and_name_history_only_when_offered():
    from xgen_agent_runtime.memory.short_term_window import WindowConfig, build_window

    turns = [
        {"role": "user", "content": "첫 지시"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "ok"}]},
        {"role": "assistant", "content": "했습니다"},
        {"role": "user", "content": "둘째"},
        {"role": "assistant", "content": "네"},
        {"role": "user", "content": "셋째"},
        {"role": "assistant", "content": "네"},
    ]
    rows = [SimpleNamespace(role=t["role"], content=t["content"]) for t in turns]
    readable, _ = build_window(rows, WindowConfig(full_turns=2, dialogue_turns=3))
    text = json.dumps(readable, ensure_ascii=False)
    assert "calls and results not shown here; ConversationHistory shows them" in text
    hidden, _ = build_window(rows, WindowConfig(full_turns=2, dialogue_turns=3, history_readable=False))
    assert "ConversationHistory" not in json.dumps(hidden, ensure_ascii=False)


def test_an_ssh_timeout_keeps_what_the_command_printed(tmp_path, monkeypatch):
    from xgen_agent_runtime.tools._ssh import SSHCommandTimeout
    from xgen_agent_runtime.tools.built_in import ssh_tools

    test_mod = pytest.importorskip("tests.unit.test_ssh_tools")

    async def fake_exec(server, command, *, timeout, cwd, sudo, **kw):
        raise SSHCommandTimeout(timeout, "step 1 ok\nstep 2 ok\n", "")

    monkeypatch.setattr(ssh_tools, "ssh_exec", fake_exec)
    res = asyncio.run(
        ssh_tools.SshRunTool().execute(
            {"server": "prod", "command": "make"}, test_mod._ctx(tmp_path, [test_mod._PW_SERVER])
        )
    )
    assert res.is_error and res.content["timed_out"] is True
    assert "step 2 ok" in res.display_text
