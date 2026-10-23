"""정직성 감사(2026-10-23) — 모델과 사용자가 받는 것이 실제와 같아야 한다.

각 시험은 감사에서 찾은 결함 하나를 고정한다. 결함이 되살아나면 결과가 사실과 반대가 되거나
(정상 답이 실패로), 잘린 것이 전체로 보이거나, 실행된 도구가 "거부됐다" 고 보고된다.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List

import pytest

from xgen_agent_runtime import PipelineState
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.core.state import TokenUsage


class _Scripted(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, responses: List[APIResponse], **kw: Any) -> None:
        super().__init__(**kw)
        self._responses = list(responses)

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        return self._responses.pop(0) if self._responses else _text("(no more)")


def _text(t: str) -> APIResponse:
    return APIResponse(
        content=[ContentBlock(type="text", text=t)],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        model="m",
    )


def _pipe(responses: List[APIResponse]):
    return runner.build_pipeline(
        name="t",
        provider="openai",
        model="m",
        api_key="k",
        llm_client=_Scripted(responses, api_key="k"),
        stream=False,
        enable_compaction=False,
    )


# ── 정상 답 속의 [Error…]·[Blocked…] 는 신호가 아니다 ─────────────────────


@pytest.mark.parametrize(
    "answer",
    [
        "오류 코드는 [Error codes](https://example.com/errors) 를 보세요.",
        "검사 결과: [Errors] 3건, 모두 고쳤습니다.",
        "[ERROR] 로 시작하는 로그 줄을 찾아 고쳤습니다.",
        "차단 목록은 [Blocked users] 메뉴에 있습니다.",
    ],
)
def test_a_correct_answer_mentioning_errors_is_not_a_failed_turn(answer):
    out = runner.run_turn(_pipe([_text(answer)]), "질문", PipelineState(session_id="s", model="m"))
    assert out == answer


def test_the_protocol_signal_at_the_end_still_works():
    from xgen_agent_runtime.stages.s09_parse.artifact.default.signals import RegexDetector
    from xgen_agent_runtime.stages.s09_parse.interface import CompletionSignal

    d = RegexDetector()
    assert d.detect("다 했습니다. [TASK_COMPLETE]")[0] == CompletionSignal.COMPLETE
    assert d.detect("승인이 필요합니다. [BLOCKED: 관리자 승인]") == (CompletionSignal.BLOCKED, "관리자 승인")
    assert d.detect("[BLOCKED: x] 뒤에 글이 더 있으면 신호가 아니다")[0] == CompletionSignal.NONE


def test_a_status_field_in_a_json_answer_is_not_a_signal():
    from xgen_agent_runtime.stages.s09_parse.artifact.default.signals import StructuredDetector
    from xgen_agent_runtime.stages.s09_parse.interface import CompletionSignal

    d = StructuredDetector()
    assert d.detect(json.dumps({"status": "error", "rows": 3}))[0] == CompletionSignal.NONE
    assert d.detect(json.dumps({"signal": "complete"}))[0] == CompletionSignal.COMPLETE


# ── 비스트리밍 답은 그 턴에 쓴 글 전부다 ─────────────────────────────────


def test_the_non_streaming_answer_keeps_what_was_written_before_the_last_step():
    first = APIResponse(
        content=[
            ContentBlock(type="text", text="## 보고서\n본문 내용"),
            ContentBlock(type="tool_use", tool_use_id="t1", tool_name="NoSuchTool", tool_input={}),
        ],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        model="m",
    )
    out = runner.run_turn(
        _pipe([first, _text("저장했습니다.")]), "보고서 써서 저장해", PipelineState(session_id="s", model="m")
    )
    assert "## 보고서\n본문 내용" in out and out.rstrip().endswith("저장했습니다.")


# ── 잘린 도구 결과는 원본을 다시 읽을 수 있다 ─────────────────────────────


def test_a_clipped_result_keeps_its_end_and_its_original(tmp_path):
    from types import SimpleNamespace

    from xgen_agent_runtime.stages.s10_tool.persistence import maybe_persist_large_result
    from xgen_agent_runtime.tools.base import ToolCapabilities, ToolContext, ToolResult
    from xgen_agent_runtime.tools.built_in.tool_result_read_tool import ToolResultReadTool

    body = "\n".join(f"line {i} " + "x" * 80 for i in range(4000)) + "\nSTDERR:\nboom\nExit code: 2"
    state = SimpleNamespace(context_window_budget=32_000, max_tokens=4096, shared={})
    ctx = ToolContext(session_id="s", working_dir=str(tmp_path / "ws"), storage_path=str(tmp_path / "store"))
    ctx.state_view = state
    out = maybe_persist_large_result(
        ToolResult(content=body),
        tool_use_id="bash_1",
        tool_name="Bash",
        capabilities=ToolCapabilities(max_result_chars=100_000),
        context=ctx,
    )
    view = out.display_text or ""
    assert "clipped" in view and 'ToolResultRead(id="bash_1"' in view
    assert view.rstrip().endswith("Exit code: 2"), "the end of the result (exit code, STDERR) is kept"

    read = asyncio.run(ToolResultReadTool().execute({"id": "bash_1", "offset": 200_000, "limit": 1000}, ctx))
    assert not read.is_error
    assert read.content.startswith(f"[Bash result bash_1: chars 200000-201000 of {len(body)}")
    assert body[200_000:201_000] in read.content


# ── 실행된 도구를 "거부됐다" 고 하지 않는다 ───────────────────────────────


def test_a_result_withheld_after_running_says_the_tool_ran():
    from xgen_agent_runtime.tools.errors import ToolError, ToolErrorCode

    err = ToolError.result_withheld("Bash", "run_budget_exceeded")
    assert err.code == ToolErrorCode.RESULT_WITHHELD
    assert "ran and its effects happened" in err.message and "Access denied" not in err.message


def test_a_missing_result_does_not_claim_the_tool_did_not_run():
    from xgen_agent_runtime.core.message_repair import repair_dangling_tool_calls

    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "x1", "name": "Bash", "input": {}}]},
    ]
    repair_dangling_tool_calls(messages)  # 제자리에서 고친다
    stub = messages[-1]["content"][0]["content"]
    assert "may or may not have run" in stub and "did not finish" not in stub


# ── 기억 층의 잘림은 표시된다 ─────────────────────────────────────────────


def test_the_session_summary_is_cut_with_a_marker():
    from xgen_agent_runtime.memory.retriever import _clip_marked

    summary = "## Summary\n" + "a" * 3000 + "\n## Open Threads\n- finish the report"
    body = _clip_marked(summary, 1000, "this conversation's summary", 'read all of it with ConversationHistory(view="summary")')
    assert len(body) <= 1000
    assert "is cut here" in body and f"of {len(summary)} chars shown" in body


# ── 창 밖의 지난 턴은 원문으로 다시 읽을 수 있다 ───────────────────────────


def _history_provider():
    from datetime import datetime, timedelta, timezone

    from xgen_agent_runtime.memory.provider import Turn

    t0 = datetime(2026, 10, 23, tzinfo=timezone.utc)
    rows = []
    for i in range(12):
        at = t0 + timedelta(minutes=3 * i)
        rows += [
            Turn(role="user", content=f"요청 {i}: 보고서 {i}번 작성", timestamp=at),
            Turn(
                role="assistant",
                content=[{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"echo {i}"}}],
                timestamp=at,
            ),
            Turn(
                role="user",
                content=[{"type": "tool_result", "tool_use_id": f"t{i}", "content": f"result-{i}-detail"}],
                timestamp=at,
            ),
            Turn(role="assistant", content=f"완료 {i}", timestamp=at),
        ]

    class _STM:
        async def recent(self, n=20):
            return rows[-n:]

        async def read_summary(self):
            return "## Summary\n긴 요약 본문"

    class _Provider:
        def stm(self):
            return _STM()

    return _Provider()


def test_conversation_history_lists_and_reads_old_turns_word_for_word():
    from xgen_agent_runtime.host.memory_tools import conversation_history

    p = _history_provider()
    listing = asyncio.run(conversation_history(p, {"view": "turns"})).content
    assert listing.startswith("[turns 1-12 of 12")
    assert "1. " in listing and "요청 0: 보고서 0번 작성" in listing and "tool calls: 1" in listing

    first = asyncio.run(conversation_history(p, {"view": "turn", "turn": 1})).content
    assert "[turn 1 of 12" in first
    assert "echo 0" in first and "result-0-detail" in first and "완료 0" in first

    found = asyncio.run(conversation_history(p, {"view": "search", "query": "result-7-detail"})).content
    assert "turn 8 (1 match)" in found

    summary = asyncio.run(conversation_history(p, {"view": "summary"})).content
    assert "긴 요약 본문" in summary


# ── 창에 맞추려고 줄인 것은 요청 사본뿐이다: 기억은 원본을 기록한다 ─────────


def test_memory_records_the_original_of_a_result_shortened_to_fit():
    from xgen_agent_runtime.core.compaction import _prune
    from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider
    from xgen_agent_runtime.memory.strategy import ProviderDrivenStrategy

    full = "line\n" * 6000 + "FINAL: 3 failed"
    state = PipelineState(session_id="s", model="m")
    state.messages = [
        {"role": "user", "content": "테스트 돌려"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "pytest"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b1", "content": full}]},
    ] + [{"role": "assistant", "content": "…"}, {"role": "user", "content": "계속"}] * 4
    _prune(state, aggressive=False, trigger="test")
    assert len(state.messages[2]["content"][0]["content"]) < len(full)  # 요청 사본은 줄었다

    provider = EphemeralMemoryProvider()
    asyncio.run(ProviderDrivenStrategy(provider).update(state))
    rows = asyncio.run(provider.stm().recent(50))
    stored = [b for r in rows if isinstance(r.content, list) for b in r.content if b.get("type") == "tool_result"]
    assert stored and stored[0]["content"] == full, "memory keeps what the tool really returned"


# ── 기억 주입 상한은 모델 창을 따른다 ──────────────────────────────────────


def test_memory_injection_follows_the_window():
    from xgen_agent_runtime.host.context_budget import memory_inject_cap_chars
    from xgen_agent_runtime.memory.provider import DEFAULT_MAX_INJECT_CHARS

    assert memory_inject_cap_chars(32_768, DEFAULT_MAX_INJECT_CHARS) == 3_276
    assert memory_inject_cap_chars(200_000, DEFAULT_MAX_INJECT_CHARS) == 20_000
    assert memory_inject_cap_chars(1_000_000, DEFAULT_MAX_INJECT_CHARS) == 100_000
    # 호스트가 정한 값은 천장이다
    assert memory_inject_cap_chars(1_000_000, 5_000) == 5_000
    assert memory_inject_cap_chars(0, DEFAULT_MAX_INJECT_CHARS) == DEFAULT_MAX_INJECT_CHARS


# ── 공급자가 알려 준 창은 다음 턴도 쓴다 ───────────────────────────────────


def test_a_window_the_provider_reported_is_used_by_later_turns(monkeypatch):
    from xgen_agent_runtime.host import token_budget as tb

    monkeypatch.setattr(tb, "_LEARNED_WINDOWS", {})
    base = "http://10.0.0.5:8000/v1/"
    assert tb.resolve_context_window("openai", "gpt-4o", base_url=base) == 128_000
    tb.learn_context_window("openai", "gpt-4o", base, 65_536)
    assert tb.resolve_context_window("openai", "gpt-4o", base_url=base.rstrip("/")) == 65_536
    # 다른 서버·모델은 그대로
    assert tb.resolve_context_window("openai", "gpt-4o", base_url="http://other/v1") == 128_000
    # 노드가 더 크게 적어도 공급자가 거절하는 크기는 쓰지 않는다
    assert tb.resolve_context_window("openai", "gpt-4o", override=200_000, base_url=base) == 65_536
    # 오래된 값은 믿지 않는다(배포가 창을 늘렸을 수 있다)
    key = tb._window_key("openai", "gpt-4o", base)
    tb._LEARNED_WINDOWS[key] = (65_536, 0.0)
    assert tb.resolve_context_window("openai", "gpt-4o", base_url=base) == 128_000


# ── 런타임 지시는 사용자의 말도, 도구의 결과도 아니다 ─────────────────────


def test_a_runtime_directive_does_not_open_a_new_user_turn():
    from types import SimpleNamespace as NS

    from xgen_agent_runtime.memory.rollup import _flatten_turn
    from xgen_agent_runtime.memory.transcript import group_logical_turns

    rows = [
        NS(role="user", content="보고서 써 줘"),
        NS(role="assistant", content="다 썼습니다"),
        NS(role="user", content="[Deliverable check — automatic, before finishing]\nreport.md 가 없습니다"),
        NS(role="assistant", content="저장했습니다"),
    ]
    turns = group_logical_turns(rows, 10)
    assert len(turns) == 1, "the check is part of the same turn"
    assert _flatten_turn(rows[2]).startswith("[runtime] ")


def test_a_turn_budget_note_is_not_recorded_as_tool_output():
    from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider
    from xgen_agent_runtime.memory.strategy import ProviderDrivenStrategy
    from xgen_agent_runtime.stages.s16_loop.turn_budget import _append_note

    state = PipelineState(session_id="s", model="m")
    state.messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "b1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b1", "content": "real output"}]},
    ]
    assert _append_note(state, "[Turn budget: wrap up now]")
    assert "[Turn budget" in state.messages[-1]["content"][0]["content"]  # the model sees it

    provider = EphemeralMemoryProvider()
    asyncio.run(ProviderDrivenStrategy(provider).update(state))
    rows = asyncio.run(provider.stm().recent(10))
    stored = [b for r in rows if isinstance(r.content, list) for b in r.content if b.get("type") == "tool_result"]
    assert stored[0]["content"] == "real output"


# ── CLI 백엔드가 안에서 돌린 도구도 기록에 남는다 ──────────────────────────


def test_tools_a_cli_ran_are_in_the_record_where_they_happened():
    from tests.unit.test_s06_chunk_forwarding import FULL_CHUNK_SCRIPT, _ChunkClient, _stage_state
    from xgen_agent_runtime.stages.s06_api import APIStage

    script = [
        {"type": "text_delta", "text": "파일을 볼게요. "},
        *FULL_CHUNK_SCRIPT,
    ]
    client = _ChunkClient(script, subprocess_backed=True)
    state = _stage_state(client)
    response = asyncio.run(APIStage().execute(None, state))

    assert response.text == "answer"  # 화면과 Stage 9 가 받는 답은 그대로
    roles = [(m["role"], [b.get("type") for b in m["content"]] if isinstance(m["content"], list) else "str") for m in state.messages]
    assert roles == [
        ("user", "str"),
        ("assistant", ["text", "tool_use"]),
        ("user", ["tool_result"]),
        ("assistant", ["text"]),
    ]
    use = state.messages[1]["content"][1]
    assert use["name"] == "Read" and use["input"] == {"file_path": "/tmp/x"}
    assert state.messages[1]["content"][0]["text"] == "파일을 볼게요. "
    assert state.messages[2]["content"][0]["content"] == "data"
    assert state.messages[3]["content"][0]["text"] == "answer"


def test_an_api_backend_keeps_one_assistant_message():
    from tests.unit.test_s06_chunk_forwarding import _ChunkClient, _response, _stage_state
    from xgen_agent_runtime.stages.s06_api import APIStage

    client = _ChunkClient([{"type": "text_delta", "text": "hi"}, {"type": "message_complete", "response": _response("hi")}])
    state = _stage_state(client)
    asyncio.run(APIStage().execute(None, state))
    assert [m["role"] for m in state.messages] == ["user", "assistant"]


# ── 압축 체크포인트는 잃은 것이 있다고 말한다 ───────────────────────────────


def test_conversation_history_reads_this_turn_before_it_is_recorded():
    from xgen_agent_runtime.host.memory_tools import conversation_history

    p = _history_provider()
    state = PipelineState(session_id="s", model="m")
    state.messages = [
        {"role": "user", "content": "[Context checkpoint] The conversation before this point ..."},
        {"role": "assistant", "content": "[Context checkpoint received; continuing from it.]"},
        {"role": "user", "content": "이번 요청: 통계 뽑기"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "q1", "name": "Bash", "input": {"command": "wc -l data.csv"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "q1", "content": "48213 data.csv"}]},
    ]
    found = asyncio.run(conversation_history(p, {"view": "search", "query": "48213"}, state=state)).content
    assert "turn 13" in found
    listing = asyncio.run(conversation_history(p, {"view": "turns"}, state=state)).content
    assert "Context checkpoint" not in listing and "이번 요청: 통계 뽑기" in listing


def test_the_checkpoint_says_what_it_left_out():
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
        CHECKPOINT_ACK,
        LLMSummaryCompactor,
    )

    state = PipelineState(session_id="s", model="m")
    state.context_window_budget = 32_768
    state.tools = [{"name": "ConversationHistory", "description": "", "input_schema": {}}]
    long_ask = "요구사항 " * 900
    state.messages = []
    for i in range(12):
        state.messages += [
            {"role": "user", "content": f"{i}번째 지시: {long_ask}"},
            {"role": "assistant", "content": f"{i}번째 답"},
        ]
    state.messages.append({"role": "user", "content": "마지막 지시"})
    asyncio.run(LLMSummaryCompactor().compact(state))
    head = state.messages[0]["content"]
    text = head if isinstance(head, str) else head[0]["text"]
    assert "A summary loses detail" in text and "ConversationHistory" in text
    assert "did not fit here and are only in the summary" in text
    assert "Understood" not in CHECKPOINT_ACK  # the runtime wrote it, and it reads that way


# ── 셸 출력은 끝(종료 코드·실패 요약)을 잃지 않는다 ─────────────────────────


def test_long_bash_output_keeps_its_end(tmp_path):
    from xgen_agent_runtime.tools.base import HOST_IS_EXECUTION_TARGET, ToolContext
    from xgen_agent_runtime.tools.built_in import bash_tool
    from xgen_agent_runtime.tools.built_in.bash_tool import BashTool

    ctx = ToolContext(working_dir=str(tmp_path), extras={HOST_IS_EXECUTION_TARGET: True})
    cmd = "python3 -c \"print('x' * 300000); print('FAILED tests/test_a.py::test_b')\"; exit 3"
    result = asyncio.run(BashTool().execute({"command": cmd}, ctx))
    assert result.is_error
    text = result.content
    assert "FAILED tests/test_a.py::test_b" in text and text.rstrip().endswith("Exit code: 3")
    assert len(text) > 300_000  # 전부 남는다: 모델에게 보이는 몫은 Stage 10 이 원본을 남기고 줄인다
    assert bash_tool._keep_ends("a" * 50 + "END", 20, "output").endswith("END")


def test_a_local_command_that_times_out_keeps_what_it_printed(tmp_path):
    from xgen_agent_runtime.tools.base import HOST_IS_EXECUTION_TARGET, ToolContext
    from xgen_agent_runtime.tools.built_in.bash_tool import BashTool

    ctx = ToolContext(working_dir=str(tmp_path), extras={HOST_IS_EXECUTION_TARGET: True})
    cmd = "echo 'step 1 done'; echo 'compile error: x' >&2; sleep 30"
    result = asyncio.run(BashTool().execute({"command": cmd, "timeout": 1000}, ctx))
    assert result.is_error
    assert "step 1 done" in result.content and "compile error: x" in result.content
    assert "timed out after 1000ms" in result.content


def test_a_forged_tools_cut_json_is_not_a_success():
    from types import SimpleNamespace as NS

    from xgen_agent_runtime.host.forged_tools import ForgedScriptTool

    tool = ForgedScriptTool(NS(name="big_report"), workspace_dir="/tmp")
    cut = '{"rows": [1, 2, 3' + "\n[xgen: output cut here: the first 200,000 of 900,000 bytes are shown; ...]\n"
    res = tool._interpret(cut, "", 0, cut=True)
    assert res.is_error and "불완전" in res.content
    whole = tool._interpret('{"rows": [1]}', "", 0)
    assert not whole.is_error and whole.content == {"rows": [1]}


# ── 실행 카드는 일어난 일을 그대로 적는다 ─────────────────────────────────


def test_an_execution_card_from_a_dict_input_is_written_and_a_stop_is_not_a_failure():
    from xgen_agent_runtime.host.execution_record import record_turn_execution
    from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider

    provider = EphemeralMemoryProvider()
    asyncio.run(
        record_turn_execution(
            provider,
            input_text={"input_str": "분기 보고서 정리", "attachments": []},
            output_text="x" * 5000,
            success=False,
            duration_ms=1200,
            session_id="s",
            cancelled=True,
        )
    )
    metas = asyncio.run(provider.notes().list(category="daily"))
    assert len(metas) == 1, "a dict input used to make the card write fail silently"
    note = asyncio.run(provider.notes().read(metas[0].ref.filename))
    assert "분기 보고서 정리" in note.body and "[⏹]" in note.body and "stopped" in note.body
    assert "[❌]" not in note.body
    assert "excerpt: the first 1,500 of 5,000 chars" in note.body


# ── 입력 맞춤이 뺀 것은 빠졌다고 말한다 ───────────────────────────────────


def test_input_fit_marks_left_out_history_and_reference_material():
    from xgen_agent_runtime.host.context_budget import RAG_DROPPED_NOTE, fit_input_to_budget

    history = []
    for i in range(30):
        history += [{"role": "user", "content": f"질문 {i} " + "가" * 400}, {"role": "assistant", "content": f"답 {i} " + "나" * 400}]
    fit = fit_input_to_budget(
        text="이번 질문",
        rag_block="참조 " * 20_000,
        system_prompt="sys",
        history=history,
        registry=None,
        provider="claude_code",
        model="m",
        max_tokens=1024,
        window=8_192,
        history_compressible=False,
    )
    assert fit.history is not None and len(fit.history) < len(history)
    first = fit.history[0]["content"]
    assert first.startswith("[Earlier in this conversation:") and "left out here" in first
    assert history[len(history) - len(fit.history)]["content"].startswith("질문"), "the record is untouched"
    assert fit.rag_block == RAG_DROPPED_NOTE or "생략" in fit.rag_block


# ── 고정 사실은 잘려도 잘렸다고, 빠져도 빠졌다고 말한다 ─────────────────────


def test_pinned_notes_are_cut_with_a_marker_and_left_out_by_name():
    from xgen_agent_runtime.memory.pinned import render_pinned

    big = "사실 " * 2000
    block = render_pinned(
        [
            ("__facts__.md", "Facts", "이름: 장하렴"),
            ("evergreen.md", "Evergreen", big),
            ("p1.md", "P1", "x" * 900),
            ("p2.md", "P2", "y" * 900),
            ("p3.md", "P3", "z" * 900),
        ],
        3000,
    )
    assert block.startswith("## Facts\n이름: 장하렴")
    assert 'memory_read("evergreen.md") shows all of it' in block
    assert 'memory_read("p2.md") shows all of it' in block  # 들어갈 만큼은 싣고 잘렸다고 말한다
    assert "1 more pinned note(s) did not fit here: p3.md" in block
    assert len(block) <= 3000 + 300
    # 첫 노트가 예산보다 커도 빈 문자열이 아니다(예전 XGEN DB provider)
    assert render_pinned([("a.md", "A", big)], 1000)


def test_the_vllm_probe_gets_the_models_key_when_the_host_takes_it():
    from xgen_agent_runtime.host.turn_executor import _vllm_probe_with_key

    seen = []

    class _New:
        def fetch_vllm_max_model_len(self, base_url, model, *, api_key=None):
            seen.append(api_key)
            return 131_072

    class _Old:
        def fetch_vllm_max_model_len(self, base_url, model):
            return 65_536

    assert _vllm_probe_with_key(_New(), "sk-1")("http://h/v1", "m") == 131_072 and seen == ["sk-1"]
    assert _vllm_probe_with_key(_Old(), "sk-1")("http://h/v1", "m") == 65_536


def test_an_oversized_stm_record_stays_a_valid_record():
    from xgen_agent_runtime.memory.providers.file.stm_store import MAX_RECORD_BYTES, _bound_record_line

    rec = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "가" * 60_000 + "END"}],
    }
    line = _bound_record_line(json.dumps(rec, ensure_ascii=False))
    back = json.loads(line)  # 예전에는 줄 가운데에서 잘려 JSON 이 깨졌다
    body = back["content"][0]["content"]
    assert len(line.encode("utf-8")) <= MAX_RECORD_BYTES
    assert body.endswith("END") and "not kept in this record" in body
