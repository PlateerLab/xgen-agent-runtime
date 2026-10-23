"""Assignments(4.99.0): 사용자의 지시는 메모리에, `assignment` 태그 노트로.

모델이 필요할 때 스스로 남기고(memory_write), 메모리는 열린 assignment 노트를 매 턴 검색과 무관하게
되살린다. 시험은 그 계약만 고정한다: 매 턴 실린다, 끝난 것은 빠진다, 잘리면 잘렸다고 한다, 같은 노트를
고쳐 쓸 수 있다, 쓰기 도구가 없는 대화에는 쓰라고 하지 않는다.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.memory.provider import NoteDraft
from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider


def _note(provider, title: str, body: str, tags: List[str]) -> str:
    meta = asyncio.run(
        provider.notes().write(NoteDraft(title=title, body=body, category="assignments", tags=tags))
    )
    return meta.ref.filename


def _stage2(provider, text: str) -> PipelineState:
    from xgen_agent_runtime.memory.retriever import MemoryAwareRetriever
    from xgen_agent_runtime.stages.s02_context.artifact.default.stage import ContextStage

    # build_pipeline 과 같은 배선: 창 출처만 provider, 기억 층은 검색기 하나가 맡는다.
    stage = ContextStage(short_term_source=provider, retriever=MemoryAwareRetriever(provider), compaction_enabled=False)
    state = PipelineState(session_id="s1")
    state.messages = [{"role": "user", "content": text}]
    asyncio.run(stage.execute(None, state))
    return state


def test_open_assignments_come_back_every_turn_even_for_continue():
    provider = EphemeralMemoryProvider()
    fn = _note(provider, "3분기 매출 보고서", "지시: 부서별 표 3개, 부가세 제외.\n한 것: 데이터 정리.", ["assignment"])
    state = _stage2(provider, "계속")
    section = state.metadata.get("memory_assignments", "")
    assert "부서별 표 3개, 부가세 제외" in section and f"(note: {fn})" in section
    assert any(e["type"] == "memory.assignments_loaded" for e in state.events)
    # 같은 노트의 내용이 검색 층에 또 실리지 않는다(볼트 지도의 파일 목록에는 이름이 있다)
    assert "부가세 제외" not in str(state.metadata.get("memory_context", ""))


def test_finished_assignments_are_not_loaded():
    provider = EphemeralMemoryProvider()
    _note(provider, "끝난 일", "완료된 지시", ["assignment", "done"])
    _note(provider, "그만둔 일", "취소된 지시", ["assignment", "dropped"])
    _note(provider, "그냥 노트", "assignment 아님", ["topic"])
    state = _stage2(provider, "계속")
    assert not state.metadata.get("memory_assignments")


def test_what_does_not_fit_says_so_and_names_the_note():
    from xgen_agent_runtime.memory.assignments import render_assignments

    notes = [(f"a{i}.md", f"일 {i}", "조건 " * 800) for i in range(6)]
    text = render_assignments(notes, 3_000)
    assert "this note is cut here" in text and 'memory_read("a0.md")' in text
    assert "did not fit here" in text and "a5.md" in text


def test_memory_write_rewrites_a_note_by_filename():
    from xgen_agent_runtime.host.memory_tools import build_memory_tools

    provider = EphemeralMemoryProvider()
    tool = {t.name: t for t in build_memory_tools(provider)}["memory_write"]

    async def go():
        made = json.loads((await tool.execute({"title": "보고서", "content": "v1", "tags": "assignment"}, None)).content)
        fn = made["filename"]
        again = await tool.execute(
            {"title": "보고서", "content": "v2: 표 3개", "tags": "assignment,done", "filename": fn}, None
        )
        assert json.loads(again.content)["status"] == "updated"
        notes = await provider.notes().list(tag="assignment")
        assert len(notes) == 1 and (await provider.notes().read(fn)).body == "v2: 표 3개"
        missing = await tool.execute({"title": "x", "content": "y", "filename": "nope.md"}, None)
        assert missing.is_error

    asyncio.run(go())


def test_the_section_and_the_guidance():
    from xgen_agent_runtime.host._constants import (
        MEMORY_AUTO_PROMPT_BLOCK,
        MEMORY_PROMPT_BLOCK,
        MEMORY_READONLY_PROMPT_BLOCK,
    )
    from xgen_agent_runtime.stages.s03_system.artifact.default.builders import AssignmentsBlock

    state = PipelineState(session_id="s1")
    state.metadata["memory_assignments"] = "## 보고서 (note: a.md)\n부가세 제외"
    block = AssignmentsBlock()
    text = block.render(state)
    assert block.volatile and text.startswith("# Assignments") and "latest instruction wins" in text
    assert "memory_write" not in text, "the section itself never asks for a tool"
    assert "tagged `assignment`" in MEMORY_PROMPT_BLOCK and "`done`" in MEMORY_PROMPT_BLOCK
    for block_text in (MEMORY_READONLY_PROMPT_BLOCK, MEMORY_AUTO_PROMPT_BLOCK):
        assert "memory_write" not in block_text


def test_end_to_end_the_request_carries_the_users_instructions():
    from xgen_agent_runtime.core.state import TokenUsage
    from xgen_agent_runtime.host import runner
    from xgen_agent_runtime.llm_client import BaseClient, ClientCapabilities
    from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock

    provider = EphemeralMemoryProvider()
    _note(provider, "3분기 매출 보고서", "사용자 지시(원문): 부서별 표 3개로, 부가세는 빼고.", ["assignment"])
    seen: List[str] = []

    class _Scripted(BaseClient):
        provider = "fake"
        capabilities = ClientCapabilities()

        async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
            seen.append(repr(request))
            return APIResponse(
                content=[ContentBlock(type="text", text="이어서 합니다.")],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                model="m",
            )

    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=_Scripted(api_key="k"),
        stream=False, enable_compaction=False, memory_provider=provider,
    )
    runner.run_turn(pipe, "계속", PipelineState(session_id="s1", model="m"))
    assert seen and "# Assignments" in seen[0] and "부서별 표 3개로, 부가세는 빼고" in seen[0]


def test_a_matching_message_does_not_bring_the_same_note_back_twice():
    provider = EphemeralMemoryProvider()
    _note(provider, "3분기 매출 보고서", "부가세 제외, 부서별 표", ["assignment"])
    state = _stage2(provider, "부가세 제외 보고서 표")
    assert "부가세 제외" in state.metadata.get("memory_assignments", "")
    assert "부가세 제외" not in str(state.metadata.get("memory_context", ""))
