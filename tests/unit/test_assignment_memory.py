"""Assignments(4.100.0): 사용자의 지시는 `assignment` 태그 메모리 노트, 찾는 것은 필요할 때.

목표 장면: 나중에 사용자가 "예전에 했던 그 편집에서 하나 고치고 싶어" 라고 하면, 메모리로 그 일을 찾아 이어 간다.
고정하는 것:
* 매 턴 싣지 않는다 — 진행 중인 대화는 이미 맥락에 있고, 노트를 또 실으면 같은 지시가 두 번 들어간다.
* 메시지가 그 일을 가리키면 평소 기억 검색이 찾는다.
* 노트를 쓴 대화는 메모리가 적는다(``written_in``), 그 대화의 원문을 다시 읽을 수 있다(호스트가 허락할 때).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.memory.provider import NoteDraft
from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider


def _stage2(provider, text: str, session: str = "s1") -> PipelineState:
    from xgen_agent_runtime.memory.retriever import MemoryAwareRetriever
    from xgen_agent_runtime.stages.s02_context.artifact.default.stage import ContextStage

    # build_pipeline 과 같은 배선: 창 출처만 provider, 기억 층은 검색기 하나가 맡는다.
    stage = ContextStage(
        short_term_source=provider, retriever=MemoryAwareRetriever(provider), compaction_enabled=False
    )
    state = PipelineState(session_id=session)
    state.messages = [{"role": "user", "content": text}]
    asyncio.run(stage.execute(None, state))
    return state


def _write_note(provider, title, body, tags):
    return asyncio.run(
        provider.notes().write(NoteDraft(title=title, body=body, category="assignments", tags=tags))
    ).ref.filename


def _tools(provider):
    from xgen_agent_runtime.host.memory_tools import build_memory_tools

    return {t.name: t for t in build_memory_tools(provider)}


def test_assignments_are_not_put_into_every_turn():
    provider = EphemeralMemoryProvider()
    _write_note(provider, "보고서 표 편집", "사용자: 부서별 표 3개, 부가세 제외", ["assignment"])
    state = _stage2(provider, "오늘 날씨 어때?")
    prompt = str(state.metadata.get("memory_context", "")) + str(state.metadata.get("memory_pinned", ""))
    assert "부가세 제외" not in prompt, "an unrelated turn does not carry the assignment"
    assert "memory_assignments" not in state.metadata


def test_a_message_about_that_work_finds_it_through_memory_search():
    provider = EphemeralMemoryProvider()
    _write_note(provider, "보고서 표 편집", "사용자: 부서별 표 3개, 부가세 제외. 한 것: report.md 표 편집", ["assignment"])
    state = _stage2(provider, "예전에 했던 보고서 표 편집에서 하나 고치고 싶어")
    assert "부가세 제외" in str(state.metadata.get("memory_context", ""))


def test_memory_stamps_the_conversation_a_note_was_written_in_and_keeps_it_on_rewrite():
    from types import SimpleNamespace

    provider = EphemeralMemoryProvider()
    tools = _tools(provider)

    async def go():
        made = json.loads(
            (await tools["memory_write"].execute(
                {"title": "보고서 표 편집", "content": "v1", "tags": "assignment"}, SimpleNamespace(session_id="conv-A")
            )).content
        )
        fn = made["filename"]
        await tools["memory_write"].execute(
            {"title": "보고서 표 편집", "content": "v2", "tags": "assignment,done", "filename": fn},
            SimpleNamespace(session_id="conv-B"),
        )
        read = json.loads((await tools["memory_read"].execute({"filename": fn}, None)).content)
        assert [w["conversation"] for w in read["written_in"]] == ["conv-A", "conv-B"]
        assert read["body"] == "v2" and len(await provider.notes().list(tag="assignment")) == 1

    asyncio.run(go())


def test_another_conversation_is_read_only_when_the_host_opens_it():
    from xgen_agent_runtime.host.memory_tools import conversation_history
    from xgen_agent_runtime.memory.provider import Turn

    other = EphemeralMemoryProvider()

    async def go():
        await other.stm().append(Turn(role="user", content="표 3개로 편집해 줘, 부가세는 빼고"))
        await other.stm().append(Turn(role="assistant", content="report.md 의 표를 고쳤습니다"))
        here = EphemeralMemoryProvider()
        state = PipelineState(session_id="conv-B")
        denied = await conversation_history(here, {"conversation": "conv-A"}, state=state)
        assert denied.is_error and "cannot be read here" in denied.content
        here.session_stm = lambda sid: other.stm() if sid == "conv-A" else None  # 호스트가 허락한 대화만
        found = await conversation_history(here, {"conversation": "conv-A", "view": "turn", "turn": 1}, state=state)
        assert "부가세는 빼고" in found.content and "report.md 의 표를 고쳤습니다" in found.content

    asyncio.run(go())


def test_the_guidance_says_how_to_find_earlier_work_and_never_asks_read_only_turns_to_write():
    from xgen_agent_runtime.host._constants import (
        MEMORY_AUTO_PROMPT_BLOCK,
        MEMORY_PROMPT_BLOCK,
        MEMORY_READONLY_PROMPT_BLOCK,
    )

    assert "tagged `assignment`" in MEMORY_PROMPT_BLOCK and "refers to earlier" in MEMORY_PROMPT_BLOCK
    assert "shown to you on every turn" not in MEMORY_PROMPT_BLOCK
    for text in (MEMORY_READONLY_PROMPT_BLOCK, MEMORY_AUTO_PROMPT_BLOCK):
        assert "memory_write" not in text


def test_end_to_end_the_edit_from_another_conversation_is_found_and_read():
    """대화 A 에서 편집을 맡기고 노트를 남긴다. 대화 B 에서 "예전에 했던 그 편집" 을 메모리로 찾아 A 의 원문까지 읽는다.

    모델의 선택(어느 도구를 부를지)은 대본이다. 시험하는 것은 메모리 쪽 길이 끝까지 이어지는가다.
    """
    from xgen_agent_runtime.core.state import TokenUsage
    from xgen_agent_runtime.host import runner
    from xgen_agent_runtime.host.memory_tools import build_memory_tools
    from xgen_agent_runtime.llm_client import BaseClient, ClientCapabilities
    from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
    from xgen_agent_runtime.tools import ToolRegistry

    conv_a = EphemeralMemoryProvider()
    conv_b = EphemeralMemoryProvider()
    conv_b._notes = conv_a._notes  # 같은 에이전트의 기억(노트), 대화별 기록(STM)
    conv_b.session_stm = lambda sid: conv_a.stm() if sid == "A" else None
    results: List[str] = []

    def _resp(*blocks):
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
        return APIResponse(content=list(blocks), stop_reason=stop, usage=TokenUsage(input_tokens=1, output_tokens=1), model="m")

    def _use(i, name, **inp):
        return ContentBlock(type="tool_use", tool_use_id=i, tool_name=name, tool_input=inp)

    def _text(t):
        return ContentBlock(type="text", text=t)

    class _Scripted(BaseClient):
        provider = "fake"
        capabilities = ClientCapabilities(supports_tools=True)

        def __init__(self, script):
            super().__init__(api_key="k")
            self._script = list(script)

        async def _send(self, request: Any, *, purpose: str = ""):
            for m in getattr(request, "messages", []) or []:
                for b in m.get("content") if isinstance(m.get("content"), list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        results.append(str(b.get("content")))
            return self._script.pop(0) if self._script else _resp(_text("(끝)"))

    def _turn(provider, session, text, script):
        registry = ToolRegistry()
        for tool in build_memory_tools(provider):
            registry.register(tool)
        pipe = runner.build_pipeline(
            name="t", provider="openai", model="m", api_key="k", llm_client=_Scripted(script),
            stream=False, enable_compaction=False, memory_provider=provider, registry=registry,
        )
        return runner.run_turn(pipe, text, PipelineState(session_id=session, model="m"))

    # 대화 A: 편집을 맡기고, 모델이 assignment 노트를 남긴다
    _turn(conv_a, "A", "report.md 의 표를 부서별 3개로 편집해 줘. 부가세는 빼고.", [
        _resp(_use("w1", "memory_write", title="report.md 표 편집", tags="assignment",
                   content="사용자: 부서별 표 3개, 부가세 제외.\n한 것: report.md 의 표 3개 편집.")),
        _resp(_text("편집했습니다.")),
    ])
    note = asyncio.run(conv_a.notes().list(tag="assignment"))[0].ref.filename

    # 대화 B: "예전에 했던 그 편집" → 메모리에서 찾고, 노트를 읽고, 그 노트를 쓴 대화의 원문을 읽는다
    _turn(conv_b, "B", "예전에 했던 그 표 편집에서 하나만 고치고 싶어", [
        _resp(_use("s1", "memory_search", query="표 편집")),
        _resp(_use("r1", "memory_read", filename=note)),
        _resp(_use("h1", "ConversationHistory", conversation="A", view="turn", turn=1)),
        _resp(_text("찾았습니다. 무엇을 고칠까요?")),
    ])
    joined = "\n".join(results)
    assert note in joined, "memory_search found the assignment"
    assert '"conversation": "A"' in joined, "memory_read says where it was written"
    assert "report.md 의 표를 부서별 3개로 편집해 줘. 부가세는 빼고." in joined, "the original words are read back"


def test_a_host_may_check_ownership_asynchronously():
    from xgen_agent_runtime.host.memory_tools import conversation_history
    from xgen_agent_runtime.memory.provider import Turn

    other = EphemeralMemoryProvider()

    async def go():
        await other.stm().append(Turn(role="user", content="원래 지시"))
        here = EphemeralMemoryProvider()

        async def _open(sid):
            return other.stm() if sid == "conv-A" else None

        here.session_stm = _open
        ok = await conversation_history(here, {"conversation": "conv-A", "view": "turn", "turn": 1}, state=None)
        assert "원래 지시" in ok.content
        denied = await conversation_history(here, {"conversation": "conv-X"}, state=None)
        assert denied.is_error

    asyncio.run(go())
