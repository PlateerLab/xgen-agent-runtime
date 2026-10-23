"""TASK 기억(4.98.0): 여러 단계·여러 턴 일을 하나의 기억 단위로.

판단은 모델(Task 도구)과 호스트(턴 끝 기록·보완)가 나눠 맡고, 회상은 매 턴의 TASK 판(검색 무관)과
TaskRecall(창 밖의 턴까지)이 맡는다. 시험은 그 계약을 하나씩 고정한다.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider
from xgen_agent_runtime.memory.tasks import (
    BOARD_KEY,
    CURRENT_KEY,
    InMemoryTaskStore,
    Task,
    TaskEvent,
    load_board,
    provider_tasks,
    record_turn,
    render_card,
)


def _run(coro):
    return asyncio.run(coro)


def _tools(provider) -> Dict[str, Any]:
    from xgen_agent_runtime.host.memory_tools import build_memory_tools

    return {t.name: t for t in build_memory_tools(provider)}


def _ctx(session: str = "conv-1", state: Any = None):
    state = state or PipelineState(session_id=session)
    return SimpleNamespace(session_id=session, state_view=state)


async def _call(tool, ctx, **kw):
    return await tool.execute(kw, ctx)


# ── 저장소 ──────────────────────────────────────────────────────────────


def test_store_keeps_one_current_task_per_conversation():
    store = InMemoryTaskStore()

    async def go():
        a = await store.create(Task(id="", title="A"))
        b = await store.create(Task(id="", title="B"))
        await store.focus(a.id, "s1")
        await store.focus(b.id, "s1")
        assert (await store.current("s1")).id == b.id
        assert (await store.get(a.id)).focus_session == ""
        assert await store.current("s2") is None
        for i in range(25):
            await store.append_event(TaskEvent(task_id=b.id, kind="note", summary=f"e{i}"))
        first = await store.events(b.id, limit=10)
        older = await store.events(b.id, limit=10, before_seq=first[-1].seq)
        assert first[0].summary == "e24" and older[0].summary == "e14"
        assert await store.count_events(b.id) == 25

    _run(go())


def test_the_ephemeral_provider_has_the_layer():
    assert provider_tasks(EphemeralMemoryProvider()) is not None
    assert provider_tasks(object()) is None


# ── Task 도구: 모델이 선언한다 ──────────────────────────────────────────


def test_open_update_switch_close_and_the_board_follows():
    provider = EphemeralMemoryProvider()
    tools = _tools(provider)
    task_tool = tools["Task"]
    ctx = _ctx()

    async def go():
        opened = await _call(
            task_tool,
            ctx,
            action="open",
            title="분기 보고서",
            goal="3분기 매출 보고서를 만든다",
            done_when="report.md 가 있고 표 3개",
            plan=[{"text": "데이터 모으기"}, {"text": "표 만들기"}, {"text": "글 쓰기"}],
        )
        assert not opened.is_error
        board = opened.state_mutations[BOARD_KEY]
        tid = opened.state_mutations[CURRENT_KEY]
        assert "## Current task in this conversation" in board and "분기 보고서" in board
        assert "1. [ ] 데이터 모으기" in board

        updated = await _call(
            task_tool,
            ctx,
            action="update",
            step={"n": 1, "status": "done"},
            decision="매출은 부가세 제외로 잡는다(사용자 확인)",
            files=["/workspace/report.md"],
            next_step="표 만들기",
        )
        board = updated.state_mutations[BOARD_KEY]
        assert "1. [done] 데이터 모으기" in board and "부가세 제외" in board and "Next step: 표 만들기" in board

        other = await _call(task_tool, ctx, action="open", title="서버 이전")
        assert "분기 보고서" in other.state_mutations[BOARD_KEY].split("## Other open tasks")[1]
        assert (await provider.tasks().get(tid)).status == "open", "the task set aside is not 'active'"

        back = await _call(task_tool, ctx, action="switch", task_id=tid)
        assert back.state_mutations[CURRENT_KEY] == tid

        closed = await _call(task_tool, ctx, action="close", outcome="done", summary="보고서 전달")
        assert (await provider.tasks().get(tid)).status == "done"
        assert "No task is current in this conversation." in closed.state_mutations[BOARD_KEY]

        listing = await _call(task_tool, ctx, action="list", include_done=True)
        assert "분기 보고서" in listing.content and "서버 이전" in listing.content

    _run(go())


def test_the_task_is_findable_as_a_note():
    provider = EphemeralMemoryProvider()
    task_tool = _tools(provider)["Task"]

    async def go():
        res = await _call(task_tool, _ctx(), action="open", title="결제 모듈 리팩터", goal="재시도 로직 정리")
        tid = res.state_mutations[CURRENT_KEY]
        note = await provider.notes().read(f"task-{tid}.md")
        assert note is not None and "결제 모듈 리팩터" in note.body and note.category == "tasks"

    _run(go())


def test_update_without_anything_is_an_error_not_a_silent_success():
    provider = EphemeralMemoryProvider()
    task_tool = _tools(provider)["Task"]

    async def go():
        await _call(task_tool, _ctx(), action="open", title="x")
        res = await _call(task_tool, _ctx(), action="update")
        assert res.is_error

    _run(go())


# ── 판과 카드: 잘리면 잘렸다고 ─────────────────────────────────────────


def test_an_oversized_card_names_what_it_left_out_and_where_it_is():
    task = Task(id="T-1", title="큰 일", decisions=[f"결정 {i} " + "가" * 200 for i in range(40)])
    card = render_card(task, max_chars=2_000)
    assert len(card) <= 2_400
    assert "older decisions left out here to fit" in card and 'TaskRecall(task_id="T-1")' in card
    assert "결정 39" in card, "the newest decisions stay"


def test_the_board_lists_other_open_tasks_with_a_count_of_the_rest():
    store = InMemoryTaskStore()

    async def go():
        for i in range(12):
            await store.create(Task(id="", title=f"일 {i}", status="open"))
        board, current = await load_board(store, "s1")
        assert current is None and "No task is current" in board
        assert "(+4 more open task(s): Task(action=\"list\"))" in board

    _run(go())


# ── 호스트: 턴 끝 기록과 보완 ─────────────────────────────────────────


def test_a_turn_on_the_current_task_is_recorded_with_its_files():
    store = InMemoryTaskStore()

    async def go():
        t = await store.create(Task(id="", title="보고서", status="active", focus_session="s1"))
        rec = await record_turn(
            store,
            session_id="s1",
            request="표 3개 만들어 줘",
            answer="표 3개를 report.md 에 넣었습니다.",
            outcome="ok",
            tool_calls=4,
            files=["/workspace/report.md"],
        )
        assert rec.id == t.id and rec.files == ["/workspace/report.md"]
        ev = (await store.events(t.id, limit=1))[0]
        assert ev.kind == "turn" and "표 3개 만들어 줘" in ev.summary and ev.data["tool_calls"] == 4

    _run(go())


def test_the_host_opens_a_task_for_unfinished_multi_step_work():
    store = InMemoryTaskStore()
    todos = [
        {"content": "스키마 확인", "status": "completed"},
        {"content": "이관 스크립트", "status": "in_progress"},
        {"content": "검증", "status": "pending"},
    ]

    async def go():
        task = await record_turn(
            store, session_id="s1", request="잡 테이블을 새 스키마로 옮겨 줘", answer="1단계 끝", outcome="ok",
            tool_calls=3, todos=todos,
        )
        assert task is not None and task.origin == "host" and task.focus_session == "s1"
        assert [s["status"] for s in task.plan] == ["done", "in_progress", "pending"]
        assert (await store.events(task.id, limit=5))[-1].kind == "opened"

    _run(go())


def test_a_quick_unrelated_question_is_not_put_on_the_current_task():
    store = InMemoryTaskStore()

    async def go():
        t = await store.create(Task(id="", title="보고서", status="active", focus_session="s1"))
        rec = await record_turn(store, session_id="s1", request="오늘 날짜가 뭐지?", answer="10월 23일", outcome="ok")
        assert rec is None and await store.count_events(t.id) == 0

    _run(go())


def test_a_one_step_question_opens_no_task():
    store = InMemoryTaskStore()

    async def go():
        task = await record_turn(store, session_id="s1", request="지금 몇 시야?", answer="10시", outcome="ok")
        assert task is None and await store.count() == 0

    _run(go())


def test_a_stopped_turn_that_used_tools_is_kept_as_a_task():
    store = InMemoryTaskStore()

    async def go():
        task = await record_turn(
            store, session_id="s1", request="로그에서 오류 원인 찾아 줘", answer="", outcome="stopped", tool_calls=2
        )
        assert task is not None and task.origin == "host"

    _run(go())


def test_the_runner_records_the_turn_and_the_plan_from_todowrite():
    from xgen_agent_runtime.host.runner import _record_task_turn

    provider = EphemeralMemoryProvider()
    state = PipelineState(session_id="s1")
    todos = [{"id": "a", "content": "읽기", "status": "completed"}, {"id": "b", "content": "고치기", "status": "pending"}]
    state.shared["executor.todos"] = todos
    state.messages = [
        {"role": "user", "content": "버그 고쳐 줘"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "TodoWrite", "input": {"todos": todos}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Write", "input": {"file_path": "/workspace/fix.py", "content": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "읽기를 마쳤고 고치는 중입니다"}]},
    ]
    loop = asyncio.new_event_loop()
    try:
        _record_task_turn(
            provider, loop, state=state, input_text={"input_str": "버그 고쳐 줘", "attachments": []},
            output_text="읽기를 마쳤고 고치는 중입니다", success=True, cancelled=False, calls=0, failures=0, blocked=0,
        )
        task = loop.run_until_complete(provider.tasks().current("s1"))
        note = loop.run_until_complete(provider.notes().read(f"task-{task.id}.md"))
    finally:
        loop.close()
    assert task.origin == "host" and task.title == "버그 고쳐 줘"
    assert task.files == ["/workspace/fix.py"]
    assert [s["text"] for s in task.plan] == ["읽기", "고치기"]
    assert note is not None


def test_a_compaction_summary_is_kept_on_the_current_task():
    from xgen_agent_runtime.core.compaction import _keep_checkpoint_on_task

    provider = EphemeralMemoryProvider()
    state = PipelineState(session_id="s1")
    state.messages = [{"role": "user", "content": "[Context checkpoint] The conversation ... 1. User requests: 보고서"}]

    async def go():
        t = await provider.tasks().create(Task(id="", title="보고서", focus_session="s1"))
        await _keep_checkpoint_on_task(state, provider, trigger="proactive", replaced=12)
        ev = (await provider.tasks().events(t.id, limit=1))[0]
        assert ev.kind == "checkpoint" and "User requests: 보고서" in ev.data["summary"]

    _run(go())


# ── 회상: 판은 매 턴, 원문은 TaskRecall ─────────────────────────────────


def test_stage2_puts_the_board_in_every_turn_and_searches_with_the_task():
    from xgen_agent_runtime.stages.s02_context.artifact.default.stage import ContextStage

    provider = EphemeralMemoryProvider()
    seen: List[str] = []

    class _Retriever:
        async def retrieve(self, query, state):
            seen.append(query)
            return []

    async def go():
        t = await provider.tasks().create(
            Task(id="", title="결제 리팩터", goal="재시도 정리", status="active", focus_session="sess")
        )
        stage = ContextStage(short_term_source=provider, retriever=_Retriever(), compaction_enabled=False)
        state = PipelineState(session_id="sess")
        state.messages = [{"role": "user", "content": "계속"}]
        await stage.execute(None, state)
        return t, state

    t, state = _run(go())
    assert state.shared[CURRENT_KEY] == t.id and "결제 리팩터" in state.shared[BOARD_KEY]
    assert seen and "결제 리팩터 재시도 정리" in seen[0], "a bare 'continue' still searches with the task"

    from xgen_agent_runtime.stages.s03_system.artifact.default.builders import TaskBoardBlock

    block = TaskBoardBlock()
    rendered = block.render(state)
    assert block.volatile and "결제 리팩터" in rendered
    # Task 도구가 없는 턴(CLI 메모리 도구 미광고)에는 부르라고 하지 않는다
    assert "Task(action=" not in rendered and "you cannot change them from here" in rendered
    state.tools = [{"name": "Task", "description": "", "input_schema": {}}]
    assert 'Task(action="update")' in block.render(state)
    # 첫 호출: Stage 3 가 블록을 그린 뒤에 state.tools 를 채운다 — 레지스트리에서 본 이름이 먼저다
    state.tools = []
    state.shared["_offered_tools"] = ["Task", "TaskRecall"]
    assert 'Task(action="update")' in block.render(state)


def test_task_recall_reads_history_and_a_turn_from_another_conversation():
    provider = EphemeralMemoryProvider()
    tools = _tools(provider)
    other_stm = provider.stm()  # 다른 대화의 기록이라고 친다

    async def go():
        from xgen_agent_runtime.memory.provider import Turn

        await other_stm.append(Turn(role="user", content="1차 초안 써 줘"))
        await other_stm.append(Turn(role="assistant", content="초안: 매출 12억"))
        t = await provider.tasks().create(Task(id="", title="보고서", focus_session="conv-2"))
        await record_turn(
            provider.tasks(), session_id="conv-old", request="1차 초안 써 줘", answer="초안: 매출 12억", outcome="ok",
            current_id=t.id, tool_calls=2,
        )
        provider.session_stm = lambda sid: other_stm  # 호스트가 다른 대화를 읽게 해 준다
        ctx = _ctx("conv-2")
        hist = await _call(tools["TaskRecall"], ctx, view="history")
        assert "conversation conv-old" in hist.content
        seq = (await provider.tasks().events(t.id, limit=1))[0].seq
        turn = await _call(tools["TaskRecall"], ctx, view="turn", seq=seq)
        assert "The turn word for word (conversation conv-old)" in turn.content and "매출 12억" in turn.content

    _run(go())


def test_the_tools_are_on_the_first_turn_surface():
    from xgen_agent_runtime.host.tool_exposure import TURN_ONE_TOOLS

    assert {"Task", "TaskRecall"} <= set(TURN_ONE_TOOLS)
    assert {"Task", "TaskRecall"} <= set(_tools(EphemeralMemoryProvider()))


@pytest.mark.parametrize("bad", [{"action": "nope"}, {"action": "switch", "task_id": "T-none"}])
def test_bad_calls_say_what_is_wrong(bad):
    provider = EphemeralMemoryProvider()

    async def go():
        return await _tools(provider)["Task"].execute(bad, _ctx())

    assert _run(go()).is_error


def test_search_does_not_bring_back_the_current_tasks_own_note():
    from xgen_agent_runtime.memory.retriever import _is_note

    hit = SimpleNamespace(key="tasks/task-T-1.md", metadata={"filename": "tasks/task-T-1.md"}, content="")
    assert _is_note(hit, "task-T-1.md") and not _is_note(hit, "task-T-2.md") and not _is_note(hit, "")


def test_the_task_note_says_it_is_rewritten_from_the_record():
    from xgen_agent_runtime.memory.tasks import task_note

    _, _, body = task_note(Task(id="T-9", title="x"))
    assert body.startswith("> Written from the record of task T-9") and "edits here do not stick" in body


# ── 끝에서 끝까지: 실제 파이프라인(build_pipeline + run_turn), 대화 둘, 턴 셋 ─────────


def test_end_to_end_tasks_carry_across_turns_and_conversations():
    from xgen_agent_runtime.host import runner
    from xgen_agent_runtime.host.memory_tools import build_memory_tools
    from xgen_agent_runtime.llm_client import BaseClient, ClientCapabilities
    from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
    from xgen_agent_runtime.core.state import TokenUsage
    from xgen_agent_runtime.tools import ToolRegistry

    provider = EphemeralMemoryProvider()
    seen: List[str] = []

    def _resp(*blocks) -> APIResponse:
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
        return APIResponse(content=list(blocks), stop_reason=stop, usage=TokenUsage(input_tokens=1, output_tokens=1), model="m")

    def _use(i: str, **inp) -> ContentBlock:
        return ContentBlock(type="tool_use", tool_use_id=i, tool_name="Task", tool_input=inp)

    def _text(t: str) -> ContentBlock:
        return ContentBlock(type="text", text=t)

    class _Scripted(BaseClient):
        provider = "fake"
        capabilities = ClientCapabilities(supports_tools=True)

        def __init__(self, script):
            super().__init__(api_key="k")
            self._script = list(script)

        async def _send(self, request, *, purpose: str = ""):
            seen.append(repr(request))
            return self._script.pop(0) if self._script else _resp(_text("(끝)"))

    def _turn(session: str, text: str, script) -> str:
        registry = ToolRegistry()
        for tool in build_memory_tools(provider):
            registry.register(tool)
        pipe = runner.build_pipeline(
            name="t", provider="openai", model="m", api_key="k", llm_client=_Scripted(script),
            stream=False, enable_compaction=False, memory_provider=provider, registry=registry,
        )
        return runner.run_turn(pipe, text, PipelineState(session_id=session, model="m"))

    # 1) 대화 A: 일을 연다
    _turn("A", "3분기 보고서 써 줘", [
        _resp(_text("TASK 로 엽니다."), _use("u1", action="open", title="3분기 보고서", goal="매출 보고서",
                                            plan=[{"text": "데이터"}, {"text": "표"}, {"text": "글"}])),
        _resp(_text("시작했습니다.")),
    ])
    task = asyncio.run(provider.tasks().current("A"))
    assert task is not None and task.title == "3분기 보고서"

    # 2) 대화 A: "계속" — 판이 요청에 실린다(검색과 무관)
    seen.clear()
    _turn("A", "계속", [
        _resp(_use("u2", action="update", step={"n": 1, "status": "done"}, next_step="표 만들기")),
        _resp(_text("데이터 단계 끝.")),
    ])
    assert "Current task in this conversation" in seen[0] and "3분기 보고서" in seen[0]
    assert "1. [done] 데이터" in seen[1], "the update reaches the very next call"

    # 3) 대화 B: 다른 대화에서 이어 간다
    seen.clear()
    _turn("B", "지난번 보고서 어디까지 했지?", [
        _resp(_use("u3", action="switch", task_id=task.id)),
        _resp(_text("데이터까지 했고 다음은 표입니다.")),
    ])
    assert "No task is current in this conversation." in seen[0] and "3분기 보고서" in seen[0]
    assert "Current task in this conversation" in seen[1]
    after = asyncio.run(provider.tasks().get(task.id))
    assert after.focus_session == "B" and after.plan[0]["status"] == "done"
    kinds = [e.kind for e in asyncio.run(provider.tasks().events(task.id, limit=50))]
    assert kinds.count("turn") == 3, kinds  # 세 턴 모두 Task 를 만졌다


def test_list_finds_a_closed_task_by_its_words_and_shared_notes_are_skipped():
    provider = EphemeralMemoryProvider()
    task_tool = _tools(provider)["Task"]

    async def go():
        opened = await _call(task_tool, _ctx(), action="open", title="3분기 매출 보고서", goal="부서별 표")
        tid = opened.state_mutations[CURRENT_KEY]
        await _call(task_tool, _ctx(), action="close", outcome="done")
        found = await _call(task_tool, _ctx("other"), action="list", query="매출 보고서")
        assert tid in found.content
        miss = await _call(task_tool, _ctx("other"), action="list", query="이관")
        assert "No task matches" in miss.content

        provider.task_notes = False  # 노트가 에이전트 공용인 provider: TASK 노트를 쓰지 않는다
        other = await _call(task_tool, _ctx(), action="open", title="비공개 일")
        assert await provider.notes().read(f"task-{other.state_mutations[CURRENT_KEY]}.md") is None

    _run(go())


def test_the_execution_card_names_the_task_its_turn_went_to():
    from xgen_agent_runtime.host.execution_record import record_turn_execution

    provider = EphemeralMemoryProvider()

    async def go():
        await record_turn_execution(
            provider, input_text="표 만들어 줘", output_text="완료", success=True, duration_ms=10,
            session_id="s1", tool_calls=2, task_id="T-ab12", task_title="3분기 보고서",
        )
        metas = await provider.notes().list(category="daily")
        note = await provider.notes().read(metas[0].ref.filename)
        assert "> **Task:** [T-ab12] 3분기 보고서" in note.body

    _run(go())
