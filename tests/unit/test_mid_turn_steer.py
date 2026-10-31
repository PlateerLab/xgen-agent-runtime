"""Mid-turn steering (core/steer.py) — the user adds a message to a RUNNING turn.

The message must reach the model at the next loop boundary (after the current
call and its tools), keep a turn that was about to complete going, and come
back rejected when the turn can no longer take it (so the host sends it as the
next turn). One terminal outcome per message, never lost.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable, List, Optional

import pytest

from xgen_agent_runtime import Pipeline, PipelineConfig, PipelineState
from xgen_agent_runtime.core.state import TokenUsage
from xgen_agent_runtime.core.steer import (
    STEER_FRAME,
    SteerInbox,
    SteerMessage,
    append_steer,
    steer_text,
)
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.stages.s01_input import InputStage
from xgen_agent_runtime.stages.s04_guard import GuardStage, IterationGuard
from xgen_agent_runtime.stages.s06_api import APIResponse, APIStage, MockProvider
from xgen_agent_runtime.stages.s06_api.retry import NoRetry
from xgen_agent_runtime.stages.s06_api.types import ContentBlock
from xgen_agent_runtime.stages.s07_token import TokenStage
from xgen_agent_runtime.stages.s09_parse import ParseStage
from xgen_agent_runtime.stages.s10_tool import ToolStage
from xgen_agent_runtime.stages.s16_loop import LoopStage, StandardLoopController
from xgen_agent_runtime.stages.s21_yield import YieldStage
from xgen_agent_runtime.tools import Tool, ToolRegistry, ToolResult

# ── inbox ────────────────────────────────────────────────────────────


def test_offer_take_and_close_reject_what_is_left() -> None:
    outcomes: list = []
    inbox = SteerInbox(on_resolve=lambda m, ok: outcomes.append((m.text, ok)))
    a = inbox.offer("first", client_msg_id="c1")
    assert a is not None and a.client_msg_id == "c1"
    assert [m.text for m in inbox.take()] == ["first"]
    b = inbox.offer("second")
    inbox.close()
    assert inbox.open is False
    assert b is not None and b.delivered is False and b.wait(0) is False
    assert outcomes == [("second", False)]
    assert inbox.offer("too late") is None


def test_empty_text_is_not_queued() -> None:
    inbox = SteerInbox()
    assert inbox.offer("   ") is None
    assert inbox.take() == []


def test_take_or_close_is_atomic() -> None:
    inbox = SteerInbox()
    inbox.offer("x")
    assert [m.text for m in inbox.take_or_close()] == ["x"]
    assert inbox.open is True  # had something: still open
    assert inbox.take_or_close() == []
    assert inbox.open is False  # nothing queued: closed in the same step
    assert inbox.offer("y") is None


def test_resolve_is_recorded_once() -> None:
    calls: list = []
    inbox = SteerInbox(on_resolve=lambda m, ok: calls.append(ok))
    msg = inbox.offer("x")
    assert msg is not None
    inbox.resolve(msg, True)
    inbox.resolve(msg, False)
    assert msg.delivered is True and calls == [True]


def test_callback_failure_does_not_escape() -> None:
    def boom(_m: SteerMessage, _ok: bool) -> None:
        raise RuntimeError("host bug")

    inbox = SteerInbox(on_resolve=boom)
    msg = inbox.offer("x")
    inbox.close()
    assert msg is not None and msg.delivered is False


def test_offer_from_many_threads_loses_nothing() -> None:
    inbox = SteerInbox()
    threads = [threading.Thread(target=lambda i=i: inbox.offer(f"m{i}")) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(m.text for m in inbox.take()) == sorted(f"m{i}" for i in range(50))


# ── how the text joins the conversation ──────────────────────────────


def test_steer_text_frames_the_users_words() -> None:
    text = steer_text([SteerMessage(text="also run the tests"), SteerMessage(text=" use pytest ")])
    assert text.startswith(STEER_FRAME)
    assert text.endswith("also run the tests\n\nuse pytest")


def test_append_after_tool_results_extends_that_user_message() -> None:
    msgs: List[dict] = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    append_steer(msgs, "NOTE")
    assert len(msgs) == 2  # no second user message in a row
    assert msgs[-1]["content"][0]["type"] == "tool_result"
    assert msgs[-1]["content"][-1] == {"type": "text", "text": "NOTE"}


def test_append_to_plain_user_text_and_after_assistant() -> None:
    msgs: List[dict] = [{"role": "user", "content": "hi"}]
    append_steer(msgs, "NOTE")
    assert msgs == [{"role": "user", "content": "hi\n\nNOTE"}]
    msgs.append({"role": "assistant", "content": "done"})
    append_steer(msgs, "MORE")
    assert msgs[-1] == {"role": "user", "content": "MORE"}


# ── a real pipeline ──────────────────────────────────────────────────


def _tool_use(name: str, inp: dict, tid: str = "tu_1") -> APIResponse:
    return APIResponse(
        content=[
            ContentBlock(
                type="tool_use",
                tool_use_id=tid,
                tool_name=name,
                tool_input=inp,
                raw={"type": "tool_use", "id": tid, "name": name, "input": inp},
            )
        ],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model="test",
    )


def _text(text: str) -> APIResponse:
    return APIResponse(
        content=[ContentBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model="test",
    )


class _Recording(MockProvider):
    """Snapshots each request's messages (the live list is mutated later) and
    can run a hook while a given call is "in flight"."""

    def __init__(self, responses: List[APIResponse]) -> None:
        super().__init__(responses)
        self.seen: List[List[dict]] = []
        self.during_call: dict[int, Callable[[], None]] = {}

    async def create_message(self, request: Any) -> APIResponse:
        self.seen.append(copy.deepcopy(list(request.messages)))
        hook = self.during_call.get(len(self.seen))
        if hook is not None:
            hook()
        return await super().create_message(request)


class _SlowTool(Tool):
    """A tool during which the user types something."""

    def __init__(self, on_run: Optional[Callable[[], None]] = None) -> None:
        self._on_run = on_run

    @property
    def name(self) -> str:
        return "work"

    @property
    def description(self) -> str:
        return "does work"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict, context: Any) -> ToolResult:
        if self._on_run is not None:
            self._on_run()
        return ToolResult(content="worked")


def _pipeline(provider: MockProvider, tool: Optional[Tool] = None, *, single_turn: bool = False) -> Pipeline:
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    p = Pipeline(PipelineConfig(name="steer-test", single_turn=single_turn))
    p.register_stage(InputStage())
    p.register_stage(GuardStage([IterationGuard(max_iterations=10)]))
    p.register_stage(APIStage(provider=provider, retry=NoRetry()))
    p.register_stage(TokenStage())
    p.register_stage(ParseStage())
    p.register_stage(ToolStage(registry=registry))
    p.register_stage(LoopStage(StandardLoopController()))
    p.register_stage(YieldStage())
    return p


def _last_user_texts(messages: List[dict]) -> List[str]:
    last = [m for m in messages if m["role"] == "user"][-1]["content"]
    if isinstance(last, str):
        return [last]
    return [b.get("text", "") for b in last if isinstance(b, dict) and b.get("type") == "text"]


@pytest.mark.asyncio
async def test_message_sent_during_a_tool_reaches_the_next_model_call() -> None:
    inbox = SteerInbox()
    sent: list = []
    provider = _Recording([_tool_use("work", {}), _text("done, with tests")])
    pipe = _pipeline(provider, _SlowTool(lambda: sent.append(inbox.offer("also run the tests"))))
    state = PipelineState(session_id="s")
    state._steer_inbox = inbox

    result = await pipe.run("fix the bug", state)

    assert result.success and result.text == "done, with tests"
    assert provider.call_count == 2
    second = provider.seen[1]
    last = second[-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"  # tool results stay first
    assert any("also run the tests" in t for t in _last_user_texts(second))
    assert not any("also run the tests" in str(m) for m in provider.seen[0])
    assert sent[0].delivered is True
    steer_events = [e for e in state.events if e["type"] == "user.steer"]
    assert len(steer_events) == 1 and steer_events[0]["data"]["text"] == "also run the tests"
    assert inbox.open is False  # the turn completed: closed


@pytest.mark.asyncio
async def test_message_during_the_final_answer_keeps_the_turn_going() -> None:
    inbox = SteerInbox()
    provider = _Recording([_text("here is the answer"), _text("answer in Korean")])
    provider.during_call[1] = lambda: inbox.offer("한국어로 답해줘")
    pipe = _pipeline(provider)
    state = PipelineState(session_id="s")
    state._steer_inbox = inbox

    result = await pipe.run("question", state)

    assert provider.call_count == 2  # one more call to answer the new message
    assert result.text == "answer in Korean"
    second = provider.seen[1]
    assert second[-2]["role"] == "assistant"  # the first answer stays in the history
    assert second[-1] == {"role": "user", "content": steer_text([SteerMessage(text="한국어로 답해줘")])}


@pytest.mark.asyncio
async def test_nothing_queued_closes_on_completion_and_late_offers_are_refused() -> None:
    inbox = SteerInbox()
    pipe = _pipeline(_Recording([_text("ok")]))
    state = PipelineState(session_id="s")
    state._steer_inbox = inbox
    await pipe.run("q", state)
    assert inbox.open is False
    assert inbox.offer("late") is None


@pytest.mark.asyncio
async def test_single_turn_runs_reject_steering() -> None:
    inbox = SteerInbox()
    provider = _Recording([_text("only answer")])
    pipe = _pipeline(provider, single_turn=True)
    state = PipelineState(session_id="s")
    state._steer_inbox = inbox
    msg_holder: list = []
    provider.during_call[1] = lambda: msg_holder.append(inbox.offer("more"))
    await pipe.run("q", state)
    assert provider.call_count == 1
    assert msg_holder[0].delivered is False


@pytest.mark.asyncio
async def test_no_inbox_changes_nothing() -> None:
    provider = _Recording([_tool_use("work", {}), _text("fine")])
    result = await _pipeline(provider, _SlowTool()).run("q", PipelineState(session_id="s"))
    assert result.text == "fine" and provider.call_count == 2


# ── stream_turn (the host bridge) ────────────────────────────────────


def test_stream_turn_announces_delivered_messages_and_closes() -> None:
    outcomes: list = []
    inbox = SteerInbox(on_resolve=lambda m, ok: outcomes.append((m.client_msg_id, ok)))
    provider = _Recording([_tool_use("work", {}), _text("final")])
    tool = _SlowTool(lambda: inbox.offer("also this", client_msg_id="c-1"))
    out = list(
        runner.stream_turn(_pipeline(provider, tool), "go", PipelineState(session_id="s"), steer_inbox=inbox)
    )
    steer = [c["data"] for c in out if isinstance(c, dict) and c.get("data", {}).get("type") == "user_steer"]
    assert len(steer) == 1
    assert steer[0]["text"] == "also this" and steer[0]["client_msg_id"] == "c-1" and steer[0]["id"]
    assert outcomes == [("c-1", True)]
    assert inbox.open is False
    assert "final" in "".join(c for c in out if isinstance(c, str))


def test_stream_turn_without_inbox_emits_no_steer_events() -> None:
    provider = _Recording([_tool_use("work", {}), _text("final")])
    out = list(runner.stream_turn(_pipeline(provider, _SlowTool()), "go", PipelineState(session_id="s")))
    assert not [c for c in out if isinstance(c, dict) and c.get("data", {}).get("type") == "user_steer"]


def test_stopped_turn_rejects_what_was_queued() -> None:
    inbox = SteerInbox()
    held: list = []
    stop = threading.Event()

    def during_tool() -> None:
        held.append(inbox.offer("never mind"))
        stop.set()  # the user pressed stop right after

    provider = _Recording([_tool_use("work", {}), _text("unreached")])
    list(
        runner.stream_turn(
            _pipeline(provider, _SlowTool(during_tool)),
            "go",
            PipelineState(session_id="s"),
            cancel_check=stop.is_set,
            steer_inbox=inbox,
        )
    )
    assert held[0].delivered is False  # comes back: the host sends it as the next turn
    assert inbox.open is False


def test_stream_turn_detaches_the_inbox_from_the_state() -> None:
    state = PipelineState(session_id="s")
    list(runner.stream_turn(_pipeline(_Recording([_text("x")])), "go", state, steer_inbox=SteerInbox()))
    assert state._steer_inbox is None



class _EndsWithoutTaking:
    """A pipeline whose turn ends while a message waits in the queue (e.g. the pass ended on
    an error / suspend verdict, which leaves the queue alone)."""

    async def run_stream(self, text: Any, state: PipelineState):
        from xgen_agent_runtime.core.pipeline import PipelineEvent

        state.begin_turn()
        state._steer_inbox.offer("late note", client_msg_id="c-9")
        yield PipelineEvent(type="text.delta", data={"text": "partial"})
        yield PipelineEvent(type="pipeline.complete", data={"result": "", "status": "completed"})

    async def aclose(self) -> None:
        pass

    def _resolved_provider_name(self, state: PipelineState) -> str:
        return "fake"


def test_stream_turn_reports_messages_it_could_not_take() -> None:
    outcomes: list = []
    inbox = SteerInbox(on_resolve=lambda m, ok: outcomes.append((m.client_msg_id, ok)))
    out = list(runner.stream_turn(_EndsWithoutTaking(), "go", PipelineState(session_id="s"), steer_inbox=inbox))
    rejected = [c["data"] for c in out if isinstance(c, dict) and c.get("data", {}).get("type") == "user_steer_rejected"]
    assert [(r["client_msg_id"], r["text"]) for r in rejected] == [("c-9", "late note")]
    assert outcomes == [("c-9", False)]
    # the rejection is announced before the turn's usage chunk (the last thing a turn yields)
    kinds = [c.get("type") if isinstance(c, dict) else "text" for c in out]
    assert kinds.index("agent_event") < len(kinds) - 1 or "usage" not in kinds
