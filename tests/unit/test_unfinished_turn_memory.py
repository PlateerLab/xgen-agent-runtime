"""끝나지 못한 턴도 대화 기억(STM)에 남는다 — 다음 턴이 "아까 그거" 를 안다(2026-10-01).

STM 기록은 Stage 18 이 맡는데, 그 단계는 파이프라인이 끝까지 가야 돈다. 사용자가 [정지] 를 누른 턴은
그 전에 끊겨 질문조차 남지 않았고, 다음 턴은 무엇을 하다 멈췄는지 몰랐다(실측: 중단된 "캐시해서 중복
호출을 막자" 다음 턴이 "어느 앱 얘기인가요?" 라고 되물었다).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List

from xgen_agent_runtime import PipelineState
from xgen_agent_runtime.events.types import PipelineEvent
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.memory.short_term_window import WINDOW_LEN_KEY, build_window
from xgen_agent_runtime.stages.s18_memory.artifact.default.stage import _STATE_LAST_RECORDED

QUESTION = "그러면 이게 cached 되게 해서 중복 호출을 방지하자"
TOOL_USE = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}
TOOL_RESULT = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "backend/ frontend/"}]}


class _Memory:
    """STM 만 흉내 — record_turn 으로 받은 것을 쌓는다."""

    def __init__(self) -> None:
        self.turns: List[Any] = []

    async def record_turn(self, turn: Any) -> None:
        self.turns.append(turn)

    async def close(self) -> None:
        return None


class _Pipeline:
    def __init__(self, script: List[Any]) -> None:
        self._script = script
        self._memory_provider = _Memory()
        self._memory_distill_spec = None

    async def run_stream(self, text: Any, state: PipelineState) -> AsyncIterator[PipelineEvent]:
        state.begin_turn()
        for item in self._script:
            if callable(item):
                item(state)
                continue
            kind, data = item
            yield PipelineEvent(type=kind, data=dict(data))

    async def aclose(self) -> None:
        return None

    def _resolved_provider_name(self, state: PipelineState) -> str:
        return "fake"


def _add(*messages: Any):
    def _apply(state: PipelineState) -> None:
        state.messages.extend(dict(m) for m in messages)

    return _apply


def _drain_until(pipe: _Pipeline, state: PipelineState, stop_after: str, text: Any = QUESTION) -> List[Any]:
    stopped = [False]
    out = []
    for chunk in runner.stream_turn(pipe, text, state, cancel_check=lambda: stopped[0]):
        out.append(chunk)
        if chunk == stop_after:
            stopped[0] = True  # 사용자가 [정지] 를 눌렀다
    return out


def _roles(turns: List[Any]) -> List[str]:
    out = []
    for t in turns:
        content = t.content
        if isinstance(content, list) and content and isinstance(content[0], dict):
            out.append(f"{t.role}:{content[0].get('type')}")
        else:
            out.append(f"{t.role}:text")
    return out


def test_멈춘_턴의_질문과_한_일이_기억에_남는다():
    pipe = _Pipeline([
        _add({"role": "user", "content": QUESTION}, TOOL_USE, TOOL_RESULT),
        ("text.delta", {"text": "확인 중"}),
        ("text.delta", {"text": "입니다"}),
    ])
    state = PipelineState(session_id="s1")
    _drain_until(pipe, state, "확인 중")
    turns = pipe._memory_provider.turns
    assert _roles(turns) == ["user:text", "assistant:tool_use", "user:tool_result", "assistant:text"]
    assert turns[0].content == QUESTION
    assert "stopped this turn" in turns[-1].content


def test_끝까지_간_턴은_여기서_다시_적지_않는다():
    pipe = _Pipeline([
        _add({"role": "user", "content": QUESTION}, {"role": "assistant", "content": "끝"}),
        ("text.delta", {"text": "끝"}),
        ("pipeline.complete", {"status": "completed"}),
    ])
    state = PipelineState(session_id="s1")
    list(runner.stream_turn(pipe, QUESTION, state))
    assert pipe._memory_provider.turns == [], "끝까지 간 턴은 Stage 18 이 기록한다"


def test_이미_기록된_부분은_다시_적지_않는다():
    def _recorded(state: PipelineState) -> None:
        # 이어 가기 조각 하나가 끝까지 가서 Stage 18 이 질문·첫 도구까지 기록했다
        state.metadata[_STATE_LAST_RECORDED] = 3

    pipe = _Pipeline([
        _add({"role": "user", "content": QUESTION}, TOOL_USE, TOOL_RESULT),
        _recorded,
        _add({"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Read", "input": {}}]}),
        ("text.delta", {"text": "다음"}),
        ("text.delta", {"text": "x"}),
    ])
    state = PipelineState(session_id="s1")
    _drain_until(pipe, state, "다음")
    assert _roles(pipe._memory_provider.turns) == ["assistant:tool_use", "assistant:text"], "질문을 두 번 세우지 않는다"


def test_질문이_상태에_오기_전에_멈췄으면_받은_입력으로_세운다():
    pipe = _Pipeline([("text.delta", {"text": "."}), ("text.delta", {"text": "x"})])
    state = PipelineState(session_id="s1")
    state.messages = [{"role": "user", "content": "지난 질문"}, {"role": "assistant", "content": "지난 답"}]
    state.metadata[WINDOW_LEN_KEY] = 2
    state.metadata[_STATE_LAST_RECORDED] = 2
    _drain_until(pipe, state, ".", text={"input_str": QUESTION, "attachments": []})
    turns = pipe._memory_provider.turns
    assert [t.content for t in turns[:1]] == [QUESTION]
    assert _roles(turns) == ["user:text", "assistant:text"]


def test_다음_턴의_창이_멈춘_턴을_올바른_모양으로_읽는다():
    pipe = _Pipeline([
        _add({"role": "user", "content": QUESTION}, TOOL_USE),  # 도구 실행 중에 멈췄다(결과 없음)
        ("text.delta", {"text": "a"}),
        ("text.delta", {"text": "b"}),
    ])
    state = PipelineState(session_id="s1")
    _drain_until(pipe, state, "a")
    window, _report = build_window(pipe._memory_provider.turns)
    texts = [m for m in window]
    assert window[0]["role"] == "user" and QUESTION in str(window[0]["content"])
    assert window[-1]["role"] == "assistant"
    # 짝 없는 tool_use 는 창이 결과로 메운다 — API 가 거절하지 않는 모양
    uses = [b for m in texts if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use"]
    results = [b for m in texts if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]
    assert {u["id"] for u in uses} == {r["tool_use_id"] for r in results}
    assert any("stopped this turn" in str(m["content"]) for m in window)
    roles = [m["role"] for m in window]
    assert all(a != b for a, b in zip(roles, roles[1:])), f"역할이 교대해야 한다: {roles}"


def test_오류로_끝난_턴도_남는다():
    pipe = _Pipeline([
        _add({"role": "user", "content": QUESTION}),
        ("pipeline.error", {"error": "boom"}),
    ])
    state = PipelineState(session_id="s1")
    list(runner.stream_turn(pipe, QUESTION, state))
    turns = pipe._memory_provider.turns
    assert _roles(turns) == ["user:text", "assistant:text"]
    assert "ended with an error" in turns[-1].content
