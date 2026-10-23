"""결정적 prune 의 비용 트리거 (Stage 2): 4.35.0 에 기본으로 켰고 4.97.0 에 기본으로 껐다.

4.35.0 은 창과 무관한 절대 임계(3만 토큰)에서 지난 도구 결과를 600자로 잘랐다. 창이 200K·1M 인
모델도 방금 받은 결과를 잘린 채 받았고, 그 잘린 것이 기억에 "도구가 돌려준 것" 으로 남았다
(2026-10-23 감사). 정확성이 먼저다: 줄임은 창이 모자랄 때만(압축 80%·호출 직전 guard). 비용 트리거는
호스트가 명시해서 켤 때만 돈다. 켜도 원본은 남는다(기억과 ToolResultRead 가 원본을 본다).
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.core.context_prune import DEFAULT_PRUNE_OVER_TOKENS
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.core.token_estimate import estimate_prompt_tokens
from xgen_agent_runtime.stages.s02_context.artifact.default.stage import ContextStage

HUGE = "판정 로그 라인 " * 1200  # ~10k chars — trim 대상(4,000자 초과)


def _tool_turn(tool_id: str, content) -> list:
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": content}]},
    ]


def _state(n_calls: int, *, content=None) -> PipelineState:
    """실사용 모양: 호출마다 **서로 다른** 큰 Bash 출력이 쌓인다(중복이 아니라
    누적이 문제였다 — 실측에서 회수분의 46% 가 절단, 5% 가 중복 축약)."""
    state = PipelineState(session_id="s", model="m")
    state.messages = [{"role": "user", "content": "긴 작업 해줘"}]
    for i in range(n_calls):
        state.messages += _tool_turn(f"t{i}", content if content is not None else f"{HUGE}{i}")
    state.context_window_budget = 200_000  # 용량 트리거(160k)는 절대 안 닿는 높이
    return state


#: 호스트가 비용 트리거를 켤 때 쓰던 값(4.35.0 의 기본).
OPT_IN = 30_000


def _stage(**kw) -> ContextStage:
    kw.setdefault("prune_over_tokens", OPT_IN)
    return ContextStage(stateless=False, **kw)


def _pruned_events(state: PipelineState) -> list:
    return [e for e in state.events if e["type"] == "context.pruned"]


# ── 트리거 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_by_default_nothing_is_cut_while_the_window_has_room():
    """창(200K)의 80% 에 한참 못 미치는 턴: 기본값으로는 아무것도 줄이지 않는다."""
    state = _state(14)
    before = estimate_prompt_tokens(state)
    assert OPT_IN < before < state.context_window_budget * 0.8
    snapshot = [str(m) for m in state.messages]

    await ContextStage(stateless=False).execute(None, state)

    assert DEFAULT_PRUNE_OVER_TOKENS == 0
    assert [str(m) for m in state.messages] == snapshot
    assert not _pruned_events(state)


@pytest.mark.asyncio
async def test_an_opted_in_cost_trigger_still_runs():
    state = _state(14)
    before = estimate_prompt_tokens(state)

    await _stage().execute(None, state)

    after = estimate_prompt_tokens(state)
    assert after < before * 0.6, (before, after)
    ev = _pruned_events(state)
    assert len(ev) == 1 and ev[0]["data"]["trigger"] == "cost"
    assert ev[0]["data"]["trimmed"] > 0
    assert ev[0]["data"]["threshold_tokens"] == OPT_IN
    assert not [e for e in state.events if e["type"] == "context.compacted"]


@pytest.mark.asyncio
async def test_an_opted_in_cut_keeps_the_original_for_memory_and_tool_result_read():
    """잘린 것은 요청 사본뿐이다. 기억은 원본을 기록하고, ToolResultRead 는 원본을 읽는다."""
    from xgen_agent_runtime.core.originals import original_of
    from xgen_agent_runtime.tools.base import ToolContext
    from xgen_agent_runtime.tools.built_in.tool_result_read_tool import ToolResultReadTool

    state = _state(14)
    first = state.messages[2]  # t0 의 결과
    full = first["content"][0]["content"]

    await _stage().execute(None, state)

    shown = first["content"][0]["content"]
    assert len(shown) < len(full) and 'ToolResultRead(id="t0")' in shown
    assert original_of(state, first)["content"][0]["content"] == full

    read = await ToolResultReadTool().execute({"id": "t0", "limit": 100_000}, ToolContext(state_view=state))
    assert not read.is_error and full in read.content
    assert read.content.startswith(f"[Bash result t0: chars 0-{len(full)} of {len(full)}; end of the result]")


@pytest.mark.asyncio
async def test_short_turn_under_the_threshold_is_untouched():
    state = _state(2)
    assert estimate_prompt_tokens(state) < OPT_IN
    snapshot = [str(m) for m in state.messages]

    await _stage().execute(None, state)

    assert [str(m) for m in state.messages] == snapshot
    assert not _pruned_events(state)


@pytest.mark.asyncio
async def test_threshold_zero_disables_the_cost_trigger():
    state = _state(14)
    await _stage(prune_over_tokens=0).execute(None, state)
    assert not _pruned_events(state)


@pytest.mark.asyncio
async def test_compaction_switch_also_gates_the_prune():
    """compaction_enabled=False 는 '이 스테이지는 이력을 건드리지 않는다' 는 계약이다."""
    state = _state(14)
    await _stage(compaction_enabled=False, prune_over_tokens=1).execute(None, state)
    assert not _pruned_events(state)


# ── 안전성 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recent_tail_and_tool_pairing_survive():
    state = _state(14)
    ids_before = [
        b["tool_use_id"]
        for m in state.messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    n_before = len(state.messages)
    last_result = state.messages[-1]["content"][0]["content"]

    await _stage().execute(None, state)

    ids_after = [
        b["tool_use_id"]
        for m in state.messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(state.messages) == n_before  # 수·순서 보존
    assert ids_after == ids_before  # tool_use_id 하나도 잃지 않는다
    assert state.messages[-1]["content"][0]["content"] == last_result  # 최근 결과는 원본


@pytest.mark.asyncio
async def test_second_pass_is_a_noop():
    """매 반복 도는 트리거라 멱등이어야 한다 — 한 번 자른 결과를 또 자르지 않는다."""
    state = _state(14)
    await _stage().execute(None, state)
    after_first = [str(m) for m in state.messages]
    state.events.clear()

    await _stage().execute(None, state)

    assert [str(m) for m in state.messages] == after_first
    assert not _pruned_events(state)


@pytest.mark.asyncio
async def test_small_results_are_never_touched_however_many():
    """짧은 결과만 쌓인 턴은 자를 것이 없다 — 임계를 넘어도 이벤트가 없다."""
    state = _state(60, content="ok")
    state.system = "x" * 200_000  # 임계는 시스템 프롬프트로 넘긴다
    state.shared.pop("_prompt_tokens_memo", None)
    assert estimate_prompt_tokens(state) > OPT_IN

    await _stage().execute(None, state)

    assert not _pruned_events(state)


# ── 설정 ───────────────────────────────────────────────────────────────


def test_config_roundtrip():
    assert ContextStage(stateless=False).get_config()["prune_over_tokens"] == DEFAULT_PRUNE_OVER_TOKENS == 0
    stage = _stage()
    assert stage.get_config()["prune_over_tokens"] == OPT_IN
    stage.update_config({"prune_over_tokens": 12_345})
    assert stage.get_config()["prune_over_tokens"] == 12_345
    stage.update_config({"prune_over_tokens": "nope"})  # 잘못된 값은 무시
    assert stage.get_config()["prune_over_tokens"] == 12_345
    stage.update_config({"prune_over_tokens": -5})
    assert stage.get_config()["prune_over_tokens"] == 0
