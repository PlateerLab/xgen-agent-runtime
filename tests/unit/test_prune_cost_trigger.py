"""결정적 prune 의 비용 트리거 (Stage 2, 4.35.0).

근거 (dev 28일, 벤치 제외): 긴 턴 입력의 66% 가 매 호출 다시 실리는 이력이고
(16회+ 턴은 71~72%), 그걸 줄이는 코드(core/context_prune.py)는 이미 있었다.
그런데 compaction 경로 안에서만 불렸고 compaction 은 윈도우×0.8 에서만 도는데,
윈도우가 200k(claude)·524k(qwen) 라 문턱이 160k·419k — 실제 최대 프롬프트는
135,487. **28일 동안 한 번도 실행되지 않았다.** 용량 트리거만 있고 비용
트리거가 없던 것이다. 그래서 윈도우와 무관한 절대 임계를 둔다.
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


def _stage(**kw) -> ContextStage:
    return ContextStage(stateless=False, **kw)


def _pruned_events(state: PipelineState) -> list:
    return [e for e in state.events if e["type"] == "context.pruned"]


# ── 트리거 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prune_runs_on_cost_even_though_capacity_never_triggers():
    """실측 재현: 윈도우 200k 의 80%(160k)에 한참 못 미치는 턴이 이력만 키운다."""
    state = _state(14)
    before = estimate_prompt_tokens(state)
    assert DEFAULT_PRUNE_OVER_TOKENS < before < state.context_window_budget * 0.8

    await _stage().execute(None, state)

    after = estimate_prompt_tokens(state)
    assert after < before * 0.6, (before, after)
    ev = _pruned_events(state)
    assert len(ev) == 1 and ev[0]["data"]["trigger"] == "cost"
    assert ev[0]["data"]["trimmed"] > 0
    assert ev[0]["data"]["tokens_after"] < ev[0]["data"]["tokens_before"]
    assert ev[0]["data"]["threshold_tokens"] == DEFAULT_PRUNE_OVER_TOKENS
    # 압축(LLM)은 돌지 않았다 — 이건 순수 함수 경로다.
    assert not [e for e in state.events if e["type"] == "context.compacted"]


@pytest.mark.asyncio
async def test_short_turn_under_the_threshold_is_untouched():
    """기본값 30,000 은 dev 28일에서 5회 이하 턴을 하나도 건드리지 않았다."""
    state = _state(2)
    assert estimate_prompt_tokens(state) < DEFAULT_PRUNE_OVER_TOKENS
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
    assert estimate_prompt_tokens(state) > DEFAULT_PRUNE_OVER_TOKENS

    await _stage().execute(None, state)

    assert not _pruned_events(state)


# ── 설정 ───────────────────────────────────────────────────────────────


def test_config_roundtrip():
    stage = _stage()
    assert stage.get_config()["prune_over_tokens"] == DEFAULT_PRUNE_OVER_TOKENS
    stage.update_config({"prune_over_tokens": 12_345})
    assert stage.get_config()["prune_over_tokens"] == 12_345
    stage.update_config({"prune_over_tokens": "nope"})  # 잘못된 값은 무시
    assert stage.get_config()["prune_over_tokens"] == 12_345
    stage.update_config({"prune_over_tokens": -5})
    assert stage.get_config()["prune_over_tokens"] == 0
