"""Compaction master switch (3.3.0) — ``ContextStage(compaction_enabled=...)``.

Hosts (e.g. the xgen-workflow agent node) expose a user-facing "컨텍스트
자동 압축" toggle. Off must mean OFF everywhere this stage compacts:

* no proactive compaction at the 80% watermark,
* no background summary scheduling,
* no deterministic prune (it only runs inside the compaction path),
* and the Stage 4 guard's auto-wired budget recovery is withheld, so a
  token-budget "compact" signal degrades to the pre-2.5.0 hard reject
  instead of compacting behind the host's back.

Retrieval / strategy / memory injection stay untouched by the switch.
"""

from __future__ import annotations

import asyncio

import pytest

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.stages.s02_context import ContextStage
from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
    TruncateCompactor,
)
from xgen_agent_runtime.stages.s04_guard import GuardStage
from xgen_agent_runtime.stages.s04_guard.artifact.default.guards import TokenBudgetGuard


def _over_budget_state(n_messages: int = 40) -> PipelineState:
    """A state whose projected size is far past 80% of a small budget."""
    state = PipelineState()
    state.context_window_budget = 1_000  # tiny window → instant pressure
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        state.messages.append({"role": role, "content": f"메시지 {i} " + ("가" * 400)})
    return state


def _events(state: PipelineState, event_type: str) -> list:
    return [e for e in state.events if e.get("type") == event_type]


def test_enabled_compacts_over_budget() -> None:
    stage = ContextStage(compactor=TruncateCompactor(keep_last=4))
    state = _over_budget_state()
    before = len(state.messages)

    asyncio.run(stage.execute(None, state))

    assert len(state.messages) < before, "80% 초과인데 압축이 안 됐다"
    assert _events(state, "context.compacted"), "context.compacted 이벤트가 없다"


def test_loop_requested_compaction_is_synchronous_and_consumed() -> None:
    stage = ContextStage(compactor=TruncateCompactor(keep_last=4))
    state = _over_budget_state()
    state.shared["context.compaction_requested"] = {
        "source": "multi_dim_budget",
        "iteration": 3,
    }

    asyncio.run(stage.execute(None, state))

    assert len(state.messages) <= 4
    assert "context.compaction_requested" not in state.shared
    compacted = _events(state, "context.compacted")
    assert compacted[-1]["data"]["trigger"] == "requested"
    assert "target_met" in compacted[-1]["data"]


def test_disabled_never_compacts() -> None:
    stage = ContextStage(
        compactor=TruncateCompactor(keep_last=4), compaction_enabled=False
    )
    state = _over_budget_state()
    before = list(state.messages)

    asyncio.run(stage.execute(None, state))

    assert state.messages == before, "compaction_enabled=False 인데 메시지가 바뀌었다"
    assert not _events(state, "context.compacted")
    assert not _events(state, "context.pruned"), "프루닝도 압축 경로다 — 꺼져야 한다"
    assert not _events(state, "context.compaction_scheduled")
    assert stage._bg_compaction is None


def test_config_roundtrip_and_update() -> None:
    stage = ContextStage()
    assert stage.get_config()["compaction_enabled"] is True

    stage.update_config({"compaction_enabled": False})
    assert stage.get_config()["compaction_enabled"] is False

    # Off via update_config must behave identically to the constructor arg.
    state = _over_budget_state()
    before = list(state.messages)
    asyncio.run(stage.execute(None, state))
    assert state.messages == before

    stage.update_config({"compaction_enabled": True})
    asyncio.run(stage.execute(None, state))
    assert len(state.messages) < len(before), "다시 켠 뒤에는 압축돼야 한다"


def test_schema_declares_toggle() -> None:
    fields = {f.name for f in ContextStage().get_config_schema().fields}
    assert "compaction_enabled" in fields


def test_guard_autowire_respects_disabled_flag() -> None:
    """Pipeline._init_state must NOT hand the guard a compactor when the
    context stage has compaction disabled — and must CLEAR a previously
    auto-wired one (re-sync contract)."""
    from xgen_agent_runtime.core.builder import PipelineBuilder

    def _build(enabled: bool):
        builder = (
            PipelineBuilder("t", api_key="k", model="claude-sonnet-4-6")
            .with_system(prompt="s")
            .with_context(compaction_enabled=enabled)
            .with_guard(guards=[TokenBudgetGuard(min_remaining_tokens=10)])
            .with_loop(max_turns=1)
        )
        return builder.build()

    p_on = _build(True)
    p_on._init_state(None)
    guard_on = p_on._stages.get(4)
    assert isinstance(guard_on, GuardStage)
    assert guard_on._budget_compactor is not None, "켜짐 → guard 회복 배선돼야 한다"

    p_off = _build(False)
    p_off._init_state(None)
    guard_off = p_off._stages.get(4)
    assert guard_off._budget_compactor is None, "꺼짐 → guard 가 몰래 압축하면 안 된다"

    # Re-sync: a guard that WAS wired must be cleared when the host turns
    # compaction off between turns (update_config path).
    p_on._stages.get(2).update_config({"compaction_enabled": False})
    p_on._init_state(None)
    assert guard_on._budget_compactor is None, "재동기화가 기존 배선을 걷어내야 한다"


def _band_state(budget: int = 1_000) -> PipelineState:
    """80–90% 구간의 상태 — 백그라운드 유예 경계를 정확히 밟는다."""
    from xgen_agent_runtime.core.token_estimate import estimate_prompt_tokens

    state = PipelineState()
    state.context_window_budget = budget
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        state.messages.append({"role": role, "content": f"m{i} " + ("가" * 165)})
    est = estimate_prompt_tokens(state)
    assert budget * 0.8 < est <= budget * 0.9, f"테스트 전제 붕괴: est={est}"
    return state


def test_background_compaction_off_forces_sync() -> None:
    """원샷 호스트 계약: bg OFF → 80–90% 구간에서도 **즉시** 압축한다.

    LLMSummaryCompactor 는 클라이언트가 없으면 정적 플레이스홀더로 강등되므로
    (self-wire 폴백), 동기 경로가 네트워크 없이 완결된다.
    """
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
        LLMSummaryCompactor,
    )

    stage = ContextStage(
        compactor=LLMSummaryCompactor(keep_recent=4), background_compaction=False
    )
    state = _band_state()
    before = len(state.messages)

    asyncio.run(stage.execute(None, state))

    assert len(state.messages) < before, "bg OFF 인데 동기 압축이 안 됐다"
    assert not _events(state, "context.compaction_scheduled"), "bg OFF 인데 스케줄됐다"
    assert stage._bg_compaction is None


def test_background_compaction_on_defers_in_band() -> None:
    """기본값(bg ON)은 80–90% 구간에서 유예한다 — TTFT 동작 보존 확인."""
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
        LLMSummaryCompactor,
    )

    async def _run() -> tuple:
        stage = ContextStage(compactor=LLMSummaryCompactor(keep_recent=4))
        state = _band_state()
        before = len(state.messages)
        await stage.execute(None, state)
        scheduled = bool(_events(state, "context.compaction_scheduled"))
        pending = stage._bg_compaction
        # 정리 — 테스트가 태스크를 남기지 않도록.
        stage.cancel_bg_compaction()
        return before, len(state.messages), scheduled, pending

    before, after, scheduled, pending = asyncio.run(_run())
    assert after == before, "유예 구간인데 즉시 압축됐다"
    assert scheduled and pending is not None


def test_background_compaction_rejects_changed_prefix_even_when_tail_matches() -> None:
    """A length+tail check accepted stale summaries after an earlier message
    was rewritten. The whole captured prefix identity is the CAS token."""
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
        LLMSummaryCompactor,
    )

    async def _run() -> bool:
        stage = ContextStage(compactor=LLMSummaryCompactor(keep_recent=4))
        state = _band_state()
        stage._schedule_bg_compaction(state)
        assert stage._bg_compaction is not None
        await stage._bg_compaction["task"]
        original_tail = state.messages[-1]
        state.messages[0] = dict(state.messages[0])
        assert state.messages[-1] is original_tail
        before = list(state.messages)
        applied = await stage._apply_bg_compaction(state)
        assert state.messages == before
        return applied

    assert asyncio.run(_run()) is False


def test_cancel_bg_compaction_reaps_pending_task() -> None:
    async def _run() -> asyncio.Task:
        stage = ContextStage()
        task = asyncio.create_task(asyncio.sleep(30))
        stage._bg_compaction = {"task": task, "len": 1, "tail_id": 0}
        stage.cancel_bg_compaction()
        assert stage._bg_compaction is None
        await asyncio.sleep(0)  # cancellation 전파
        return task

    task = asyncio.run(_run())
    assert task.cancelled(), "pending bg 태스크가 회수되지 않았다"


def test_pipeline_aclose_cancels_bg_compaction() -> None:
    """원샷 호스트의 turn teardown(aclose)이 유예 태스크를 정리한다."""
    from xgen_agent_runtime.core.builder import PipelineBuilder

    async def _run() -> asyncio.Task:
        pipeline = (
            PipelineBuilder("t", api_key="k", model="claude-sonnet-4-6")
            .with_system(prompt="s")
            .with_context()
            .with_loop(max_turns=1)
            .build()
        )
        ctx = pipeline._stages.get(2)
        task = asyncio.create_task(asyncio.sleep(30))
        ctx._bg_compaction = {"task": task, "len": 1, "tail_id": 0}
        await pipeline.aclose()
        await asyncio.sleep(0)
        return task

    task = asyncio.run(_run())
    assert task.cancelled(), "aclose 가 bg 압축 태스크를 회수하지 않았다"


def test_disabled_still_retrieves_memory() -> None:
    """The switch only kills compaction — retrieval/injection must survive."""
    from xgen_agent_runtime.stages.s02_context.types import MemoryChunk

    class _OneChunk:
        name = "one"
        description = "test"

        async def retrieve(self, query: str, state: PipelineState) -> list:
            return [MemoryChunk(key="k", content="기억", source="test", relevance_score=1.0)]

    stage = ContextStage(retriever=_OneChunk(), compaction_enabled=False)
    state = PipelineState()
    state.context_window_budget = 1_000
    state.messages.append({"role": "user", "content": "질문"})

    asyncio.run(stage.execute(None, state))

    assert state.memory_refs, "compaction off 이 retrieval 까지 죽였다"
    assert state.metadata.get("memory_context"), "메모리 주입이 사라졌다"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
