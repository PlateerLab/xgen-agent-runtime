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
from typing import Any

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
