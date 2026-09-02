"""Resumable slice semantics for long-running agent tasks."""

from __future__ import annotations

from typing import Any

import pytest

from xgen_agent_runtime import CONTINUE_RUN, Pipeline, PipelineConfig, PipelineState
from xgen_agent_runtime.core.stage import Stage
from xgen_agent_runtime.host.runner import run_turn, stream_turn
from xgen_agent_runtime.session.persistence import FileSessionPersistence
from xgen_agent_runtime.stages.s01_input import InputStage
from xgen_agent_runtime.stages.s16_loop import LoopController, LoopDecision, LoopStage
from xgen_agent_runtime.stages.s20_persist import EveryNTurnsFrequency, PersistStage
from xgen_agent_runtime.stages.s20_persist.interface import Persister


class _AlwaysContinue(LoopController):
    @property
    def name(self) -> str:
        return "always_continue"

    def decide(self, state: PipelineState) -> str:
        return LoopDecision.CONTINUE


class _CompleteOnThirdPass(Stage[Any, Any]):
    @property
    def name(self) -> str:
        return "counting"

    @property
    def order(self) -> int:
        return 6

    @property
    def category(self) -> str:
        return "test"

    async def execute(self, input: Any, state: PipelineState) -> Any:
        count = int(state.metadata.get("passes", 0)) + 1
        state.metadata["passes"] = count
        state.accumulate_cost(0.1)
        if count >= 3:
            state.loop_decision = LoopDecision.COMPLETE
            state.final_text = "done"
        return input


def _pipeline() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(max_iterations=2))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(_CompleteOnThirdPass())
    pipeline.register_stage(LoopStage(_AlwaysContinue()))
    return pipeline


@pytest.mark.asyncio
async def test_iteration_cap_suspends_then_continues_without_duplicate_user_message() -> None:
    pipeline = _pipeline()
    state = PipelineState(session_id="slice")

    first = await pipeline.run("do the long task", state)
    assert first.success is False
    assert first.status == "suspended"
    assert first.termination_reason == "max_iterations_per_slice"
    assert first.resumable is True
    assert state.messages == [{"role": "user", "content": "do the long task"}]
    assert state.total_cost_usd == pytest.approx(0.2)
    assert state.session_cost_usd == pytest.approx(0.2)

    second = await pipeline.run(CONTINUE_RUN, state)
    assert second.success is True
    assert second.status == "completed"
    assert second.text == "done"
    assert state.messages == [{"role": "user", "content": "do the long task"}]
    # Cost is task/turn cumulative across slices; session accounting adds
    # only the new delta instead of double-counting the first slice.
    assert state.total_cost_usd == pytest.approx(0.3)
    assert state.session_cost_usd == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_continue_run_rejects_non_suspended_state() -> None:
    pipeline = _pipeline()
    with pytest.raises(ValueError, match="status='suspended'"):
        await pipeline.run(CONTINUE_RUN, PipelineState(session_id="fresh"))


class _CapturePersister(Persister):
    def __init__(self) -> None:
        self.records = []

    @property
    def name(self) -> str:
        return "capture"

    async def write(self, record, state) -> None:  # noqa: ANN001
        self.records.append(record)


@pytest.mark.asyncio
async def test_suspended_slice_forces_checkpoint_even_off_frequency() -> None:
    persister = _CapturePersister()
    stage = PersistStage(persister=persister, frequency=EveryNTurnsFrequency(5))
    state = PipelineState(session_id="checkpoint", iteration=2)
    state.mark_suspended("max_iterations_per_slice")

    await stage.execute(None, state)

    assert len(persister.records) == 1
    record = persister.records[0]
    assert state.checkpoint_id == record.checkpoint_id
    assert record.payload["run_status"] == "suspended"
    assert record.payload["resumable"] is True


def test_file_session_persistence_round_trips_suspended_status(tmp_path) -> None:  # noqa: ANN001
    persistence = FileSessionPersistence(str(tmp_path))
    state = PipelineState(session_id="persisted")
    state.add_message("user", "work")
    state.total_cost_usd = 1.25
    state.session_cost_usd = 1.25
    state._accounted_turn_cost_usd = 1.25
    state.mark_suspended("max_iterations_per_slice", detail="slice boundary")

    persistence.save("persisted", state)
    restored = persistence.load("persisted")

    assert restored is not None
    assert restored.run_status == "suspended"
    assert restored.termination_reason == "max_iterations_per_slice"
    assert restored.resumable is True
    assert restored.total_cost_usd == pytest.approx(1.25)
    assert restored.session_cost_usd == pytest.approx(1.25)
    assert restored._accounted_turn_cost_usd == pytest.approx(1.25)
    assert not list((tmp_path / "persisted").glob("*.tmp"))


def test_host_bridge_auto_continues_a_suspended_slice() -> None:
    state = PipelineState(session_id="host-continuation")
    output = run_turn(
        _pipeline(),
        "do the long task",
        state,
        max_continuation_slices=1,
    )

    assert output == "done"
    assert state.run_status == "completed"
    assert state.metadata["passes"] == 3


def test_streaming_host_bridge_announces_then_auto_continues() -> None:
    state = PipelineState(session_id="host-stream-continuation")

    chunks = list(
        stream_turn(
            _pipeline(),
            "do the long task",
            state,
            max_continuation_slices=1,
        )
    )

    progress = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and chunk.get("type") == "agent_event"
        and chunk.get("data", {}).get("type") == "task_progress"
    ]
    assert len(progress) == 1
    assert progress[0]["data"]["reason"] == "max_iterations_per_slice"
    assert "done" in [chunk for chunk in chunks if isinstance(chunk, str)]
    assert state.run_status == "completed"
