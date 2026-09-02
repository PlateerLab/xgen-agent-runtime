from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from xgen_agent_runtime import Pipeline, PipelineConfig, PipelineState
from xgen_agent_runtime.core.rollout_recorder import RolloutRecorder
from xgen_agent_runtime.stages.s01_input import InputStage
from xgen_agent_runtime.stages.s06_api import APIStage, MockProvider
from xgen_agent_runtime.stages.s09_parse import ParseStage
from xgen_agent_runtime.stages.s21_yield import YieldStage


def _pipeline(
    *,
    input_stage: InputStage | None = None,
    provider: MockProvider | None = None,
) -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="rollout-recorder-test"))
    pipeline.register_stage(input_stage or InputStage())
    pipeline.register_stage(APIStage(provider=provider or MockProvider(default_text="ok")))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_run_records_complete_correlated_event_stream(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorder = RolloutRecorder(path)
    pipeline = _pipeline()
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )

    result = await pipeline.run("hello", PipelineState(session_id="session-1"))
    await recorder.shutdown()

    records = _records(path)
    assert result.state is not None
    assert records[0]["type"] == "pipeline.start"
    assert records[-1]["type"] == "pipeline.complete"
    assert [record["seq"] for record in records] == sorted(record["seq"] for record in records)
    assert {record["session_id"] for record in records} == {"session-1"}
    assert len({record["run_id"] for record in records}) == 1
    assert not pipeline._run_rollout_recorders
    assert not pipeline._run_rollout_failures


@pytest.mark.asyncio
async def test_run_stream_flushes_terminal_event_before_return(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    recorder = RolloutRecorder(path)
    pipeline = _pipeline()
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )

    streamed = [
        event
        async for event in pipeline.run_stream(
            "hello", PipelineState(session_id="stream-session")
        )
    ]

    records = _records(path)
    assert streamed[-1].type == "pipeline.complete"
    assert records[-1]["type"] == "pipeline.complete"
    assert records[-1]["seq"] == streamed[-1].seq
    assert not pipeline._run_rollout_recorders
    await recorder.shutdown()


@pytest.mark.asyncio
async def test_terminal_event_reaches_bus_only_after_durability_barrier() -> None:
    class _OrderingRecorder:
        def __init__(self) -> None:
            self.flushed = False

        def record_nowait(self, event: Any) -> None:
            self.flushed = False

        async def flush(self) -> None:
            self.flushed = True

    recorder = _OrderingRecorder()
    pipeline = _pipeline()
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )
    observed: list[bool] = []
    pipeline.on("pipeline.complete", lambda _event: observed.append(recorder.flushed))

    result = await pipeline.run("hello")

    assert result.success is True
    assert observed == [True]


@pytest.mark.asyncio
async def test_concurrent_runs_share_recorder_without_reordering(tmp_path: Path) -> None:
    class _YieldingProvider(MockProvider):
        async def create_message(self, request: Any) -> Any:
            await asyncio.sleep(0.01)
            return await super().create_message(request)

    path = tmp_path / "concurrent.jsonl"
    recorder = RolloutRecorder(path)
    pipeline = _pipeline(provider=_YieldingProvider(default_text="ok"))
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )

    result_a, result_b = await asyncio.gather(
        pipeline.run("alpha", PipelineState(session_id="session-a")),
        pipeline.run("beta", PipelineState(session_id="session-b")),
    )
    await recorder.shutdown()

    records = _records(path)
    assert result_a.success is result_b.success is True
    assert [record["seq"] for record in records] == sorted(record["seq"] for record in records)
    run_ids = {record["run_id"] for record in records}
    assert len(run_ids) == 2
    for run_id in run_ids:
        per_run = [record["type"] for record in records if record["run_id"] == run_id]
        assert per_run[0] == "pipeline.start"
        assert per_run[-1] == "pipeline.complete"


@pytest.mark.asyncio
async def test_prepopulated_states_can_use_separate_recorders(tmp_path: Path) -> None:
    recorder_a = RolloutRecorder(tmp_path / "a.jsonl")
    recorder_b = RolloutRecorder(tmp_path / "b.jsonl")
    pipeline = _pipeline()
    state_a = PipelineState(session_id="session-a")
    state_b = PipelineState(session_id="session-b")
    state_a.session_runtime = SimpleNamespace(rollout_recorder=recorder_a)
    state_b.session_runtime = SimpleNamespace(rollout_recorder=recorder_b)

    await asyncio.gather(pipeline.run("alpha", state_a), pipeline.run("beta", state_b))
    await asyncio.gather(recorder_a.shutdown(), recorder_b.shutdown())

    assert {record["session_id"] for record in _records(recorder_a.path)} == {"session-a"}
    assert {record["session_id"] for record in _records(recorder_b.path)} == {"session-b"}


@pytest.mark.asyncio
async def test_cancellation_flushes_the_accepted_prefix(tmp_path: Path) -> None:
    started = asyncio.Event()

    class _BlockingInput(InputStage):
        async def execute(self, input: Any, state: PipelineState) -> Any:
            started.set()
            await asyncio.Event().wait()

    path = tmp_path / "cancelled.jsonl"
    recorder = RolloutRecorder(path)
    pipeline = _pipeline(input_stage=_BlockingInput())
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )
    state = PipelineState(session_id="cancelled-session")
    task = asyncio.create_task(pipeline.run("hello", state))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await recorder.shutdown()

    assert [record["type"] for record in _records(path)] == [
        "pipeline.start",
        "stage.enter",
    ]
    assert state._turn_in_flight is False
    assert not pipeline._run_rollout_recorders


@pytest.mark.asyncio
async def test_aclose_flushes_prefix_before_active_run_is_reaped(tmp_path: Path) -> None:
    started = asyncio.Event()

    class _BlockingInput(InputStage):
        async def execute(self, input: Any, state: PipelineState) -> Any:
            started.set()
            await asyncio.Event().wait()

    path = tmp_path / "aclose.jsonl"
    recorder = RolloutRecorder(path)
    pipeline = _pipeline(input_stage=_BlockingInput())
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )
    task = asyncio.create_task(pipeline.run("hello"))
    await started.wait()

    await pipeline.aclose()
    with pytest.raises(asyncio.CancelledError):
        await task
    await recorder.shutdown()

    assert [record["type"] for record in _records(path)] == [
        "pipeline.start",
        "stage.enter",
    ]
    assert not pipeline._run_rollout_recorders


@pytest.mark.asyncio
async def test_bounded_queue_overflow_fails_run_instead_of_dropping(tmp_path: Path) -> None:
    path = tmp_path / "overflow.jsonl"
    recorder = RolloutRecorder(path, queue_size=1)
    pipeline = _pipeline()
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )

    result = await pipeline.run("hello")
    await recorder.shutdown()

    assert result.success is False
    assert result.error is not None and "queue is full" in result.error
    assert [record["type"] for record in _records(path)] == ["pipeline.start"]


@pytest.mark.asyncio
async def test_stage_failure_is_recorded_and_flushed(tmp_path: Path) -> None:
    class _FailingInput(InputStage):
        async def execute(self, input: Any, state: PipelineState) -> Any:
            raise ValueError("injected stage failure")

    path = tmp_path / "error.jsonl"
    recorder = RolloutRecorder(path)
    pipeline = _pipeline(input_stage=_FailingInput())
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )

    result = await pipeline.run("hello")
    await recorder.shutdown()

    records = _records(path)
    assert result.success is False
    assert records[-1]["type"] == "pipeline.error"
    assert any(record["type"] == "stage.error" for record in records)


def test_invalid_rollout_recorder_fails_before_claiming_state() -> None:
    pipeline = _pipeline()
    state = PipelineState()
    state.session_runtime = SimpleNamespace(rollout_recorder=object())

    with pytest.raises(TypeError, match="record_nowait.*flush"):
        pipeline._init_state(state)

    assert state._turn_in_flight is False
    assert not pipeline._run_rollout_recorders


@pytest.mark.asyncio
async def test_record_failure_uses_existing_pipeline_error_channel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingRecorder:
        def __init__(self) -> None:
            self.flush_calls = 0

        def record_nowait(self, event: Any) -> None:
            raise RuntimeError("injected admission failure")

        async def flush(self) -> None:
            self.flush_calls += 1

    recorder = _FailingRecorder()
    pipeline = _pipeline()
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )

    with caplog.at_level("ERROR"):
        result = await pipeline.run("hello")

    assert result.success is False
    assert result.error is not None and "injected admission failure" in result.error
    assert recorder.flush_calls >= 1
    assert any(event.type == "pipeline.error" for event in pipeline._event_journal)
    assert "rollout recording failed" in caplog.text
    assert caplog.text.count("rollout recording failed") == 1
    assert not pipeline._run_rollout_recorders


@pytest.mark.asyncio
async def test_flush_failure_is_not_silently_reported_as_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FlushFailingRecorder:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def record_nowait(self, event: Any) -> None:
            self.events.append(event)

        async def flush(self) -> None:
            raise OSError("injected flush failure")

    recorder = _FlushFailingRecorder()
    pipeline = _pipeline()
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=recorder)
    )

    bus_terminals: list[str] = []
    pipeline.on("pipeline.complete", lambda event: bus_terminals.append(event.type))
    pipeline.on("pipeline.error", lambda event: bus_terminals.append(event.type))
    with caplog.at_level("ERROR"):
        result = await pipeline.run("hello")

    assert result.success is False
    assert result.error is not None and "injected flush failure" in result.error
    assert [event.type for event in recorder.events][-1] == "pipeline.complete"
    assert any(event.type == "pipeline.error" for event in pipeline._event_journal)
    assert bus_terminals == ["pipeline.error"]
    assert "rollout recording failed" in caplog.text
    assert caplog.text.count("rollout recording failed") == 1
    assert not pipeline._run_rollout_failures


@pytest.mark.asyncio
async def test_stream_reports_flush_failure_without_publishing_complete() -> None:
    class _FlushFailingRecorder:
        def record_nowait(self, event: Any) -> None:
            pass

        async def flush(self) -> None:
            raise OSError("stream flush failure")

    pipeline = _pipeline()
    pipeline.attach_runtime(
        session_runtime=SimpleNamespace(rollout_recorder=_FlushFailingRecorder())
    )

    events = [event async for event in pipeline.run_stream("hello")]

    assert [event.type for event in events if event.type.startswith("pipeline.")] == [
        "pipeline.start",
        "pipeline.error",
    ]
    assert events[-1].data["error"] == "stream flush failure"
