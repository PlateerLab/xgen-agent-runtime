from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from xgen_agent_runtime.core.rollout_recorder import (
    RolloutBackpressureError,
    RolloutRecorder,
)
from xgen_agent_runtime.events.types import PipelineEvent


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorder_requires_running_event_loop(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="running event loop"):
        RolloutRecorder(tmp_path / "rollout.jsonl")


@pytest.mark.asyncio
async def test_flush_writes_ordered_immutable_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rollout.jsonl"
    recorder = RolloutRecorder(path)
    event = PipelineEvent(
        type="tool.complete",
        stage="tool",
        iteration=2,
        data={"text": "안녕"},
        session_id="session-1",
        run_id="run-1",
        seq=7,
    )

    await recorder.record(event)
    event.data["text"] = "mutated-after-record"
    await recorder.record({"type": "pipeline.complete", "seq": 8})
    assert not path.exists()  # deferred until the first durability barrier

    await recorder.flush()

    records = _read_jsonl(path)
    assert [record["type"] for record in records] == ["tool.complete", "pipeline.complete"]
    assert records[0]["data"] == {"text": "안녕"}
    assert records[0]["seq"] == 7
    await recorder.shutdown()


@pytest.mark.asyncio
async def test_resume_preserves_existing_records_and_repairs_missing_newline(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(b'{"type":"existing"}')
    recorder = RolloutRecorder(path)

    await recorder.record({"type": "new"})
    await recorder.shutdown()

    assert path.read_bytes() == b'{"type":"existing"}\n{"type":"new"}\n'


@pytest.mark.asyncio
async def test_shutdown_is_idempotent_and_rejects_new_records(tmp_path: Path) -> None:
    recorder = RolloutRecorder(tmp_path / "rollout.jsonl")
    await recorder.record({"type": "one"})

    await asyncio.gather(recorder.shutdown(), recorder.shutdown())
    await recorder.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        await recorder.record({"type": "late"})
    with pytest.raises(RuntimeError, match="shut down"):
        recorder.record_nowait({"type": "late"})


@pytest.mark.asyncio
async def test_cancelled_shutdown_still_reaps_writer(tmp_path: Path) -> None:
    recorder = RolloutRecorder(tmp_path / "rollout.jsonl")
    await recorder.record({"type": "one"})
    shutdown = asyncio.create_task(recorder.shutdown())
    shutdown.cancel()

    with pytest.raises(asyncio.CancelledError):
        await shutdown

    await recorder.shutdown()


@pytest.mark.asyncio
async def test_record_nowait_reports_bounded_backpressure(tmp_path: Path) -> None:
    recorder = RolloutRecorder(tmp_path / "rollout.jsonl", queue_size=1)
    recorder.record_nowait({"type": "one"})

    with pytest.raises(RolloutBackpressureError, match="queue is full"):
        recorder.record_nowait({"type": "two"})

    await recorder.shutdown()


@pytest.mark.asyncio
async def test_flush_failure_keeps_pending_suffix_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xgen_agent_runtime.core import rollout_recorder as module

    path = tmp_path / "rollout.jsonl"
    recorder = RolloutRecorder(path)
    await recorder.record({"type": "one"})
    original = module._WriterState._write_pending_once
    attempts = 0

    def fail_twice(self: object, *, durable: bool) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OSError("injected disk failure")
        original(self, durable=durable)  # type: ignore[arg-type]

    monkeypatch.setattr(module._WriterState, "_write_pending_once", fail_twice)

    with pytest.raises(OSError, match="injected disk failure"):
        await recorder.flush()
    assert not path.exists()

    await recorder.flush()
    assert _read_jsonl(path) == [{"type": "one"}]
    await recorder.shutdown()


@pytest.mark.asyncio
async def test_flush_retries_fsync_barrier_after_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xgen_agent_runtime.core import rollout_recorder as module

    path = tmp_path / "rollout.jsonl"
    recorder = RolloutRecorder(path)
    await recorder.record({"type": "one"})
    original_fsync = module.os.fsync
    attempts = 0

    def fail_first_fsync(fd: int) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", fail_first_fsync)

    await recorder.flush()

    assert attempts == 2
    assert _read_jsonl(path) == [{"type": "one"}]
    await recorder.shutdown()


@pytest.mark.asyncio
async def test_terminal_writer_failure_is_reported_instead_of_hanging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xgen_agent_runtime.core import rollout_recorder as module

    recorder = RolloutRecorder(tmp_path / "rollout.jsonl")

    def crash(self: object, *, durable: bool) -> None:
        raise AssertionError("injected writer crash")

    monkeypatch.setattr(module._WriterState, "write_pending", crash)
    await recorder.record({"type": "one"})

    with pytest.raises(RuntimeError, match="writer failed") as exc_info:
        await asyncio.wait_for(recorder.flush(), timeout=1)
    assert isinstance(exc_info.value.__cause__, AssertionError)


@pytest.mark.asyncio
async def test_persistent_shutdown_failure_reaps_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xgen_agent_runtime.core import rollout_recorder as module

    recorder = RolloutRecorder(tmp_path / "blocked" / "rollout.jsonl")

    def always_fail(self: object, *, durable: bool) -> None:
        raise OSError("persistent disk failure")

    monkeypatch.setattr(module._WriterState, "_write_pending_once", always_fail)
    await recorder.record({"type": "one"})

    with pytest.raises(OSError, match="persistent disk failure"):
        await recorder.shutdown()

    assert recorder._writer_task.done()
    await recorder.shutdown()  # terminal failure is still idempotently closed
    with pytest.raises(RuntimeError, match="shut down"):
        await recorder.record({"type": "late"})


@pytest.mark.asyncio
async def test_empty_shutdown_does_not_materialize_file(tmp_path: Path) -> None:
    path = tmp_path / "rollout.jsonl"
    recorder = RolloutRecorder(path)

    await recorder.shutdown()

    assert not path.exists()


@pytest.mark.asyncio
async def test_invalid_configuration_and_record_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="queue_size"):
        RolloutRecorder(tmp_path / "rollout.jsonl", queue_size=0)

    recorder = RolloutRecorder(tmp_path / "rollout.jsonl")
    with pytest.raises(TypeError, match="PipelineEvent or Mapping"):
        await recorder.record("not-a-record")  # type: ignore[arg-type]
    await recorder.shutdown()
