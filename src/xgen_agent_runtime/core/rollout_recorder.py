"""Durable, append-only JSONL recording for pipeline rollouts.

The recorder deliberately owns its file handle in one background task.  Producers
only serialize and enqueue records, so event publication never performs blocking
filesystem I/O and every accepted record reaches the file in queue order.

This module is intentionally not re-exported from the package root.  Pipeline
integration is an opt-in runtime concern and must not widen the established public
``run``/``run_stream`` or event/state contracts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, Final

from xgen_agent_runtime.events.types import PipelineEvent

_ADD: Final = "add"
_FLUSH: Final = "flush"
_SHUTDOWN: Final = "shutdown"


class RolloutBackpressureError(RuntimeError):
    """Raised when a synchronous producer fills the bounded recorder queue."""


class _Command:
    __slots__ = ("kind", "lines", "ack")

    def __init__(
        self,
        kind: str,
        *,
        lines: tuple[bytes, ...] = (),
        ack: asyncio.Future[None] | None = None,
    ) -> None:
        self.kind = kind
        self.lines = lines
        self.ack = ack


class _WriterState:
    """Mutable file state accessed exclusively by the writer task."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file: BinaryIO | None = None
        self.pending: list[bytes] = []

    def add(self, lines: tuple[bytes, ...]) -> None:
        self.pending.extend(lines)

    def write_pending(self, *, durable: bool) -> None:
        """Write the pending suffix, reopening and retrying once on I/O failure."""
        try:
            self._write_pending_once(durable=durable)
        except OSError:
            self.close()
            self._write_pending_once(durable=durable)

    def _write_pending_once(self, *, durable: bool) -> None:
        if not self.pending and self.file is None:
            # A previous attempt can write+flush every line and then fail at
            # fsync. The written prefix is no longer pending, but the retry
            # must reopen the file and repeat the durability barrier.
            if not durable or not self.path.exists():
                return
        self._ensure_open()
        assert self.file is not None

        written_count = 0
        try:
            for line in self.pending:
                self.file.write(line)
                written_count += 1
            self.file.flush()
            if durable:
                os.fsync(self.file.fileno())
        finally:
            # A successfully returned write belongs to the persisted prefix even
            # when a later line or barrier fails.  Retrying only the unwritten
            # suffix avoids duplicating the confirmed prefix.
            if written_count:
                del self.pending[:written_count]

    def _ensure_open(self) -> None:
        if self.file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        try:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            if size:
                file.seek(-1, os.SEEK_END)
                if file.read(1) != b"\n":
                    file.seek(0, os.SEEK_END)
                    file.write(b"\n")
            file.seek(0, os.SEEK_END)
        except BaseException:
            file.close()
            raise
        self.file = file

    def close(self) -> None:
        file, self.file = self.file, None
        if file is not None:
            file.close()


class RolloutRecorder:
    """Record immutable snapshots as ordered JSONL through a bounded queue.

    ``record`` applies asynchronous backpressure when the queue is full.
    ``record_nowait`` is provided for synchronous event funnels and fails loudly
    with :class:`RolloutBackpressureError` instead of silently dropping a record.
    A successful ``flush`` or ``shutdown`` includes ``fsync`` and therefore forms
    a durability barrier for all records accepted before that call.
    """

    def __init__(self, path: str | os.PathLike[str], *, queue_size: int = 256) -> None:
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1 (got {queue_size})")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("RolloutRecorder must be created inside a running event loop") from exc

        self._path = Path(path)
        self._queue: asyncio.Queue[_Command] = asyncio.Queue(maxsize=queue_size)
        self._terminal_failure: BaseException | None = None
        self._writer_task = loop.create_task(
            self._writer_loop(), name=f"rollout-recorder:{self._path.name}"
        )
        # Retrieve terminal exceptions even when no caller makes another API
        # call, retaining the cause without an asyncio teardown warning.
        self._writer_task.add_done_callback(self._capture_terminal_failure)
        self._stopped = False
        self._shutdown_lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def record(self, value: PipelineEvent | Mapping[str, Any]) -> None:
        """Queue one immutable record, waiting for bounded capacity if necessary."""
        self._ensure_running()
        command = _Command(_ADD, lines=(_encode_record(value),))
        await self._put(command)

    def record_nowait(self, value: PipelineEvent | Mapping[str, Any]) -> None:
        """Queue one record synchronously or raise without dropping it silently."""
        self._ensure_running()
        command = _Command(_ADD, lines=(_encode_record(value),))
        try:
            self._queue.put_nowait(command)
        except asyncio.QueueFull as exc:
            raise RolloutBackpressureError(
                f"rollout recorder queue is full for {self._path}"
            ) from exc

    async def flush(self) -> None:
        """Wait until all preceding records are flushed and fsynced."""
        self._ensure_running()
        await self._barrier(_FLUSH)

    async def shutdown(self) -> None:
        """Durably drain records and stop the writer; safe to call repeatedly."""
        async with self._shutdown_lock:
            if self._stopped:
                return
            self._ensure_running()
            barrier = asyncio.create_task(self._barrier(_SHUTDOWN))
            try:
                await asyncio.shield(barrier)
            except asyncio.CancelledError:
                # Once the shutdown command is admitted it must be reaped: if
                # cancellation left ``_stopped`` false after the writer exited,
                # every later shutdown would report an unexpected task death.
                await barrier
                self._stopped = True
                await self._writer_task
                raise
            self._stopped = True
            await self._writer_task

    def _ensure_running(self) -> None:
        if self._stopped:
            raise RuntimeError("rollout recorder is shut down")
        if self._terminal_failure is not None:
            raise RuntimeError("rollout recorder writer failed") from self._terminal_failure
        if self._writer_task.done():
            # The callback normally captures the exception immediately, but this
            # branch also covers an unexpected clean exit.
            self._capture_terminal_failure(self._writer_task)
            if self._terminal_failure is not None:
                raise RuntimeError("rollout recorder writer failed") from self._terminal_failure
            raise RuntimeError("rollout recorder writer stopped unexpectedly")

    async def _put(self, command: _Command) -> None:
        put_task = asyncio.create_task(self._queue.put(command))
        try:
            done, _ = await asyncio.wait(
                {put_task, self._writer_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if put_task in done:
                await put_task
                return
            self._ensure_running()
        finally:
            if not put_task.done():
                put_task.cancel()
                await asyncio.gather(put_task, return_exceptions=True)

    async def _barrier(self, kind: str) -> None:
        ack = asyncio.get_running_loop().create_future()
        await self._put(_Command(kind, ack=ack))
        done, _ = await asyncio.wait({ack, self._writer_task}, return_when=asyncio.FIRST_COMPLETED)
        if ack in done:
            await ack
            return
        self._ensure_running()

    async def _writer_loop(self) -> None:
        state = _WriterState(self._path)
        try:
            while True:
                command = await self._queue.get()
                try:
                    if command.kind == _ADD:
                        state.add(command.lines)
                        # Match Codex's deferred materialization: an unused
                        # recorder creates no empty file.  Once materialized,
                        # keep writes flowing without making every record fsync.
                        if state.file is not None:
                            await asyncio.to_thread(state.write_pending, durable=False)
                    elif command.kind == _FLUSH:
                        await asyncio.to_thread(state.write_pending, durable=True)
                        assert command.ack is not None
                        command.ack.set_result(None)
                    elif command.kind == _SHUTDOWN:
                        await asyncio.to_thread(state.write_pending, durable=True)
                        assert command.ack is not None
                        command.ack.set_result(None)
                        return
                    else:  # pragma: no cover - commands are module-private
                        raise RuntimeError(f"unknown rollout recorder command: {command.kind}")
                except OSError as exc:
                    # Keep both the writer and pending suffix alive so a later
                    # flush/shutdown can retry after a transient filesystem fault.
                    if command.ack is not None and not command.ack.done():
                        command.ack.set_exception(exc)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._terminal_failure = exc
            raise
        finally:
            try:
                await asyncio.to_thread(state.close)
            except RuntimeError:
                # The interpreter/event loop may already be tearing down.
                state.close()

    def _capture_terminal_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            self._terminal_failure = asyncio.CancelledError()
            return
        try:
            failure = task.exception()
        except BaseException as exc:  # pragma: no cover - defensive Task API guard
            failure = exc
        if failure is not None:
            self._terminal_failure = failure


def _encode_record(value: PipelineEvent | Mapping[str, Any]) -> bytes:
    if isinstance(value, PipelineEvent):
        payload: Mapping[str, Any] = dataclasses.asdict(value)
    elif isinstance(value, Mapping):
        payload = value
    else:  # type checkers protect callers; keep runtime failure actionable.
        raise TypeError("rollout records must be PipelineEvent or Mapping values")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return encoded + b"\n"


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    return str(value)
