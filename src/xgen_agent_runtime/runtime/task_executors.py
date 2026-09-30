"""Background task executors (PR-A.1.3).

A :class:`BackgroundTaskExecutor` knows how to run **one kind** of
task. It is invoked by :class:`BackgroundTaskRunner` once per
submitted :class:`TaskRecord`. The runner owns scheduling, lifecycle,
and output persistence; the executor owns the actual work.

Built-in executors:

* :class:`LocalBashExecutor` — runs ``payload['command']`` via shell.

(``LocalAgentExecutor`` — a sub-agent run as a background task — was
removed in 4.71.0 together with sub-agent orchestration.)

Hosts that need additional task kinds implement
:class:`BackgroundTaskExecutor` and pass their executor map to
:class:`BackgroundTaskRunner`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator

from xgen_agent_runtime.runtime.tasks import TaskRecord


class BackgroundTaskExecutor(ABC):
    """One executor handles one task ``kind``.

    Implementations yield output bytes as work progresses. The runner
    persists each chunk via :meth:`TaskRegistry.append_output`. Raise
    on failure — the runner catches the exception and marks the task
    ``FAILED`` with ``str(exc)``.
    """

    @abstractmethod
    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        """Execute the task. Yield output chunks as they become
        available. The runner appends each chunk to the registry's
        per-task output buffer.
        """
        ...


class LocalBashExecutor(BackgroundTaskExecutor):
    """Runs ``record.payload['command']`` via the shell.

    Uses ``asyncio.create_subprocess_shell`` so chained pipes work.
    Stdout + stderr are merged so the user sees errors interleaved
    with normal output. Non-zero exit raises ``RuntimeError``.
    """

    def __init__(self, *, max_output_bytes: int = 64 * 1024 * 1024) -> None:
        self._max_output_bytes = max_output_bytes

    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        command = record.payload.get("command")
        if not command:
            raise ValueError("local_bash task requires payload['command']")
        # SCRUBBED env — a background task has no ToolContext/sandbox handle, so
        # it can't route to the agent's session, but it must NOT inherit the
        # backend's full secret-bearing os.environ. Any per-task env travels in
        # the payload (never platform secrets).
        from xgen_agent_runtime.tools.built_in.bash_tool import _scrubbed_env

        payload_env = record.payload.get("env")
        proc = await asyncio.create_subprocess_shell(
            command,
            env=_scrubbed_env(payload_env if isinstance(payload_env, dict) else None),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        emitted = 0
        try:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                emitted += len(chunk)
                yield chunk
                if emitted >= self._max_output_bytes:
                    proc.kill()
                    raise RuntimeError(
                        f"local_bash exceeded max_output_bytes={self._max_output_bytes}"
                    )
        finally:
            # Make sure we never leave a zombie even if the consumer
            # cancelled or raised.
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        rc = proc.returncode if proc.returncode is not None else await proc.wait()
        if rc != 0:
            raise RuntimeError(f"local_bash exited rc={rc}")


__all__ = [
    "BackgroundTaskExecutor",
    "LocalBashExecutor",
]
