"""Framework runtime layer — background workers, schedulers, lifecycles.

Modules here run **outside** the synchronous pipeline path. They are
service-instantiated at startup (FastAPI lifespan, CLI bootstrap,
SDK bootstrap) and torn down at shutdown.

Public surface:

* :class:`BackgroundTaskExecutor` — ABC for one type of background
  task. Yields output bytes; raises on failure.
* :class:`LocalBashExecutor` — runs a shell command via subprocess.
* :class:`BackgroundTaskRunner` — owns the :class:`asyncio.Task`
  futures, talks to a :class:`TaskRegistry` for state + output
  persistence, and supports submit / stop / shutdown.
* :mod:`~xgen_agent_runtime.runtime.tasks` — the task record model
  (:class:`TaskRecord` / :class:`TaskStatus` / :class:`TaskFilter`)
  and the :class:`TaskRegistry` backends (:class:`InMemoryRegistry`,
  :class:`FileBackedRegistry`). Relocated from the retired Stage 13
  in 4.71.0.
"""

from xgen_agent_runtime.runtime.task_executors import (
    BackgroundTaskExecutor,
    LocalBashExecutor,
)
from xgen_agent_runtime.runtime.task_runner import BackgroundTaskRunner
from xgen_agent_runtime.runtime.tasks import (
    FileBackedRegistry,
    InMemoryRegistry,
    TaskFilter,
    TaskRecord,
    TaskRegistry,
    TaskStatus,
)

__all__ = [
    "BackgroundTaskExecutor",
    "BackgroundTaskRunner",
    "FileBackedRegistry",
    "InMemoryRegistry",
    "LocalBashExecutor",
    "TaskFilter",
    "TaskRecord",
    "TaskRegistry",
    "TaskStatus",
]
