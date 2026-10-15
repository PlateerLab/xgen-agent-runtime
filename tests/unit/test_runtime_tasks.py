"""Task record model + InMemoryRegistry (``xgen_agent_runtime.runtime.tasks``).

These types lived in Stage 13 (``stages/s13_task_registry``) until 4.71.0,
when the stage was retired with sub-agent orchestration. The record model
and registries survive for the background-task runtime (runner, cron,
``/tasks``); the stage/policy tests went with the stage.
"""

from __future__ import annotations

from xgen_agent_runtime.runtime.tasks import (
    InMemoryRegistry,
    TaskRecord,
    TaskRegistry,
    TaskStatus,
)


def test_old_stage_module_is_gone():
    """The relocation is a move, not a copy — no second home to drift."""
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("xgen_agent_runtime.stages.s13_task_registry")


def test_registry_is_a_plain_abc_with_default_identity():
    """No longer a Stage-13 Strategy: a minimal custom backend only needs
    the storage methods; ``name`` / ``description`` have defaults."""

    class _Minimal(TaskRegistry):
        def __init__(self):
            self._rows = {}

        def register(self, record):
            self._rows[record.task_id] = record

        def get(self, task_id):
            return self._rows.get(task_id)

        def update_status(self, task_id, status, *, result=None, error=None):
            rec = self._rows.get(task_id)
            if rec is not None:
                rec.mark(status, result=result, error=error)
            return rec

        def list_all(self):
            return list(self._rows.values())

        def remove(self, task_id):
            return self._rows.pop(task_id, None) is not None

    reg = _Minimal()
    assert reg.name == "_Minimal"
    assert reg.description == ""
    reg.register(TaskRecord(task_id="t"))
    assert reg.by_status() == {"pending": [reg.get("t")]}
    assert InMemoryRegistry().name == "in_memory"


# ── TaskRecord / TaskStatus ────────────────────────────────────────────


class TestTaskRecord:
    def test_default_status_pending(self):
        r = TaskRecord(task_id="t1")
        assert r.status == TaskStatus.PENDING
        assert r.is_terminal is False

    def test_mark_running_sets_started_at(self):
        r = TaskRecord(task_id="t1")
        r.mark(TaskStatus.RUNNING)
        assert r.started_at is not None

    def test_mark_done_sets_completed_at_and_result(self):
        r = TaskRecord(task_id="t1")
        r.mark(TaskStatus.DONE, result={"value": 42})
        assert r.is_terminal
        assert r.completed_at is not None
        assert r.result == {"value": 42}

    def test_mark_failed_sets_error(self):
        r = TaskRecord(task_id="t1")
        r.mark(TaskStatus.FAILED, error="boom")
        assert r.error == "boom"
        assert r.is_terminal

    def test_to_dict_round_trip_keys(self):
        r = TaskRecord(task_id="t1", kind="K", payload={"x": 1})
        r.mark(TaskStatus.DONE, result="ok")
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["kind"] == "K"
        assert d["status"] == "done"
        assert d["payload"] == {"x": 1}
        assert d["completed_at"] is not None


# ── InMemoryRegistry ───────────────────────────────────────────────────


class TestInMemoryRegistry:
    def test_register_and_get(self):
        r = InMemoryRegistry()
        rec = TaskRecord(task_id="t1")
        r.register(rec)
        assert r.get("t1") is rec

    def test_get_unknown_returns_none(self):
        assert InMemoryRegistry().get("ghost") is None

    def test_re_register_replaces(self):
        r = InMemoryRegistry()
        a = TaskRecord(task_id="t1", kind="A")
        b = TaskRecord(task_id="t1", kind="B")
        r.register(a)
        r.register(b)
        assert r.get("t1") is b

    def test_update_status(self):
        r = InMemoryRegistry()
        r.register(TaskRecord(task_id="t1"))
        updated = r.update_status("t1", TaskStatus.DONE, result="ok")
        assert updated.status == TaskStatus.DONE
        assert updated.result == "ok"

    def test_update_unknown_returns_none(self):
        assert InMemoryRegistry().update_status("ghost", TaskStatus.DONE) is None

    def test_remove(self):
        r = InMemoryRegistry()
        r.register(TaskRecord(task_id="t1"))
        assert r.remove("t1") is True
        assert r.remove("t1") is False

    def test_by_status_groups(self):
        r = InMemoryRegistry()
        r.register(TaskRecord(task_id="t1", status=TaskStatus.PENDING))
        r.register(TaskRecord(task_id="t2", status=TaskStatus.DONE))
        r.register(TaskRecord(task_id="t3", status=TaskStatus.PENDING))
        out = r.by_status()
        assert sorted(out.keys()) == ["done", "pending"]
        assert {t.task_id for t in out["pending"]} == {"t1", "t3"}
