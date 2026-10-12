"""Built-in control slash commands tests (PR-A.2.3)."""

from __future__ import annotations


import pytest

from xgen_agent_runtime.slash_commands import (
    SlashCategory,
    SlashCommandRegistry,
    SlashContext,
)
from xgen_agent_runtime.slash_commands.built_in import (
    CancelCommand,
    CompactCommand,
    ConfigCommand,
    ModelCommand,
    PresetInfoCommand,
    TasksCommand,
    install_built_in_commands,
)
from xgen_agent_runtime.runtime.tasks import (
    InMemoryRegistry,
    TaskRecord,
    TaskStatus,
)


class _FakePipeline:
    def __init__(self, **kwargs):
        self._strategies: dict = kwargs.pop("strategies", {})
        self.manifest = kwargs.pop("manifest", None) or type(
            "M", (), {"preset_name": "worker_adaptive", "model": "claude-haiku",
                      "preset_metadata": {"role": "general"}}
        )()
        self.stages = kwargs.pop("stages", [])
        self._stop_called = False
        self._set_model_value: str | None = None

    def get_strategy(self, name):
        return self._strategies.get(name)

    async def stop(self):
        self._stop_called = True

    def set_model(self, m):
        self._set_model_value = m


# ── /tasks ───────────────────────────────────────────────────────────


class TestTasks:
    @pytest.mark.asyncio
    async def test_lists_tasks(self):
        registry = InMemoryRegistry()
        registry.register(TaskRecord(task_id="t1", kind="K"))
        running = TaskRecord(task_id="t2", kind="K")
        running.mark(TaskStatus.RUNNING)
        registry.register(running)
        ctx = SlashContext(extras={"task_registry": registry})
        result = await TasksCommand().execute([], ctx)
        assert "t1" in result.content
        assert "t2" in result.content

    @pytest.mark.asyncio
    async def test_filter_by_status(self):
        registry = InMemoryRegistry()
        registry.register(TaskRecord(task_id="pending"))
        running = TaskRecord(task_id="r")
        running.mark(TaskStatus.RUNNING)
        registry.register(running)
        ctx = SlashContext(extras={"task_registry": registry})
        result = await TasksCommand().execute(["running"], ctx)
        assert "r" in result.content
        assert "pending" not in result.content

    @pytest.mark.asyncio
    async def test_no_registry(self):
        result = await TasksCommand().execute([], SlashContext())
        assert result.success is False

    @pytest.mark.asyncio
    async def test_invalid_status_arg(self):
        ctx = SlashContext(extras={"task_registry": InMemoryRegistry()})
        result = await TasksCommand().execute(["bogus"], ctx)
        assert result.success is False
        assert "Unknown status" in result.content

    @pytest.mark.asyncio
    async def test_empty_with_filter(self):
        registry = InMemoryRegistry()
        ctx = SlashContext(extras={"task_registry": registry})
        result = await TasksCommand().execute(["running"], ctx)
        assert "No running tasks" in result.content


# ── /cancel ──────────────────────────────────────────────────────────


class TestCancel:
    @pytest.mark.asyncio
    async def test_calls_stop(self):
        pipeline = _FakePipeline()
        ctx = SlashContext(pipeline=pipeline)
        result = await CancelCommand().execute([], ctx)
        assert result.success is True
        assert pipeline._stop_called is True

    @pytest.mark.asyncio
    async def test_no_pipeline(self):
        result = await CancelCommand().execute([], SlashContext())
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_stop_method(self):
        class _Bare:
            pass
        ctx = SlashContext(pipeline=_Bare())
        result = await CancelCommand().execute([], ctx)
        assert result.success is False


# ── /compact ─────────────────────────────────────────────────────────


class TestCompact:
    """4.89.0: ``/compact`` queues a manual compaction on the live state; Stage 2 runs it at the
    next pass even when the context is far from the window."""

    @pytest.mark.asyncio
    async def test_queues_a_manual_request_that_stage_2_runs(self):
        from xgen_agent_runtime.core.shared_keys import SharedKeys
        from xgen_agent_runtime.core.state import PipelineState
        from xgen_agent_runtime.stages.s02_context import ContextStage
        from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
            TruncateCompactor,
        )

        state = PipelineState()
        state.messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(12)
        ]
        result = await CompactCommand().execute([], SlashContext(extras={"state": state}))
        assert result.success is True
        assert state.shared[SharedKeys.CONTEXT_COMPACTION_REQUEST]["source"] == "manual"

        await ContextStage(compactor=TruncateCompactor(keep_last=4)).execute(None, state)

        assert len(state.messages) < 12
        compacted = [e for e in state.events if e["type"] == "context.compacted"]
        assert compacted and compacted[-1]["data"]["trigger"] == "manual"
        assert SharedKeys.CONTEXT_COMPACTION_REQUEST not in state.shared

    @pytest.mark.asyncio
    async def test_without_a_state_reports_failure(self):
        result = await CompactCommand().execute([], SlashContext(pipeline=_FakePipeline()))
        assert result.success is False


# ── /config ──────────────────────────────────────────────────────────


class _FakeStage:
    def __init__(self, name: str, slots: dict):
        self.name = name
        self._slots = slots

    def get_strategy_slots(self):
        return self._slots


class TestConfig:
    @pytest.mark.asyncio
    async def test_renders_slots(self):
        slot_obj = type("S", (), {"strategy": object()})()
        stage = _FakeStage("api", {"completion": slot_obj})
        ctx = SlashContext(pipeline=_FakePipeline(stages=[stage]))
        result = await ConfigCommand().execute([], ctx)
        assert "api" in result.content
        assert "completion" in result.content

    @pytest.mark.asyncio
    async def test_no_stages(self):
        ctx = SlashContext(pipeline=_FakePipeline(stages=[]))
        result = await ConfigCommand().execute([], ctx)
        assert result.success is False


# ── /model ───────────────────────────────────────────────────────────


class TestModel:
    @pytest.mark.asyncio
    async def test_no_args_shows_current(self):
        ctx = SlashContext(pipeline=_FakePipeline())
        result = await ModelCommand().execute([], ctx)
        assert "claude-haiku" in result.content

    @pytest.mark.asyncio
    async def test_sets_new_model(self):
        pipeline = _FakePipeline()
        ctx = SlashContext(pipeline=pipeline)
        result = await ModelCommand().execute(["claude-sonnet-4-6"], ctx)
        assert result.success is True
        assert pipeline._set_model_value == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_rejects_unknown_prefix(self):
        ctx = SlashContext(pipeline=_FakePipeline())
        result = await ModelCommand().execute(["gpt-4"], ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_custom_allow_prefix(self):
        pipeline = _FakePipeline()
        ctx = SlashContext(pipeline=pipeline, extras={"model_allow": ["gpt-"]})
        result = await ModelCommand().execute(["gpt-4"], ctx)
        assert result.success is True


# ── /preset-info ─────────────────────────────────────────────────────


class TestPresetInfo:
    @pytest.mark.asyncio
    async def test_renders_preset_metadata(self):
        ctx = SlashContext(pipeline=_FakePipeline())
        result = await PresetInfoCommand().execute([], ctx)
        assert "worker_adaptive" in result.content
        assert "role" in result.content
        assert "general" in result.content

    @pytest.mark.asyncio
    async def test_alias_resolves(self):
        from xgen_agent_runtime.slash_commands.registry import (
            get_default_registry,
        )
        # The default singleton was populated on import.
        cmd = get_default_registry().resolve("preset_info")
        assert cmd is not None


# ── install_built_in_commands ────────────────────────────────────────


class TestInstall:
    def test_installs_twelve(self):
        reg = SlashCommandRegistry()
        count = install_built_in_commands(reg)
        assert count == 12
        for name in (
            "cost", "status", "help", "memory", "context",
            "config", "preset-info", "tasks",
            "clear", "cancel", "compact", "model",
        ):
            assert reg.resolve(name) is not None

    def test_categories_split_correctly(self):
        reg = SlashCommandRegistry()
        install_built_in_commands(reg)
        intros = {c.name for c in reg.list_by_category(SlashCategory.INTROSPECTION)}
        controls = {c.name for c in reg.list_by_category(SlashCategory.CONTROL)}
        assert "cost" in intros
        assert "status" in intros
        assert "tasks" in intros
        assert "clear" in controls
        assert "cancel" in controls
        assert "model" in controls
