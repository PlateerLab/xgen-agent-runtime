"""4.71.0 — Stage 12 (agent) / Stage 13 (task_registry) retired.

Sub-agent orchestration was removed for good. Two promises survive it:

* **Stored manifests keep loading.** An environment saved before the
  removal still lists ``agent`` / ``task_registry`` entries, a
  ``subagents`` section, and maybe the Stage 14 ``agent_evaluation``
  evaluator. Those parts are dropped (or rewritten) with a warning —
  never an error, never a strict-build failure.
* **The numbering is stable.** Orders 12 / 13 stay reserved; 14–21 keep
  their numbers and a 1..21 walk (``describe()``) keeps its shape.
"""

from __future__ import annotations

import copy
import logging

import pytest

from xgen_agent_runtime import (
    RETIRED_STAGE_ORDERS,
    EnvironmentManifest,
    Pipeline,
    PipelineBuilder,
    build_manifest,
    validate_manifest,
)
from xgen_agent_runtime.core.artifact import STAGE_MODULES, create_stage, is_retired_stage

_ENV_LOGGER = "xgen_agent_runtime.core.environment"

# The two entries every pre-4.71.0 ``build_manifest("worker_adaptive")``
# emitted (verbatim from the 4.70.0 vendored layout fixture).
_OLD_AGENT_ENTRY = {
    "order": 12,
    "name": "agent",
    "active": True,
    "artifact": "default",
    "strategies": {"orchestrator": "subagent_type"},
    "strategy_configs": {},
    "config": {"max_delegations": 4},
    "tool_binding": None,
    "model_override": None,
    "chain_order": {},
}
_OLD_TASK_REGISTRY_ENTRY = {
    "order": 13,
    "name": "task_registry",
    "active": True,
    "artifact": "default",
    "strategies": {"registry": "in_memory", "policy": "fire_and_forget"},
    "strategy_configs": {},
    "config": {},
    "tool_binding": None,
    "model_override": None,
    "chain_order": {},
}
_OLD_SUBAGENTS = [
    {"agent_type": "researcher", "description": "Looks things up", "provider": None},
]


def _old_manifest_dict() -> dict:
    """A 4.70.0-shaped stored environment: retired stages + subagents."""
    data = build_manifest("worker_adaptive", provider="anthropic").to_dict()
    data["stages"] = sorted(
        data["stages"] + [copy.deepcopy(_OLD_AGENT_ENTRY), copy.deepcopy(_OLD_TASK_REGISTRY_ENTRY)],
        key=lambda e: e["order"],
    )
    data["subagents"] = copy.deepcopy(_OLD_SUBAGENTS)
    return data


# ── Stored manifests keep loading ────────────────────────────────────


class TestOldManifestLoads:
    def test_from_dict_drops_retired_entries_with_one_warning(self, caplog):
        data = _old_manifest_dict()
        before = copy.deepcopy(data)
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            manifest = EnvironmentManifest.from_dict(data)
        orders = [s["order"] for s in manifest.stages]
        assert 12 not in orders and 13 not in orders
        assert orders == [o for o in range(1, 22) if o not in (12, 13)]
        assert not hasattr(manifest, "subagents")
        assert "subagents" not in manifest.to_dict()
        retired = [r for r in caplog.records if "retired" in r.getMessage()]
        assert len(retired) == 1
        msg = retired[0].getMessage()
        assert "agent (order 12)" in msg and "task_registry (order 13)" in msg
        assert "1 'subagents'" in msg
        # from_dict never edits the caller's payload.
        assert data == before

    def test_from_dict_result_builds_strict(self):
        manifest = EnvironmentManifest.from_dict(_old_manifest_dict())
        pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=True)
        assert pipeline.get_stage(12) is None
        assert pipeline.get_stage(13) is None
        assert pipeline.get_stage(14) is not None

    def test_in_memory_manifest_with_retired_entries_builds_strict(self, caplog):
        """A manifest object that never went through from_dict (built in
        memory) is cleaned by from_manifest itself — without mutating it."""
        manifest = EnvironmentManifest.from_dict(_old_manifest_dict())
        manifest.stages = sorted(
            manifest.stages + [dict(_OLD_AGENT_ENTRY), dict(_OLD_TASK_REGISTRY_ENTRY)],
            key=lambda e: e["order"],
        )
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=True)
        assert pipeline.get_stage(12) is None and pipeline.get_stage(13) is None
        assert any("Pipeline.from_manifest" in r.getMessage() for r in caplog.records)
        # The caller's object still carries what it declared.
        assert {12, 13} <= {s["order"] for s in manifest.stages}

    def test_empty_subagents_key_is_silent(self, caplog):
        """Every pre-4.71.0 to_dict wrote ``"subagents": []`` — no noise."""
        data = build_manifest("vtuber", provider="anthropic").to_dict()
        data["subagents"] = []
        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            EnvironmentManifest.from_dict(data)
        assert not [r for r in caplog.records if "retired" in r.getMessage()]
        assert "unknown" not in caplog.text

    def test_module_name_entries_are_retired_too(self):
        data = build_manifest("vtuber", provider="anthropic").to_dict()
        data["stages"].append({"order": 12, "name": "s12_agent", "active": False})
        data["stages"].append({"order": 13, "name": "", "active": False})
        manifest = EnvironmentManifest.from_dict(data)
        assert {12, 13}.isdisjoint({s["order"] for s in manifest.stages})

    def test_v2_migration_no_longer_pads_task_registry(self):
        legacy = {"version": "2.0", "stages": [], "metadata": {"id": "env_v2"}}
        manifest = EnvironmentManifest.from_dict(legacy)
        assert sorted(s["order"] for s in manifest.stages) == [11, 15, 19, 20]

    def test_validate_manifest_reports_retired_entry_as_warning(self):
        manifest = EnvironmentManifest.from_dict(_old_manifest_dict())
        manifest.stages = manifest.stages + [dict(_OLD_AGENT_ENTRY)]
        issues = [i for i in validate_manifest(manifest) if i.code == "stage.retired"]
        assert len(issues) == 1
        assert issues[0].severity == "warning" and issues[0].stage_order == 12
        assert not [i for i in validate_manifest(manifest) if i.severity == "error"]


class TestRetiredEvaluator:
    """s14 ``agent_evaluation`` only scored Stage 12 output — gone with it."""

    def _with_s14(self, mutate) -> dict:
        data = build_manifest("worker_adaptive", provider="anthropic").to_dict()
        for entry in data["stages"]:
            if entry["name"] == "evaluate":
                mutate(entry)
        return data

    def test_slot_selection_falls_back_to_signal_based(self, caplog):
        def mutate(entry):
            entry["strategies"]["strategy"] = "agent_evaluation"

        with caplog.at_level(logging.WARNING, logger=_ENV_LOGGER):
            manifest = EnvironmentManifest.from_dict(self._with_s14(mutate))
        s14 = next(s for s in manifest.stages if s["name"] == "evaluate")
        assert s14["strategies"]["strategy"] == "signal_based"
        assert "strategy" not in s14["strategy_configs"]
        assert "agent_evaluation" in caplog.text
        Pipeline.from_manifest(manifest, api_key="sk-test", strict=True)

    def test_chain_member_is_dropped(self):
        def mutate(entry):
            entry["strategy_configs"]["strategy"]["evaluators"].append("agent_evaluation")

        manifest = EnvironmentManifest.from_dict(self._with_s14(mutate))
        s14 = next(s for s in manifest.stages if s["name"] == "evaluate")
        assert s14["strategy_configs"]["strategy"]["evaluators"] == [
            "binary_classify",
            "signal_based",
        ]
        Pipeline.from_manifest(manifest, api_key="sk-test", strict=True)


# ── The numbering is stable ──────────────────────────────────────────


class TestRetiredSlots:
    def test_constant_names_the_two_slots(self):
        assert RETIRED_STAGE_ORDERS == {12: "agent", 13: "task_registry"}

    def test_stage_modules_skip_but_do_not_renumber(self):
        assert 12 not in STAGE_MODULES and 13 not in STAGE_MODULES
        assert STAGE_MODULES[14] == "s14_evaluate"
        assert STAGE_MODULES[21] == "s21_yield"

    @pytest.mark.parametrize("ident", [12, "12", "agent", "task_registry", "s13_task_registry"])
    def test_create_stage_names_the_retirement(self, ident):
        assert is_retired_stage(ident)
        with pytest.raises(ValueError, match="retired"):
            create_stage(ident)

    def test_live_names_are_not_retired(self):
        for ident in (11, 14, "evaluate", "tool", "s14_evaluate"):
            assert not is_retired_stage(ident)

    def test_describe_still_walks_all_21_orders(self):
        pipeline = Pipeline.from_manifest(
            build_manifest("worker_adaptive", provider="anthropic"), api_key="sk-test"
        )
        rows = pipeline.describe()
        assert [r.order for r in rows] == list(range(1, 22))
        by_order = {r.order: r for r in rows}
        assert by_order[12].name == "agent" and not by_order[12].is_active
        assert by_order[13].name == "task_registry" and not by_order[13].is_active

    def test_builder_has_no_retired_hooks(self):
        builder = PipelineBuilder("x")
        assert not hasattr(builder, "with_agent")
        assert not hasattr(builder, "with_task_registry")
