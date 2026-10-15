"""2.2.0 Wave 3 — manifest ``memory`` section.

Audit §1-1's last first-class gaps: the memory provider was
host-code-only. It became an optional manifest section — absent →
empty (full back-compat), serialized via ``to_dict``/``from_dict``,
checked by ``validate_manifest`` with the ``memory.*`` issue codes.

The sibling ``subagents`` section was retired in 4.71.0 with sub-agent
orchestration; stored payloads that still carry it load with it dropped
(see ``test_retired_stage_manifests.py``).
"""

from __future__ import annotations

import logging

import pytest

from xgen_agent_runtime import validate_manifest
from xgen_agent_runtime.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    ToolsSnapshot,
)

from tests._fixtures.manifest_entries import required_stage_entries


def _manifest(*, memory=None) -> EnvironmentManifest:
    """Minimal valid manifest plus the Wave 3 section under test."""
    return EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_w3", name="wave3"),
        stages=required_stage_entries(),
        tools=ToolsSnapshot(),
        memory=dict(memory or {}),
    )


def _codes(issues, severity=None):
    return [i.code for i in issues if severity is None or i.severity == severity]


def _by_code(issues, code):
    found = [i for i in issues if i.code == code]
    assert found, f"expected an issue with code {code!r}; got {[i.code for i in issues]}"
    return found[0]


_MEMORY_BLOCK = {"provider": "file", "config": {"root": "/tmp/mem"}}


# ── Serialization round-trip ─────────────────────────────────


class TestSerialization:
    def test_roundtrip_preserves_memory_section(self):
        m = _manifest(memory=_MEMORY_BLOCK)
        reloaded = EnvironmentManifest.from_dict(m.to_dict())
        assert reloaded.memory == _MEMORY_BLOCK
        assert reloaded.to_dict() == m.to_dict()

    def test_to_dict_no_longer_writes_subagents(self):
        assert "subagents" not in _manifest().to_dict()

    def test_absent_sections_default_empty(self):
        """Pre-Wave-3 payloads carry no memory key — full back-compat."""
        data = _manifest().to_dict()
        del data["memory"]
        reloaded = EnvironmentManifest.from_dict(data)
        assert reloaded.memory == {}

    def test_none_payload_values_coerce_to_empty(self):
        data = _manifest().to_dict()
        data["subagents"] = None
        data["memory"] = None
        reloaded = EnvironmentManifest.from_dict(data)
        assert reloaded.memory == {}
        assert not hasattr(reloaded, "subagents")

    def test_sections_are_known_keys_not_warned(self, caplog):
        """The from_dict hygiene pass must not flag the memory section, nor
        the empty ``subagents: []`` every pre-4.71.0 manifest carries."""
        data = _manifest(memory=_MEMORY_BLOCK).to_dict()
        data["subagents"] = []
        with caplog.at_level(logging.WARNING, logger="xgen_agent_runtime.core.environment"):
            EnvironmentManifest.from_dict(data)
        assert "unknown" not in caplog.text
        assert "retired" not in caplog.text

    def test_v1_migration_still_lands_empty_sections(self):
        """Legacy payloads chain through v1→v2→v3 and gain the defaults."""
        legacy = {"version": "1.0", "stages": [], "metadata": {"id": "env_old"}}
        m = EnvironmentManifest.from_dict(legacy)
        assert m.memory == {}


# ── validate_manifest: tools decoys (review B6) ──────────────


class TestValidateToolsDecoys:
    def test_adhoc_data_is_warning(self):
        m = _manifest()
        m.tools = ToolsSnapshot(adhoc=[{"name": "calc", "code": "..."}])
        issue = _by_code(validate_manifest(m), "tools.adhoc_unconsumed")
        assert issue.severity == "warning"
        assert issue.field == "tools.adhoc"
        assert "does not consume" in issue.message

    def test_scope_data_is_warning(self):
        m = _manifest()
        m.tools = ToolsSnapshot(scope={"default": "session"})
        issue = _by_code(validate_manifest(m), "tools.scope_unconsumed")
        assert issue.severity == "warning"
        assert issue.field == "tools.scope"
        assert "does not consume" in issue.message

    def test_consumed_tool_fields_stay_clean(self):
        """built_in / mcp_servers / external ARE consumed — no decoy
        warnings for them, and an empty adhoc/scope stays silent."""
        m = _manifest()
        m.tools = ToolsSnapshot(
            built_in=["*"],
            mcp_servers=[{"name": "bridge", "url": "http://localhost:1"}],
            external=["host_tool"],
        )
        codes = _codes(validate_manifest(m))
        assert "tools.adhoc_unconsumed" not in codes
        assert "tools.scope_unconsumed" not in codes


# ── validate_manifest: memory ────────────────────────────────


class TestValidateMemory:
    def test_clean_file_block_no_issues(self):
        issues = validate_manifest(_manifest(memory=_MEMORY_BLOCK))
        assert [c for c in _codes(issues) if c.startswith("memory.")] == []

    def test_empty_block_no_issues(self):
        issues = validate_manifest(_manifest(memory={}))
        assert [c for c in _codes(issues) if c.startswith("memory.")] == []

    def test_unknown_provider_is_error(self):
        issues = validate_manifest(_manifest(memory={"provider": "redis"}))
        issue = _by_code(issues, "memory.unknown_provider")
        assert issue.severity == "error"
        assert "redis" in issue.message

    def test_missing_provider_is_error(self):
        issues = validate_manifest(_manifest(memory={"config": {"root": "/x"}}))
        assert _by_code(issues, "memory.missing_provider").severity == "error"

    def test_unaccepted_config_key_is_warning(self):
        block = {"provider": "file", "config": {"root": "/x", "shards": 4}}
        issues = validate_manifest(_manifest(memory=block))
        issue = _by_code(issues, "memory.unknown_config_key")
        assert issue.severity == "warning"
        assert "shards" in issue.message

    def test_stray_top_level_key_is_warning(self):
        """A 'root' beside 'provider' is the forgot-to-nest mistake."""
        block = {"provider": "file", "root": "/x", "config": {"root": "/x"}}
        issues = validate_manifest(_manifest(memory=block))
        issue = _by_code(issues, "memory.unknown_key")
        assert issue.severity == "warning"
        assert "root" in issue.message

    @pytest.mark.parametrize("provider", ["ephemeral", "file", "sql", "composite"])
    def test_all_builtin_provider_names_recognised(self, provider):
        issues = validate_manifest(_manifest(memory={"provider": provider}))
        assert "memory.unknown_provider" not in _codes(issues)
