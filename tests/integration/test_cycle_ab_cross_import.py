"""Cycle A+B cross-import audit (PR-C.1).

Verifies every public surface from new-executor-uplift cycles A+B
is reachable through its top-level package import. Catches
``__init__.py`` re-export drift before downstream Geny-side wiring
chases a broken import.

Treat any failure here as a regression: the contract with adopters
(Geny / future hosts) is "import the package, get the symbol".
"""

from __future__ import annotations

import importlib
from typing import Tuple

import pytest


# ── Cycle A executor side: 1.1.0 surface ──────────────────────────────


CYCLE_A_EXPORTS: Tuple[Tuple[str, str], ...] = (
    # P0.1 task lifecycle
    ("xgen_agent_runtime.stages.s13_task_registry", "TaskRegistry"),
    ("xgen_agent_runtime.stages.s13_task_registry", "TaskRecord"),
    ("xgen_agent_runtime.stages.s13_task_registry", "TaskFilter"),
    ("xgen_agent_runtime.stages.s13_task_registry", "TaskStatus"),
    ("xgen_agent_runtime.stages.s13_task_registry", "InMemoryRegistry"),
    ("xgen_agent_runtime.stages.s13_task_registry", "FileBackedRegistry"),
    ("xgen_agent_runtime.runtime", "BackgroundTaskRunner"),
    ("xgen_agent_runtime.runtime", "BackgroundTaskExecutor"),
    ("xgen_agent_runtime.runtime", "LocalBashExecutor"),
    ("xgen_agent_runtime.runtime", "LocalAgentExecutor"),
    ("xgen_agent_runtime.tools.built_in", "AgentTool"),
    ("xgen_agent_runtime.tools.built_in", "TaskCreateTool"),
    ("xgen_agent_runtime.tools.built_in", "TaskGetTool"),
    ("xgen_agent_runtime.tools.built_in", "TaskListTool"),
    ("xgen_agent_runtime.tools.built_in", "TaskUpdateTool"),
    ("xgen_agent_runtime.tools.built_in", "TaskOutputTool"),
    ("xgen_agent_runtime.tools.built_in", "TaskStopTool"),
    # P0.2 slash commands
    ("xgen_agent_runtime.slash_commands", "SlashCommand"),
    ("xgen_agent_runtime.slash_commands", "SlashCommandRegistry"),
    ("xgen_agent_runtime.slash_commands", "SlashContext"),
    ("xgen_agent_runtime.slash_commands", "SlashResult"),
    ("xgen_agent_runtime.slash_commands", "SlashCategory"),
    ("xgen_agent_runtime.slash_commands", "parse_slash"),
    ("xgen_agent_runtime.slash_commands", "get_default_registry"),
    ("xgen_agent_runtime.slash_commands.md_template", "MdTemplateCommand"),
    ("xgen_agent_runtime.slash_commands.md_template", "load_md_command"),
    # P0.3 tool catalog
    ("xgen_agent_runtime.tools.built_in", "AskUserQuestionTool"),
    ("xgen_agent_runtime.tools.built_in", "PushNotificationTool"),
    ("xgen_agent_runtime.tools.built_in", "MCPTool"),
    ("xgen_agent_runtime.tools.built_in", "ListMcpResourcesTool"),
    ("xgen_agent_runtime.tools.built_in", "ReadMcpResourceTool"),
    ("xgen_agent_runtime.tools.built_in", "McpAuthTool"),
    ("xgen_agent_runtime.tools.built_in", "EnterWorktreeTool"),
    ("xgen_agent_runtime.tools.built_in", "ExitWorktreeTool"),
    ("xgen_agent_runtime.tools.built_in", "LSPTool"),
    ("xgen_agent_runtime.tools.built_in", "REPLTool"),
    ("xgen_agent_runtime.tools.built_in", "BriefTool"),
    ("xgen_agent_runtime.tools.built_in", "ConfigTool"),
    ("xgen_agent_runtime.tools.built_in", "MonitorTool"),
    ("xgen_agent_runtime.tools.built_in", "SendUserFileTool"),
    ("xgen_agent_runtime.tools.built_in", "SendMessageTool"),
    ("xgen_agent_runtime.notifications", "NotificationEndpoint"),
    ("xgen_agent_runtime.notifications", "NotificationEndpointRegistry"),
    ("xgen_agent_runtime.channels", "UserFileChannel"),
    ("xgen_agent_runtime.channels", "SendMessageChannel"),
    ("xgen_agent_runtime.channels", "SendMessageChannelRegistry"),
    ("xgen_agent_runtime.channels", "StdoutSendMessageChannel"),
    # P0.4 cron
    ("xgen_agent_runtime.cron", "CronJob"),
    ("xgen_agent_runtime.cron", "CronJobStatus"),
    ("xgen_agent_runtime.cron", "CronJobStore"),
    ("xgen_agent_runtime.cron", "InMemoryCronJobStore"),
    ("xgen_agent_runtime.cron", "FileBackedCronJobStore"),
    ("xgen_agent_runtime.cron", "CronRunner"),
    ("xgen_agent_runtime.tools.built_in", "CronCreateTool"),
    ("xgen_agent_runtime.tools.built_in", "CronDeleteTool"),
    ("xgen_agent_runtime.tools.built_in", "CronListTool"),
)


# ── Cycle B executor side: 1.2.0 surface ──────────────────────────────


CYCLE_B_EXPORTS: Tuple[Tuple[str, str], ...] = (
    # P1.1 in-process hooks (method on existing class — verified separately)
    # P1.2 auto-compaction
    ("xgen_agent_runtime.stages.s19_summarize", "FrequencyPolicy"),
    ("xgen_agent_runtime.stages.s19_summarize", "NeverPolicy"),
    ("xgen_agent_runtime.stages.s19_summarize", "EveryNTurnsPolicy"),
    ("xgen_agent_runtime.stages.s19_summarize", "OnContextFillPolicy"),
    ("xgen_agent_runtime.stages.s19_summarize", "FrequencyAwareSummarizerProxy"),
    # P1.3 settings
    ("xgen_agent_runtime.settings", "SettingsLoader"),
    ("xgen_agent_runtime.settings", "get_default_loader"),
    ("xgen_agent_runtime.settings", "register_section"),
    # P1.4 skill schema (extra fields on existing class — verified separately)
    # P1.5 permission modes (enum members — verified separately)
)


@pytest.mark.parametrize("module,name", CYCLE_A_EXPORTS + CYCLE_B_EXPORTS)
def test_public_surface_importable(module: str, name: str):
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module} missing public name {name!r}"


# ── Method / attr surface checks ──────────────────────────────────────


def test_hook_runner_has_register_in_process():
    """PR-B.1.1 — register_in_process must be on HookRunner instance."""
    from xgen_agent_runtime.hooks.runner import HookRunner
    assert hasattr(HookRunner, "register_in_process")


def test_hook_runner_has_list_in_process_handlers():
    from xgen_agent_runtime.hooks.runner import HookRunner
    assert hasattr(HookRunner, "list_in_process_handlers")


def test_skill_metadata_has_richer_fields():
    """PR-B.4.1 — category / effort / examples on SkillMetadata."""
    from dataclasses import fields
    from xgen_agent_runtime.skills.types import SkillMetadata
    field_names = {f.name for f in fields(SkillMetadata)}
    for new_field in ("category", "effort", "examples"):
        assert new_field in field_names, f"SkillMetadata missing {new_field!r}"


def test_permission_mode_has_new_modes():
    """PR-B.5.1 — ACCEPT_EDITS / DONT_ASK enum members."""
    from xgen_agent_runtime.permission.types import PermissionMode
    assert PermissionMode("acceptEdits") is PermissionMode.ACCEPT_EDITS
    assert PermissionMode("dontAsk") is PermissionMode.DONT_ASK


def test_permission_edit_tools_set_exported():
    from xgen_agent_runtime.permission.types import EDIT_TOOLS
    assert "Write" in EDIT_TOOLS
    assert "Edit" in EDIT_TOOLS
    assert "NotebookEdit" in EDIT_TOOLS


def test_built_in_tool_classes_includes_all_new():
    """The registry mapping must enumerate every tool added in 1.1.0."""
    from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_CLASSES
    new_tools = (
        "Agent", "AskUserQuestion", "PushNotification",
        "MCP", "ListMcpResources", "ReadMcpResource", "McpAuth",
        "EnterWorktree", "ExitWorktree",
        "LSP", "REPL", "Brief",
        "Config", "Monitor", "SendUserFile", "SendMessage",
        "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskOutput", "TaskStop",
        "CronCreate", "CronDelete", "CronList",
    )
    missing = [t for t in new_tools if t not in BUILT_IN_TOOL_CLASSES]
    assert not missing, f"BUILT_IN_TOOL_CLASSES missing: {missing}"


def test_built_in_tool_features_groups_present():
    from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_FEATURES
    expected_groups = {
        "agent", "tasks", "interaction", "notification",
        "mcp", "worktree", "dev", "operator", "messaging", "cron",
    }
    missing = expected_groups - set(BUILT_IN_TOOL_FEATURES)
    assert not missing, f"BUILT_IN_TOOL_FEATURES missing groups: {missing}"
