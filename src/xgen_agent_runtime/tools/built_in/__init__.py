"""Built-in tools for file system operations, shell execution, and search.

These tools provide the core capabilities that an agent needs to interact
with the local environment — reading/writing files, running commands,
and searching codebases. They ship with the executor so every consumer
gets a working tool surface without having to reimplement filesystem
access against the :class:`~xgen_agent_runtime.tools.base.Tool` ABC.

:data:`BUILT_IN_TOOL_CLASSES` maps each tool's registry name to its
class; it is the single source of truth consumed by
``Pipeline.from_manifest_async`` when resolving
``manifest.tools.built_in`` entries.

:data:`BUILT_IN_TOOL_FEATURES` groups those same tools by capability
family (``filesystem`` / ``shell`` / ``web`` / ``workflow``). Use
:func:`get_builtin_tools` with the ``features=`` kwarg to select a
subset without hardcoding tool names.
"""

from typing import Dict, Iterable, List, Optional, Tuple, Type

from xgen_agent_runtime.tools.base import Tool
from xgen_agent_runtime.tools.built_in.agent_tool import AgentTool
from xgen_agent_runtime.tools.built_in.delegation_guide_tool import (
    DELEGATION_FAMILY,
    DelegationGuideTool,
)
from xgen_agent_runtime.tools.built_in.subagent_tools import (
    SubAgentAssignTool,
    SubAgentInboxReadTool,
    SubAgentListTool,
    SubAgentSpawnTool,
    SubAgentStopTool,
)
from xgen_agent_runtime.tools.built_in.ask_user_question_tool import (
    AskUserQuestionTool,
    QuestionCancelled,
)
from xgen_agent_runtime.tools.built_in.mcp_wrapper_tools import (
    ListMcpResourcesTool,
    MCPTool,
    McpAuthTool,
    ReadMcpResourceTool,
)
from xgen_agent_runtime.tools.built_in.push_notification_tool import (
    PushNotificationTool,
)
from xgen_agent_runtime.tools.built_in.dev_tools import (
    BriefTool,
    LSPTool,
    REPLTool,
)
from xgen_agent_runtime.tools.built_in.operator_tools import (
    ConfigTool,
    MonitorTool,
    SendUserFileTool,
)
from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
from xgen_agent_runtime.tools.built_in.workspace_tools import (
    SandboxFetchTool,
    SandboxInfoTool,
    SandboxPutTool,
    WorkspaceInfoTool,
)
from xgen_agent_runtime.tools.built_in.cron_tools import (
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
)
from xgen_agent_runtime.tools.built_in.send_message_tool import SendMessageTool
from xgen_agent_runtime.tools.built_in.worktree_tools import (
    EnterWorktreeTool,
    ExitWorktreeTool,
)
from xgen_agent_runtime.tools.built_in.task_tools import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
)
from xgen_agent_runtime.tools.built_in.write_tool import WriteTool
from xgen_agent_runtime.tools.built_in.edit_tool import EditTool
from xgen_agent_runtime.tools.built_in.bash_tool import BashTool
from xgen_agent_runtime.tools.built_in.glob_tool import GlobTool
from xgen_agent_runtime.tools.built_in.grep_tool import GrepTool
from xgen_agent_runtime.tools.built_in.web_fetch_tool import WebFetchTool
from xgen_agent_runtime.tools.built_in.web_search_tool import WebSearchTool
from xgen_agent_runtime.tools.built_in.todo_write_tool import TodoWriteTool
from xgen_agent_runtime.tools.built_in.notebook_edit_tool import NotebookEditTool
from xgen_agent_runtime.tools.built_in.tool_search_tool import ToolSearchTool
from xgen_agent_runtime.tools.built_in.plan_mode_tools import (
    EnterPlanModeTool,
    ExitPlanModeTool,
)
from xgen_agent_runtime.tools.built_in.env_tools import EnvTool

# Google Workspace — native Gmail/Calendar/Drive/Tasks tools. Read the OAuth token
# from ``ctx.extras['google']``; gated via required_config_keys → hidden until the
# host marks ``feature:google_connected`` satisfied.
from xgen_agent_runtime.tools.built_in.google_tools import GOOGLE_TOOL_CLASSES

# Atlassian — native Jira/Confluence tools. Read credentials from
# ``ctx.extras['atlassian']``; gated via required_config_keys → hidden until
# the host marks ``feature:atlassian_connected`` satisfied.
from xgen_agent_runtime.tools.built_in.atlassian_tools import ATLASSIAN_TOOL_CLASSES

# Browser — AI-native web exploration on the an-web engine (semantic snapshots,
# per-session tabs, embedded V8; no Chromium). an-web itself imports lazily —
# 'pip install xgen-agent-runtime[browser]' (Python >= 3.12).
from xgen_agent_runtime.tools.built_in.browser_tools import (
    BROWSER_FAMILY,
    BROWSER_TOOL_CLASSES,
    BrowserActTool,
    BrowserBackTool,
    BrowserCloseTool,
    BrowserEvalTool,
    BrowserExtractTool,
    BrowserNavigateTool,
    BrowserSnapshotTool,
)

# Doc — office documents (docx/xlsx/pptx) on the edit2docs engine: addressable
# outlines, deterministic edits, generation. Lazy import — 'pip install
# xgen-agent-runtime[docs]'.
from xgen_agent_runtime.tools.built_in.doc_tools import (
    DOC_FAMILY,
    DOC_TOOL_CLASSES,
    DocAnalyzeTool,
    DocApplyEditsTool,
    DocArrangeTool,
    DocBuildTool,
    DocEditTool,
    DocGenerateTool,
    DocGuideTool,
    DocRenderTool,
    DocXmlEditTool,
    DocXmlReadTool,
)

# NOT in BUILT_IN_TOOL_CLASSES: SandboxExecTool is instantiated per Sandbox Tool
# Pack (with a spec + a live SandboxHandle), not activated by a manifest name.
from xgen_agent_runtime.tools.built_in.sandbox_exec_tool import SandboxExecTool

# SSH — run commands / move files on the session's pre-configured servers.
# Gated on feature:ssh_enabled; degrades to an install-hint error when the
# optional ``asyncssh`` dependency is absent.
from xgen_agent_runtime.tools.built_in.audio_tools import AUDIO_TOOL_CLASSES
from xgen_agent_runtime.tools.built_in.ssh_tools import (
    SSH_FAMILY,
    SSH_TOOL_CLASSES,
    SshDownloadTool,
    SshListServersTool,
    SshRunTool,
    SshUploadTool,
)


BUILT_IN_TOOL_CLASSES: Dict[str, Type[Tool]] = {
    "Read": ReadTool,
    "Write": WriteTool,
    "Edit": EditTool,
    "Bash": BashTool,
    "Glob": GlobTool,
    "Grep": GrepTool,
    "WebFetch": WebFetchTool,
    "WebSearch": WebSearchTool,
    "TodoWrite": TodoWriteTool,
    "NotebookEdit": NotebookEditTool,
    "ToolSearch": ToolSearchTool,
    "EnterPlanMode": EnterPlanModeTool,
    "ExitPlanMode": ExitPlanModeTool,
    "Agent": AgentTool,
    "AskUserQuestion": AskUserQuestionTool,
    "PushNotification": PushNotificationTool,
    "MCP": MCPTool,
    "ListMcpResources": ListMcpResourcesTool,
    "ReadMcpResource": ReadMcpResourceTool,
    "McpAuth": McpAuthTool,
    "EnterWorktree": EnterWorktreeTool,
    "ExitWorktree": ExitWorktreeTool,
    "LSP": LSPTool,
    "REPL": REPLTool,
    "Brief": BriefTool,
    "Config": ConfigTool,
    "Monitor": MonitorTool,
    "SendUserFile": SendUserFileTool,
    "WorkspaceInfo": WorkspaceInfoTool,
    "SandboxInfo": SandboxInfoTool,
    "SandboxPut": SandboxPutTool,
    "SandboxFetch": SandboxFetchTool,
    "SendMessage": SendMessageTool,
    "CronCreate": CronCreateTool,
    "CronDelete": CronDeleteTool,
    "CronList": CronListTool,
    "TaskCreate": TaskCreateTool,
    "TaskGet": TaskGetTool,
    "TaskList": TaskListTool,
    "TaskUpdate": TaskUpdateTool,
    "TaskOutput": TaskOutputTool,
    "TaskStop": TaskStopTool,
    "SubAgentSpawn": SubAgentSpawnTool,
    "SubAgentAssign": SubAgentAssignTool,
    "SubAgentList": SubAgentListTool,
    "SubAgentStop": SubAgentStopTool,
    "SubAgentInboxRead": SubAgentInboxReadTool,
    # Delegation gateway — the skill guide across DelegateTask/SubAgent*/Task*
    # (Guide + compact members, DocGuide 동형 점진공개).
    "DelegationGuide": DelegationGuideTool,
    # Self-modifying environment — one lean dispatcher; detailed guidance lives
    # in the bundled ``environment`` skill (progressive disclosure).
    "env": EnvTool,
    # Google Workspace (gated on feature:google_connected — hidden until the host
    # injects OAuth creds + marks Google connected).
    **GOOGLE_TOOL_CLASSES,
    # Atlassian (gated on feature:atlassian_connected — hidden until the host
    # injects a site URL + API token and marks Atlassian connected).
    **ATLASSIAN_TOOL_CLASSES,
    # Browser (an-web) — semantic web exploration; degrades to an install-hint
    # error when the optional an-web dependency is absent.
    **BROWSER_TOOL_CLASSES,
    # Doc (edit2docs) — office document engine; same lazy-import contract.
    **DOC_TOOL_CLASSES,
    # SSH — command/SFTP on the session's configured servers (gated on
    # feature:ssh_enabled); lazy-imports asyncssh with an install-hint fallback.
    **SSH_TOOL_CLASSES,
    **AUDIO_TOOL_CLASSES,
}


# Feature groupings keep the catalog navigable as it grows. A tool may
# belong to exactly one family — the boundary is "which capability bucket
# does this power?", not "which source directory does it live in?" Hosts
# selecting by feature get a stable API even as we add, rename, or split
# individual tools.
#: 스킬 게이트웨이 — ``{문 이름: 그 문이 여는 방}``.
#:
#: 계층 표면의 규약은 두 줄이다: 기본 명령은 턴 1에 바로 서고, 나머지는 **문
#: 하나**만 서서 부르면 그 방이 열린다. 그 두 번째 줄이 네 모듈에 흩어진 습관으로
#: 남아 있는 동안, 문 하나(DelegationGuide)는 지도만 돌려주고 방을 잠가 둔 채였다
#: — 가이드가 "DelegateTask 를 써라" 고 말해 놓고 그 이름은 부를 수 없었다.
#:
#: 여기 선언하면 규약이 검사 대상이 된다(tests/unit/test_skill_gateways.py):
#: 표에 있는 문은 실제로 자기 방을 열어야 하고, 여는 이름은 자기 설명이 약속한
#: 이름이어야 한다. 새 패밀리를 만들면서 문만 만들고 여는 것을 잊으면 테스트가
#: 먼저 말한다.
#:
#: ``JobGuide`` 는 xgen-workflow 에 산다(서버 스케줄러에 묶여 있다) — 같은 규약을
#: 그쪽 테스트가 고정한다.
SKILL_GATEWAYS: Dict[str, Tuple[str, ...]] = {
    "DelegationGuide": DELEGATION_FAMILY,
    "BrowserGuide": BROWSER_FAMILY,
    "DocGuide": DOC_FAMILY,
    # 목록 도구가 곧 문이다 — SshRun 이 받는 서버 '이름' 의 유일한 출처다.
    "SshListServers": SSH_FAMILY,
}


BUILT_IN_TOOL_FEATURES: Dict[str, List[str]] = {
    "filesystem": ["Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit"],
    "shell": ["Bash"],
    "web": ["WebFetch", "WebSearch"],
    # Interactive web exploration (an-web engine) — JS-rendered pages,
    # per-session tabs, semantic snapshots. Distinct from "web" (one-shot
    # fetch/search) so hosts can enable them independently.
    "browser": list(BROWSER_TOOL_CLASSES.keys()),
    # Office documents (edit2docs engine) — outline/edit/preview/generate.
    "documents": list(DOC_TOOL_CLASSES.keys()),
    "workflow": ["TodoWrite"],
    "meta": ["ToolSearch", "EnterPlanMode", "ExitPlanMode"],
    "agent": ["Agent"],
    "subagent": [
        # 게이트웨이 먼저 — 위임 3표면(DelegateTask/SubAgent*/Task*)의 결정
        # 지도를 요청 시 공개한다 (Guide + 컴팩트 멤버 규약).
        "DelegationGuide",
        "SubAgentSpawn",
        "SubAgentAssign",
        "SubAgentList",
        "SubAgentStop",
        "SubAgentInboxRead",
    ],
    "tasks": ["TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskOutput", "TaskStop"],
    "interaction": ["AskUserQuestion"],
    "notification": ["PushNotification"],
    "mcp": ["MCP", "ListMcpResources", "ReadMcpResource", "McpAuth"],
    "worktree": ["EnterWorktree", "ExitWorktree"],
    "dev": ["LSP", "REPL", "Brief"],
    "operator": ["Config", "Monitor", "SendUserFile"],
    # The session's two file spaces: inspect the host-side files workspace,
    # check the sandbox, and move files between them.
    "workspace": ["WorkspaceInfo", "SandboxInfo", "SandboxPut", "SandboxFetch"],
    "messaging": ["SendMessage"],
    "cron": ["CronCreate", "CronDelete", "CronList"],
    "environment": ["env"],
    "google": list(GOOGLE_TOOL_CLASSES.keys()),
    # Jira + Confluence control on the configured Atlassian site.
    "atlassian": list(ATLASSIAN_TOOL_CLASSES.keys()),
    # Remote server ops over SSH/SFTP — run commands, transfer files, sudo.
    "ssh": list(SSH_TOOL_CLASSES.keys()),
    # Workspace audio → text bridge (STT). Gated on feature:stt_enabled.
    "audio": list(AUDIO_TOOL_CLASSES.keys()),
}


def get_builtin_tools(
    *,
    features: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Type[Tool]]:
    """Return a ``{tool_name: tool_class}`` mapping.

    Selection:
        * No args → every tool in :data:`BUILT_IN_TOOL_CLASSES`.
        * ``features=[...]`` → the union of every tool in those
          feature families (see :data:`BUILT_IN_TOOL_FEATURES`). An
          unknown feature name raises ``KeyError`` so typos surface
          at the call site rather than silently dropping tools.
        * ``names=[...]`` → exactly those tool names. An unknown name
          raises ``KeyError``. Can be combined with ``features`` to
          subtract or add specific entries from the feature union.

    The returned dict is fresh — callers may mutate it without
    affecting the registry constants.

    Examples:
        >>> sorted(get_builtin_tools(features=["filesystem"]).keys())
        ['Edit', 'Glob', 'Grep', 'Read', 'Write']

        >>> sorted(get_builtin_tools(features=["web"], names=["Read"]).keys())
        ['Read', 'WebFetch', 'WebSearch']
    """
    selected: Dict[str, Type[Tool]] = {}

    if features is None and names is None:
        return dict(BUILT_IN_TOOL_CLASSES)

    if features is not None:
        for feat in features:
            if feat not in BUILT_IN_TOOL_FEATURES:
                raise KeyError(
                    f"unknown built-in feature {feat!r}; "
                    f"known: {sorted(BUILT_IN_TOOL_FEATURES.keys())}"
                )
            for tool_name in BUILT_IN_TOOL_FEATURES[feat]:
                selected[tool_name] = BUILT_IN_TOOL_CLASSES[tool_name]

    if names is not None:
        for name in names:
            if name not in BUILT_IN_TOOL_CLASSES:
                raise KeyError(
                    f"unknown built-in tool {name!r}; known: {sorted(BUILT_IN_TOOL_CLASSES.keys())}"
                )
            selected[name] = BUILT_IN_TOOL_CLASSES[name]

    return selected


__all__ = [
    "SKILL_GATEWAYS",
    "AgentTool",
    "ATLASSIAN_TOOL_CLASSES",
    "BROWSER_TOOL_CLASSES",
    "DOC_TOOL_CLASSES",
    "SSH_TOOL_CLASSES",
    "AUDIO_TOOL_CLASSES",
    "SshListServersTool",
    "SshRunTool",
    "SshUploadTool",
    "SshDownloadTool",
    "DocAnalyzeTool",
    "DocApplyEditsTool",
    "DocArrangeTool",
    "DocBuildTool",
    "DocEditTool",
    "DocGenerateTool",
    "DocGuideTool",
    "DocRenderTool",
    "DocXmlEditTool",
    "DocXmlReadTool",
    "BrowserActTool",
    "BrowserBackTool",
    "BrowserCloseTool",
    "BrowserEvalTool",
    "BrowserExtractTool",
    "BrowserNavigateTool",
    "BrowserSnapshotTool",
    "SubAgentSpawnTool",
    "SubAgentAssignTool",
    "SubAgentListTool",
    "SubAgentStopTool",
    "SubAgentInboxReadTool",
    "AskUserQuestionTool",
    "BriefTool",
    "ConfigTool",
    "CronCreateTool",
    "CronDeleteTool",
    "CronListTool",
    "EnterWorktreeTool",
    "ExitWorktreeTool",
    "LSPTool",
    "ListMcpResourcesTool",
    "MonitorTool",
    "REPLTool",
    "SendMessageTool",
    "SendUserFileTool",
    "MCPTool",
    "McpAuthTool",
    "PushNotificationTool",
    "QuestionCancelled",
    "ReadMcpResourceTool",
    "ReadTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TaskUpdateTool",
    "WriteTool",
    "EditTool",
    "BashTool",
    "GlobTool",
    "GrepTool",
    "WebFetchTool",
    "WebSearchTool",
    "TodoWriteTool",
    "NotebookEditTool",
    "ToolSearchTool",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "EnvTool",
    "SandboxExecTool",
    "WorkspaceInfoTool",
    "SandboxInfoTool",
    "SandboxPutTool",
    "SandboxFetchTool",
    "BUILT_IN_TOOL_CLASSES",
    "BUILT_IN_TOOL_FEATURES",
    "get_builtin_tools",
]
