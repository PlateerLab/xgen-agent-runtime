"""Tool system — registration, routing, execution, composition."""

from xgen_agent_runtime.tools.base import (
    Tool,
    ToolResult,
    ToolContext,
    ToolCapabilities,
    PermissionDecision,
    build_tool,
)
from xgen_agent_runtime.tools.registry import ToolRegistry
from xgen_agent_runtime.tools.adhoc import (
    AdhocTool,
    AdhocToolDefinition,
    AdhocToolFactory,
    HttpToolConfig,
    ScriptToolConfig,
    TemplateToolConfig,
    CompositeToolConfig,
    CompositeStep,
)
from xgen_agent_runtime.tools.composer import ToolComposer, ToolInfo, ToolPreset
from xgen_agent_runtime.tools.providers import AdhocToolProvider
from xgen_agent_runtime.tools.scope import ToolScope, ToolScopeRule, ToolScopeManager
from xgen_agent_runtime.tools.sandbox import ToolSandbox, SandboxConfig, SandboxPolicy

# XGeny sandbox primitives — the session an agent's code runs in. Public so
# hosts can build sandboxed tools. ``_xgeny_sandbox`` only depends on stdlib, so
# this import is cycle-safe (unlike importing the built_in package here).
# ``SandboxExecTool`` itself lives in ``xgen_agent_runtime.tools.built_in`` to keep
# this module free of the built-in import cycle.
from xgen_agent_runtime.tools._xgeny_sandbox import (
    ExecResult,
    SandboxError,
    SandboxPathError,
    XgenySandbox,
    sandbox_path,
    sandbox_readonly_roots,
    sandbox_root,
    sb_read_bytes,
    sb_run,
    sb_write_bytes,
)
from xgen_agent_runtime.tools.plugin import (
    TOOL_ENTRY_POINT_GROUP,
    ToolPluginRegistry,
    discover_tool_plugins,
    register_tool_plugins,
)

__all__ = [
    # Base
    "Tool",
    "ToolResult",
    "ToolContext",
    "ToolCapabilities",
    "PermissionDecision",
    "build_tool",
    "ToolRegistry",
    # Ad-hoc
    "AdhocTool",
    "AdhocToolDefinition",
    "AdhocToolFactory",
    "HttpToolConfig",
    "ScriptToolConfig",
    "TemplateToolConfig",
    "CompositeToolConfig",
    "CompositeStep",
    # Composer
    "ToolComposer",
    "ToolInfo",
    "ToolPreset",
    # Providers
    "AdhocToolProvider",
    # Scope
    "ToolScope",
    "ToolScopeRule",
    "ToolScopeManager",
    # Sandbox (policy)
    "ToolSandbox",
    "SandboxConfig",
    "SandboxPolicy",
    # XGeny sandbox session — for sandboxed tools
    "ExecResult",
    "SandboxError",
    "SandboxPathError",
    "XgenySandbox",
    "sandbox_path",
    "sandbox_readonly_roots",
    "sandbox_root",
    "sb_run",
    "sb_read_bytes",
    "sb_write_bytes",
    # Plugin discovery (entry-point group: xgen_agent_runtime.tools)
    "TOOL_ENTRY_POINT_GROUP",
    "ToolPluginRegistry",
    "discover_tool_plugins",
    "register_tool_plugins",
]


# Lazy import for built-in tools to avoid circular dependencies
def get_built_in_registry(working_dir: str = "", **kwargs) -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with all built-in tools.

    Args:
        working_dir: Working directory for file operations.
        **kwargs: Additional keyword arguments passed to ToolContext fields
            (storage_path, env_vars, allowed_paths).

    Returns:
        ToolRegistry with Read, Write, Edit, Bash, Glob, Grep registered.
    """
    from xgen_agent_runtime.tools.built_in import (
        ReadTool,
        WriteTool,
        EditTool,
        BashTool,
        GlobTool,
        GrepTool,
    )

    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(EditTool())
    registry.register(BashTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry
