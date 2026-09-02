"""Tool Sandbox — execution-time security boundary for tool invocations.

Wraps tool execution with:
  - Path validation (chroot-style)
  - Timeout enforcement
  - Output size limits
  - Network policy (advisory)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.core._head_tail_buffer import HeadTailBuffer
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether *path* is *root* or one of its real descendants."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass
class SandboxConfig:
    """Sandbox execution constraints."""

    allowed_paths: Optional[List[str]] = None
    network_policy: str = "allow"  # "allow" | "deny" | "restrict"
    allowed_hosts: Optional[List[str]] = None
    max_execution_time: int = 120  # seconds
    max_output_size: int = 1_000_000  # bytes
    env_vars: Optional[Dict[str, str]] = None


class ToolSandbox:
    """Wraps tool execution with security constraints."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self._config = config or SandboxConfig()

    @property
    def config(self) -> SandboxConfig:
        return self._config

    async def execute_tool(
        self,
        tool: Tool,
        input: Dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute a tool within the sandbox constraints."""
        # 1. Path validation
        if self._config.allowed_paths is not None:
            error = self._validate_paths(input, context)
            if error:
                return ToolResult(content=error, is_error=True)

        # 2. Enrich context with sandbox env vars.
        # Override ONLY the two fields this wrapper actually changes and
        # carry every other field through untouched via ``replace``. A
        # field-by-field rebuild silently dropped the runtime handles the
        # host attaches — most critically ``sandbox`` (file/shell tools
        # then run on the pod instead of the agent's session), but also
        # ``environment``/``hook_runner``/``permission_rules``/``event_emit``
        # — and would keep dropping any field added to ToolContext later.
        if self._config.env_vars:
            ctx_env = dict(context.env_vars or {})
            ctx_env.update(self._config.env_vars)
            context = replace(
                context,
                env_vars=ctx_env,
                allowed_paths=self._config.allowed_paths or context.allowed_paths,
            )

        # 3. Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(input, context),
                timeout=self._config.max_execution_time,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                content=(f"Tool '{tool.name}' timed out after {self._config.max_execution_time}s"),
                is_error=True,
            )
        except Exception as e:
            return ToolResult(
                content=f"Tool '{tool.name}' error: {e}",
                is_error=True,
            )

        # 4. Output size limit
        if isinstance(result.content, str):
            encoded = result.content.encode("utf-8")
            if len(encoded) > self._config.max_output_size:
                output = HeadTailBuffer(self._config.max_output_size)
                output.push_chunk(encoded)
                result = ToolResult(
                    content=output.to_bytes_with_omission_marker().decode(
                        "utf-8", errors="replace"
                    ),
                    is_error=result.is_error,
                    metadata=result.metadata,
                )

        return result

    def _validate_paths(self, input: Dict[str, Any], context: ToolContext) -> Optional[str]:
        """Check that file-related inputs are within allowed paths."""
        if not self._config.allowed_paths:
            return None

        path_keys = ("path", "file_path", "directory", "command")
        for key in path_keys:
            if key not in input:
                continue
            val = input[key]
            if not isinstance(val, str):
                continue

            # Skip command-type args (they aren't pure paths)
            if key == "command":
                continue

            candidate = Path(context.working_dir, val) if not os.path.isabs(val) else Path(val)
            resolved = candidate.resolve()
            allowed_roots = [Path(path).resolve() for path in self._config.allowed_paths]
            if not any(_path_is_within(resolved, root) for root in allowed_roots):
                return (
                    f"Path '{val}' resolves to '{resolved}' which is outside "
                    f"allowed paths: {self._config.allowed_paths}"
                )

        return None


# ── Preset policies ─────────────────────────────────────────


class SandboxPolicy:
    """Pre-defined sandbox configurations."""

    @staticmethod
    def strict(working_dir: str = ".") -> ToolSandbox:
        """Minimal permissions."""
        return ToolSandbox(
            SandboxConfig(
                allowed_paths=[os.path.realpath(working_dir)],
                network_policy="deny",
                max_execution_time=30,
                max_output_size=100_000,
            )
        )

    @staticmethod
    def standard(working_dir: str = ".") -> ToolSandbox:
        """Standard security."""
        return ToolSandbox(
            SandboxConfig(
                allowed_paths=[os.path.realpath(working_dir)],
                network_policy="restrict",
                max_execution_time=120,
                max_output_size=1_000_000,
            )
        )

    @staticmethod
    def permissive() -> ToolSandbox:
        """Permissive — local development."""
        return ToolSandbox(
            SandboxConfig(
                allowed_paths=None,
                network_policy="allow",
                max_execution_time=600,
                max_output_size=10_000_000,
            )
        )
