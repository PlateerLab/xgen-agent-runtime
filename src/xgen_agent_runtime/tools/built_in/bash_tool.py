"""BashTool — execute shell commands."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Host env vars the model's shell is allowed to inherit (audit S3). The
# non-sandbox path used ``os.environ.copy()``, handing every backend
# secret (ANTHROPIC_API_KEY, GENY_AUTH_SECRET, DB URLs, …) to any command
# the model runs. We inherit only a benign base; the host injects anything
# the workload legitimately needs via ``ToolContext.env_vars``. Set
# ``GENY_BASH_INHERIT_ENV=1`` to restore the old full-inherit behavior for
# a fully-trusted single-tenant deployment.
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TERM",
        "TZ",
        "TMPDIR",
        "PWD",
        "HOSTNAME",
        "DISPLAY",
        "COLUMNS",
        "LINES",
    }
)


def _scrubbed_env(extra: Dict[str, str] | None) -> Dict[str, str]:
    if os.environ.get("GENY_BASH_INHERIT_ENV", "").strip() in ("1", "true", "yes"):
        env = os.environ.copy()
    else:
        env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS or k.startswith("LC_")}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    if extra:
        env.update(extra)
    return env


_DEFAULT_TIMEOUT_MS = 120_000  # 2 minutes
_MAX_TIMEOUT_MS = 600_000  # 10 minutes
_MAX_OUTPUT = 100_000  # characters


class BashTool(Tool):
    """Execute a bash command and return stdout/stderr.

    Commands run in the session's working directory with configurable
    timeout and environment variable injection.
    """

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command in your own isolated sandbox session — a "
            "dedicated Linux environment that is separate from the server "
            "hosting this conversation. This sandbox is where all your work "
            "runs: you have full read/write access to your working directory "
            "and home directory, and you can install the dependencies you need "
            "(for example `pip install ...`, `uv pip install ...`, "
            "`npm install ...`) into your session. Returns stdout, stderr, and "
            "exit code. Commands run in your working directory with a "
            "configurable timeout."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in milliseconds (default: {_DEFAULT_TIMEOUT_MS}, max: {_MAX_TIMEOUT_MS}).",
                    "minimum": 1000,
                    "maximum": _MAX_TIMEOUT_MS,
                },
            },
            "required": ["command"],
        }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        command = input.get("command", "").strip()
        if not command:
            return ToolResult(content="command must not be empty", is_error=True)

        timeout_ms = min(input.get("timeout", _DEFAULT_TIMEOUT_MS), _MAX_TIMEOUT_MS)
        timeout_s = timeout_ms / 1000.0

        # Sandbox: run the command in the agent's XGeny session instead
        # of on the host. Same output shaping as the host path below.
        if context.sandbox is not None:
            from xgen_agent_runtime.tools._xgeny_sandbox import sb_run

            try:
                exit_code, stdout, stderr = await sb_run(
                    context.sandbox,
                    command,
                    workdir=context.working_dir or "/workspace",
                    env=context.env_vars,
                    timeout_s=timeout_s,
                )
            except asyncio.TimeoutError:
                return ToolResult(content=f"Command timed out after {timeout_ms}ms", is_error=True)
            except Exception as e:  # noqa: BLE001
                return ToolResult(content=f"Sandbox exec failed: {e}", is_error=True)
            if len(stdout) > _MAX_OUTPUT:
                stdout = stdout[:_MAX_OUTPUT] + "\n\n... (truncated)"
            if len(stderr) > _MAX_OUTPUT:
                stderr = stderr[:_MAX_OUTPUT] + "\n\n... (truncated)"
            parts = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"STDERR:\n{stderr}")
            if exit_code != 0:
                parts.append(f"Exit code: {exit_code}")
            return ToolResult(
                content="\n".join(parts) if parts else "(no output)",
                is_error=exit_code != 0,
                metadata={"exit_code": exit_code, "sandboxed": True},
            )

        # No sandbox attached. By design every XGeny agent has one, so this
        # is a degraded path (feature disabled via XGENY_SANDBOX_EXEC, or a
        # non-agent context) that runs the command on the HOST/pod, not in an
        # isolated session. Never fail silently: warn and tag the result so
        # the degradation is visible in logs and telemetry rather than the
        # command quietly touching the serving pod.
        logger.warning(
            "Bash executing on the HOST (no sandbox attached to ToolContext); "
            "command will run on the serving pod, not in an isolated session. "
            "This is a degraded path — check that the agent's sandbox session "
            "is being propagated into the tool dispatch context."
        )
        cwd = context.working_dir or None

        # Build a SCRUBBED environment (audit S3): a benign base +
        # host-injected env_vars, never the backend's full secret-bearing
        # os.environ.
        env = _scrubbed_env(context.env_vars)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        except OSError as e:
            return ToolResult(content=f"Failed to start process: {e}", is_error=True)

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return ToolResult(
                content=f"Command timed out after {timeout_ms}ms",
                is_error=True,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0

        # Truncate very large output
        if len(stdout) > _MAX_OUTPUT:
            stdout = stdout[:_MAX_OUTPUT] + f"\n\n... (truncated, {len(stdout_bytes)} bytes total)"
        if len(stderr) > _MAX_OUTPUT:
            stderr = stderr[:_MAX_OUTPUT] + f"\n\n... (truncated, {len(stderr_bytes)} bytes total)"

        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        if exit_code != 0:
            parts.append(f"Exit code: {exit_code}")

        output = "\n".join(parts) if parts else "(no output)"

        return ToolResult(
            content=output,
            is_error=exit_code != 0,
            metadata={"exit_code": exit_code, "sandboxed": False, "execution_environment": "host"},
        )
