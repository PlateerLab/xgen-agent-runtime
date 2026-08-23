"""BashTool — execute shell commands."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Dict, FrozenSet, Mapping, Optional

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

# Windows additions (desktop host — the connector sidecar runs this tool
# directly on the user's PC). A child spawned without ``SystemRoot`` fails
# to initialise Winsock/CRT, ``COMSPEC``/``PATHEXT`` are needed for the
# shell to resolve commands at all, and ``HOME`` is normally unset there
# (``USERPROFILE`` is the home). Kept as a local fallback table; the CLI
# runtime's authoritative Windows whitelist is reused when importable.
_SAFE_ENV_KEYS_WINDOWS_FALLBACK = frozenset(
    {
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "SYSTEMDRIVE",
        "USERNAME",
    }
)


def _windows_env_keys() -> FrozenSet[str]:
    """Windows whitelist — ``_cli_runtime``'s table when importable (one
    source of truth with the CLI subprocess env), else the local fallback."""
    try:
        from xgen_agent_runtime.llm_client._cli_runtime import _ENV_WHITELIST_WINDOWS

        return frozenset(_ENV_WHITELIST_WINDOWS) | _SAFE_ENV_KEYS_WINDOWS_FALLBACK
    except Exception:  # noqa: BLE001 — import cycle / layout drift: fall back
        return _SAFE_ENV_KEYS_WINDOWS_FALLBACK


def _scrubbed_env(
    extra: Optional[Mapping[str, str]],
    *,
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> Dict[str, str]:
    """Benign base env for the host-path subprocess.

    Platform-aware: on Windows (``platform == "win32"``) the process
    bootstrap variables are whitelisted too and matching is
    case-insensitive (``Path`` vs ``PATH``, ``SystemRoot`` vs
    ``SYSTEMROOT`` — the parent's spelling is preserved); ``HOME`` is
    mapped from ``USERPROFILE`` when unset so ``~``/``$HOME`` resolve. The
    ``environ``/``platform`` knobs exist for tests — production reads
    ``os.environ`` / ``sys.platform``.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform
    is_windows = plat == "win32"

    if str(source.get("GENY_BASH_INHERIT_ENV", "")).strip() in ("1", "true", "yes"):
        env: Dict[str, str] = dict(source)
    elif is_windows:
        allowed_ci = {k.upper() for k in (_SAFE_ENV_KEYS | _windows_env_keys())}
        env = {
            k: v
            for k, v in source.items()
            if k.upper() in allowed_ci or k.upper().startswith("LC_")
        }
        # PATH must exist under SOME spelling; only synthesise when absent.
        if not any(k.upper() == "PATH" for k in env):
            system_root = next((v for k, v in env.items() if k.upper() == "SYSTEMROOT"), "")
            if system_root:
                env["PATH"] = f"{system_root}\\System32;{system_root}"
        if not any(k.upper() == "HOME" for k in env):
            profile = next((v for k, v in env.items() if k.upper() == "USERPROFILE"), "")
            if profile:
                env["HOME"] = profile
    else:
        env = {k: v for k, v in source.items() if k in _SAFE_ENV_KEYS or k.startswith("LC_")}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    if extra:
        env.update(extra)
    return env


def _host_shell_argv(command: str, *, platform: Optional[str] = None) -> Optional[list]:
    """호스트 실행(샌드박스 없음)에서 명령을 돌릴 shell argv.

    Windows 는 bash 가 없다 — 커넥터 로컬 셸 도구와 동일하게 **PowerShell** 로 돈다
    (``powershell.exe -NoProfile -NonInteractive -Command <cmd>``). PowerShell 이 없으면
    (매우 드묾) ``cmd.exe /d /s /c`` 로 폴백. POSIX 는 None 을 돌려 기존 경로
    (``create_subprocess_shell`` = ``/bin/sh -c``)를 그대로 쓴다.

    반환 None → create_subprocess_shell(command) (POSIX).
    반환 [file, *args] → create_subprocess_exec(*argv) (Windows).
    """
    plat = sys.platform if platform is None else platform
    if plat != "win32":
        return None
    pwsh = _which_windows("powershell.exe") or _which_windows("pwsh.exe")
    if pwsh:
        return [pwsh, "-NoProfile", "-NonInteractive", "-Command", command]
    comspec = os.environ.get("ComSpec") or os.environ.get("COMSPEC") or "cmd.exe"
    return [comspec, "/d", "/s", "/c", command]


def _which_windows(name: str) -> Optional[str]:
    """PATH 에서 실행 파일을 찾는다(Windows). 없으면 None — shutil.which 얇은 래퍼."""
    try:
        import shutil

        return shutil.which(name)
    except Exception:  # noqa: BLE001
        return None


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
            "Execute a shell command in your execution host — the place where "
            "all your work runs. Depending on how this turn is executed that "
            "is either your own isolated sandbox session on the server, or "
            "the user's PC itself — directly inside the synchronized workspace "
            "folder — when the turn runs locally; the environment section of "
            "your prompt says which. Either way you "
            "have read/write access to your working directory and can install "
            "the dependencies you need (for example `pip install ...`, "
            "`uv pip install ...`, `npm install ...`) into that environment — "
            "be mindful that on a local PC this touches the user's real "
            "machine. Returns stdout, stderr, and exit code. Commands run in "
            "your working directory with a configurable timeout. Shell: on "
            "Linux/macOS (server sandbox, or a local Unix PC) commands run in a "
            "POSIX shell — use bash/sh syntax. On a local Windows PC they run in "
            "PowerShell — use PowerShell syntax (e.g. `Get-ChildItem`, `$env:VAR`, "
            "`;` to chain) rather than bash-isms."
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

        # Windows 호스트(커넥터 로컬)는 bash 가 없으므로 PowerShell 로 돈다 — 셸 선택은
        # _host_shell_argv 가 캡슐화한다(POSIX 는 None → 기존 /bin/sh 경로).
        shell_argv = _host_shell_argv(command)
        try:
            if shell_argv is not None:
                proc = await asyncio.create_subprocess_exec(
                    *shell_argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
            else:
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
