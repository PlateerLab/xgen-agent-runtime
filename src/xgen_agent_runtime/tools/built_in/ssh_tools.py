"""Built-in SSH tools — run commands and move files on the session's
pre-configured servers.

The host (Geny) records the user's servers per session (see
``_ssh_store.SSHServerStore``). Each tool takes a server by NAME and resolves
the credential internally, so the agent operates servers without ever handling
a password or key. Gated on ``feature:ssh_enabled`` so the family stays hidden
until the host provisions SSH for the session.

Tools:
  * ``SshListServers`` — what servers can I reach? (names + host/user, no secrets)
  * ``SshRun``         — run a shell command on a named server (optional sudo)
  * ``SshUpload``      — copy a session-storage file → server (SFTP)
  * ``SshDownload``    — copy a server file → session storage (SFTP)
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.tools.base import (
    Tool,
    ToolCapabilities,
    ToolResult,
    execution_backend_unavailable,
)
from xgen_agent_runtime.tools._ssh import (
    SSHConfigError,
    SSHUnavailableError,
    sftp_get,
    sftp_put,
    ssh_exec,
)
from xgen_agent_runtime.tools.built_in._skill_gateway import open_family, with_opened
from xgen_agent_runtime.tools.built_in._ssh_store import SSHServerStore
from xgen_agent_runtime.tools.fs import FsAccessError, tool_fs

_SSH_FEATURE_KEY = "feature:ssh_enabled"
_MAX_OUTPUT = 100_000  # chars per stream, matching BashTool
_DEFAULT_TIMEOUT = 60.0
_MAX_TIMEOUT = 900.0
_MAX_TRANSFER_BYTES = 100 * 1024 * 1024


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content={"error": {"code": code, "message": message}}, is_error=True)


def _store(context: Any) -> SSHServerStore:
    return SSHServerStore.from_context(context)


def _resolve_or_error(store: SSHServerStore, name: Optional[str]):
    name = (name or "").strip()
    if not name:
        return None, _err("NO_SERVER", "Provide 'server' (a configured server name).")
    server = store.target(name)
    if server is None:
        avail = ", ".join(store.names()) or "(none configured)"
        return None, _err(
            "UNKNOWN_SERVER",
            f"No SSH server named '{name}'. Available: {avail}. "
            "Use SshListServers to see configured servers.",
        )
    return server, None


def _truncate(text: str) -> tuple[str, bool]:
    """The start and the end of ``text``, with a marker in the text itself where it was cut.

    The cut used to keep only the start, and said so only in the ``truncated`` field of the result
    object, which the model never reads (it reads ``display_text``): a long remote log lost its end
    (the error) and looked complete (2026-10-23 audit)."""
    if len(text) <= _MAX_OUTPUT:
        return text, False
    head = _MAX_OUTPUT * 3 // 4
    tail = _MAX_OUTPUT - head
    cut = len(text) - head - tail
    return (
        f"{text[:head]}\n\n[... {cut:,} chars cut here; the start and the end are shown. "
        f"Redirect long output to a file on the server and read it in parts. ...]\n\n{text[-tail:]}",
        True,
    )


def _guarded_local(context: Any, path: str):
    """Resolve *path* through the call's explicit file capability."""
    fs = tool_fs(context)
    if getattr(fs, "kind", "") == "unavailable":
        return None, None, execution_backend_unavailable("SSH file access")
    try:
        resolved = fs.resolve(path)
    except (PermissionError, ValueError) as exc:
        return None, None, _err("PATH_ESCAPE", str(exc))
    return fs, resolved, None


class _SSHToolBase(Tool):
    """Shared feature gate for the SSH family."""

    def required_config_keys(self) -> List[str]:
        # Host gate — hidden until Geny marks the session SSH-provisioned.
        return [_SSH_FEATURE_KEY]


#: 이 문 뒤의 방 — SshListServers 를 부르면 열린다. 목록 도구가 곧 게이트웨이인
#: 이유는 단순하다: SshRun/SshUpload/SshDownload 는 전부 서버 '이름' 을 받는데,
#: 그 이름의 유일한 출처가 이 목록이다.
SSH_FAMILY = ("SshRun", "SshUpload", "SshDownload")


class SshListServersTool(_SSHToolBase):
    """List the servers this session can SSH into (no secrets).

    Also the family's gateway — see :data:`SSH_FAMILY`.
    """

    @property
    def name(self) -> str:
        return "SshListServers"

    @property
    def description(self) -> str:
        return (
            "START HERE to reach a REMOTE SERVER (deploy, restart a service, "
            "read production logs) — lists this session's SSH hosts and OPENS "
            "SshRun / SshUpload / SshDownload. Use a server's 'name'; "
            "credentials stay server-side, jump hosts are dialled for you."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(read_only=True, concurrency_safe=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        servers = _store(context).list_public()
        if not servers:
            # 열 것이 없다. SshRun 은 등록된 서버 '이름' 으로만 도는 도구라,
            # 서버가 하나도 없는 세션에서 문을 여는 것은 쓸 수 없는 동사를
            # 표면에 올리는 일일 뿐이다.
            return ToolResult(
                content={"servers": []},
                display_text="No SSH servers are configured for this session.",
            )
        # 이 도구가 이 패밀리의 문이다 — 목록이 이름을 알려 주고, 그 이름을 쓰는
        # 동사를 같은 답에서 연다. 예전에는 "use SshRun" 이라고 말해 놓고 그
        # 이름은 아직 deferred 라, CLI 백엔드에서 모델이 곧바로 부르면 클라이언트가
        # 로컬에서 거절했다 (No such tool available).
        opened = open_family(context, SSH_FAMILY)
        lines = []
        for s in servers:
            line = f"- {s['name']}: {s['user']}@{s['host']}:{s['port']} [{s['auth']}]"
            # 경유 경로는 실패를 읽는 데 필수다 — 명령이 잘못된 건지 경로가 끊긴 건지.
            if s.get("via"):
                line += " via " + " → ".join(s["via"])
            if s.get("description"):
                line += f" — {s['description']}"
            lines.append(line)
        return ToolResult(
            content={"servers": servers},
            display_text=with_opened("Configured SSH servers:\n" + "\n".join(lines), opened),
            metadata={"opened": opened},
        )


class SshRunTool(_SSHToolBase):
    """Run a shell command on a named server over SSH."""

    @property
    def name(self) -> str:
        return "SshRun"

    @property
    def description(self) -> str:
        return (
            "Run a shell command on a REMOTE server over SSH — deploy, restart "
            "a service, tail logs. Pass 'server' (from SshListServers) and "
            "'command'. Optional: 'cwd', 'timeout', 'sudo' (uses the stored "
            "password). Returns stdout, stderr, exit code."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured server name."},
                "command": {"type": "string", "description": "Shell command to run remotely."},
                "cwd": {"type": "string", "description": "Remote working directory (optional)."},
                "timeout": {
                    "type": "number",
                    "description": f"Max seconds to wait (default {int(_DEFAULT_TIMEOUT)}, max {int(_MAX_TIMEOUT)}).",
                },
                "sudo": {"type": "boolean", "description": "Run under sudo -S (default false)."},
            },
            "required": ["server", "command"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(network_egress=True, interrupt="cancel")

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        store = _store(context)
        server, err = _resolve_or_error(store, input.get("server"))
        if err is not None:
            return err
        command = str(input.get("command") or "").strip()
        if not command:
            return _err("NO_COMMAND", "Provide a non-empty 'command'.")
        try:
            timeout = float(input.get("timeout") or _DEFAULT_TIMEOUT)
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT
        timeout = max(1.0, min(timeout, _MAX_TIMEOUT))
        cwd = input.get("cwd") or None
        sudo = bool(input.get("sudo"))
        if sudo and not server.get("password"):
            return _err(
                "NO_SUDO_PASSWORD",
                f"Server '{server.get('name')}' has no stored password, so sudo "
                "cannot be used (key-only auth). Configure a password to enable sudo.",
            )

        name = server.get("name")
        try:
            rc, out, err_out = await ssh_exec(
                server,
                command,
                timeout=timeout,
                cwd=cwd,
                sudo=sudo,
                resolver=store.resolve,
            )
        except SSHUnavailableError as exc:
            return _err("SSH_UNAVAILABLE", str(exc))
        except SSHConfigError as exc:
            return _err("BAD_SERVER", f"Server '{name}': {exc}")
        except asyncio.TimeoutError:
            return _err("TIMEOUT", f"Command on '{name}' exceeded {timeout:.0f}s.")
        except Exception as exc:  # noqa: BLE001 — auth/network → structured error
            return _err("SSH_ERROR", f"{type(exc).__name__}: {exc}")

        out_s, out_trunc = _truncate(out)
        err_s, err_trunc = _truncate(err_out)
        content = {
            "server": name,
            "exit_code": rc,
            "stdout": out_s,
            "stderr": err_s,
            "truncated": bool(out_trunc or err_trunc),
        }
        header = f"[{name}] exit={rc}" + (" (sudo)" if sudo else "")
        body = out_s if out_s else ""
        if err_s:
            body += ("\n" if body else "") + "stderr:\n" + err_s
        display = header + ("\n" + body if body else "")
        return ToolResult(
            content=content,
            is_error=(rc != 0),
            display_text=display,
            metadata={"exit_code": rc, "server": name},
        )


class SshUploadTool(_SSHToolBase):
    """Upload a session-storage file to a server via SFTP."""

    @property
    def name(self) -> str:
        return "SshUpload"

    @property
    def description(self) -> str:
        return (
            "Copy a file from this session's storage to a REMOTE server via "
            "SFTP — how you ship a build artifact or config to deploy it. Pass "
            "'server', 'local_path' (relative to session storage), and "
            "'remote_path' (absolute path on the server)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured server name."},
                "local_path": {
                    "type": "string",
                    "description": "File in session storage to upload.",
                },
                "remote_path": {"type": "string", "description": "Destination path on the server."},
            },
            "required": ["server", "local_path", "remote_path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(network_egress=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        store = _store(context)
        server, err = _resolve_or_error(store, input.get("server"))
        if err is not None:
            return err
        fs, _resolved, lerr = _guarded_local(context, str(input.get("local_path") or ""))
        if lerr is not None:
            return lerr
        remote_path = str(input.get("remote_path") or "").strip()
        if not remote_path:
            return _err("NO_REMOTE_PATH", "Provide 'remote_path'.")
        name = server.get("name")
        try:
            assert fs is not None
            async with fs.materialize(str(input.get("local_path") or "")) as local:
                if not local.is_file():
                    return _err("NOT_FOUND", f"Local file not found: {input.get('local_path')}")
                size = local.stat().st_size
                if size > _MAX_TRANSFER_BYTES:
                    return _err(
                        "TOO_LARGE",
                        f"File exceeds {_MAX_TRANSFER_BYTES // (1024 * 1024)} MB transfer cap.",
                    )
                await sftp_put(server, str(local), remote_path, resolver=store.resolve)
                local_name = local.name
        except SSHUnavailableError as exc:
            return _err("SSH_UNAVAILABLE", str(exc))
        except FileNotFoundError:
            return _err("NOT_FOUND", f"Local file not found: {input.get('local_path')}")
        except FsAccessError:
            return execution_backend_unavailable("SSH file access")
        except SSHConfigError as exc:
            return _err("BAD_SERVER", f"Server '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001
            return _err("SFTP_ERROR", f"{type(exc).__name__}: {exc}")
        return ToolResult(
            content={"server": name, "uploaded": remote_path, "bytes": size},
            display_text=f"Uploaded {local_name} → {name}:{remote_path}",
        )


class SshDownloadTool(_SSHToolBase):
    """Download a server file into session storage via SFTP."""

    @property
    def name(self) -> str:
        return "SshDownload"

    @property
    def description(self) -> str:
        return (
            "Copy a file from a REMOTE server into this session's storage via "
            "SFTP — how you pull a production log or config back to inspect it. "
            "Pass 'server', 'remote_path' (path on the server), and "
            "'local_path' (relative to session storage)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured server name."},
                "remote_path": {"type": "string", "description": "File path on the server."},
                "local_path": {"type": "string", "description": "Destination in session storage."},
            },
            "required": ["server", "remote_path", "local_path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(network_egress=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        store = _store(context)
        server, err = _resolve_or_error(store, input.get("server"))
        if err is not None:
            return err
        fs, resolved, lerr = _guarded_local(context, str(input.get("local_path") or ""))
        if lerr is not None:
            return lerr
        remote_path = str(input.get("remote_path") or "").strip()
        if not remote_path:
            return _err("NO_REMOTE_PATH", "Provide 'remote_path'.")
        name = server.get("name")
        try:
            assert fs is not None
            with tempfile.TemporaryDirectory(prefix="xgen-sftp-") as scratch:
                local = Path(scratch) / (Path(resolved or "download").name or "download")
                await sftp_get(server, remote_path, str(local), resolver=store.resolve)
                if not local.is_file():
                    return _err("SFTP_ERROR", "SFTP download produced no local file.")
                size = local.stat().st_size
                if size > _MAX_TRANSFER_BYTES:
                    return _err(
                        "TOO_LARGE",
                        f"File exceeds {_MAX_TRANSFER_BYTES // (1024 * 1024)} MB transfer cap.",
                    )
                await fs.commit(local, str(input.get("local_path") or ""))
        except SSHUnavailableError as exc:
            return _err("SSH_UNAVAILABLE", str(exc))
        except FsAccessError:
            return execution_backend_unavailable("SSH file access")
        except SSHConfigError as exc:
            return _err("BAD_SERVER", f"Server '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001
            return _err("SFTP_ERROR", f"{type(exc).__name__}: {exc}")
        return ToolResult(
            content={"server": name, "downloaded": resolved, "bytes": size},
            display_text=f"Downloaded {name}:{remote_path} → {input.get('local_path')} ({size} bytes)",
        )


SSH_TOOL_CLASSES: Dict[str, type] = {
    "SshListServers": SshListServersTool,
    "SshRun": SshRunTool,
    "SshUpload": SshUploadTool,
    "SshDownload": SshDownloadTool,
}

__all__ = [
    "SSH_FAMILY",
    "SshListServersTool",
    "SshRunTool",
    "SshUploadTool",
    "SshDownloadTool",
    "SSH_TOOL_CLASSES",
]
