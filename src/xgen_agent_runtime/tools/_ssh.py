"""Low-level SSH primitives for the built-in SSH tools.

Pure-async, built on ``asyncssh`` (native asyncio — no thread offload, so it
drops straight into a tool's ``async def execute``). Mirrors the shape of
``tools/_sandbox.py``: small helpers that open a connection, run a command (or
transfer a file), and return a ``(rc, stdout, stderr)`` triple, with all output
shaping/truncation left to the caller.

``asyncssh`` is an OPTIONAL dependency (extra ``ssh``). It is imported lazily so
the core install stays lean; callers get :class:`SSHUnavailableError` (a clean
install hint) when it is absent, never an ``ImportError`` at import time.

A "server" here is a plain dict — the credential record resolved by name from
the per-session store (see ``tools/built_in/_ssh_store.py``):

    {"name", "host", "port"?, "user", "password"?, "private_key"?,
     "passphrase"?, "description"?, "strict_host_key"?, "jump"?}

At least one of ``password`` / ``private_key`` must be present. The password
never leaves this module (it is fed to the transport / to ``sudo -S`` on
stdin), so the agent — which only ever passes a server *name* — never handles
it.

Jump hosts (bastions)
---------------------
``jump`` is an ordered list of **other configured server names** — the path
taken to reach this host, nearest hop first::

    {"name": "db", "host": "10.0.0.9", "jump": ["bastion", "inner-gw"]}

We open ``bastion`` first, tunnel ``inner-gw`` through it, then tunnel ``db``
through that. Naming other records (rather than inlining a nested credential)
means one machine is described **once** and can serve as both a bastion and a
target — and a rotated password is changed in one place. Resolution needs the
store, so the entry points take a ``resolver`` callable; without one, a record
that declares ``jump`` is refused rather than silently connected directly (a
direct dial usually fails, but when the bastion's network is flat it could
*succeed* against the wrong path — that must never happen quietly).
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class SSHUnavailableError(RuntimeError):
    """Raised when the optional ``asyncssh`` dependency is not installed."""


class SSHConfigError(ValueError):
    """Raised when a server record is malformed (missing host/creds, bad key)."""


_INSTALL_HINT = (
    "SSH tools require the `asyncssh` package. Install it via "
    "`pip install xgen-agent-runtime[ssh]`."
)


def _require_asyncssh():
    try:
        import asyncssh  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SSHUnavailableError(_INSTALL_HINT) from exc
    return asyncssh


def _connect_kwargs(
    server: Dict[str, Any], *, connect_timeout: float
) -> Tuple[str, Dict[str, Any]]:
    """Build ``(host, asyncssh.connect kwargs)`` from a server record.

    Host-key verification is DISABLED by default (``known_hosts=None``) so a
    user's own freshly-provisioned server "just works" on first contact; set
    ``strict_host_key: true`` on the record to opt back into verification.
    """
    asyncssh = _require_asyncssh()
    host = str(server.get("host") or "").strip()
    if not host:
        raise SSHConfigError("server record has no 'host'")
    try:
        port = int(server.get("port") or 22)
    except (TypeError, ValueError):
        raise SSHConfigError(f"invalid 'port': {server.get('port')!r}")
    username = str(server.get("user") or server.get("username") or "").strip() or None

    kwargs: Dict[str, Any] = {
        "port": port,
        "username": username,
        "connect_timeout": connect_timeout,
    }
    # known_hosts=None disables host-key checking; opt-in strict mode uses the
    # default system known_hosts.
    if not server.get("strict_host_key"):
        kwargs["known_hosts"] = None

    password = server.get("password") or None
    private_key = server.get("private_key") or None
    if private_key:
        passphrase = server.get("passphrase") or None
        try:
            key = asyncssh.import_private_key(private_key, passphrase)
        except Exception as exc:  # noqa: BLE001 — surface a clean config error
            raise SSHConfigError(f"invalid private_key: {exc}") from exc
        kwargs["client_keys"] = [key]
        if password:  # allow key + password (e.g. key + sudo password)
            kwargs["password"] = password
    elif password:
        kwargs["password"] = password
    else:
        raise SSHConfigError("server record has neither 'password' nor 'private_key'")

    return host, kwargs


#: Resolves a configured server *name* to its full record (or ``None``).
Resolver = Callable[[str], Optional[Dict[str, Any]]]

#: Hard ceiling on how many bastions may precede a host. Deep chains are almost
#: always a config mistake, and each hop multiplies the connect timeout.
MAX_JUMP_DEPTH = 8


def jump_names(server: Dict[str, Any]) -> List[str]:
    """The declared jump path of *server*, nearest hop first.

    Tolerates the shapes people actually type: a list, a single string, or a
    comma-separated string. Blank entries are dropped rather than turned into a
    lookup for the empty name.
    """
    raw = server.get("jump") or server.get("jump_via") or server.get("proxy_jump")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        parts: Sequence[Any] = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        parts = raw
    else:
        raise SSHConfigError(f"invalid 'jump' (want a list of server names): {raw!r}")
    out: List[str] = []
    for part in parts:
        name = str(part or "").strip()
        if name:
            out.append(name)
    return out


def resolve_chain(server: Dict[str, Any], resolver: Optional[Resolver]) -> List[Dict[str, Any]]:
    """Expand *server* into the full dial order: ``[hop1, hop2, …, server]``.

    A record with no ``jump`` resolves to just ``[server]``, so callers can use
    one code path for both.

    Refuses three things that would otherwise fail late and confusingly:
    a jump name nothing resolves to; a loop (``a`` via ``b`` via ``a``), which
    would hang until every hop timed out; and a chain past
    :data:`MAX_JUMP_DEPTH`.
    """
    chain: List[Dict[str, Any]] = []
    seen: List[str] = []

    def walk(node: Dict[str, Any], depth: int) -> None:
        name = str(node.get("name") or "").strip() or "(unnamed)"
        if name in seen:
            path = " → ".join([*seen, name])
            raise SSHConfigError(f"jump host loop: {path}")
        if depth > MAX_JUMP_DEPTH:
            raise SSHConfigError(
                f"jump path is deeper than {MAX_JUMP_DEPTH} hops (starting at '{name}')"
            )
        seen.append(name)
        hops = jump_names(node)
        if hops and resolver is None:
            raise SSHConfigError(
                f"server '{name}' is reached via {', '.join(hops)} but no jump-host "
                "resolver is available in this session"
            )
        for hop in hops:
            nxt = resolver(hop) if resolver else None
            if nxt is None:
                raise SSHConfigError(
                    f"server '{name}' declares jump host '{hop}', which is not a configured server"
                )
            walk(nxt, depth + 1)
        chain.append(node)

    walk(server, 0)
    return chain


@contextlib.asynccontextmanager
async def _open(
    server: Dict[str, Any],
    *,
    connect_timeout: float,
    resolver: Optional[Resolver] = None,
):
    """Open a connection to *server*, tunnelling through its jump hosts.

    Yields the connection to the **final** host. Every bastion opened on the way
    is closed on exit, in reverse order — asyncssh does not take ownership of a
    connection passed as ``tunnel``, so if we dropped those references the
    sockets would leak for the life of the session.
    """
    asyncssh = _require_asyncssh()
    chain = resolve_chain(server, resolver)
    async with contextlib.AsyncExitStack() as stack:
        tunnel = None
        for hop in chain:
            host, kwargs = _connect_kwargs(hop, connect_timeout=connect_timeout)
            if tunnel is not None:
                kwargs["tunnel"] = tunnel
            try:
                conn = await stack.enter_async_context(await asyncssh.connect(host, **kwargs))
            except (SSHUnavailableError, SSHConfigError):
                raise
            except Exception as exc:  # noqa: BLE001 — say WHICH hop failed
                if hop is not chain[-1]:
                    raise ConnectionError(
                        f"jump host '{hop.get('name') or host}': {_friendly_error(exc)}"
                    ) from exc
                raise
            tunnel = conn
        yield tunnel


def _remote_command(
    command: str, *, cwd: Optional[str], sudo: bool, password: Optional[str]
) -> Tuple[str, Optional[str]]:
    """Compose the remote command string + optional stdin (sudo password)."""
    inner = command
    if cwd:
        inner = f"cd {shlex.quote(cwd)} && {command}"
    if sudo:
        # -S reads the password from stdin; -p '' suppresses the prompt so it
        # can't pollute stdout. Wrap in a login shell so pipes/&&/redirs work.
        remote = f"sudo -S -p '' /bin/sh -c {shlex.quote(inner)}"
        return remote, (password or "") + "\n"
    return inner, None


async def ssh_exec(
    server: Dict[str, Any],
    command: str,
    *,
    timeout: float = 60.0,
    cwd: Optional[str] = None,
    sudo: bool = False,
    connect_timeout: float = 15.0,
    resolver: Optional[Resolver] = None,
) -> Tuple[int, str, str]:
    """Run *command* on *server*; return ``(exit_code, stdout, stderr)``.

    ``sudo=True`` runs it under ``sudo -S`` with the server's stored password
    fed on stdin — the agent never sees the password.
    """
    async with _open(server, connect_timeout=connect_timeout, resolver=resolver) as conn:
        remote, run_input = _remote_command(
            command, cwd=cwd, sudo=sudo, password=server.get("password")
        )
        result = await asyncio.wait_for(
            conn.run(remote, input=run_input, check=False), timeout=timeout
        )
        rc = result.exit_status
        # exit_status is None on signal termination; map to 137-style non-zero.
        if rc is None:
            rc = 128 + int(result.exit_signal[1]) if getattr(result, "exit_signal", None) else 1
        return int(rc), str(result.stdout or ""), str(result.stderr or "")


async def ssh_test_connection(
    server: Dict[str, Any],
    *,
    connect_timeout: float = 15.0,
    resolver: Optional[Resolver] = None,
) -> Dict[str, Any]:
    """Open a connection and run a trivial command to verify reachability +
    auth. Returns ``{success, latency_ms, error?, hops?}`` — never raises.

    ``hops`` names the dial order actually used, so a failed test tells the user
    *where* it broke instead of just "it broke".
    """
    start = time.monotonic()
    hops: List[str] = []
    try:
        hops = [str(h.get("name") or h.get("host") or "?") for h in resolve_chain(server, resolver)]
    except SSHConfigError as exc:
        return {"success": False, "error": f"config: {exc}"}
    try:
        async with _open(server, connect_timeout=connect_timeout, resolver=resolver) as conn:
            r = await asyncio.wait_for(
                conn.run("echo geny-ssh-ok", check=False), timeout=connect_timeout
            )
            latency_ms = (time.monotonic() - start) * 1000.0
            ok = (r.exit_status == 0) and ("geny-ssh-ok" in str(r.stdout or ""))
            out: Dict[str, Any] = {
                "success": bool(ok),
                "latency_ms": round(latency_ms, 1),
                "hops": hops,
            }
            if not ok:
                out["error"] = "connected but test command failed"
            return out
    except SSHUnavailableError as exc:
        return {"success": False, "error": str(exc), "hops": hops}
    except SSHConfigError as exc:
        return {"success": False, "error": f"config: {exc}", "hops": hops}
    except asyncio.TimeoutError:
        return {"success": False, "error": f"timed out after {connect_timeout:.0f}s", "hops": hops}
    except ConnectionError as exc:  # a named jump host failed — say which
        return {"success": False, "error": str(exc), "hops": hops}
    except Exception as exc:  # noqa: BLE001 — auth/network/etc → friendly string
        return {"success": False, "error": _friendly_error(exc), "hops": hops}


async def sftp_put(
    server: Dict[str, Any],
    local_path: str,
    remote_path: str,
    *,
    connect_timeout: float = 15.0,
    resolver: Optional[Resolver] = None,
) -> None:
    async with _open(server, connect_timeout=connect_timeout, resolver=resolver) as conn:
        async with conn.start_sftp_client() as sftp:
            await sftp.put(local_path, remote_path)


async def sftp_get(
    server: Dict[str, Any],
    remote_path: str,
    local_path: str,
    *,
    connect_timeout: float = 15.0,
    resolver: Optional[Resolver] = None,
) -> None:
    async with _open(server, connect_timeout=connect_timeout, resolver=resolver) as conn:
        async with conn.start_sftp_client() as sftp:
            await sftp.get(remote_path, local_path)


def _friendly_error(exc: Exception) -> str:
    """Map common asyncssh/OS errors to a short, safe message (no secrets)."""
    name = type(exc).__name__
    msg = str(exc).strip() or name
    lowered = msg.lower()
    if "permission denied" in lowered or "authentication" in lowered or name == "PermissionDenied":
        return "authentication failed (check user / password / key)"
    if "name or service not known" in lowered or "nodename nor servname" in lowered:
        return "host not found (check host / DNS)"
    if "connection refused" in lowered:
        return "connection refused (is SSH listening on that host/port?)"
    if "timed out" in lowered or name == "TimeoutError":
        return "connection timed out"
    return f"{name}: {msg}"


__all__ = [
    "SSHUnavailableError",
    "SSHConfigError",
    "MAX_JUMP_DEPTH",
    "Resolver",
    "jump_names",
    "resolve_chain",
    "ssh_exec",
    "ssh_test_connection",
    "sftp_put",
    "sftp_get",
]
