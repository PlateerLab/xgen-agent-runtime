"""Per-session SSH server credential store.

The host hands the user's configured servers to the session via
``ToolContext.extras["ssh"]["servers"]``. **When that key is present it is the
whole truth for the turn** — including when it is an empty list, which is how a
host says "this user turned SSH off / has no servers".

A file at ``<storage_path>/ssh/servers.json`` is only a fallback for a session
with no host injection at all (the executor run standalone). Writing to it is
opt-in (``extras["ssh"]["persist"] = True``) and **off by default**, because the
file holds decrypted credentials at rest for the life of the session directory.
A host that re-injects every turn — which is what makes a rotated password or a
revoked server take effect on the next turn — gains nothing from that file and
would only risk a stale copy outliving the change. So when persistence is off we
also delete any file an earlier version left behind.

The split that keeps secrets away from the agent:

* :meth:`SSHServerStore.list_public` returns only non-secret metadata (name,
  host, port, user, description, auth kind, jump path) — safe to show the model.
* :meth:`SSHServerStore.resolve` returns the FULL record (including password /
  private key), used ONLY inside a tool to open the connection. The agent
  passes a server *name*; it never sees or handles the credential.

:meth:`SSHServerStore.resolve` doubles as the **jump-host resolver** handed to
``_ssh``: a record reached through a bastion names that bastion, and the
connection layer looks it up here. That is why the store is passed down rather
than a single record — a bastion is just another entry, described once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_SECRET_KEYS = ("password", "private_key", "passphrase")


def _server_dir(storage_path: Optional[str]) -> Optional[Path]:
    if not storage_path:
        return None
    return Path(storage_path).resolve() / "ssh"


def _auth_kind(server: Dict[str, Any]) -> str:
    has_pw = bool(server.get("password"))
    has_key = bool(server.get("private_key"))
    if has_pw and has_key:
        return "password+key"
    if has_key:
        return "key"
    if has_pw:
        return "password"
    return "none"


class SSHServerStore:
    """A per-session view of the configured SSH servers, keyed by name."""

    def __init__(self, servers: Optional[List[Dict[str, Any]]] = None) -> None:
        self._by_name: Dict[str, Dict[str, Any]] = {}
        for s in servers or []:
            name = str(s.get("name") or "").strip()
            if not name:
                continue
            self._by_name[name] = dict(s)

    # ── construction ────────────────────────────────────────────────
    @classmethod
    def from_context(cls, context: Any) -> "SSHServerStore":
        """Build the store for a session.

        The host-injected list (``context.extras["ssh"]["servers"]``) wins
        whenever the ``ssh`` extra is present — **even if it is empty**. That
        distinction is the whole point: "the host says zero servers" and "no host
        is speaking" are different states, and reading a stale file for the first
        one would let a revoked server keep working after the user removed it.

        The file is consulted only in the second state, and written only when the
        host opts in (see the module docstring).
        """
        extras = getattr(context, "extras", None) or {}
        ssh_extra = extras.get("ssh") if isinstance(extras, dict) else None
        has_extra = isinstance(ssh_extra, dict) and "servers" in ssh_extra
        injected = ssh_extra.get("servers") if has_extra else None
        if has_extra and not isinstance(injected, list):
            injected = []

        sdir = _server_dir(getattr(context, "storage_path", None))

        if has_extra:
            if sdir is not None:
                if ssh_extra.get("persist") and injected:
                    cls._write_file(sdir, injected)
                else:
                    # Decrypted credentials must not outlive the turn that needed
                    # them when nobody asked us to keep them.
                    cls._remove_file(sdir)
            return cls(injected or [])

        # No host is speaking — fall back to whatever record exists on disk.
        if sdir is not None:
            servers = cls._read_file(sdir)
            if servers:
                return cls(servers)
        return cls([])

    # ── lookups ─────────────────────────────────────────────────────
    def names(self) -> List[str]:
        """Targetable server names (what the agent is told it can use)."""
        return [n for n, s in self._by_name.items() if s.get("listable") is not False]

    def is_empty(self) -> bool:
        return not self._by_name

    def resolve(self, name: str) -> Optional[Dict[str, Any]]:
        """Full record (WITH secrets) for internal connection use only.

        Resolves hop-only records too — this doubles as the jump-host resolver,
        and a route must work even when one of its bastions is not itself a
        permitted destination. Destination permission is enforced by
        :meth:`target`, which the tools use.
        """
        return self._by_name.get(str(name).strip())

    def target(self, name: str) -> Optional[Dict[str, Any]]:
        """Full record, but only for servers the agent may aim at.

        A hop-only record (``listable: false``) is not a destination — returning
        it here would let the agent run commands on a bastion the user disabled.
        """
        record = self.resolve(name)
        if record is not None and record.get("listable") is False:
            return None
        return record

    def list_public(self) -> List[Dict[str, Any]]:
        """Non-secret metadata for the servers the agent may target.

        ``via`` is the declared jump path. The agent needs to see it: when a
        command fails, "reached through bastion" is the difference between
        debugging the command and debugging the network.

        Records flagged ``listable: false`` are omitted. Those are hops the host
        included **only** so a jump path still resolves — typically a bastion the
        user switched off. Hiding them keeps a disabled server from being used as
        a destination while the route through it keeps working; without the split,
        turning off one bastion would silently kill every host behind it.
        """
        from xgen_agent_runtime.tools._ssh import jump_names

        out: List[Dict[str, Any]] = []
        for name, s in self._by_name.items():
            if s.get("listable") is False:
                continue
            try:
                via = jump_names(s)
            except Exception:  # noqa: BLE001 — a malformed record must not hide the rest
                via = []
            out.append(
                {
                    "name": name,
                    "host": s.get("host", ""),
                    "port": int(s.get("port") or 22),
                    "user": s.get("user") or s.get("username") or "",
                    "description": s.get("description", "") or "",
                    "auth": _auth_kind(s),
                    "via": via,
                }
            )
        return out

    # ── file I/O ────────────────────────────────────────────────────
    @staticmethod
    def _read_file(sdir: Path) -> List[Dict[str, Any]]:
        path = sdir / "servers.json"
        try:
            if not path.is_file():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if isinstance(data, dict):  # tolerate {"servers": [...]}
            data = data.get("servers")
        return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []

    @staticmethod
    def _write_file(sdir: Path, servers: List[Dict[str, Any]]) -> None:
        try:
            sdir.mkdir(parents=True, exist_ok=True)
            path = sdir / "servers.json"
            tmp = sdir / "servers.json.tmp"
            tmp.write_text(
                json.dumps(list(servers), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Credentials on disk — keep them owner-only.
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, path)
        except OSError:
            pass  # best-effort record; the in-memory view still works

    @staticmethod
    def _remove_file(sdir: Path) -> None:
        """Drop a previously persisted credential file (best effort)."""
        for name in ("servers.json", "servers.json.tmp"):
            try:
                (sdir / name).unlink()
            except OSError:
                pass


__all__ = ["SSHServerStore"]
