#!/usr/bin/env python3
"""Stdio MCP server for **local CLI turns** — the connector/desktop counterpart
of xgen-workflow's ``scripts/connector_mcp_bridge.py``.

Why this exists
---------------
CLI backends (claude_code / codex) own their own agent loop and can only see
external tools through MCP. Their **native** tools are fully blocked (server and
local alike), so what this shim carries is not a supplement — it is the agent's
entire tool surface: files, shell, documents, browser, memory, WorkflowSelf.

The shim is a thin, dumb proxy. It does not build or filter tools; it forwards
each JSON-RPC envelope to whatever endpoint its env points at:

  * **Local turn** — ``http://127.0.0.1:<port>/rpc``, the loopback server that
    :class:`~xgen_agent_runtime.host.local_tool_mcp.LocalToolMcpServer` opened
    **inside the turn process**. That server owns the live ``ToolRegistry`` and
    ``ToolContext`` this turn assembled, so the CLI sees exactly the SDK
    surface, executing at exactly the SDK location (``sandbox=None`` → this PC,
    ``working_dir`` = the synced workspace, guarded by ``allowed_paths``).
  * **Server turn** — the workflow repo's own stdio bridge plays this role
    (``scripts/connector_mcp_bridge.py``); same protocol, same shape, but its
    ToolContext carries the runner sandbox so the same tools run there instead.

Design (mirrors the proven server shim):
  - **Stdlib only** (``urllib``) and synchronous line-by-line stdio: one
    JSON-RPC envelope per line, no pipelining.
  - HTTP/transport failures become MCP-shaped JSON-RPC errors so the CLI never
    sees a crashed bridge.
  - ``tools/call`` responses carrying ``_meta.genyToolsChanged`` trigger a
    ``notifications/tools/list_changed`` push so a tool created mid-turn is
    callable in the same turn.

Env (set by the host when spawning the CLI with --mcp-config):
  XGEN_MCP_URL        — base URL (REQUIRED; loopback origin for local turns)
  XGEN_MCP_PATH       — RPC path (REQUIRED)
  XGEN_MCP_TOKEN      — ephemeral bearer token (REQUIRED)
  XGEN_MCP_TIMEOUT_S  — per-RPC HTTP timeout (default 300)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

_URL = (os.environ.get("XGEN_MCP_URL", "") or "").rstrip("/")
_PATH = os.environ.get("XGEN_MCP_PATH", "") or ""
_TOKEN = os.environ.get("XGEN_MCP_TOKEN", "") or ""
_TIMEOUT = float(os.environ.get("XGEN_MCP_TIMEOUT_S", "300") or 300)


def _arm_parent_death_signal() -> None:
    """Best-effort (Linux): SIGTERM us if the parent CLI dies, so a crashed CLI
    doesn't orphan this bridge holding an HTTP connection open."""
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:  # noqa: BLE001 — not Linux / libc missing; EOF still cleans up
        pass


def _write_response(response: dict) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _err(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _endpoint() -> str:
    path = _PATH if _PATH.startswith("/") else f"/{_PATH}"
    return f"{_URL}{path}"


def _forward(envelope: dict) -> dict:
    """POST one JSON-RPC envelope to the server, translating transport failures
    into MCP-shaped JSON-RPC errors so the CLI never sees a crash."""
    req_id = envelope.get("id")
    if not _URL or not _PATH or not _TOKEN:
        return _err(req_id, -32603, "bridge misconfigured: missing XGEN_MCP_URL/PATH/TOKEN")

    payload = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        _endpoint(),
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return _err(req_id, -32603, f"HTTP {exc.code}: {body[:200]}")
        # Pass through only a real JSON-RPC envelope (the endpoint's own error
        # response). Anything else — e.g. FastAPI's {"detail": ...} on a 401 —
        # is wrapped so the CLI never sees a non-protocol body.
        if isinstance(parsed, dict) and (
            "error" in parsed or "result" in parsed or parsed.get("jsonrpc")
        ):
            return parsed
        detail = parsed.get("detail") if isinstance(parsed, dict) else None
        return _err(req_id, -32603, f"HTTP {exc.code}: {detail or str(parsed)[:200]}")
    except urllib.error.URLError as exc:
        return _err(req_id, -32603, f"transport error: {exc.reason}")
    except Exception as exc:  # noqa: BLE001
        return _err(req_id, -32603, f"bridge error: {exc}")

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return _err(req_id, -32603, f"invalid JSON response: {body[:200]}")


def main() -> int:
    _arm_parent_death_signal()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"cli_mcp_shim: malformed JSON: {line[:120]}\n")
            sys.stderr.flush()
            continue
        response = _forward(envelope)
        _write_response(response)

        # Same-turn tool activation: if a tools/call changed the advertised
        # surface the server stamps result._meta.genyToolsChanged; push the MCP
        # list_changed notification so the CLI re-lists within the same turn.
        try:
            if (
                envelope.get("method") == "tools/call"
                and isinstance(response.get("result"), dict)
                and (response["result"].get("_meta") or {}).get("genyToolsChanged")
            ):
                _write_response({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        except Exception:  # noqa: BLE001 — a notification must not break the proxy
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
