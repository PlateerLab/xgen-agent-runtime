"""로컬 CLI 내장 MCP shim — 실제 서브프로세스로 프로토콜을 검증한다.

이 shim 은 CLI(claude_code/codex)가 로컬 턴에서 도구를 보는 **유일한 경로**다.
stdio JSON-RPC 를 서버의 connector-MCP RPC 엔드포인트로 전달하고, 응답을 그대로
돌려준다. 여기서 깨지면 로컬 CLI 는 도구가 하나도 없는 채로 답한다 — 그래서
목(mock)이 아니라 진짜 프로세스 + 진짜 HTTP 서버로 검증한다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple


def _run_shim(lines: str, *, port: int, token: str = "tok-abc", path: str = "/rpc") -> List[Any]:
    env = dict(
        os.environ,
        XGEN_MCP_URL=f"http://127.0.0.1:{port}",
        XGEN_MCP_PATH=path,
        XGEN_MCP_TOKEN=token,
        XGEN_MCP_TIMEOUT_S="10",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "xgen_agent_runtime.host.cli_mcp_shim"],
        input=lines,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]


class _Recorder:
    """요청을 기록하고 정해진 응답을 주는 최소 HTTP 서버."""

    def __init__(self, responder) -> None:
        self.seen: List[Tuple[str, str, str]] = []
        recorder = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                envelope: Dict[str, Any] = json.loads(self.rfile.read(length).decode())
                recorder.seen.append(
                    (self.path, self.headers.get("Authorization", ""), envelope.get("method", ""))
                )
                status, payload = responder(envelope)
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:  # 조용히
                return

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()


def test_shim_forwards_envelopes_with_auth_and_path() -> None:
    def responder(envelope):
        return 200, {
            "jsonrpc": "2.0",
            "id": envelope.get("id"),
            "result": {"tools": [{"name": "mcp_local_BrowserTabs"}]},
        }

    rec = _Recorder(responder)
    try:
        out = _run_shim(
            '{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n',
            port=rec.port,
            path="/api/internal/connector-mcp/42/rpc",
        )
    finally:
        rec.close()
    assert out[0]["result"]["tools"][0]["name"] == "mcp_local_BrowserTabs"
    path, auth, method = rec.seen[0]
    assert path == "/api/internal/connector-mcp/42/rpc"
    assert auth == "Bearer tok-abc"
    assert method == "tools/list"


def test_shim_pushes_list_changed_when_surface_changed() -> None:
    """turns 중 도구가 생기면(_meta.genyToolsChanged) 같은 턴에 재조회하게 통지."""

    def responder(envelope):
        return 200, {
            "jsonrpc": "2.0",
            "id": envelope.get("id"),
            "result": {"content": [{"type": "text", "text": "ok"}],
                       "_meta": {"genyToolsChanged": True}},
        }

    rec = _Recorder(responder)
    try:
        out = _run_shim('{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{}}\n', port=rec.port)
    finally:
        rec.close()
    assert out[0]["result"]["content"][0]["text"] == "ok"
    assert out[1] == {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}


def test_shim_turns_http_errors_into_jsonrpc_errors() -> None:
    """CLI 는 프로토콜 응답만 이해한다 — 401/500 이 크래시로 새어 나가면 안 된다."""

    def responder(envelope):
        return 401, {"detail": "bearer token required"}

    rec = _Recorder(responder)
    try:
        out = _run_shim('{"jsonrpc":"2.0","id":3,"method":"tools/list"}\n', port=rec.port)
    finally:
        rec.close()
    assert out[0]["id"] == 3
    assert out[0]["error"]["code"] == -32603
    assert "401" in out[0]["error"]["message"]


def test_shim_reports_misconfiguration_and_bad_input() -> None:
    # 토큰이 없으면 전송하지 않고 프로토콜 에러로 알린다.
    env = dict(os.environ, XGEN_MCP_URL="http://127.0.0.1:1", XGEN_MCP_PATH="/rpc", XGEN_MCP_TOKEN="")
    proc = subprocess.run(
        [sys.executable, "-m", "xgen_agent_runtime.host.cli_mcp_shim"],
        input='{"jsonrpc":"2.0","id":1,"method":"tools/list"}\nnot-json\n',
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0
    lines = [json.loads(x) for x in proc.stdout.strip().splitlines() if x.strip()]
    assert lines[0]["error"]["code"] == -32603 and "misconfigured" in lines[0]["error"]["message"]
    # 깨진 입력 줄은 건너뛰고 프로세스는 살아 있는다(응답 1건뿐).
    assert len(lines) == 1 and "malformed JSON" in proc.stderr
