"""로컬 CLI 턴을 위한 **루프백 MCP 서버** — 런타임 레지스트리를 그대로 노출한다.

왜 필요한가
-----------
CLI 백엔드(claude_code/codex)는 자기 에이전트 루프를 소유해서 런타임의
``ToolRegistry`` 를 **보지 못한다**(turn_executor 가 CLI 경로에서 registry 를
버린다). 그래서 CLI 에게 도구를 주는 표준 경로는 MCP 뿐이다. 서버 실행은 그
표면을 workflow 의 stdio 브릿지로 만들어 주지만, 로컬 실행에는 그런 게 없었다.

이 모듈이 그 빈자리를 채운다: **턴 프로세스(사이드카) 안에서** 살아 있는
레지스트리와 ToolContext 를 127.0.0.1 루프백 JSON-RPC 로 노출하고, CLI 는
:mod:`xgen_agent_runtime.host.cli_mcp_shim` 을 통해 그 앞단에 붙는다.

이 구조라야 지켜지는 것들:
  * **도구 표면이 SDK 경로와 동일** — 같은 host 메서드로 조립한 같은 레지스트리다.
  * **실행 위치가 정확** — 같은 ``ToolContext``(sandbox=None, working_dir=동기화
    workspace, allowed_paths 가드)를 쓰므로 파일이 엉뚱한 곳에 생기지 않는다.
  * **세션 유지** — 도구 실행을 전용 이벤트 루프 **하나**에 마샬링한다. an-web
    브라우저 세션처럼 루프에 묶인 상태가 호출 간에 살아남아야 하기 때문
    (호출마다 asyncio.run 하면 매번 새 탭이 된다).

프로토콜은 서버 브릿지(controller/tools/connectorMcpInternal.py)와 **동일**하게
맞춘다 — 같은 CLI 가 두 경로에서 같은 응답을 봐야 한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("xgen_agent_runtime.host.local_tool_mcp")

#: 서버 브릿지와 같은 MCP 버전 — 두 경로의 CLI 동작이 갈리면 안 된다.
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "xgen-local-tools"
SERVER_VERSION = "1"

#: 도구 1회 실행 상한(초). CLI 자체 타임아웃이 더 짧게 잡히므로 여기서는
#: '영원히 매달리지 않는다' 정도의 안전망만 둔다(Bash 기본 상한이 10분).
_CALL_TIMEOUT_S = 3600.0


def _text_of(result: Any) -> Tuple[str, bool]:
    """ToolResult → (text, is_error). 서버 execute_cli_bridge_tool 과 같은 규약."""
    content = getattr(result, "content", "")
    if isinstance(content, str) and content:
        text = content
    elif content:
        text = json.dumps(content, ensure_ascii=False, default=str)
    else:
        text = str(getattr(result, "display_text", None) or "")
    return text, bool(getattr(result, "is_error", False))


class LocalToolMcpServer:
    """살아 있는 레지스트리를 루프백 JSON-RPC(MCP)로 노출하는 서버."""

    def __init__(self, registry: Any, tool_context: Any) -> None:
        self._registry = registry
        self._tool_context = tool_context
        self._token = secrets.token_urlsafe(32)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._base_url = ""

    # ── 수명주기 ──────────────────────────────────────────────────────
    def start(self) -> Tuple[str, str]:
        """서버를 띄우고 ``(base_url, token)`` 을 돌려준다."""
        loop_ready = threading.Event()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop_ready.set()
            loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run_loop, name="xgen-local-mcp-loop", daemon=True
        )
        self._loop_thread.start()
        if not loop_ready.wait(timeout=10.0):
            raise RuntimeError("로컬 MCP 실행 루프를 시작하지 못했습니다")

        server = self  # 핸들러에서 참조

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                auth = self.headers.get("Authorization", "")
                if auth != f"Bearer {server._token}":
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    envelope = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception:  # noqa: BLE001 — 파싱 실패도 프로토콜 응답으로
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "parse error"},
                        },
                    )
                    return
                self._send(200, server._dispatch(envelope))

            def _send(self, status: int, payload: Dict[str, Any]) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:  # 접근 로그 억제
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.daemon_threads = True
        port = self._httpd.server_address[1]
        self._base_url = f"http://127.0.0.1:{port}"
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, name="xgen-local-mcp-http", daemon=True
        )
        self._http_thread.start()
        logger.info(
            "local MCP: 루프백 도구 서버 시작 %s (도구 %d종)",
            self._base_url,
            len(self._registry.list_names()),
        )
        return self._base_url, self._token

    def stop(self) -> None:
        """턴 종료 시 회수 — 절대 raise 하지 않는다(정리 실패가 턴을 깨면 안 된다)."""
        try:
            if self._httpd is not None:
                self._httpd.shutdown()
                self._httpd.server_close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:  # noqa: BLE001
            pass
        logger.info("local MCP: 루프백 도구 서버 종료")

    # ── JSON-RPC 디스패치 (서버 브릿지와 같은 메서드 집합) ─────────────
    def _dispatch(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        req_id = envelope.get("id")
        method = str(envelope.get("method") or "")
        params = envelope.get("params") or {}

        def ok(result: Any) -> Dict[str, Any]:
            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        def err(code: int, message: str) -> Dict[str, Any]:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

        if method == "initialize":
            return ok(
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": False, "subscribe": False},
                        "prompts": {"listChanged": False},
                        "logging": {},
                    },
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                }
            )
        if method in (
            "notifications/initialized",
            "notifications/cancelled",
            "logging/setLevel",
            "ping",
        ):
            return ok({})
        if method == "resources/list":
            return ok({"resources": []})
        if method == "resources/templates/list":
            return ok({"resourceTemplates": []})
        if method == "prompts/list":
            return ok({"prompts": []})

        if method == "tools/list":
            tools = []
            for api in self._registry.to_api_format(exposed_only=False):
                try:
                    tools.append(
                        {
                            "name": api["name"],
                            "description": api.get("description") or "",
                            "inputSchema": api.get("input_schema")
                            or {"type": "object", "properties": {}},
                        }
                    )
                except Exception:  # noqa: BLE001 — 하나가 목록 전체를 죽이지 않는다
                    continue
            return ok({"tools": tools})

        if method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            if not name:
                return err(-32602, "missing tool name")
            if self._registry.get(name) is None:
                return ok(
                    {
                        "content": [{"type": "text", "text": f"Tool '{name}' not found"}],
                        "isError": True,
                    }
                )
            # 같은 턴 안에서 도구를 만들어 곧바로 쓰는 흐름(ForgeTool 류) — 호출
            # 전후의 이름 집합이 달라지면 _meta 를 찍어 shim 이 list_changed 를
            # 밀게 한다. 안 찍으면 CLI 는 턴이 끝날 때까지 새 도구를 못 본다.
            before = set(self._registry.list_names())
            try:
                text, is_error = self._call_tool(name, dict(arguments))
            except Exception as exc:  # noqa: BLE001
                logger.warning("local MCP: tools/call '%s' 실패: %s", name, exc, exc_info=True)
                return err(-32603, str(exc))
            result: Dict[str, Any] = {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            }
            if set(self._registry.list_names()) != before:
                result["_meta"] = {"genyToolsChanged": True}
            return ok(result)

        return err(-32601, f"method not found: {method}")

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        """전용 루프에 마샬링해 실행 — 세션 상태(브라우저 탭 등)가 호출 간 유지된다."""
        from xgen_agent_runtime.stages.s10_tool import RegistryRouter

        loop = self._loop
        if loop is None:
            raise RuntimeError("local MCP 실행 루프가 없습니다")
        future = asyncio.run_coroutine_threadsafe(
            RegistryRouter(self._registry).route(name, arguments, self._tool_context), loop
        )
        result = future.result(timeout=_CALL_TIMEOUT_S)
        return _text_of(result)
