"""로컬 실행(사이드카)에서 외부 MCP 서버를 **런타임이 직접** 연결해 에이전트에 노출한다.

서버 실행 경로는 커넥터의 reverse-WS 브릿지로 MCP 도구를 프록시하지만, 로컬 실행에선
사이드카가 곧 로컬이므로 런타임 MCP 매니저(``tools.mcp.manager``)로 직접 스폰/연결한다.

**이벤트 루프 경계**: 턴은 ``runner.stream_turn`` 이 만드는 사설 루프에서 돌고, 도구 목록은
그 루프가 생기기 **전**(``build_pipeline`` → ``host.build_connector_mcp_tools``)에 만들어진다.
그래서 MCP 연결은 프로세스 상주 **전용 백그라운드 루프**에서 유지하고, 도구 호출은
``run_coroutine_threadsafe`` 로 그 루프에 넘겨 턴 루프에서 ``await`` (``wrap_future``) 한다 —
MCP 세션의 루프와 턴 루프가 달라도 안전하다. 연결은 설정 해시로 캐시해 매 턴 재스폰을 피한다.

**방어**: 어떤 실패도 턴을 깨지 않는다 — 연결/검색 실패는 빈 목록으로 degrade(무 MCP).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Mapping, Optional

from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger("xgen_agent_runtime.host.connector_mcp_local")

_CONNECT_TIMEOUT_S = 40.0
_CALL_TIMEOUT_S = 180.0


def _norm_configs(raw: Any) -> List[Dict[str, Any]]:
    """커넥터가 넘긴 MCP 서버 설정(list of dict)을 정규화 — enabled=False/이름 없음 제외."""
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for c in raw:
        if not isinstance(c, Mapping):
            continue
        name = str(c.get("name") or "").strip()
        if not name or c.get("enabled") is False:
            continue
        url = str(c.get("url") or "").strip()
        transport = str(c.get("transport") or ("http" if url else "stdio")).strip() or "stdio"
        out.append(
            {
                "name": name,
                "transport": transport,
                "command": str(c.get("command") or "").strip(),
                "args": [str(x) for x in (c.get("args") or [])],
                "env": {str(k): str(v) for k, v in (c.get("env") or {}).items()},
                "url": url,
                "headers": {str(k): str(v) for k, v in (c.get("headers") or {}).items()},
            }
        )
    out.sort(key=lambda c: c["name"])
    return out


def _config_key(configs: List[Dict[str, Any]]) -> str:
    return json.dumps(configs, sort_keys=True, ensure_ascii=False)


def _to_runtime_config(c: Mapping[str, Any]) -> Any:
    from xgen_agent_runtime.tools.mcp.manager import MCPServerConfig

    return MCPServerConfig(
        name=str(c.get("name") or ""),
        command=str(c.get("command") or ""),
        args=list(c.get("args") or []),
        env=dict(c.get("env") or {}),
        transport=str(c.get("transport") or "stdio"),
        url=str(c.get("url") or ""),
        headers=dict(c.get("headers") or {}),
    )


class _ProxyMcpTool(Tool):
    """MCPToolAdapter 를 감싸 실행을 **백그라운드 MCP 루프**로 프록시한다.

    스키마/이름은 어댑터의 동기 속성에서 그대로 가져오고, ``execute`` 만 교차 루프로
    넘긴다(어댑터의 async execute 를 MCP 세션 루프에서 돌리고 턴 루프에서 wrap_future).
    """

    def __init__(self, adapter: Tool, loop: asyncio.AbstractEventLoop) -> None:
        self._a = adapter
        self._loop = loop

    @property
    def name(self) -> str:
        return self._a.name

    @property
    def description(self) -> str:
        return self._a.description

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._a.input_schema

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        try:
            return self._a.capabilities(input)
        except Exception:  # noqa: BLE001
            return ToolCapabilities()

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            fut = asyncio.run_coroutine_threadsafe(self._a.execute(input, context), self._loop)
            return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=_CALL_TIMEOUT_S)
        except asyncio.TimeoutError:
            return ToolResult(content=f"MCP 도구 '{self._a.name}' 시간 초과", is_error=True)
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"MCP 도구 '{self._a.name}' 실패: {e}", is_error=True)


class _LocalConnectorMcp:
    """프로세스 상주 싱글턴 — 백그라운드 asyncio 루프에 MCP 매니저를 얹어 유지한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._manager: Any = None
        self._key: str = ""
        self._tools: List[Tool] = []

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop and self._loop.is_running():
            return self._loop
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run, name="connector-mcp-local", daemon=True)
        t.start()
        self._loop = loop
        self._thread = t
        return loop

    def _submit(self, coro: Any, timeout: float) -> Any:
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=timeout)

    def tools_for(self, raw_configs: Any) -> List[Tool]:
        """설정에 맞춰 (재)연결하고 프록시 도구 목록을 돌려준다. 실패는 [] 로 degrade."""
        configs = _norm_configs(raw_configs)
        if not configs:
            # 설정이 사라졌으면 기존 연결을 정리한다.
            with self._lock:
                if self._manager is not None:
                    self._teardown()
            return []
        key = _config_key(configs)
        with self._lock:
            if key == self._key and self._tools:
                return list(self._tools)
            try:
                self._reconcile(configs, key)
                return list(self._tools)
            except Exception as exc:  # noqa: BLE001 — 절대 턴을 깨지 않는다
                logger.warning("connector_mcp_local: 연결 실패로 무 MCP 진행: %s", exc)
                return []

    def _reconcile(self, configs: List[Dict[str, Any]], key: str) -> None:
        from xgen_agent_runtime.tools.mcp.manager import MCPManager

        loop = self._ensure_loop()
        # 설정이 바뀌었으면 기존 연결을 내리고 새로 세운다(단순·안전; 서버 수 적음).
        if self._manager is not None:
            self._teardown()
        manager = MCPManager()
        rt_configs = {c["name"]: _to_runtime_config(c) for c in configs}
        self._submit(manager.connect_all(rt_configs), _CONNECT_TIMEOUT_S)
        adapters: List[Tool] = self._submit(manager.discover_tools(), _CONNECT_TIMEOUT_S)
        self._manager = manager
        self._key = key
        self._tools = [_ProxyMcpTool(a, loop) for a in (adapters or [])]
        logger.info(
            "connector_mcp_local: MCP 서버 %d개 연결, 도구 %d개 노출",
            len(configs),
            len(self._tools),
        )

    def _teardown(self) -> None:
        mgr = self._manager
        self._manager = None
        self._key = ""
        self._tools = []
        if mgr is not None:
            try:
                self._submit(mgr.disconnect_all(), 10.0)
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        with self._lock:
            self._teardown()
            loop = self._loop
            self._loop = None
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:  # noqa: BLE001
                pass


_INSTANCE: Optional[_LocalConnectorMcp] = None


def _instance() -> _LocalConnectorMcp:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _LocalConnectorMcp()
    return _INSTANCE


def connector_mcp_tools(raw_configs: Any) -> List[Tool]:
    """커넥터가 넘긴 외부 MCP 서버 설정 → 에이전트에 붙일 프록시 도구 목록(방어적)."""
    try:
        return _instance().tools_for(raw_configs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("connector_mcp_local: tools 빌드 실패(무시): %s", exc)
        return []


def shutdown() -> None:
    """데몬 종료 시 MCP 연결·백그라운드 루프 정리."""
    if _INSTANCE is not None:
        _INSTANCE.shutdown()
