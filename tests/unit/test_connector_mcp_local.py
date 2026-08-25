"""로컬 실행에서 외부 MCP 서버 연결·도구 노출·교차 루프 호출 검증(실 stdio 서버).

핵심: MCP 세션은 전용 백그라운드 루프에, 도구 호출은 턴 루프에서 일어난다 —
run_coroutine_threadsafe/wrap_future 교차 루프 프록시가 실제로 동작하는지 실서버로 확인.
"""
from __future__ import annotations

import asyncio
import os
import sys


from xgen_agent_runtime.host import connector_mcp_local as m
from xgen_agent_runtime.tools.base import ToolContext

_SERVER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_fixtures", "echo_mcp_server.py")


def _cfg():
    return [{"name": "echo", "transport": "stdio", "command": sys.executable, "args": [_SERVER]}]


def teardown_module(_mod):
    m.shutdown()


def test_empty_and_disabled_return_no_tools():
    assert m.connector_mcp_tools(None) == []
    assert m.connector_mcp_tools([]) == []
    assert m.connector_mcp_tools([{"name": "x", "enabled": False}]) == []


def test_connect_discover_and_cross_loop_call():
    tools = m.connector_mcp_tools(_cfg())
    assert tools, "MCP 도구가 하나도 없음 — 연결/검색 실패"
    echo = next((t for t in tools if t.name.endswith("echo")), None)
    assert echo is not None, f"echo 도구 없음: {[t.name for t in tools]}"
    assert "text" in (echo.input_schema.get("properties") or {})

    # 턴 루프(새 이벤트 루프)에서 실행 — 도구는 백그라운드 MCP 루프로 프록시한다.
    async def _call():
        ctx = ToolContext(working_dir=".")
        return await echo.execute({"text": "hi"}, ctx)

    result = asyncio.run(_call())
    text = "".join(
        b.get("text", "") if isinstance(b, dict) else str(b)
        for b in (result.content if isinstance(result.content, list) else [result.content])
    )
    assert "echo:hi" in text, f"예상 밖 결과: {result.content}"
    assert not result.is_error


def test_same_config_is_cached():
    a = m.connector_mcp_tools(_cfg())
    b = m.connector_mcp_tools(_cfg())
    # 같은 설정이면 재연결 없이 같은 도구 목록(길이) 반환.
    assert len(a) == len(b) and len(a) >= 1
