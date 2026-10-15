"""CLI 백엔드(claude_code·codex)의 도구 표면 — SDK 경로와 **같은 객체**다.

왜 있나
-------
CLI 는 자기 에이전트 루프를 소유하고 외부 도구를 MCP 로만 받는다. 예전 서버는 그 MCP 표면을 **따로
조립**했다(xgen-workflow 의 CLI 브릿지 재조립). 같은 에이전트가 provider 에 따라 다른 도구를 받았다:

* 게스트·고정본 턴에서 SDK 는 Bash·Write·Edit 를 뺐는데 CLI 는 그대로 줬다.
* SDK 에만 자기확장의 문(SelfExtendGuide)과 숨김 목록이 있었다.
* 기기 도구 스키마·이름 충돌 규칙·메모리 노출 방식이 달랐다.
* 실행 경로가 Stage 10 을 거치지 않아 읽기 장부·반복 실패 차단·거부 존중이 CLI 에서만 빠졌다.

그래서 턴 조립(:class:`~xgen_agent_runtime.host.turn_executor.AgentTurnExecutor`)은 provider 와 무관하게
**하나의** 레지스트리와 도구 컨텍스트를 만들고, CLI 턴이면 그것을 이 객체로 묶어 호스트에 넘긴다. 호스트의
MCP 브릿지는 이 객체를 **그대로** 광고(:meth:`tools_list`)하고 실행(:meth:`call`)한다. 광고는 레지스트리의
노출분(턴 1 표면 + 이번 턴에 연 것), 실행은 Stage 10 과 같은 함수(``ToolStage.dispatch_calls``)다.

실행은 **턴의 이벤트 루프**에서 한다(:meth:`bind_loop`). 메모리 provider·기기 도구 같은 객체가 그 루프에서
만들어지고 쓰이므로(SDK 경로의 Stage 10 이 그렇다), 브릿지 요청이 도착한 서빙 루프에서 부르면 루프 소속이
어긋난다. 파이프라인이 CLI 출력을 기다리는 동안 턴 루프는 비어 있어 넘겨받은 호출을 바로 돌린다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("xgen_agent_runtime.host.tool_surface")


class TurnToolSurface:
    """한 턴의 도구 표면 — 레지스트리 + 도구 컨텍스트 + 턴 상태."""

    def __init__(
        self,
        *,
        registry: Any,
        tool_context: Any,
        state: Any,
        server_name: str = "connector",
    ) -> None:
        from xgen_agent_runtime.stages.s10_tool.artifact.default.stage import ToolStage
        from xgen_agent_runtime.tools.base import ToolContext

        self.registry = registry
        self.tool_context = tool_context if tool_context is not None else ToolContext()
        self.state = state
        self.server_name = server_name
        #: SDK 파이프라인의 Stage 10 과 같은 클래스 — 같은 가드·같은 라우터·같은 문 열기.
        self._stage = ToolStage(registry=registry, context=self.tool_context)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        #: 표면 검사(기록 복원·문 도달성)를 마지막으로 돌린 레지스트리 버전.
        self._checked_version: Any = object()
        #: CLI 는 자기 도구 사건을 스트림으로 알린다 — 여기서 파이프라인 사건으로 또 내면 화면에 두 번
        #: 뜬다. 진단용으로만 붙잡아 둔다(최근 것 몇 개).
        self.events: List[Tuple[str, Dict[str, Any]]] = []

    # ── 수명 ─────────────────────────────────────────────────────────

    def bind_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        """턴 루프를 알린다(러너가 루프를 만들 때 부르고, 닫기 전에 None 으로 부른다)."""
        self._loop = loop

    @property
    def running(self) -> bool:
        loop = self._loop
        return loop is not None and not loop.is_closed()

    # ── 광고 ─────────────────────────────────────────────────────────

    def _check_surface(self) -> None:
        """SDK Stage 3 가 표면을 굳히기 전에 하는 두 검사를 같은 함수로 — 레지스트리가 바뀔 때만.

        기록에서 모델이 쓴 도구를 다시 열고(``tools.gates.restore_from_history``), 숨긴 가족에
        보이는 문이 없으면 연다(``reachability_fixes``). CLI 는 Stage 3 에 레지스트리가 없으므로
        여기서 하지 않으면 두 경로의 첫 화면이 달라진다.
        """
        version = getattr(self.registry, "version", None)
        if version is not None and version == self._checked_version:
            return
        from xgen_agent_runtime.stages.s03_system.artifact.default.stage import (
            _enforce_gate_reachability,
            _restore_from_history,
        )

        restored = _restore_from_history(self.registry, getattr(self.state, "messages", None))
        if restored:
            self._sink("tool.surface_restored", {"opened": restored})
        repaired = _enforce_gate_reachability(self.registry)
        if repaired:
            self._sink("tool.gate_reachability_repaired", {"opened": repaired})
        self._checked_version = getattr(self.registry, "version", None)

    def exposed_names(self) -> Tuple[str, ...]:
        """지금 모델에게 보이는 도구 이름 — 표면 변화 감지(list_changed)에 쓴다."""
        try:
            self._check_surface()
            return tuple(sorted(getattr(t, "name", "") for t in self.registry.list_exposed()))
        except Exception:  # noqa: BLE001
            return ()

    def tools_list(self) -> List[Dict[str, Any]]:
        """MCP ``tools/list`` 결과 — 노출분의 이름·설명·입력 스키마."""
        try:
            self._check_surface()
        except Exception:  # noqa: BLE001 — 검사가 목록을 막지 않는다
            logger.debug("tool surface: 표면 검사 실패", exc_info=True)
        from xgen_agent_runtime.tools.definition import api_definition

        out: List[Dict[str, Any]] = []
        for tool in self.registry.list_exposed():
            try:
                # SDK Stage 3 가 보내는 것과 같은 함수 — 이름·설명·스키마가 글자 하나까지 같다.
                fmt = api_definition(tool)
            except Exception:  # noqa: BLE001 — 도구 하나가 목록 전체를 죽이지 않는다
                logger.warning(
                    "tool surface: %s 스키마 생성 실패 (스킵)",
                    getattr(tool, "name", "?"),
                    exc_info=True,
                )
                continue
            out.append(
                {
                    "name": fmt.get("name"),
                    "description": fmt.get("description") or "",
                    "inputSchema": fmt.get("input_schema") or {"type": "object", "properties": {}},
                }
            )
        return out

    # ── 실행 ─────────────────────────────────────────────────────────

    def _sink(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append((event_type, dict(data or {})))
        if len(self.events) > 200:
            del self.events[:100]

    async def _dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        call = {
            "tool_use_id": f"cli_{uuid.uuid4().hex[:16]}",
            "tool_name": str(name),
            "tool_input": dict(arguments or {}),
        }
        try:
            results = await self._stage.dispatch_calls([call], self.state, add_event=self._sink)
        except Exception as exc:  # noqa: BLE001 — 실행 실패도 모델에게 돌아가는 결과다
            logger.warning("tool surface: %s 실행 실패: %s", name, exc, exc_info=True)
            return {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True}
        result = results[0] if results else {"content": "", "is_error": True}
        return to_mcp_result(result)

    async def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP ``tools/call`` — 어느 루프·스레드에서 불러도 된다. 결과는 MCP 모양."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return {
                "content": [{"type": "text", "text": "Error: this turn is no longer running."}],
                "isError": True,
            }
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            return await self._dispatch(name, arguments)
        future = asyncio.run_coroutine_threadsafe(self._dispatch(name, arguments), loop)
        return await asyncio.wrap_future(future)


def to_mcp_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 10 ``tool_result`` 블록 → MCP ``tools/call`` 결과(텍스트·이미지 블록)."""
    content = result.get("content")
    is_error = bool(result.get("is_error"))
    blocks: List[Dict[str, Any]] = []
    if isinstance(content, str):
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
                continue
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "image":
                source = block.get("source") if isinstance(block.get("source"), dict) else {}
                data = source.get("data") or block.get("data")
                if data:
                    blocks.append(
                        {
                            "type": "image",
                            "data": str(data),
                            "mimeType": str(
                                source.get("media_type") or block.get("mimeType") or "image/png"
                            ),
                        }
                    )
                continue
            if kind == "text" or "text" in block:
                blocks.append({"type": "text", "text": str(block.get("text") or "")})
            else:
                blocks.append(
                    {"type": "text", "text": json.dumps(block, ensure_ascii=False, default=str)}
                )
    elif content is not None:
        blocks.append(
            {"type": "text", "text": json.dumps(content, ensure_ascii=False, default=str)}
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return {"content": blocks, "isError": is_error}


__all__ = ["TurnToolSurface", "to_mcp_result"]
