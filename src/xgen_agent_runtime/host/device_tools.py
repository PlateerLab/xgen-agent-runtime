"""사용자 기기 도구 — 데스크톱·CLI·VSCode(``local``), 휴대폰(``mobile``), 웹 [폴더](``web``).

기기 앱은 자기 도구를 MCP 카탈로그(``{server, name, description, inputSchema}``)로 서버에 올리고, 호출은
서버가 기기로 돌려보낸다(xgen-workflow ``ConnectorMcpBridge``). 이 모듈은 그 카탈로그 한 줄을 **런타임
도구 하나**로 만든다. 서버 SDK 경로와 CLI 경로(claude_code·codex)가 같은 레지스트리의 같은 도구 객체를
쓰므로(``host.tool_surface``), 기기 도구의 계약도 여기 한 곳에 있다.

지키는 것
---------
* **스키마를 그대로 준다.** 예전 서버 어댑터는 JSON Schema 를 pydantic 모델로 다시 만들며 ``type``·
  ``description``·``default`` 만 남겼다 — ``enum``·``items``·중첩 속성이 전부 사라져, 예컨대
  ``BrowserTabs`` 의 ``action`` 이 list/close/activate 를 받는다는 사실이 모델에게서 없어졌다(13개 도구,
  2026-09-30 감사). CLI 경로는 원본을 보냈으므로 같은 도구가 provider 마다 다른 계약이었다.
* **결과의 이미지를 살린다.** MCP 결과의 이미지 블록은 런타임 이미지 블록으로 옮긴다(화면 캡처 등).
* **실패는 실패로 표시한다.** ``isError`` 는 ``is_error`` 가 된다 — 반복 실패 가드가 그걸 센다.
* **사람의 거부는 거부로.** 거부 결과는 구조화 머리말로 바꿔 Stage 10 이 같은 동작을 다시 묻지 않게 한다.
* **기계를 헷갈리면 부르기 전에 말한다.** sandbox 경로를 기기 도구에 주면 기기로 보내지 않고 바로
  "그건 sandbox 경로다" 라고 돌려준다(반대 방향은 ``stages/s10_tool/second_machine``).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from xgen_agent_runtime.host.local_folders import device_tool_name, model_tool_name
from xgen_agent_runtime.tools import Tool, ToolResult, build_tool

logger = logging.getLogger("xgen_agent_runtime.host.device_tools")

#: 기기 호출 — ``(원래 도구 이름, 인자) → 브릿지 결과`` (``{"ok", "result", "error"}``).
DeviceCall = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]

#: 기기 도구 입력에서 경로를 담는 키 (데스크톱·휴대폰·웹 [폴더] 도구가 쓰는 이름).
_PATH_KEYS = ("path", "cwd", "dir", "directory", "file", "file_path", "folder", "target")

WRONG_MACHINE = (
    "[Wrong machine] {path} is in your sandbox, not on the user's device. This tool acts only on "
    "the folders connected on the user's device. Use your own tools (Read, Write, Edit, Glob, Grep, "
    "Bash) for sandbox paths."
)


def _object_schema(schema: Any) -> Dict[str, Any]:
    """MCP ``inputSchema`` 를 그대로 쓰되 최상위가 object 인지만 보장한다."""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    out = dict(schema)
    if out.get("type") != "object":
        out["type"] = "object"
    if not isinstance(out.get("properties"), dict):
        out["properties"] = {}
    return out


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").rstrip("/")


def _sandbox_path(args: Dict[str, Any], working_dir: str) -> Optional[str]:
    """인자 중 sandbox 작업 폴더 안의 경로. 없으면 None."""
    root = _norm(working_dir)
    if not root or root == "/" or len(root) < 4:
        return None
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            v = _norm(value.strip())
            if v == root or v.startswith(root + "/"):
                return value.strip()
    return None


def _blocks(content: Any) -> List[Dict[str, Any]]:
    """MCP content → 런타임 블록(text / image base64). 알 수 없는 블록은 텍스트로."""
    out: List[Dict[str, Any]] = []
    if not isinstance(content, list):
        return out
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            out.append({"type": "text", "text": str(block.get("text") or "")})
        elif kind == "image":
            source = block.get("source") if isinstance(block.get("source"), dict) else {}
            data = source.get("data") or block.get("data") or ""
            mime = (
                source.get("media_type")
                or block.get("mimeType")
                or block.get("mime_type")
                or "image/png"
            )
            if isinstance(data, str) and data.startswith("data:"):
                data = data.split(",", 1)[-1]
            if data:
                out.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": str(mime), "data": str(data)},
                    }
                )
        else:
            import json

            out.append({"type": "text", "text": json.dumps(block, ensure_ascii=False, default=str)})
    return out


def _as_content(blocks: List[Dict[str, Any]]) -> Any:
    """텍스트뿐이면 문자열 하나로(기존 결과 모양), 이미지가 있으면 블록 목록 그대로."""
    if any(b.get("type") == "image" for b in blocks):
        return blocks
    return "\n".join(b.get("text", "") for b in blocks)


def to_tool_result(name: str, payload: Any) -> ToolResult:
    """브릿지 결과(``{"ok", "result", "error"}``) → :class:`ToolResult`."""
    from xgen_agent_runtime.host.tools import _denial_in_text, _denied_result

    if not isinstance(payload, dict):
        return ToolResult(content=str(payload))
    if not payload.get("ok", False):
        err = str(payload.get("error") or "Unknown device error")
        denied = _denial_in_text(err)
        if denied is not None:
            return _denied_result(name, denied, None)
        return ToolResult(content=f"Error: {err}", is_error=True)
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        blocks = _blocks(result["content"])
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if result.get("isError"):
            denied = _denial_in_text(text)
            if denied is not None:
                return _denied_result(name, denied, None)
            return ToolResult(
                content=_as_content(blocks) or "(tool reported failure with no message)",
                is_error=True,
            )
        return ToolResult(content=_as_content(blocks))
    if isinstance(result, str):
        return ToolResult(content=result)
    import json

    try:
        return ToolResult(content=json.dumps(result, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return ToolResult(content=str(result))


def build_device_tool(
    *,
    server: str,
    tool: str,
    description: str,
    input_schema: Any,
    call: DeviceCall,
) -> Tool:
    """기기 카탈로그 한 줄 → 런타임 도구. 모델 이름은 :func:`model_tool_name` 이 정한다."""
    name = model_tool_name(server, tool)
    raw_tool = device_tool_name(server, str(tool))
    schema = _object_schema(input_schema)

    async def _execute(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        args = dict(tool_input or {})
        wrong = _sandbox_path(args, str(getattr(ctx, "working_dir", "") or ""))
        if wrong is not None:
            return ToolResult(content=WRONG_MACHINE.format(path=wrong), is_error=True)
        try:
            payload = await call(raw_tool, args)
        except Exception as exc:  # noqa: BLE001 — 기기 호출 실패는 모델에게 돌아간다
            logger.warning("device tool %s failed: %s", name, exc)
            return ToolResult(content=f"Error: {exc}", is_error=True)
        return to_tool_result(name, payload)

    return build_tool(
        name=name,
        description=str(description or f"Device tool {tool} on {server}"),
        input_schema=schema,
        execute=_execute,
    )


def build_device_guide(*, name: str, description: str, text: str) -> Tool:
    """기기 도구 가족의 **문**(예: ``mcp_local_BrowserGuide``) — 부르면 지도를 돌려준다.

    여는 일은 문 표(:mod:`xgen_agent_runtime.tools.gates`)가 라우터에서 한다 — 문이 성공하면 같은
    접두의 가족이 열린다.
    """

    async def _execute(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        return ToolResult(content=text)

    return build_tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        execute=_execute,
    )


__all__ = [
    "DeviceCall",
    "WRONG_MACHINE",
    "build_device_guide",
    "build_device_tool",
    "to_tool_result",
]
