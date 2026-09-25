"""TOOL port → geny-executor ``Tool`` adapter.

At runtime the xgen ``TOOL`` port delivers LangChain tool objects
(``BaseTool``/``StructuredTool``, e.g. from mcp_loader / api_tool_loader),
possibly nested in lists (``multi: True`` ports; loader nodes
return ``List[BaseTool]``). Skill payloads arrive as dicts carrying a
``dispatch_tool``, and a few legacy nodes emit plain
``{"name": ..., "func": ...}`` dicts.

Everything is normalized into geny-executor native ``Tool``s (via
``build_tool``) inside a ``ToolRegistry`` so the engine owns dispatch,
permission checks and tool events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from xgen_agent_runtime.tools import Tool, ToolRegistry, ToolResult, build_tool

logger = logging.getLogger("editor.geny_bridge.tools")

_EMPTY_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}


def _sanitize_name(name: Any) -> str:
    """Provider-safe tool name (same rule as the agent helper's sanitizer)."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name or ""))
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unnamed_tool"


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _object_schema(properties: Dict[str, Any], required: Any = None) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": dict(properties or {})}
    if required:
        schema["required"] = list(required)
    return schema


def _json_schema_of(lc_tool: Any) -> Dict[str, Any]:
    """Extract a JSON-schema ``input_schema`` from a LangChain-ish tool.

    ``args_schema`` may be a pydantic model class OR a raw JSON-schema dict
    (mcp_loader passes the MCP ``inputSchema`` dict straight through), and
    ``.args`` is the BaseTool property returning the properties map. Tools
    with lazy/broken schemas degrade to an empty object schema rather than
    being dropped.
    """
    raw = getattr(lc_tool, "args_schema", None)
    if isinstance(raw, dict):
        return _object_schema(raw.get("properties", {}), raw.get("required"))
    if raw is not None:
        for attr in ("model_json_schema", "schema"):
            fn = getattr(raw, attr, None)
            if callable(fn):
                try:
                    full = fn()
                    return _object_schema(full.get("properties", {}), full.get("required"))
                except Exception:  # noqa: BLE001 - schema extraction is best-effort
                    break
    try:
        props = lc_tool.args
        if isinstance(props, dict):
            return _object_schema(props)
    except Exception:  # noqa: BLE001
        pass
    return dict(_EMPTY_SCHEMA)


def _looks_like_langchain_tool(obj: Any) -> bool:
    if isinstance(obj, (dict, list, tuple, set, str, bytes)):
        return False
    return bool(getattr(obj, "name", None)) and any(
        callable(getattr(obj, attr, None)) for attr in ("ainvoke", "invoke", "run")
    )


async def _invoke_langchain(lc_tool: Any, tool_input: Dict[str, Any]) -> Any:
    """Call a LangChain tool with the whole input dict (BaseTool convention)."""
    if callable(getattr(lc_tool, "ainvoke", None)):
        return await lc_tool.ainvoke(tool_input)
    if callable(getattr(lc_tool, "invoke", None)):
        return await asyncio.to_thread(lc_tool.invoke, tool_input)
    if callable(getattr(lc_tool, "run", None)):
        return await asyncio.to_thread(lc_tool.run, tool_input)
    raise RuntimeError(f"tool {getattr(lc_tool, 'name', lc_tool)!r} has no invoke method")


#: Adapted tools may declare a family they open, like the built-in guides do
#: (``_skill_gateway``): ``lc_tool.metadata[OPENS_FAMILY_KEY] = [names]``.
OPENS_FAMILY_KEY = "opens_family"


def _opens_family(lc_tool: Any) -> List[str]:
    meta = getattr(lc_tool, "metadata", None)
    names = meta.get(OPENS_FAMILY_KEY) if isinstance(meta, dict) else None
    return [str(n) for n in names] if isinstance(names, (list, tuple)) else []


#: 사람이 거부했다는 구조화 코드 — 도구가 던진 예외의 ``code``/``error_code`` 또는 메시지 머리말.
#: 커넥터·MCP 서버는 이 중 하나로 알려 주면 된다(권장: 예외 ``code="user_denied"``, 또는
#: 메시지를 ``user_denied:`` 로 시작).
_DENIAL_CODES = frozenset({"user_denied", "access_denied", "denied_by_user"})

#: ⚠ **임시 호환 — DeX 1.56 의 거부 문구.** 지금 커넥터는 구조화 코드 없이 한국어 문장만 보낸다.
#: 이 문구를 알아보지 않으면 denial_guard 가 거부를 모르고, 모델은 같은 ``rm -rf`` 를 거부된 뒤에도
#: 거듭 불러 확인 창을 세 번 띄운다(2026-09-23 dev, trace 46745).
#: **제거 조건: DeX 가 거부를 ``user_denied`` 코드로 보내기 시작하면 이 튜플을 지운다.**
_LEGACY_DENIAL_PHRASES = ("사용자가 이 명령의 실행을 거부했습니다",)

_DENIAL_GUIDANCE = (
    "The user refused this action. It was NOT done. Do not attempt it again in this turn "
    "(not even with different quoting), and do not reach the same effect another way. "
    "Tell the user it was not done and ask how to proceed."
)


#: 어댑터가 실패를 **결과 문자열**로 알리는 머리말 — LangChain 도구엔 오류 플래그가 없어서
#: workflow 커넥터 어댑터는 MCP ``isError=True`` 를 ``"Error: <본문>"`` 으로 돌려준다(예외 아님).
_ERROR_TEXT_HEAD = re.compile(r"^\s*error\b\s*:?\s*", re.IGNORECASE)


def _denial_message(exc: BaseException) -> Optional[str]:
    """사람이 거부한 것이면 그 사유, 아니면 None."""
    code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
    if isinstance(code, str) and code.strip().lower() in _DENIAL_CODES:
        return str(exc).strip() or code
    return _denial_in_text(str(exc))


def _denial_in_error_text(text: str) -> Optional[str]:
    """``"Error: …"`` 로 돌아온 결과가 사람의 거부면 그 사유.

    오류 머리말이 있을 때만 본다 — 성공한 결과가 우연히 거부 문구를 담은 것(파일 내용 등)은
    거부가 아니다.
    """
    m = _ERROR_TEXT_HEAD.match(text or "")
    return _denial_in_text(text[m.end() :]) if m else None


def _denial_in_text(text: str) -> Optional[str]:
    msg = (text or "").strip()
    head = msg.split(":", 1)[0].strip().lower()
    if head in _DENIAL_CODES:
        return msg.split(":", 1)[1].strip() if ":" in msg else msg
    if any(p in msg for p in _LEGACY_DENIAL_PHRASES):
        return msg
    return None


def _denied_result(name: str, denied: str, result_sink: Optional[Dict[str, str]]) -> ToolResult:
    """사람의 거부는 오류가 아니라 답이다 — 구조화 머리말로 바꿔 Stage 10 이 알아보게 한다
    (stages/s10_tool/denial_guard.py: 같은 턴에 같은 동작을 다시 묻지 않는다). 예외로 오든
    ``"Error: …"`` 결과 문자열로 오든 같은 모양으로 모은다."""
    logger.info("geny_bridge: tool %s denied by user: %s", name, denied)
    text = f"ERROR user_denied: {denied}\n{_DENIAL_GUIDANCE}"
    if result_sink is not None:
        result_sink[name] = text
    return ToolResult(content=text, is_error=True)


def _wrap_langchain(lc_tool: Any, result_sink: Optional[Dict[str, str]]) -> Tool:
    name = _sanitize_name(getattr(lc_tool, "name", type(lc_tool).__name__))
    description = str(getattr(lc_tool, "description", "") or "")
    family = _opens_family(lc_tool)

    async def _execute(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        try:
            output = await _invoke_langchain(lc_tool, dict(tool_input or {}))
        except Exception as exc:  # noqa: BLE001 - tool errors go back to the model, never crash the loop
            denied = _denial_message(exc)
            if denied is not None:
                return _denied_result(name, denied, result_sink)
            logger.warning("geny_bridge: tool %s failed: %s", name, exc)
            text = f"Error: {exc}"
            if result_sink is not None:
                result_sink[name] = text
            return ToolResult(content=text, is_error=True)
        text = _stringify(output)
        denied = _denial_in_error_text(text)
        if denied is not None:
            return _denied_result(name, denied, result_sink)
        if family:
            # 안내 도구가 가리킨 도구들을 이 턴에 실제로 연다 — 내장 안내 도구와 같은 규약.
            # 안내만 하고 열지 않으면 모델은 부를 수 없는 이름을 받고, 약한 모델은 안내
            # 도구만 되풀이한다(2026-09-09 dev: gpt-4.1 이 커넥터 브라우저 안내를 100회).
            from xgen_agent_runtime.tools.built_in._skill_gateway import open_family, with_opened

            text = with_opened(text, open_family(ctx, family))
        if result_sink is not None:
            result_sink[name] = text
        return ToolResult(content=text)

    return build_tool(
        name=name,
        description=description,
        input_schema=_json_schema_of(lc_tool),
        execute=_execute,
    )


def _wrap_callable_dict(spec: Dict[str, Any], result_sink: Optional[Dict[str, str]]) -> Tool:
    func = spec.get("func") or spec.get("function")
    name = _sanitize_name(spec.get("name"))
    description = str(spec.get("description", "") or "")
    schema = spec.get("input_schema") or spec.get("args_schema")
    input_schema = (
        _object_schema(schema.get("properties", {}), schema.get("required"))
        if isinstance(schema, dict)
        else dict(_EMPTY_SCHEMA)
    )

    async def _execute(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        try:
            output = func(**(tool_input or {}))
            if asyncio.iscoroutine(output):
                output = await output
        except Exception as exc:  # noqa: BLE001
            denied = _denial_message(exc)
            if denied is not None:
                return _denied_result(name, denied, result_sink)
            logger.warning("geny_bridge: tool %s failed: %s", name, exc)
            text = f"Error: {exc}"
            if result_sink is not None:
                result_sink[name] = text
            return ToolResult(content=text, is_error=True)
        text = _stringify(output)
        denied = _denial_in_error_text(text)
        if denied is not None:
            return _denied_result(name, denied, result_sink)
        if result_sink is not None:
            result_sink[name] = text
        return ToolResult(content=text)

    return build_tool(
        name=name, description=description, input_schema=input_schema, execute=_execute
    )


def _flatten(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _adapt_one(obj: Any, result_sink: Optional[Dict[str, str]]) -> List[Tool]:
    if obj is None:
        return []
    if isinstance(obj, Tool):
        return [obj]
    if isinstance(obj, dict):
        # Skill payload: the real tool travels under "dispatch_tool".
        if obj.get("dispatch_tool") is not None:
            return [
                t for item in _flatten(obj["dispatch_tool"]) for t in _adapt_one(item, result_sink)
            ]
        if obj.get("name") and callable(obj.get("func") or obj.get("function")):
            return [_wrap_callable_dict(obj, result_sink)]
        logger.warning(
            "geny_bridge: cannot adapt tool dict with keys %s — skipping", sorted(obj.keys())
        )
        return []
    if _looks_like_langchain_tool(obj):
        return [_wrap_langchain(obj, result_sink)]
    logger.warning(
        "geny_bridge: cannot adapt tool object of type %s — skipping", type(obj).__name__
    )
    return []


def adapt_tools(
    port_value: Any,
    *,
    result_sink: Optional[Dict[str, str]] = None,
    registry: Optional[ToolRegistry] = None,
    core: "bool | Callable[[str], bool]" = True,
) -> Optional[ToolRegistry]:
    """Adapt everything on the Tools port into a ``ToolRegistry``.

    ``result_sink`` (tool name → last stringified output) lets the streaming
    bridge attach result content to xgen ``tool_result`` events — geny-executor's
    ``tool.call_complete`` event carries name/is_error/duration (+ the
    failure reason when it failed), not the successful output.

    ``core=False`` registers the tools *deferred* (2.42.0 exposure model):
    schemas stay out of the LLM request until a ``ToolSearch`` hit activates
    them — the token-saving mode for large tool sets. ``core`` may instead be a
    predicate over the tool name, for a surface where some of a batch belongs on
    the first turn and the rest sits behind a gateway (the connector's browser
    tools, say) — a batch is rarely all one thing.

    Returns ``None`` when nothing usable was connected (and no registry was
    passed in), so callers can skip tool stages entirely.
    """
    tools = [t for item in _flatten(port_value) for t in _adapt_one(item, result_sink)]
    if not tools and registry is None:
        return None
    registry = registry if registry is not None else ToolRegistry()
    decide = core if callable(core) else (lambda _name, _c=bool(core): _c)
    for tool in tools:
        registry.register(tool, core=bool(decide(getattr(tool, "name", ""))))
    return registry
