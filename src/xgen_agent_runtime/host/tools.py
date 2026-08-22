"""TOOL port → geny-executor ``Tool`` adapter.

At runtime the xgen ``TOOL`` port delivers LangChain tool objects
(``BaseTool``/``StructuredTool``, e.g. from mcp_loader / api_tool_loader /
a2ui_tool), possibly nested in lists (``multi: True`` ports; loader nodes
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
from typing import Any, Dict, List, Optional

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


def _wrap_langchain(lc_tool: Any, result_sink: Optional[Dict[str, str]]) -> Tool:
    name = _sanitize_name(getattr(lc_tool, "name", type(lc_tool).__name__))
    description = str(getattr(lc_tool, "description", "") or "")

    async def _execute(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        try:
            output = await _invoke_langchain(lc_tool, dict(tool_input or {}))
        except Exception as exc:  # noqa: BLE001 - tool errors go back to the model, never crash the loop
            logger.warning("geny_bridge: tool %s failed: %s", name, exc)
            text = f"Error: {exc}"
            if result_sink is not None:
                result_sink[name] = text
            return ToolResult(content=text, is_error=True)
        text = _stringify(output)
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
    input_schema = _object_schema(schema.get("properties", {}), schema.get("required")) if isinstance(schema, dict) else dict(_EMPTY_SCHEMA)

    async def _execute(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        try:
            output = func(**(tool_input or {}))
            if asyncio.iscoroutine(output):
                output = await output
        except Exception as exc:  # noqa: BLE001
            logger.warning("geny_bridge: tool %s failed: %s", name, exc)
            text = f"Error: {exc}"
            if result_sink is not None:
                result_sink[name] = text
            return ToolResult(content=text, is_error=True)
        text = _stringify(output)
        if result_sink is not None:
            result_sink[name] = text
        return ToolResult(content=text)

    return build_tool(name=name, description=description, input_schema=input_schema, execute=_execute)


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
            return [t for item in _flatten(obj["dispatch_tool"]) for t in _adapt_one(item, result_sink)]
        if obj.get("name") and callable(obj.get("func") or obj.get("function")):
            return [_wrap_callable_dict(obj, result_sink)]
        logger.warning("geny_bridge: cannot adapt tool dict with keys %s — skipping", sorted(obj.keys()))
        return []
    if _looks_like_langchain_tool(obj):
        return [_wrap_langchain(obj, result_sink)]
    logger.warning("geny_bridge: cannot adapt tool object of type %s — skipping", type(obj).__name__)
    return []


def adapt_tools(
    port_value: Any,
    *,
    result_sink: Optional[Dict[str, str]] = None,
    registry: Optional[ToolRegistry] = None,
    core: bool = True,
) -> Optional[ToolRegistry]:
    """Adapt everything on the Tools port into a ``ToolRegistry``.

    ``result_sink`` (tool name → last stringified output) lets the streaming
    bridge attach result content to xgen ``tool_result`` events — geny-executor's
    ``tool.call_complete`` event carries only name/is_error/duration.

    ``core=False`` registers the tools *deferred* (2.42.0 exposure model):
    schemas stay out of the LLM request until a ``ToolSearch`` hit activates
    them — the token-saving mode for large tool sets.

    Returns ``None`` when nothing usable was connected (and no registry was
    passed in), so callers can skip tool stages entirely.
    """
    tools = [t for item in _flatten(port_value) for t in _adapt_one(item, result_sink)]
    if not tools and registry is None:
        return None
    registry = registry if registry is not None else ToolRegistry()
    for tool in tools:
        registry.register(tool, core=core)
    return registry
