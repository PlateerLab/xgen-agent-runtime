"""Structured error types for tool dispatch.

The router (``RegistryRouter``) converts every failure mode into a
``ToolError`` with a stable ``code`` so the model — and any downstream
consumer — can reason about the failure without parsing free-form
English. Tool implementations that want to signal a structured failure
raise ``ToolFailure`` from ``execute`` (or ``run`` on the Geny side);
the router catches it and emits the matching ``ToolError``.

No string fallbacks are kept here; the old ``"Unknown tool: X"`` /
``"Tool 'X' failed: ..."`` strings are gone as of v0.22.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from xgen_agent_runtime.tools.base import ToolResult


class ToolErrorCode(str, Enum):
    """Stable identifiers for tool failure modes."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_INPUT = "invalid_input"
    TOOL_CRASHED = "tool_crashed"
    TRANSPORT = "transport_error"
    ACCESS_DENIED = "access_denied"


@dataclass(frozen=True)
class ToolError:
    """Structured description of a tool failure.

    Keep ``message`` concise — it's surfaced on the first line of the
    tool_result so the model can pattern-match it. Put everything else
    (paths, expected values, server name, …) in ``details``.
    """

    code: ToolErrorCode
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Return the wire representation used inside ``ToolResult.content``."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": dict(self.details),
            }
        }

    @classmethod
    def unknown_tool(cls, name: str, *, known: Optional[Iterable[str]] = None) -> "ToolError":
        details: Dict[str, Any] = {"tool_name": name}
        if known is not None:
            details["known_tools"] = sorted(known)
        return cls(
            code=ToolErrorCode.UNKNOWN_TOOL,
            message=f"Unknown tool: {name}",
            details=details,
        )

    @classmethod
    def invalid_input(
        cls, tool_name: str, reason: str, *, path: Optional[str] = None
    ) -> "ToolError":
        details: Dict[str, Any] = {"tool_name": tool_name, "reason": reason}
        if path is not None:
            details["path"] = path
        return cls(
            code=ToolErrorCode.INVALID_INPUT,
            message=f"Invalid input for '{tool_name}': {reason}",
            details=details,
        )

    @classmethod
    def tool_crashed(cls, tool_name: str, exc: BaseException) -> "ToolError":
        return cls(
            code=ToolErrorCode.TOOL_CRASHED,
            message=f"Tool '{tool_name}' crashed: {type(exc).__name__}: {exc}",
            details={
                "tool_name": tool_name,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    @classmethod
    def access_denied(
        cls, tool_name: str, reason: str = "binding disallows this tool"
    ) -> "ToolError":
        return cls(
            code=ToolErrorCode.ACCESS_DENIED,
            message=f"Access denied for '{tool_name}': {reason}",
            details={"tool_name": tool_name, "reason": reason},
        )

    @classmethod
    def transport(cls, server_name: str, reason: str) -> "ToolError":
        return cls(
            code=ToolErrorCode.TRANSPORT,
            message=f"MCP transport error on '{server_name}': {reason}",
            details={"server": server_name, "reason": reason},
        )


class ToolFailure(Exception):
    """Raised by tool implementations to report a structured failure.

    Preferred over returning a JSON blob with an ``"error"`` field —
    the router bridges this into a ``ToolError`` with the given
    ``code`` (default ``TOOL_CRASHED``) and preserves ``details``.

    Example::

        raise ToolFailure(
            "rate limit exceeded",
            code=ToolErrorCode.TRANSPORT,
            details={"retry_after": 30},
        )
    """

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode = ToolErrorCode.TOOL_CRASHED,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error = ToolError(code=code, message=message, details=details or {})


def make_error_result(err: ToolError) -> "ToolResult":
    """Wrap a ``ToolError`` into a ``ToolResult`` ready for the API layer.

    Kept out of ``base.py`` to avoid a circular import; routers call
    this helper from within ``stages.s10_tool``.
    """
    from xgen_agent_runtime.tools.base import ToolResult

    return ToolResult(
        content=err.to_payload(),
        is_error=True,
        metadata={"error_code": err.code.value},
    )


_INT_RE = None
_NUM_RE = None


def coerce_input(schema: Dict[str, Any], payload: Any) -> Any:
    """스키마가 숫자·불리언을 원하는데 모델이 **문자열로** 보낸 값을 바로잡는다.

    실측 (2026-09-16/17 dev, claude-sonnet-4-6): ``max_results: "3"`` 하나로
    ``'3' is not of type 'integer'`` 가 나 같은 호출이 6~15번 반복됐다. 뜻이 분명한
    변환만 한다 — 정수 문자열 → integer, 숫자 문자열 → number, "true"/"false" →
    boolean. 애매한 값은 건드리지 않고 검증기가 원래대로 거절하게 둔다.
    원본은 바꾸지 않고, 바뀐 것이 없으면 같은 객체를 돌려준다.
    """
    import re

    global _INT_RE, _NUM_RE
    if _INT_RE is None:
        _INT_RE = re.compile(r"^\s*-?\d+\s*$")
        _NUM_RE = re.compile(r"^\s*-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")

    if not isinstance(schema, dict):
        return payload
    types = schema.get("type")
    types = [types] if isinstance(types, str) else list(types or [])

    if isinstance(payload, str) and "string" not in types:
        if "integer" in types and _INT_RE.match(payload):
            return int(payload.strip())
        if "number" in types and _NUM_RE.match(payload):
            text = payload.strip()
            return int(text) if _INT_RE.match(text) else float(text)
        if "boolean" in types and payload.strip().lower() in ("true", "false"):
            return payload.strip().lower() == "true"
        if "array" in types or "object" in types:
            parsed = _parse_json_container(payload, types)
            if parsed is not None:
                return parsed
        return payload

    if isinstance(payload, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return payload
        out = None
        for key, value in payload.items():
            sub = props.get(key)
            if not isinstance(sub, dict):
                continue
            fixed = coerce_input(sub, value)
            if fixed is not value:
                if out is None:
                    out = dict(payload)
                out[key] = fixed
        return payload if out is None else out

    if isinstance(payload, list) and isinstance(schema.get("items"), dict):
        fixed_items = [coerce_input(schema["items"], v) for v in payload]
        if any(a is not b for a, b in zip(fixed_items, payload)):
            return fixed_items
    return payload


#: 프로바이더가 돌려준 tool-call 인자 JSON 을 끝내 해석하지 못했을 때, 그 원본을
#: 담아 두는 키. 빈 ``{}`` 로 뭉개면 "모델이 인자 없이 불렀다" 로 보여 필수 필드
#: 누락이라는 **틀린 진단**이 모델에게 돌아간다 — 실제로는 우리가 흘린 것이다.
UNPARSED_ARGUMENTS_KEY = "__xgen_unparsed_arguments__"


def _parse_json_container(text: str, types: List[str]) -> Any:
    """``"[...]"`` / ``"{...}"`` 문자열이 스키마가 원하는 컨테이너면 풀어 준다.

    모델이 배열·객체 자리에 **JSON 을 문자열로** 넣는 일이 꾸준히 있다
    (실측 30일 dev: TodoWrite·ToolBatch·ForgeTool·DocApplyEdits 12건). 문자열을
    허용하는 스키마는 건드리지 않고, 풀었을 때 타입이 맞는 경우에만 바꾼다.
    """
    import json

    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, list) and "array" in types:
        return parsed
    if isinstance(parsed, dict) and "object" in types:
        return parsed
    return None


def _value_fits(subschema: Dict[str, Any], value: Any) -> bool:
    """*value* 가 *subschema* 를 통과하는가 (통과 못 하면 False)."""
    import jsonschema

    try:
        jsonschema.validate(instance=value, schema=subschema)
    except jsonschema.ValidationError:
        return False
    except Exception:  # noqa: BLE001 — 깨진 스키마는 판단 보류
        return False
    return True


def _name_akin(a: str, b: str) -> bool:
    """이름이 서로를 품고 있는가 — ``file_path``↔``path``, ``prompt``↔``positive_prompt``."""
    x = "".join(ch for ch in a.lower() if ch.isalnum())
    y = "".join(ch for ch in b.lower() if ch.isalnum())
    if not x or not y:
        return False
    return x in y or y in x


def _pick_alias(
    key: str, candidates: List[str], subschema: Dict[str, Any], payload: Dict[str, Any]
) -> Optional[str]:
    """빠진 *key* 자리에 넣어도 되는 후보를 고른다 — 없으면 None."""
    fits = [c for c in candidates if _value_fits(subschema, payload[c])]
    if not fits:
        return None
    if len(fits) == 1:
        return fits[0]
    akin = [c for c in fits if _name_akin(key, c)]
    return akin[0] if len(akin) == 1 else None


def repair_missing_required(
    schema: Dict[str, Any], payload: Dict[str, Any]
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """필수 필드가 비었는데 **스키마에 없는 키**가 와 있으면 그 자리로 옮긴다.

    모델은 도구 목록 안에서 이름을 섞는다 — 실측 30일 dev: ``DocRender``/
    ``DocAnalyze``/``mcp_local_ReadFile`` 에 ``file_path``(스키마는 ``path``),
    ``mcp_local_Open`` 에 ``path``(스키마는 ``target``), ``comfyui_test_anima`` 에
    ``prompt``(스키마는 ``positive_prompt``). 6개 도구·4개 도메인에서 같은 모양이다.

    판단은 **스키마만** 본다 — 도메인 단어 목록도, 도구별 예외도 없다.
    옮기는 조건은 셋이다: (1) 그 키가 스키마에 아예 없어서 어차피 버려질 값이고,
    (2) 값이 빠진 필드의 서브스키마를 통과하며, (3) 후보가 하나로 좁혀진다
    (둘 이상이면 이름이 서로를 품는 쪽 하나만). 애매하면 손대지 않고 검증기가
    거절하게 둔다.

    돌려주는 것: ``(고친 payload, 모델에게 알려 줄 문장들)`` 또는 None.
    """
    if not isinstance(schema, dict) or not isinstance(payload, dict):
        return None
    props = schema.get("properties")
    required = schema.get("required")
    if not isinstance(props, dict) or not isinstance(required, list):
        return None

    missing = [k for k in required if isinstance(k, str) and k not in payload]
    unknown = [k for k in payload if isinstance(k, str) and k not in props]
    if not missing or not unknown:
        return None

    out = dict(payload)
    notes: List[str] = []
    remaining = list(unknown)
    for key in missing:
        sub = props.get(key)
        if not isinstance(sub, dict) or not remaining:
            continue
        cand = _pick_alias(key, remaining, sub, out)
        if cand is None:
            continue
        out[key] = out.pop(cand)
        remaining.remove(cand)
        notes.append(f"'{cand}' is not a parameter of this tool — used it as '{key}'.")
    if not notes:
        return None
    return out, notes


def describe_validation_failure(schema: Dict[str, Any], payload: Any, message: str) -> str:
    """필수 필드 누락 오류에 **무엇이 필요하고 무엇을 보냈는지**를 붙인다.

    ``'path' is a required property`` 만 돌려주면 모델은 자기가 보낸 것이
    무엇이었는지 모른 채 턴 전체를 다시 생각한다. 필요한 이름과 보낸 이름을
    나란히 보여 주면 이름을 섞은 경우 한 번에 고친다 — 몇 토큰이면 된다.
    """
    if "is a required property" not in message:
        return message
    if not isinstance(schema, dict) or not isinstance(payload, dict):
        return message
    required = [k for k in schema.get("required") or [] if isinstance(k, str)]
    if not required:
        return message
    sent = ", ".join(str(k) for k in payload) or "(nothing)"
    return f"{message}. required: {', '.join(required)}; you sent: {sent}"


def validate_input(schema: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Validate *payload* against JSON Schema *schema*.

    Raises ``jsonschema.ValidationError`` on failure. Returns ``None``
    on success. The caller (router) converts validation errors into
    ``ToolError.invalid_input``.

    ``jsonschema`` is a required dependency as of v0.22.0.
    """
    import jsonschema

    jsonschema.validate(instance=payload, schema=schema)
