"""모델에게 가는 도구 정의(이름·설명·입력 스키마)를 **한 곳**에서 만든다 — provider 무관하게 같다.

왜 필요한가
-----------
같은 레지스트리가 SDK 경로(요청 ``tools`` 배열)와 CLI 경로(MCP ``tools/list`` → claude_code·codex)로
나간다. CLI 는 받은 정의를 자기 규칙으로 **고쳐서** 모델에게 보낸다. 2026-09-30 실측(가짜 모델 서버로
실제 요청 캡처):

Claude Code 2.1.280
  * 도구 설명을 2048자에서 자르고 ``… [truncated]`` 를 붙인다. 입력 스키마는 그대로 보낸다.
  * 이름의 ``[A-Za-z0-9_-]`` 밖 문자를 ``_`` 로 바꾼다(한글 이름 둘이 ``____``·``_____`` 로 뭉개진다).
  * 이름 길이는 자르지 않는다 — ``mcp__connector__`` 16자 + 이름이 64자를 넘으면 API 가 요청 전체를 거절한다.

Codex 0.159.2
  * 스키마 직렬화가 약 5,000바이트를 넘으면 **모든** 속성 설명과 최상위 설명을 지운다.
  * ``default``·``title``·``format``·``pattern``·``minimum``·``maximum``·``minLength``·``maxLength``·
    ``multipleOf``·``uniqueItems``·``examples`` 는 항상 지운다. ``const`` 는 ``enum`` 하나로 바꾼다.
  * 타입이 없는 속성의 설명을 지운다. 도구 설명은 길어도 온전하다.

그리고 런타임 자신도 노드 도구 스키마의 ``$defs`` 를 버려 ``$ref`` 가 깨진 스키마를 보내고 있었다
(``host.tools._json_schema_of``).

그래서 여기서 정한다
--------------------
* 이름은 48자 이하 ``[A-Za-z0-9_-]`` 이고 한 턴 안에서 유일하다(:func:`safe_tool_name`).
* 스키마는 자기완결이다 — 로컬 ``$ref`` 를 풀어 넣고 ``$defs`` 를 뺀다(:func:`normalize_input_schema`).
* 제약 키워드(기본값·형식·범위…)는 속성 설명 끝에 글로도 적는다 — 키워드를 지우는 CLI 에서도 모델이 안다.
* 스키마의 최상위 설명은 도구 설명으로 옮긴다(Codex 가 지운다).
* 설명은 2,000자, 스키마는 4,000바이트 안에 든다(:func:`fit_definition`). 넘친 원문은 잃지 않는다 —
  잘린 정의는 끝에 "전체는 ToolSearch 로" 라고 적고, ToolSearch 가 원문(:func:`full_reference`)을 돌려준다.

SDK 경로도 같은 정의를 받는다(``ToolRegistry.to_api_format``). 한 경로만 고치면 provider 에 따라 같은
도구가 다른 계약이 된다 — 이 모듈이 없애려는 바로 그 상태다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: 모델이 보는 이름의 상한 — API 한도 64 에서 CLI 가 붙이는 ``mcp__connector__``(16자)를 뺀 값.
MAX_TOOL_NAME = 48
#: 도구 설명 상한 — Claude Code 가 2048자에서 자른다.
DESCRIPTION_BUDGET = 2000
#: 입력 스키마 직렬화 상한 — Codex 가 약 5,000바이트를 넘으면 설명을 전부 지운다.
SCHEMA_BUDGET = 4000
#: 스키마가 넘칠 때 속성 설명을 줄이는 단계.
_PROPERTY_DESCRIPTION_STEPS = (400, 200, 80)

_NAME_BAD = re.compile(r"[^A-Za-z0-9_-]+")
_REF_PREFIXES = ("#/$defs/", "#/definitions/")
#: 스키마 노드에서 "스키마 여럿을 담는 사전" 인 키 — 이 안의 키는 **이름**이지 키워드가 아니다.
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
#: 스키마 하나를 담는 키.
_SCHEMA_ONE = (
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
)
#: 스키마 목록을 담는 키.
_SCHEMA_LIST = ("anyOf", "oneOf", "allOf", "prefixItems")
_MAX_DEPTH = 24


# ── 이름 ─────────────────────────────────────────────────────────────


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def safe_tool_name(raw: Any, *, taken: Iterable[str] = ()) -> str:
    """모델에게 보일 도구 이름 — ``[A-Za-z0-9_-]``, :data:`MAX_TOOL_NAME` 이하, ``taken`` 과 겹치지 않음.

    ASCII 밖 글자만으로 된 이름(``한글도구``)은 ``tool_<해시>`` 가 된다 — 글자를 ``_`` 로 바꾸면 서로 다른
    이름이 한 이름으로 뭉개진다. 길면 앞부분 + ``_<해시>`` 로 줄인다(해시는 원래 이름에서 — 같은 이름은 늘
    같은 결과). 이미 쓰인 이름이면 ``_2``·``_3`` … 을 붙인다.
    """
    text = str(raw or "").strip()
    cleaned = _NAME_BAD.sub("_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    # ASCII 밖 글자(한글 등)만 "잃은 글자" 로 센다 — 점·공백 같은 구두점은 ``_`` 로 바꿔도 뜻이 남는다.
    lost = sum(1 for ch in text if ord(ch) > 127)
    if not cleaned or (lost and lost * 2 >= len(text)):
        # 글자 대부분이 사라졌다 — 남은 조각으로는 도구를 가를 수 없다.
        base = (
            f"tool_{cleaned}_{_short_hash(text)}"
            if cleaned
            else f"tool_{_short_hash(text or 'tool')}"
        )
    elif lost:
        base = f"{cleaned}_{_short_hash(text)}"
    else:
        base = cleaned
    if len(base) > MAX_TOOL_NAME:
        base = f"{base[: MAX_TOOL_NAME - 7].rstrip('_-')}_{_short_hash(text)}"
    used = set(taken)
    if base not in used:
        return base
    for n in range(2, 1000):
        suffix = f"_{n}"
        cand = f"{base[: MAX_TOOL_NAME - len(suffix)]}{suffix}"
        if cand not in used:
            return cand
    return f"{base[: MAX_TOOL_NAME - 7]}_{_short_hash(text + str(len(used)))}"


def cap_tool_name(name: str) -> str:
    """이미 안전한 문자로 된 이름의 길이만 :data:`MAX_TOOL_NAME` 안으로 (앞부분 + ``_<해시>``)."""
    text = str(name or "")
    if len(text) <= MAX_TOOL_NAME:
        return text
    return f"{text[: MAX_TOOL_NAME - 7].rstrip('_-')}_{_short_hash(text)}"


# ── 스키마 ───────────────────────────────────────────────────────────


def _resolve_refs(node: Any, defs: Dict[str, Any], stack: Tuple[str, ...], depth: int) -> Any:
    """로컬 ``$ref`` 를 풀어 넣는다. 순환이면 그 자리는 빈 객체 스키마로 둔다."""
    if depth > _MAX_DEPTH:
        return {}
    if isinstance(node, list):
        return [_resolve_refs(v, defs, stack, depth + 1) for v in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        rest = {k: v for k, v in node.items() if k != "$ref"}
        target: Any = None
        key = ""
        for prefix in _REF_PREFIXES:
            if ref.startswith(prefix):
                key = ref[len(prefix) :]
                target = defs.get(key)
                break
        if target is None or key in stack:
            # 풀 수 없는 참조(외부·순환) — 깨진 참조를 보내지 않는다. 설명은 남긴다.
            out = {"type": "object"} if key in stack else {}
            out.update(_resolve_refs(rest, defs, stack, depth + 1))
            return out
        resolved = _resolve_refs(copy.deepcopy(target), defs, stack + (key,), depth + 1)
        if isinstance(resolved, dict):
            merged = dict(resolved)
            merged.update(_resolve_refs(rest, defs, stack, depth + 1))
            return merged
        return resolved
    return {k: _resolve_refs(v, defs, stack, depth + 1) for k, v in node.items()}


def _num(value: Any) -> Optional[str]:
    """제약 값(숫자)만 — draft-04 의 불리언 ``exclusiveMinimum`` 같은 것은 None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return json.dumps(value)


def _constraint_note(node: Dict[str, Any]) -> str:
    """Codex 가 지우는 제약 키워드를 글로 — 이미 설명에 그 말이 있으면 되풀이하지 않는다."""
    desc = str(node.get("description") or "")
    parts: List[Tuple[str, str]] = []
    if "default" in node:
        parts.append(
            ("Default", f"Default: {json.dumps(node['default'], ensure_ascii=False, default=str)}.")
        )
    if isinstance(node.get("format"), str):
        parts.append(("Format", f"Format: {node['format']}."))
    if isinstance(node.get("pattern"), str):
        parts.append(("Pattern", f"Pattern: {node['pattern']}."))
    bounds: List[str] = []
    for key, sym in (
        ("minimum", ">="),
        ("exclusiveMinimum", ">"),
        ("maximum", "<="),
        ("exclusiveMaximum", "<"),
    ):
        num = _num(node.get(key))
        if num is not None:
            bounds.append(f"{sym} {num}")
    if bounds:
        parts.append(("Range", f"Range: {', '.join(bounds)}."))
    lengths: List[str] = []
    for key, word in (("minLength", "at least"), ("maxLength", "at most")):
        num = _num(node.get(key))
        if num is not None:
            lengths.append(f"{word} {num}")
    if lengths:
        parts.append(("Length", f"Length: {' and '.join(lengths)} characters."))
    multiple = _num(node.get("multipleOf"))
    if multiple is not None:
        parts.append(("Multiple of", f"Multiple of {multiple}."))
    if node.get("uniqueItems") is True:
        parts.append(("unique", "Items must be unique."))
    examples = node.get("examples")
    if isinstance(examples, list) and examples:
        shown = ", ".join(json.dumps(e, ensure_ascii=False, default=str) for e in examples[:3])
        parts.append(("Example", f"Examples: {shown}."))
    return " ".join(text for word, text in parts if word.lower() not in desc.lower())


def _annotate(node: Any, depth: int = 0) -> Any:
    """스키마 노드를 돌며 ``title`` 을 빼고 제약을 설명에 적는다. 속성 **이름**(``title`` 등)은 건드리지 않는다."""
    if depth > _MAX_DEPTH or not isinstance(node, dict):
        return node
    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue  # 스키마 키워드 title — 모델에게 새 정보가 없고 CLI 가 지운다
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {name: _annotate(sub, depth + 1) for name, sub in value.items()}
        elif key in _SCHEMA_ONE and isinstance(value, dict):
            out[key] = _annotate(value, depth + 1)
        elif key in _SCHEMA_LIST and isinstance(value, list):
            out[key] = [_annotate(v, depth + 1) for v in value]
        else:
            out[key] = value
    note = _constraint_note(out)
    if note:
        desc = str(out.get("description") or "").rstrip()
        out["description"] = f"{desc} {note}".strip() if desc else note
    return out


def normalize_input_schema(schema: Any) -> Dict[str, Any]:
    """어느 provider 에 보내도 같은 뜻인 입력 스키마.

    * 최상위는 ``type: object`` + ``properties`` 사전.
    * 로컬 ``$ref`` 는 풀어 넣고 ``$defs``/``definitions``/``$schema``/``$id`` 는 뺀다.
    * 스키마 키워드 ``title`` 은 뺀다. 기본값·형식·범위 같은 제약은 키워드를 두고 설명에도 적는다.
    * ``required`` 는 실제 있는 속성만.
    """
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    try:
        src = json.loads(json.dumps(schema, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}}
    defs: Dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        if isinstance(src.get(key), dict):
            defs.update(src[key])
    for key in ("$defs", "definitions", "$schema", "$id"):
        src.pop(key, None)
    resolved = _resolve_refs(src, defs, (), 0)
    out = _annotate(resolved if isinstance(resolved, dict) else {})
    out["type"] = "object"
    if not isinstance(out.get("properties"), dict):
        out["properties"] = {}
    required = out.get("required")
    if isinstance(required, list):
        kept = [r for r in required if isinstance(r, str) and r in out["properties"]]
        if kept:
            out["required"] = list(dict.fromkeys(kept))
        else:
            out.pop("required", None)
    elif "required" in out:
        out.pop("required", None)
    return out


# ── 예산 맞추기 ──────────────────────────────────────────────────────


def _size(schema: Dict[str, Any]) -> int:
    return len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    for sep in ("\n", ". ", "; ", ", ", " "):
        i = cut.rfind(sep)
        if i >= limit // 2:
            cut = cut[: i + (1 if sep.strip() else 0)]
            break
    return cut.rstrip() + "…"


def _cap_descriptions(node: Any, limit: int, depth: int = 0) -> bool:
    """스키마 안의 모든 설명을 ``limit`` 자로. 줄였으면 True."""
    if depth > _MAX_DEPTH or not isinstance(node, dict):
        return False
    changed = False
    desc = node.get("description")
    if isinstance(desc, str) and len(desc) > limit:
        node["description"] = _shorten(desc, limit)
        changed = True
    for key, value in node.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            for sub in value.values():
                changed = _cap_descriptions(sub, limit, depth + 1) or changed
        elif key in _SCHEMA_ONE and isinstance(value, dict):
            changed = _cap_descriptions(value, limit, depth + 1) or changed
        elif key in _SCHEMA_LIST and isinstance(value, list):
            for sub in value:
                changed = _cap_descriptions(sub, limit, depth + 1) or changed
    return changed


def _more_note(name: str) -> str:
    return f' (Full reference: ToolSearch(query="{name}").)'


def fit_definition(name: str, description: Any, schema: Any) -> Tuple[str, Dict[str, Any], bool]:
    """(설명, 스키마, 줄였는가). 두 CLI 의 한도 안에 들고, 줄였으면 원문을 어디서 보는지 적는다."""
    text = str(description or "").strip()
    fitted = normalize_input_schema(schema)
    top = fitted.pop("description", None)
    if isinstance(top, str) and top.strip() and top.strip() not in text:
        # Codex 는 스키마 최상위 설명을 지운다 — 도구 설명으로 옮겨야 모두에게 닿는다.
        text = f"{text}\n\n{top.strip()}" if text else top.strip()
    if not text:
        props = list(fitted.get("properties") or {})
        text = f"{name}." + (f" Inputs: {', '.join(props[:12])}." if props else "")
    trimmed = False
    if _size(fitted) > SCHEMA_BUDGET:
        for limit in _PROPERTY_DESCRIPTION_STEPS:
            trimmed = _cap_descriptions(fitted, limit) or trimmed
            if _size(fitted) <= SCHEMA_BUDGET:
                break
    note = _more_note(name)
    if len(text) > DESCRIPTION_BUDGET or trimmed:
        room = DESCRIPTION_BUDGET - len(note)
        text = _shorten(text, room) + note if len(text) > room else text + note
        trimmed = True
    return text, fitted, trimmed


def api_definition(tool: Any) -> Dict[str, Any]:
    """레지스트리 도구 하나 → 모델에게 가는 정의 ``{name, description, input_schema}``."""
    raw = tool.to_api_format()
    description, schema, _ = fit_definition(
        str(raw.get("name") or ""), raw.get("description"), raw.get("input_schema")
    )
    return {"name": raw.get("name"), "description": description, "input_schema": schema}


def full_reference(tool: Any) -> Optional[str]:
    """정의가 줄었으면 원문(설명 + 속성별 설명·제약). 줄지 않았으면 None — ToolSearch 가 붙인다."""
    raw = tool.to_api_format()
    name = str(raw.get("name") or "")
    _, _, trimmed = fit_definition(name, raw.get("description"), raw.get("input_schema"))
    if not trimmed:
        return None
    schema = normalize_input_schema(raw.get("input_schema"))
    lines = [f"Full reference for {name}:", str(raw.get("description") or "").strip()]
    top = schema.get("description")
    if isinstance(top, str) and top.strip():
        lines.append(top.strip())
    props = schema.get("properties") or {}
    if props:
        lines.append("Parameters:")
        required = set(schema.get("required") or [])
        for pname, pschema in props.items():
            kind = pschema.get("type") if isinstance(pschema, dict) else None
            desc = (pschema.get("description") if isinstance(pschema, dict) else "") or ""
            flag = " (required)" if pname in required else ""
            lines.append(f"- {pname}{f' [{kind}]' if kind else ''}{flag}: {desc}".rstrip(": "))
    return "\n".join(line for line in lines if line)


__all__ = [
    "DESCRIPTION_BUDGET",
    "MAX_TOOL_NAME",
    "SCHEMA_BUDGET",
    "api_definition",
    "cap_tool_name",
    "fit_definition",
    "full_reference",
    "normalize_input_schema",
    "safe_tool_name",
]
