"""표준 형식 ↔ OpenAI Responses API(``/v1/responses``).

왜 Responses 인가 (2026-10-01 dev 실측, ``llm_client/openai.py`` 의 ``RESPONSES_FAMILIES``)
-----------------------------------------------------------------------------------
GPT-5.4 부터 Chat Completions 는 **함수 도구와 생각(reasoning_effort)을 함께 받지 않는다** —
``Function tools with reasoning_effort are not supported for <model> in /v1/chat/completions.
To use function tools, use /v1/responses or set reasoning_effort to 'none'.`` gpt-6-sol 은 기본
강도가 medium 이라 생각 값을 보내지 않아도 도구만 있으면 같은 400 이고, gpt-6-astra·gpt-6.1-sol 은
``none`` 도 받지 않아 Chat Completions 로는 도구를 아예 쓸 수 없다. Responses 는 모든 강도에서 도구를
받았다(gpt-5.2·5.4·5.5·6-sol·6-astra·6.1-sol, 도구 호출 → 결과 → 답까지).

모양의 차이
-----------
- 시스템 프롬프트 → ``instructions``. 대화 → ``input`` 항목 목록.
- 도구 정의는 평평하다: ``{"type":"function","name","description","parameters","strict"}``.
- 모델의 도구 호출 = ``function_call`` 항목(``call_id``), 결과 = ``function_call_output`` 항목.
- 생각 = ``reasoning`` 항목. 상태 없이(``store=False``) 부르므로 ``encrypted_content`` 를 받아 다음
  호출에 **그 자리 그대로** 돌려준다(도구 호출 앞의 생각을 모델이 이어 간다). 표준 형식에서는
  ``{"type":"reasoning","provider":"openai","model":…,"summary":…,"item":{…}}`` 블록으로 산다 —
  다른 provider 의 번역기는 이 블록을 버린다(``_canonical``). 같은 모델에만 돌려준다(다른 모델의
  암호화된 생각은 풀 수 없다).
- 어시스턴트 글에는 ``phase``(commentary·final_answer)가 붙는다 — 도구 사이의 중간 말을 마지막
  답으로 오해하지 않게 그대로 돌려준다(``_meta.openai_phase``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from xgen_agent_runtime.llm_client.translators._canonical import (
    _file_block_to_text_fallback,
    _tool_result_text_and_images,
    blocks_to_text,
    materialize_local_image_block,
    split_tool_results,
)

#: 표준 형식의 생각 블록 종류(이 모듈이 만든다).
REASONING_BLOCK = "reasoning"
#: 그 블록의 provider 표시 — 다른 번역기가 이것을 보고 버린다.
REASONING_PROVIDER = "openai"


# ── 도구 ─────────────────────────────────────────────────────────────


def canonical_tools_to_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """표준 도구 정의 → Responses 함수 도구.

    ``strict`` 는 **명시적으로 False** — 빼면 Responses 는 스키마를 strict 로 바꿔 보려 하는데(모든
    속성 required·additionalProperties false), 선택 인자가 있는 도구의 의미가 달라진다. Chat
    Completions 의 기본(non-strict)과 같게 둔다.
    """
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            "strict": False,
        }
        for t in tools
    ]


def canonical_tool_choice_to_responses(choice: Optional[Dict[str, Any]]) -> Any:
    """표준 tool_choice → Responses tool_choice (``auto``·``required``·``none``·지정 함수)."""
    if choice is None:
        return "auto"
    kind = choice.get("type", "auto")
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "name": choice["name"]}
    return "auto"


def response_format_to_responses(rf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """표준 response_format → Responses ``text.format``.

    표준형은 ``json_schema`` 에 스키마 자체를 담는다(Chat Completions 쪽 변환과 같은 계약). ``strict``
    는 넣지 않는다 — strict 는 모든 속성을 required 로 요구해 선택 필드가 있는 스키마를 거절한다.
    """
    kind = rf.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind != "json_schema":
        return None
    body = rf.get("json_schema")
    if not isinstance(body, dict) or not body:
        return None
    if isinstance(body.get("schema"), dict) and body.get("name"):
        name, schema = str(body["name"]), body["schema"]
    else:
        name, schema = str(body.get("title") or "response"), body
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "response"
    return {"type": "json_schema", "name": name, "schema": schema, "strict": False}


# ── 입력(대화) ────────────────────────────────────────────────────────


def _image_part(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    block = materialize_local_image_block(block)
    source = block.get("source") or {}
    if source.get("type") == "base64" and source.get("data"):
        url = f"data:{source.get('media_type') or 'image/png'};base64,{source['data']}"
    elif source.get("type") == "url" and source.get("url"):
        url = source["url"]
    else:
        return None
    meta = block.get("_meta") if isinstance(block.get("_meta"), dict) else {}
    return {
        "type": "input_image",
        "image_url": url,
        "detail": block.get("detail") or meta.get("detail") or "auto",
    }


def _user_parts(content: List[Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append({"type": "input_text", "text": str(block)})
            continue
        kind = block.get("type")
        if kind == "text":
            if block.get("text"):
                parts.append({"type": "input_text", "text": block["text"]})
        elif kind == "image":
            part = _image_part(block)
            if part is not None:
                parts.append(part)
        elif kind == "file":
            parts.append({"type": "input_text", "text": _file_block_to_text_fallback(block)})
    return parts


def _reasoning_item(block: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """돌려줄 수 있는 생각 항목 — 같은 모델이 만든 것, 암호화된 내용이 있는 것만."""
    if block.get("provider") != REASONING_PROVIDER:
        return None
    if model and block.get("model") and block.get("model") != model:
        return None
    item = block.get("item")
    if not isinstance(item, dict) or not item.get("encrypted_content"):
        return None
    out = {k: v for k, v in item.items() if k in ("type", "id", "summary", "encrypted_content")}
    out["type"] = "reasoning"
    out.setdefault("summary", [])
    return out


def _assistant_text_item(text: str, phase: Optional[str]) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    if phase:
        item["phase"] = phase
    return item


def canonical_to_responses_input(
    messages: List[Dict[str, Any]],
    system: Any = "",
    *,
    model: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """표준 대화 + 시스템 → (``instructions``, ``input`` 항목).

    어시스턴트 블록은 **받은 순서 그대로** 항목이 된다 — 생각 항목은 그 뒤의 도구 호출 바로 앞에
    있어야 한다(순서가 바뀌면 "function_call was provided without its required reasoning item").
    생각 항목을 돌려주지 못하는 경우(다른 모델이 만든 것·이미 지운 것)에는 도구 호출 항목에 id 를
    달지 않는다 — id 가 붙은 호출은 짝이 되는 생각 항목을 요구한다.
    """
    instructions = blocks_to_text(system) if system else ""
    items: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            if not isinstance(content, list):
                if str(content):
                    items.append({"role": "user", "content": str(content)})
                continue
            tool_results, other = split_tool_results(content)
            lifted: List[Dict[str, Any]] = []
            for tr in tool_results:
                text, images = _tool_result_text_and_images(tr)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(tr.get("tool_use_id", "")),
                        "output": text or "(the tool returned an image; see the next message)",
                    }
                )
                lifted.extend(images)
            if lifted:
                parts = _user_parts(lifted)
                if parts:
                    items.append({"role": "user", "content": parts})
            if other:
                parts = _user_parts(other)
                if parts:
                    items.append({"role": "user", "content": parts})
            continue
        if role != "assistant":
            continue
        if not isinstance(content, list):
            if str(content):
                items.append(_assistant_text_item(str(content), None))
            continue
        paired = False  # 바로 앞의 생각 항목을 돌려줬는가
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == REASONING_BLOCK:
                item = _reasoning_item(block, model)
                if item is not None:
                    items.append(item)
                    paired = True
                continue
            if kind == "text":
                text = block.get("text") or ""
                if text:
                    meta = block.get("_meta") if isinstance(block.get("_meta"), dict) else {}
                    items.append(_assistant_text_item(text, meta.get("openai_phase")))
                continue
            if kind == "tool_use":
                call: Dict[str, Any] = {
                    "type": "function_call",
                    "call_id": str(block.get("id", "")),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}) or {}, ensure_ascii=False),
                }
                meta = block.get("_meta") if isinstance(block.get("_meta"), dict) else {}
                if paired and meta.get("openai_item_id"):
                    call["id"] = meta["openai_item_id"]
                items.append(call)
    return instructions, items


def strip_reasoning_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """생각 항목과 도구 호출의 항목 id 를 뗀 입력 — 생각을 돌려주다 거절됐을 때 다시 보낼 것."""
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            continue
        if isinstance(item, dict) and item.get("type") == "function_call" and "id" in item:
            item = {k: v for k, v in item.items() if k != "id"}
        out.append(item)
    return out


# ── 출력 ─────────────────────────────────────────────────────────────


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dump(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()
    return {}


def reasoning_summary_text(item: Any) -> str:
    parts = []
    for part in _get(item, "summary", None) or []:
        text = _get(part, "text", "")
        if text:
            parts.append(str(text))
    return "\n\n".join(parts)


def stop_reason_of(response: Any, *, has_tool_calls: bool) -> str:
    """Responses 의 끝 상태 → 표준 stop_reason."""
    if has_tool_calls:
        return "tool_use"
    if _get(response, "status", "") == "incomplete":
        reason = _get(_get(response, "incomplete_details", None), "reason", "")
        if reason == "max_output_tokens":
            return "max_tokens"
        if reason == "content_filter":
            return "content_filter"
        return str(reason or "max_tokens")
    return "end_turn"


__all__ = [
    "REASONING_BLOCK",
    "REASONING_PROVIDER",
    "canonical_to_responses_input",
    "canonical_tool_choice_to_responses",
    "canonical_tools_to_responses",
    "reasoning_summary_text",
    "response_format_to_responses",
    "stop_reason_of",
    "strip_reasoning_items",
]
