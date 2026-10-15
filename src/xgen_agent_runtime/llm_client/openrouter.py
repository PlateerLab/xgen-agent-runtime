"""OpenRouter client.

OpenRouter 는 **한 키·한 주소**로 여러 회사의 모델(Claude·GPT·Gemini·DeepSeek·Qwen …)을
OpenAI Chat Completions 모양으로 부르게 해 주는 중계다 (https://openrouter.ai/docs).
그래서 이 클라이언트는 :class:`OpenAIClient` 를 그대로 쓰고, OpenAI 표준 밖에 있는 것만 더한다.

무엇을 더하나 (OpenRouter 문서, 2026-10 확인)
-------------------------------------------
1. **생각 이어 가기** — 생각하는 모델은 답에 ``message.reasoning_details`` (스트림은
   ``delta.reasoning_details``)를 싣는다. 도구를 부른 뒤 다음 호출에 이것을 **고치지 않고 그대로**
   돌려줘야 모델이 도구 호출 앞의 생각을 이어 간다 ("Preserving Reasoning"). 표준 형식에서는
   ``{"type": "reasoning", "provider": "openrouter", "model", "summary", "details"}`` 블록으로
   산다 — OpenAI Responses 의 생각 블록과 같은 종류라 다른 provider 의 번역기는 이미 버린다.
   같은 모델에만 돌려준다(대화 중에 모델을 바꾸면 다른 모델의 생각을 보내지 않는다).
2. **200 으로 오는 실패** — 제공자가 헤더를 보낸 뒤 실패하면 상태는 200 이고 본문에 ``error`` 만
   있다(``choices`` 없음). 스트림에서는 ``error`` 를 실은 청크가 오고 SDK 가 ``openai.APIError``
   로 올린다. 둘 다 ``error.code``(HTTP 와 같은 숫자)로 분류해 사람이 읽을 문구를 남긴다.
3. **비용** — 모든 응답의 ``usage.cost``(USD)를 :attr:`TokenUsage.cost_usd` 로 옮긴다.

모델 이름은 ``author/slug`` 다 (``anthropic/claude-sonnet-5.5``, ``qwen/qwen3.6-27b``).
생각 강도는 ``reasoning`` 하나로 보낸다 — OpenRouter 가 그 모델의 방식으로 옮긴다
(:func:`xgen_agent_runtime.llm_client.thinking.thinking_spec` 의 ``openrouter`` 갈래).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.core.errors import APIError, ErrorCategory, is_context_overflow_message
from xgen_agent_runtime.core.state import TokenUsage
from xgen_agent_runtime.llm_client.openai import OpenAIClient
from xgen_agent_runtime.llm_client.translators import canonical_messages_to_openai
from xgen_agent_runtime.llm_client.types import APIRequest, APIResponse, ContentBlock

logger = logging.getLogger(__name__)

#: 기본 주소. 지역 라우팅(``eu.openrouter.ai``)이나 사내 프록시는 ``base_url`` 로 바꾼다.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: 표준 형식의 생각 블록 종류 — OpenAI Responses 의 것과 같다(다른 번역기가 이미 버린다).
REASONING_BLOCK = "reasoning"
#: 그 블록의 provider 표시 — 이 클라이언트만 돌려준다.
REASONING_PROVIDER = "openrouter"

#: 스트림 조각에서 이어 붙이는 글 필드(``reasoning.text``·``reasoning.summary``·``reasoning.encrypted``).
_DETAIL_TEXT_KEYS = ("text", "summary", "data")


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """SDK 모델·dict 어느 쪽이든 필드 하나. SDK 가 모르는 필드는 ``model_extra`` 에 산다."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return default


def _as_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            out = dump()
        except Exception:  # noqa: BLE001 — 덤프 실패는 그 조각을 버린다
            return {}
        return dict(out) if isinstance(out, dict) else {}
    return {}


def merge_reasoning_details(fragments: List[Any]) -> List[Dict[str, Any]]:
    """스트림 조각들 → 완성된 ``reasoning_details``.

    한 생각이 여러 청크로 나뉘어 온다(문서: "built by concatenating all chunks in order"). 같은
    ``index``·같은 ``type`` 의 조각은 글(``text``·``summary``·``data``)을 이어 붙이고, 서명·id·format
    처럼 끝에 한 번 오는 값은 비어 있지 않은 마지막 값을 쓴다. ``index`` 가 없으면 바로 앞 조각과
    ``type`` 이 같을 때 이어 붙인다.
    """
    merged: List[Dict[str, Any]] = []
    by_index: Dict[Any, Dict[str, Any]] = {}
    for raw in fragments:
        frag = _as_dict(raw)
        if not frag:
            continue
        idx = frag.get("index")
        if idx is not None:
            target = by_index.get(idx)
        elif merged and merged[-1].get("index") is None:
            target = merged[-1]
        else:
            target = None
        if target is None or target.get("type") != frag.get("type"):
            target = dict(frag)
            merged.append(target)
            if idx is not None:
                by_index[idx] = target
            continue
        for key, value in frag.items():
            if key in _DETAIL_TEXT_KEYS and isinstance(value, str):
                target[key] = str(target.get(key) or "") + value
            elif value not in (None, ""):
                target[key] = value
    return merged


def reasoning_text(details: List[Dict[str, Any]], fallback: Any = None) -> str:
    """사람에게 보일 생각 글 — ``reasoning.text`` 와 ``reasoning.summary``. 암호화된 것은 보이지 않는다."""
    parts: List[str] = []
    for d in details:
        kind = d.get("type")
        if kind == "reasoning.text" and isinstance(d.get("text"), str):
            parts.append(d["text"])
        elif kind == "reasoning.summary" and isinstance(d.get("summary"), str):
            parts.append(d["summary"])
    text = "\n\n".join(p for p in parts if p)
    if not text and isinstance(fallback, str):
        text = fallback
    return text


def reasoning_block(details: List[Dict[str, Any]], text: str, model: str) -> Optional[ContentBlock]:
    """생각 블록 하나 — 돌려줄 것도 보일 것도 없으면 ``None``."""
    if not details and not text:
        return None
    return ContentBlock(
        type=REASONING_BLOCK,
        thinking_text=text or None,
        raw={
            "type": REASONING_BLOCK,
            "provider": REASONING_PROVIDER,
            "model": model,
            "summary": text,
            "details": details,
        },
    )


def assistant_reasoning_fields(content: Any, model: str) -> Dict[str, Any]:
    """어시스턴트 메시지에 다시 실을 생각 — 같은 모델이 만든 OpenRouter 생각 블록만.

    ``reasoning_details`` 가 있으면 그것을, 글만 있으면 ``reasoning`` 을 싣는다(문서: 글만 내는
    모델은 ``reasoning`` 문자열로 돌려줘도 된다).
    """
    details: List[Dict[str, Any]] = []
    texts: List[str] = []
    for block in content if isinstance(content, list) else ():
        if not isinstance(block, dict) or block.get("type") != REASONING_BLOCK:
            continue
        if block.get("provider") != REASONING_PROVIDER:
            continue
        if model and block.get("model") and block.get("model") != model:
            continue
        found = block.get("details")
        if isinstance(found, list):
            details.extend(d for d in found if isinstance(d, dict))
        elif isinstance(block.get("summary"), str) and block["summary"]:
            texts.append(block["summary"])
    if details:
        return {"reasoning_details": details}
    if texts:
        return {"reasoning": "\n\n".join(texts)}
    return {}


def error_from_body(error: Any, *, fallback: str = "") -> APIError:
    """OpenRouter 의 ``{"error": {"code", "message", "metadata"}}`` → :class:`APIError`.

    ``code`` 는 HTTP 상태와 같은 숫자다(문서 "Errors and debugging"). 402 는 크레딧 부족·키 한도다.
    """
    message = str(_field(error, "message") or fallback or "OpenRouter request failed")
    metadata = _field(error, "metadata")
    provider_name = _field(metadata, "provider_name")
    if provider_name:
        message = f"{message} (provider: {provider_name})"
    raw_code = _field(error, "code")
    try:
        code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        code = None

    if code in (401, 403):
        category = ErrorCategory.AUTH
    elif code == 402:
        # 돈이 모자라다 — 다시 보내도 풀리지 않는다.
        category = ErrorCategory.TERMINAL
        message = f"OpenRouter 크레딧 또는 키 한도가 부족합니다: {message}"
    elif code == 408:
        category = ErrorCategory.TIMEOUT
    elif code == 429:
        category = ErrorCategory.RATE_LIMITED
    elif code is not None and code >= 500:
        category = ErrorCategory.SERVER_ERROR
    elif is_context_overflow_message(message):
        category = ErrorCategory.TOKEN_LIMIT
    elif code == 400:
        category = ErrorCategory.BAD_REQUEST
    else:
        category = ErrorCategory.UNKNOWN
    details = {"code": code, "metadata": metadata} if metadata else {"code": code}
    return APIError(message, category=category, status_code=code, details=details)


class OpenRouterClient(OpenAIClient):
    """OpenRouter — OpenAI Chat Completions + 생각 이어 가기·200 실패·비용."""

    provider = "openrouter"
    _sdk_module = "openai"
    #: ``developer`` 는 OpenAI 의 역할이다. OpenRouter 뒤의 모델 대부분은 ``system`` 을 쓴다.
    system_role = "system"
    capabilities = replace(OpenAIClient.capabilities, supports_cost_usage=True)

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=(base_url or "").strip() or OPENROUTER_BASE_URL,
            default_headers=default_headers,
            event_sink=event_sink,
        )

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        """연결만 미리 연다 — ``/models`` 는 수백 개 모델(약 1MB)이라 키 정보(``/key``)를 묻는다."""
        import asyncio

        import httpx

        try:
            client = self._get_client()
            await asyncio.wait_for(client.get("/key", cast_to=httpx.Response), timeout=timeout_s)
            return True
        except Exception:  # noqa: BLE001 — warmup is best-effort by contract
            logger.debug("openrouter: warmup failed", exc_info=True)
            return False

    # ── 요청 ──────────────────────────────────────────────────────────

    def _chat_messages(self, request: APIRequest) -> List[Dict[str, Any]]:
        model = request.model
        return canonical_messages_to_openai(
            request.messages,
            request.system,
            system_role=self.system_role,
            assistant_extras=lambda content: assistant_reasoning_fields(content, model),
        )

    def _heal_request_kwargs(
        self, kwargs: Dict[str, Any], exc: BaseException
    ) -> Optional[Dict[str, Any]]:
        """OpenAI 의 고침에 하나를 더한다 — 이 모델이 받지 않는 생각 설정이면 빼고 다시 보낸다.

        예: 늘 생각하는 모델에 끄기(``enabled: false``), 받지 않는 강도. 고른 생각 값은 이 요청에서
        쓰이지 않는다(모델 기본으로 돈다).
        """
        healed = super()._heal_request_kwargs(kwargs, exc)
        if healed is not None:
            return healed
        extra = kwargs.get("extra_body")
        if not isinstance(extra, dict) or "reasoning" not in extra:
            return None
        msg = str(getattr(exc, "message", "") or exc).lower()
        if "reasoning" not in msg and "thinking" not in msg and "effort" not in msg:
            return None
        retry = dict(kwargs)
        rest = {k: v for k, v in extra.items() if k != "reasoning"}
        if rest:
            retry["extra_body"] = rest
        else:
            retry.pop("extra_body", None)
        logger.warning(
            "openrouter: model %r refused the reasoning setting %r — retrying with the model default",
            kwargs.get("model"),
            extra.get("reasoning"),
        )
        return retry

    # ── 응답 ──────────────────────────────────────────────────────────

    def _parse_response(self, raw: Any) -> APIResponse:
        error = _field(raw, "error")
        choices = _field(raw, "choices")
        if error is not None or not choices:
            raise error_from_body(error, fallback="OpenRouter returned no choices")
        response = super()._parse_response(raw)
        message = _field(choices[0], "message")
        details = merge_reasoning_details(list(_field(message, "reasoning_details") or []))
        text = reasoning_text(details, _field(message, "reasoning"))
        block = reasoning_block(details, text, response.model or "")
        if block is not None:
            response.content.insert(0, block)
        return response

    def _on_stream_delta(self, delta: Any, extras: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        fragments = _field(delta, "reasoning_details")
        if isinstance(fragments, list) and fragments:
            extras.setdefault("reasoning_details", []).extend(fragments)
            shown = reasoning_text([_as_dict(f) for f in fragments])
            if shown:
                events.append({"type": "thinking_delta", "text": shown})
            return events
        # reasoning_details 없이 글만 내는 모델.
        plain = _field(delta, "reasoning")
        if isinstance(plain, str) and plain:
            extras["reasoning"] = str(extras.get("reasoning") or "") + plain
            events.append({"type": "thinking_delta", "text": plain})
        return events

    def _stream_leading_blocks(self, extras: Dict[str, Any], model: str) -> List[ContentBlock]:
        details = merge_reasoning_details(list(extras.get("reasoning_details") or []))
        text = reasoning_text(details, extras.get("reasoning"))
        block = reasoning_block(details, text, model)
        return [block] if block is not None else []

    def _parse_usage(self, usage_data: Any) -> TokenUsage:
        usage = super()._parse_usage(usage_data)
        cost = _field(usage_data, "cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            usage.cost_usd = float(cost)
        return usage

    def _classify_error(self, e: Exception) -> APIError:
        """스트림 도중의 ``error`` 청크(SDK 의 ``openai.APIError``)는 본문의 ``code`` 로 분류한다."""
        try:
            import openai
        except ImportError:  # pragma: no cover — SDK 없는 환경
            return super()._classify_error(e)
        if isinstance(e, openai.APIError) and not isinstance(e, openai.APIStatusError):
            body = getattr(e, "body", None)
            if isinstance(body, dict) and ("code" in body or "message" in body):
                return error_from_body(body, fallback=str(e))
        return super()._classify_error(e)


__all__ = [
    "OPENROUTER_BASE_URL",
    "OpenRouterClient",
    "assistant_reasoning_fields",
    "error_from_body",
    "merge_reasoning_details",
    "reasoning_text",
]
