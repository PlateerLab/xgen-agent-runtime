"""GenAI span — LLM 호출 1건(``chat {model}``)과 도구 실행 1건(``execute_tool {tool}``).

이름 ``{...}`` 자리에는 저카디널리티 값(모델·도구 이름)만 들어간다 — 규약이 허용하는 예외이고,
trace UI 의 operation 목록에서 바로 읽힌다. span 수명은 ``spans`` 에 맡기고, 여기서는 무엇을
어떤 키로 남길지만 정한다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from xgen_agent_runtime.observability import attributes as attrs
from xgen_agent_runtime.observability.spans import (
    SpanHandle,
    client_kind,
    current_span,
    internal_kind,
)


def _span_name(operation: str, target: str) -> str:
    target = (target or "").strip()
    return f"{operation} {target}" if target else operation


# ── LLM 호출 ────────────────────────────────────────────────────────────────


@contextmanager
def llm_call_span(
    *,
    model: str,
    provider: str = "",
    stream: bool = False,
    message_count: Optional[int] = None,
    has_tools: Optional[bool] = None,
) -> Iterator[SpanHandle]:
    """LLM 호출 1건(재시도 포함). 응답은 :func:`record_llm_response` 로 남긴다."""
    attributes: Dict[str, Any] = {
        attrs.GEN_AI_OPERATION_NAME: attrs.OP_CHAT,
        attrs.GEN_AI_PROVIDER_NAME: provider,
        attrs.GEN_AI_REQUEST_MODEL: model,
        attrs.XGEN_LLM_STREAM: bool(stream),
        attrs.XGEN_LLM_MESSAGE_COUNT: message_count,
        attrs.XGEN_LLM_HAS_TOOLS: has_tools,
    }
    with current_span(_span_name(attrs.OP_CHAT, model), client_kind(), attributes) as span:
        yield span


def _usage_attributes(usage: Any) -> Dict[str, Any]:
    """호출 1건의 토큰. 캐시 토큰은 0 이면 생략한다 — 캐시를 안 쓰는 모델이 대부분이라 잡음이다."""
    payload: Dict[str, Any] = {
        attrs.GEN_AI_USAGE_INPUT_TOKENS: int(getattr(usage, "input_tokens", 0) or 0),
        attrs.GEN_AI_USAGE_OUTPUT_TOKENS: int(getattr(usage, "output_tokens", 0) or 0),
    }
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    if cache_read:
        payload[attrs.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read
    if cache_write:
        payload[attrs.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = cache_write
    return payload


def record_llm_response(span: SpanHandle, response: Any) -> None:
    """응답의 모델·종료 사유·토큰·크기를 호출 span 에 남긴다. 본문은 남기지 않는다."""
    if not span.recording or response is None:
        return
    payload: Dict[str, Any] = {
        attrs.GEN_AI_RESPONSE_MODEL: getattr(response, "model", None),
        attrs.GEN_AI_RESPONSE_ID: getattr(response, "message_id", None),
        attrs.XGEN_LLM_OUTPUT_LENGTH: len(getattr(response, "text", "") or ""),
        attrs.XGEN_LLM_TOOL_CALLS: len(getattr(response, "tool_calls", None) or []),
    }
    stop_reason = getattr(response, "stop_reason", "")
    if stop_reason:
        payload[attrs.GEN_AI_RESPONSE_FINISH_REASONS] = [str(stop_reason)]
    usage = getattr(response, "usage", None)
    if usage is not None:
        payload.update(_usage_attributes(usage))
    span.set(payload)


# ── 도구 실행 ───────────────────────────────────────────────────────────────


@contextmanager
def tool_span(*, tool_name: str) -> Iterator[SpanHandle]:
    """도구 실행 1건. 결과는 :func:`record_tool_result` 로 남긴다."""
    attributes: Dict[str, Any] = {
        attrs.GEN_AI_OPERATION_NAME: attrs.OP_EXECUTE_TOOL,
        attrs.GEN_AI_TOOL_NAME: tool_name,
        attrs.GEN_AI_TOOL_TYPE: attrs.TOOL_TYPE_FUNCTION,
    }
    with current_span(
        _span_name(attrs.OP_EXECUTE_TOOL, tool_name), internal_kind(), attributes
    ) as span:
        yield span


def _result_size(result: Any) -> Optional[int]:
    content = getattr(result, "display_text", None) or getattr(result, "content", None)
    if isinstance(content, (str, bytes, list, dict)):
        return len(content)
    return None


def record_tool_result(span: SpanHandle, result: Any) -> None:
    """결과의 성패와 크기만 남긴다. 본문은 남기지 않는다."""
    if not span.recording or result is None:
        return
    is_error = bool(getattr(result, "is_error", False))
    span.set(
        {
            attrs.XGEN_TOOL_IS_ERROR: is_error,
            attrs.XGEN_TOOL_RESULT_SIZE: _result_size(result),
        }
    )
    if is_error:
        span.fail("tool_error")


__all__ = ["llm_call_span", "record_llm_response", "record_tool_result", "tool_span"]
