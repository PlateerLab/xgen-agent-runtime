"""OpenAI client — Responses API(추론 모델) 와 Chat Completions(그 밖).

Ported from the former :class:`OpenAIProvider` in
``stages/s06_api/artifact/openai/providers.py``. Translators are
imported from :mod:`xgen_agent_runtime.llm_client.translators`, which
re-exports from the s06_api module during the PR-3→PR-4 bridge.

어느 표면으로 부르는가 (2026-10-01)
-----------------------------------
OpenAI 공식 엔드포인트의 추론 모델(``RESPONSES_FAMILIES``)은 **Responses API** 로 부른다. GPT-5.4 부터
Chat Completions 는 함수 도구와 생각을 함께 받지 않는다(400 "Function tools with reasoning_effort are not
supported … use /v1/responses"). gpt-6-sol 은 기본 강도가 medium 이라 생각을 고르지 않아도 도구만 있으면
실패했고, gpt-6-astra·gpt-6.1-sol 은 끌 수도 없어 Chat Completions 로는 도구를 쓸 수 없었다. dev 실측에서
Responses 는 10개 모델의 모든 강도에서 도구 호출 → 결과 → 답까지 받았다(``translators/_responses``).

그 밖(gpt-4.x, 공식이 아닌 base_url, Azure·vLLM·호환 서버 하위 클래스)은 Chat Completions 그대로다. 그쪽에서
같은 400 이 나면 생각을 끄고 한 번 다시 보낸다(``_heal_request_kwargs``) — 답은 하되 경고를 남긴다.
운영 스위치 ``XGEN_OPENAI_API_SURFACE``: ``auto``(기본)·``responses``(공식이 아닌 base_url 도 Responses)·``chat``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlsplit

from xgen_agent_runtime.core.errors import APIError, ErrorCategory
from xgen_agent_runtime.core.state import TokenUsage
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.translators import (
    canonical_messages_to_openai,
    canonical_thinking_to_openai,
    canonical_tool_choice_to_openai,
    canonical_tools_to_openai,
    normalize_stop_reason,
)
from xgen_agent_runtime.llm_client.translators._responses import (
    REASONING_BLOCK,
    REASONING_PROVIDER,
    canonical_to_responses_input,
    canonical_tool_choice_to_responses,
    canonical_tools_to_responses,
    reasoning_summary_text,
    response_format_to_responses,
    stop_reason_of,
    strip_reasoning_items,
)
from xgen_agent_runtime.llm_client.types import APIRequest, APIResponse, ContentBlock


logger = logging.getLogger(__name__)


# ── 어느 표면으로 부르는가 ─────────────────────────────────────────────

#: Responses API 로 부르는 모델 묶음(접두사). 추론 모델 — 생각과 도구를 함께 쓰려면 Responses 여야 한다.
RESPONSES_FAMILIES: tuple[str, ...] = ("gpt-5", "gpt-6", "o1", "o3", "o4")


#: 생각이 시작된 뒤 첫 글자까지 기다리는 상한(초). 높은 강도는 몇 분씩 생각한다 — 그 사이 스트림
#: 감시(무응답 상한)가 끊지 않게 살아 있다는 신호를 보낸다. 이 시간이 지나면 보내지 않는다.
def _reasoning_wait_s() -> float:
    try:
        return max(0.0, float(os.getenv("XGEN_LLM_REASONING_TIMEOUT_S", "900")))
    except ValueError:
        return 900.0


#: 살아 있다는 신호의 간격(초) — 스트림 감시의 무응답 상한(기본 120초)보다 짧게.
_HEARTBEAT_S = 20.0


def _api_surface() -> str:
    mode = str(os.getenv("XGEN_OPENAI_API_SURFACE", "auto") or "auto").strip().lower()
    return mode if mode in ("auto", "responses", "chat") else "auto"


def _is_official_endpoint(base_url: Optional[str]) -> bool:
    """OpenAI 공식 엔드포인트인가 — 비었거나 ``*.openai.com``. 사내 게이트웨이·프록시는 Responses 를
    모를 수 있어 Chat Completions 로 둔다(``XGEN_OPENAI_API_SURFACE=responses`` 로 바꿀 수 있다)."""
    if not base_url:
        return True
    host = (urlsplit(str(base_url)).hostname or "").lower()
    return host == "api.openai.com" or host.endswith(".openai.com")


def model_uses_responses(model: str) -> bool:
    name = str(model or "").strip().lower()
    return any(name.startswith(prefix) for prefix in RESPONSES_FAMILIES)


_TOOLS_WITH_REASONING = "function tools with reasoning_effort are not supported"


# ── ``max_tokens`` → ``max_completion_tokens`` migration ────────────
#
# OpenAI's reasoning families reject the classic ``max_tokens`` kwarg:
#
#   ``Unsupported parameter: 'max_tokens' is not supported with this
#     model. Use 'max_completion_tokens' instead.``
#
# The audit (§1-4) flagged this drift as "already real in prod" with
# zero defense on the OpenAI boundary. Two layers, mirroring the
# Anthropic pattern proven in 2.1.2/2.1.3:
#
#   1. Proactive — the static prefix table below sends
#      ``max_completion_tokens`` up front for families known to demand
#      it. Prefix match so dated/sized variants (``o3-mini``,
#      ``gpt-5.2-codex``) ride along without a code change.
#   2. Reactive — ``_heal_request_kwargs`` rebuilds + retries once when
#      the 400 names the rename, covering whatever family ships after
#      this table goes stale. Every reactive heal warns loudly so the
#      table gets refreshed instead of paying the retry forever.
_MAX_COMPLETION_TOKENS_PREFIXES: tuple[str, ...] = (
    "o1",
    "o3",
    "o4",
    "gpt-5",
)


def _model_requires_max_completion_tokens(model: str) -> bool:
    """True iff ``model`` belongs to a family known to reject
    ``max_tokens`` in favour of ``max_completion_tokens``."""
    return any(model.startswith(prefix) for prefix in _MAX_COMPLETION_TOKENS_PREFIXES)


# ── Reasoning families reject sampling params ───────────────────────
#
# The same o-series / gpt-5 families that demand ``max_completion_tokens``
# also refuse the classic sampler knobs — the live 400 reads
#
#   ``Unsupported value: 'temperature' does not support 0.2 with this
#     model. Only the default (1) value is supported.``
#
# (and the analogous ``Unsupported parameter: 'top_p' is not supported
# with this model.``). Mirror of Anthropic's
# ``_model_rejects_sampling_params``: drop the kwargs at the boundary
# (INFO log so an operator who pinned a temperature sees why it was
# ignored) and keep the reactive heal below as the safety net for
# families the table doesn't know yet. ``min_p``/``top_k`` never reach
# this client (``drops`` declares ``top_k``), so the key list is just
# the two the Chat Completions request actually carries.
_SAMPLING_PARAM_KEYS: tuple[str, ...] = ("temperature", "top_p")


def _model_rejects_sampling_params(model: str) -> bool:
    """True iff ``model`` belongs to a reasoning family that rejects
    ``temperature``/``top_p`` (same prefix table as the
    ``max_completion_tokens`` rename — the two quirks ship together)."""
    return _model_requires_max_completion_tokens(model)


def _response_format_to_openai(rf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """표준 response_format → OpenAI Chat Completions 형식.

    표준형은 ``json_schema`` 에 스키마 자체를 담고(CLI 백엔드 계약과 같다), OpenAI 형식은
    ``{"name": ..., "schema": ...}`` 로 감싼다. 이미 감싼 형태는 그대로 둔다. ``strict`` 는 넣지
    않는다 — OpenAI 의 strict 는 모든 속성을 required 로 요구해 선택 필드가 있는 스키마를 거절한다.
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
        return {"type": "json_schema", "json_schema": dict(body)}
    name = str(body.get("title") or "response")
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "response"
    return {"type": "json_schema", "json_schema": {"name": name, "schema": body}}


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """SDK 객체와 dict 를 같은 방식으로 읽는다(테스트 가짜·다른 SDK 판)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dump_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump(exclude_none=True))
        except TypeError:
            return dict(dump())
    return {}


class OpenAIClient(BaseClient):
    """OpenAI Chat Completions API client.

    Requires: ``pip install xgen-agent-runtime[openai]``
    """

    provider = "openai"
    _sdk_module = "openai"
    capabilities = ClientCapabilities(
        supports_thinking=False,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=False,
        supports_system_prompt=True,
        supports_structured_output=True,
        supports_session_continuity=False,
        supports_mcp_passthrough=False,
        supports_budget_limit=False,
        supports_token_usage=True,
        supports_cost_usage=False,
        is_subprocess=False,
        requires_workspace=False,
        streaming_granularity="token",
        drops=("thinking_enabled", "top_k"),
    )

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._client: Optional[Any] = None

    def configure(self, **kwargs: Any) -> None:
        super().configure(**kwargs)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI client requires the 'openai' package. "
                    "Install with: pip install xgen-agent-runtime[openai]"
                ) from e
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            # 시간 상한·재시도는 한 곳(llm_client.timeouts)에서.
            from xgen_agent_runtime.llm_client.timeouts import sdk_client_kwargs

            import openai as _openai_sdk

            kwargs.update(sdk_client_kwargs(_openai_sdk))
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        """Establish the httpx pool before the first real call.

        ``GET /v1/models`` is served by OpenAI and by every compatible
        server this package targets (vLLM, Ollama, LM Studio), so the
        subclasses inherit this unchanged.
        """
        import asyncio

        try:
            client = self._get_client()
            await asyncio.wait_for(client.models.list(), timeout=timeout_s)
            return True
        except Exception:  # noqa: BLE001 — warmup is best-effort by contract
            logger.debug("%s: warmup failed", self.provider, exc_info=True)
            return False

    def _heal_request_kwargs(
        self, kwargs: Dict[str, Any], exc: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Self-heal two reasoning-family 400s.

        1. ``max_tokens`` → ``max_completion_tokens`` rename. Triggers
           only when the message names the problem — it must mention
           ``max_tokens`` *and* either say the param is not supported
           or name the replacement kwarg. The live phrasing (verified
           against openai 2.x):

             ``Unsupported parameter: 'max_tokens' is not supported with
               this model. Use 'max_completion_tokens' instead.``

        2. ``temperature``/``top_p`` rejection — the message names the
           sampling kwarg as unsupported (``Unsupported value:
           'temperature' does not support 0.2 with this model`` /
           ``Unsupported parameter: 'top_p' is not supported with this
           model``); the named key is stripped.

        Both cover reasoning families the static
        ``_MAX_COMPLETION_TOKENS_PREFIXES`` table doesn't know yet.
        Pure: always returns a fresh dict or ``None``.
        """
        raw_msg = str(getattr(exc, "message", "") or exc)
        msg = raw_msg.lower()

        if "input" in kwargs and "messages" not in kwargs:
            return self._heal_responses_kwargs(kwargs, raw_msg)

        # Class 3 — 도구 + 생각 거절(GPT-5.4 이후의 Chat Completions). 공식 엔드포인트의 추론 모델은
        # Responses 로 가므로 여기 오는 것은 프록시·Azure·스위치로 Chat Completions 를 고른 경우다.
        # 생각을 끄고 다시 보낸다 — 답은 하되, 고른 생각 강도는 이 요청에서 쓰이지 않는다(경고가 남는다).
        if (
            _TOOLS_WITH_REASONING in msg
            and kwargs.get("tools")
            and kwargs.get("reasoning_effort") != "none"
        ):
            retry = dict(kwargs)
            retry["reasoning_effort"] = "none"
            for key in _SAMPLING_PARAM_KEYS:
                retry.pop(key, None)
            return retry

        # Class 4 — 이 모델이 받지 않는 생각 강도. 400 이 받는 값을 알려 주면 가장 가까운 값으로.
        if "reasoning_effort" in kwargs and "supported values" in msg:
            from xgen_agent_runtime.llm_client.thinking import nearest_supported_effort

            better = nearest_supported_effort(raw_msg, str(kwargs.get("reasoning_effort") or ""))
            if better and better != kwargs.get("reasoning_effort"):
                retry = dict(kwargs)
                retry["reasoning_effort"] = better
                return retry

        # Class 5 — 생각 파라미터 자체를 모르는 모델(gpt-4.1 등).
        if "reasoning_effort" in kwargs and (
            "unrecognized request argument supplied: reasoning_effort" in msg
            or "'reasoning_effort' is not supported" in msg
        ):
            retry = dict(kwargs)
            retry.pop("reasoning_effort", None)
            return retry

        # Class 2 — sampling-param rejection. The 400 names the kwarg
        # (``'temperature' does not support 0.2 with this model`` /
        # ``'top_p' is not supported with this model``); strip every
        # named sampling key that is actually in the request and retry.
        # Checked first because a request can carry both quirks and the
        # API reports only one per round-trip — each heal fixes the one
        # it was told about, the retry surfaces the next.
        if ("not supported" in msg or "unsupported" in msg) and "max_tokens" not in msg:
            named = [
                key
                for key in _SAMPLING_PARAM_KEYS
                if key in kwargs
                and (f"'{key}'" in msg or f"`{key}`" in msg or f" {key} " in f" {msg} ")
            ]
            if named:
                retry = dict(kwargs)
                for key in named:
                    retry.pop(key, None)
                return retry

        # Class 1 — ``max_tokens`` → ``max_completion_tokens`` rename.
        if "max_tokens" not in kwargs:
            return None
        if "max_tokens" not in msg:
            return None
        if "max_completion_tokens" not in msg and "not supported" not in msg:
            return None
        retry = dict(kwargs)
        retry["max_completion_tokens"] = retry.pop("max_tokens")
        return retry

    def _uses_responses(self, model: str) -> bool:
        """이 요청을 Responses API 로 보내는가 — 모듈 머리말의 규칙. 하위 클래스(Azure·vLLM·호환
        서버)는 provider 이름이 달라 언제나 Chat Completions 다."""
        if self.provider != "openai":
            return False
        mode = _api_surface()
        if mode == "chat":
            return False
        if mode == "responses":
            return True
        return _is_official_endpoint(self._base_url) and model_uses_responses(model)

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        if self._uses_responses(request.model):
            kwargs = self._build_responses_kwargs(request)
            raw = await self._invoke_with_heal(
                client.responses.create,
                kwargs,
                purpose=purpose or "responses.create",
            )
            return self._parse_responses_response(raw, request.model)
        kwargs = self._build_kwargs(request)
        raw = await self._invoke_with_heal(
            client.chat.completions.create,
            kwargs,
            purpose=purpose or "chat.completions.create",
        )
        return self._parse_response(raw)

    async def create_message_stream(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        client = self._get_client()
        if self._uses_responses(request.model):
            async for event in self._stream_responses(client, request, purpose=purpose):
                yield event
            return
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        # Without this flag the Chat Completions stream sends NO usage
        # chunk at all — the harvesting branch below ran for months while
        # every streamed call aggregated $0 (audit §2.5: CostBudgetGuard
        # and both hosts' cost displays silently neutralized). The usage
        # arrives as a final chunk with empty ``choices``.
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_content = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        model = request.model
        finish_reason = ""
        usage_data: Optional[Any] = None

        # The SDK validates the request and raises the 400 at ``create()``
        # time even with ``stream=True`` — so the retry-on-heal wrapper
        # can guard stream setup exactly like the non-streaming path.
        stream = await self._invoke_with_heal(
            client.chat.completions.create,
            kwargs,
            purpose=purpose or "chat.completions.create(stream)",
        )

        try:
            async for chunk in stream:
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                if hasattr(chunk, "model") and chunk.model:
                    model = chunk.model

                if delta and delta.content:
                    accumulated_content += delta.content
                    yield {"type": "text_delta", "text": delta.content}

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = accumulated_tool_calls[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

        except Exception as e:
            raise self._classify_error(e) from e

        blocks: List[ContentBlock] = []
        if accumulated_content:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=accumulated_content,
                    raw={"type": "text", "text": accumulated_content},
                )
            )
        for tc in accumulated_tool_calls.values():
            tool_input = self._parse_tool_arguments(tc["arguments"])
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=tc["id"],
                    tool_name=tc["name"],
                    tool_input=tool_input,
                    raw={
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tool_input,
                    },
                )
            )

        response = APIResponse(
            content=blocks,
            stop_reason=normalize_stop_reason(finish_reason, "openai"),
            usage=self._parse_usage(usage_data),
            model=model,
            raw=self._provenance(),
        )
        yield {"type": "message_complete", "response": response}

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        """Canonical APIRequest → OpenAI Chat Completions kwargs."""
        messages = canonical_messages_to_openai(request.messages, request.system)

        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if request.max_tokens:
            # Reasoning families reject the classic kwarg — see the
            # ``_MAX_COMPLETION_TOKENS_PREFIXES`` block at module top.
            # Sending the right name up front avoids burning a 400 +
            # retry on every single call to o-series / gpt-5 models.
            if _model_requires_max_completion_tokens(request.model):
                logger.debug(
                    "openai: model %r takes max_completion_tokens — sent %d via the renamed kwarg",
                    request.model,
                    request.max_tokens,
                )
                kwargs["max_completion_tokens"] = request.max_tokens
            else:
                kwargs["max_tokens"] = request.max_tokens
        # Reasoning families refuse ``temperature``/``top_p`` outright —
        # see ``_model_rejects_sampling_params`` at module top. Drop at
        # the boundary (mirrors the Anthropic client) so an env that
        # ships an explicit temperature keeps working on o-series / gpt-5.
        rejects_sampling = _model_rejects_sampling_params(request.model)
        if request.temperature is not None:
            if rejects_sampling:
                logger.info(
                    "openai: dropped 'temperature'=%r — model %r refuses this sampling param",
                    request.temperature,
                    request.model,
                )
            else:
                kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            if rejects_sampling:
                logger.info(
                    "openai: dropped 'top_p'=%r — model %r refuses this sampling param",
                    request.top_p,
                    request.model,
                )
            else:
                kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        if request.tools:
            kwargs["tools"] = canonical_tools_to_openai(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = canonical_tool_choice_to_openai(request.tool_choice)

        if request.thinking:
            effort = canonical_thinking_to_openai(request.thinking)
            if effort:
                kwargs["reasoning_effort"] = effort
        if request.thinking_level:
            self._apply_thinking_level(kwargs, request)
        if "max_completion_tokens" in kwargs and self.provider in ("openai", "azure_foundry"):
            # 생각 토큰은 출력 상한 안에서 자리를 먹는다 — 높은 강도에서 빈 답으로 끝나지 않게 더 둔다.
            from xgen_agent_runtime.llm_client.thinking import openai_output_budget, thinking_spec

            level = request.thinking_level or (
                "off"
                if kwargs.get("reasoning_effort") == "none"
                else kwargs.get("reasoning_effort")
            )
            kwargs["max_completion_tokens"] = openai_output_budget(
                thinking_spec(self.thinking_provider(), request.model),
                level,
                kwargs["max_completion_tokens"],
            )

        # 구조화 출력 — 표준 요청({"type": "json_schema", "json_schema": <스키마>})을 OpenAI 전송 형식으로.
        # 예전엔 전달하지 않아 스키마가 지시문에만 있었고, Qwen(vLLM) 은 키를 지어내 메모리 사실 추출이
        # 매번 schema_mismatch 로 버려졌다(dev 60일: 메모리가 쌓인 Qwen 에이전트 59개 중 사실 장부 0개).
        if request.response_format and self.capabilities.supports_structured_output:
            wire = _response_format_to_openai(request.response_format)
            if wire is not None:
                kwargs["response_format"] = wire

        return kwargs

    def _apply_thinking_level(self, kwargs: Dict[str, Any], request: APIRequest) -> None:
        """생각의 표준 값 → OpenAI ``reasoning_effort`` 또는 vLLM 채팅 템플릿 값(llm_client.thinking 표).

        OpenAI 는 effort 가 ``none`` 이 아니면 temperature·top_p 를 받지 않는다 — 함께 내려놓는다.
        """
        from xgen_agent_runtime.llm_client.thinking import (
            openai_effort,
            thinking_spec,
            vllm_request,
        )

        spec = thinking_spec(self.thinking_provider(), request.model)
        level = str(request.thinking_level)
        if spec.via == "openai_effort":
            kwargs["reasoning_effort"] = openai_effort(level)
            if level != "off":
                for key in ("temperature", "top_p"):
                    kwargs.pop(key, None)
            return
        shaped = vllm_request(spec, level)
        if "reasoning_effort" in shaped:
            kwargs["reasoning_effort"] = shaped["reasoning_effort"]
        if "extra_body" in shaped:
            extra = dict(kwargs.get("extra_body") or {})
            for key, value in shaped["extra_body"].items():
                if isinstance(value, dict):
                    merged = dict(extra.get(key) or {})
                    merged.update(value)
                    extra[key] = merged
                else:
                    extra[key] = value
            kwargs["extra_body"] = extra

    # ── Responses API ────────────────────────────────────────────────────

    def _build_responses_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        """표준 요청 → ``responses.create`` 인자. 상태 없이(``store=False``) 부르고 생각은 암호화된 채로 받는다."""
        from xgen_agent_runtime.llm_client.thinking import (
            openai_effort,
            openai_output_budget,
            thinking_spec,
        )

        instructions, items = canonical_to_responses_input(
            request.messages, request.system, model=request.model
        )
        kwargs: Dict[str, Any] = {"model": request.model, "input": items, "store": False}
        if instructions:
            kwargs["instructions"] = instructions
        if request.tools:
            kwargs["tools"] = canonical_tools_to_responses(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = canonical_tool_choice_to_responses(request.tool_choice)

        spec = thinking_spec(self.thinking_provider(), request.model)
        level: Optional[str] = request.thinking_level or None
        if not level and request.thinking:
            level = canonical_thinking_to_openai(request.thinking)
        reasoning: Dict[str, Any] = {}
        if level and spec.kind == "levels":
            reasoning["effort"] = openai_effort(level)
        effective = level or (spec.default if spec.kind == "levels" else "")
        if spec.kind == "levels" and effective not in ("", "off", "none"):
            # 생각의 요약을 받아 화면에 "생각" 으로 흘린다(스트림의 thinking_delta).
            reasoning["summary"] = "auto"
        if reasoning:
            kwargs["reasoning"] = reasoning
        if spec.kind == "levels" or model_uses_responses(request.model):
            kwargs["include"] = ["reasoning.encrypted_content"]

        if request.max_tokens:
            kwargs["max_output_tokens"] = openai_output_budget(spec, level, request.max_tokens)
        # 생각하는 동안에는 temperature·top_p 를 받지 않는다(OpenAI: 생각을 끈 때만).
        if effective in ("", "off", "none") and not _model_rejects_sampling_params(request.model):
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            logger.debug("openai responses: stop sequences are not supported — dropped")
        if request.response_format and self.capabilities.supports_structured_output:
            wire = response_format_to_responses(request.response_format)
            if wire is not None:
                kwargs["text"] = {"format": wire}
        return kwargs

    def _heal_responses_kwargs(
        self, kwargs: Dict[str, Any], raw_msg: str
    ) -> Optional[Dict[str, Any]]:
        """Responses 의 400 중 문구가 원인을 말하는 것만 고쳐 한 번 다시 보낸다."""
        msg = raw_msg.lower()
        reasoning = kwargs.get("reasoning") if isinstance(kwargs.get("reasoning"), dict) else None

        # 받지 않는 생각 강도 — 받는 값 중 가장 가까운 것.
        if reasoning and reasoning.get("effort") and "supported values" in msg:
            from xgen_agent_runtime.llm_client.thinking import nearest_supported_effort

            better = nearest_supported_effort(raw_msg, str(reasoning["effort"]))
            if better and better != reasoning["effort"]:
                retry = dict(kwargs)
                retry["reasoning"] = {**reasoning, "effort": better}
                if better == "none":
                    retry["reasoning"].pop("summary", None)
                return retry

        # 생각 파라미터를 모르는 모델.
        if "reasoning" in kwargs and "reasoning.effort" in msg and "not supported" in msg:
            retry = {k: v for k, v in kwargs.items() if k not in ("reasoning", "include")}
            return retry

        # 생각 요약을 받을 수 없는 조직(검증 전) — 요약 없이.
        if reasoning and "summary" in reasoning and "summar" in msg:
            retry = dict(kwargs)
            retry["reasoning"] = {k: v for k, v in reasoning.items() if k != "summary"}
            if not retry["reasoning"]:
                retry.pop("reasoning")
            return retry

        # 돌려준 생각 항목을 받지 못했다(다른 모델·만료·짝이 안 맞음) — 생각 항목 없이.
        items = kwargs.get("input")
        if (
            isinstance(items, list)
            and any(isinstance(i, dict) and i.get("type") == "reasoning" for i in items)
            and ("reasoning" in msg or "encrypted" in msg)
        ):
            retry = dict(kwargs)
            retry["input"] = strip_reasoning_items(items)
            return retry

        # 출력 상한이 모델 한도를 넘었다 — 알려 준 한도로(없으면 절반).
        if "max_output_tokens" in msg and kwargs.get("max_output_tokens"):
            current = int(kwargs["max_output_tokens"])
            limits = [
                int(n) for n in re.findall(r"\b(\d{4,7})\b", raw_msg) if 1024 <= int(n) < current
            ]
            retry = dict(kwargs)
            retry["max_output_tokens"] = max(limits) if limits else max(1024, current // 2)
            return retry

        # 샘플링 파라미터 거절.
        if "not supported" in msg or "unsupported" in msg:
            named = [
                k
                for k in _SAMPLING_PARAM_KEYS
                if k in kwargs and (f"'{k}'" in msg or f" {k} " in f" {msg} ")
            ]
            if named:
                return {k: v for k, v in kwargs.items() if k not in named}
        return None

    def _parse_responses_response(self, raw: Any, model: str) -> APIResponse:
        """Responses 응답 → 표준 응답. 블록은 출력 항목의 순서 그대로(생각 → 글/도구 호출)."""
        blocks: List[ContentBlock] = []
        has_calls = False
        for item in _get_attr(raw, "output", None) or []:
            kind = _get_attr(item, "type", "")
            if kind == "reasoning":
                dumped = _dump_item(item)
                summary = reasoning_summary_text(item)
                kept: Dict[str, Any] = {
                    k: dumped[k] for k in ("id", "summary", "encrypted_content") if k in dumped
                }
                kept["type"] = "reasoning"
                blocks.append(
                    ContentBlock(
                        type=REASONING_BLOCK,
                        thinking_text=summary or None,
                        raw={
                            "type": REASONING_BLOCK,
                            "provider": REASONING_PROVIDER,
                            "model": model,
                            "summary": summary,
                            "item": kept,
                        },
                    )
                )
            elif kind == "message":
                parts = []
                for part in _get_attr(item, "content", None) or []:
                    ptype = _get_attr(part, "type", "")
                    if ptype == "output_text":
                        parts.append(str(_get_attr(part, "text", "") or ""))
                    elif ptype == "refusal":
                        parts.append(str(_get_attr(part, "refusal", "") or ""))
                text = "".join(parts)
                if not text:
                    continue
                raw_block: Dict[str, Any] = {"type": "text", "text": text}
                phase = _get_attr(item, "phase", None)
                if phase:
                    raw_block["_meta"] = {"openai_phase": str(phase)}
                blocks.append(ContentBlock(type="text", text=text, raw=raw_block))
            elif kind == "function_call":
                has_calls = True
                call_id = str(_get_attr(item, "call_id", "") or "")
                name = str(_get_attr(item, "name", "") or "")
                tool_input = self._parse_tool_arguments(_get_attr(item, "arguments", "") or "")
                raw_call: Dict[str, Any] = {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": tool_input,
                }
                item_id = _get_attr(item, "id", None)
                if item_id:
                    raw_call["_meta"] = {"openai_item_id": str(item_id)}
                blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=call_id,
                        tool_name=name,
                        tool_input=tool_input,
                        raw=raw_call,
                    )
                )
        provenance = self._provenance()
        provenance["response"] = raw
        provenance["api"] = "responses"
        return APIResponse(
            content=blocks,
            stop_reason=stop_reason_of(raw, has_tool_calls=has_calls),
            usage=self._parse_responses_usage(_get_attr(raw, "usage", None)),
            model=str(_get_attr(raw, "model", "") or model),
            message_id=str(_get_attr(raw, "id", "") or ""),
            raw=provenance,
        )

    @staticmethod
    def _parse_responses_usage(usage: Any) -> TokenUsage:
        if usage is None:
            return TokenUsage()
        details = _get_attr(usage, "input_tokens_details", None)
        cached = _get_attr(details, "cached_tokens", 0) if details is not None else 0
        return TokenUsage(
            input_tokens=int(_get_attr(usage, "input_tokens", 0) or 0),
            output_tokens=int(_get_attr(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(cached or 0),
        )

    async def _stream_responses(
        self, client: Any, request: APIRequest, *, purpose: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """Responses 스트림 → 표준 스트림 청크(``text_delta``·``thinking_delta``·``message_complete``).

        생각은 몇 분씩 걸릴 수 있고 그동안 서버는 아무것도 보내지 않는다. 생각이 시작되면(생각 항목이
        열리면) 빈 ``thinking_delta`` 로 "응답이 시작됐다" 를 알리고, 그 뒤 ``_HEARTBEAT_S`` 마다
        ``heartbeat`` 청크를 보내 스트림 감시가 무응답으로 끊지 않게 한다(``XGEN_LLM_REASONING_TIMEOUT_S``
        까지). 최종 응답은 ``response.completed``·``response.incomplete`` 가 실어 오는 응답 전체로 만든다.
        """
        kwargs = self._build_responses_kwargs(request)
        kwargs["stream"] = True
        stream = await self._invoke_with_heal(
            client.responses.create,
            kwargs,
            purpose=purpose or "responses.create(stream)",
        )
        final: Any = None
        reasoning_started_at: Optional[float] = None
        summary_parts = 0
        iterator = stream.__aiter__()
        pending: Optional[asyncio.Future] = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(iterator.__anext__())
                wait_s = _HEARTBEAT_S if reasoning_started_at is not None else None
                done, _ = await asyncio.wait({pending}, timeout=wait_s)
                if not done:
                    if time.monotonic() - (reasoning_started_at or 0.0) <= _reasoning_wait_s():
                        yield {"type": "heartbeat"}
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    break
                finally:
                    pending = None
                etype = str(_get_attr(event, "type", "") or "")
                if etype == "response.output_text.delta":
                    reasoning_started_at = None
                    delta = _get_attr(event, "delta", "")
                    if delta:
                        yield {"type": "text_delta", "text": str(delta)}
                elif etype == "response.reasoning_summary_text.delta":
                    delta = _get_attr(event, "delta", "")
                    if delta:
                        yield {"type": "thinking_delta", "text": str(delta)}
                elif etype == "response.reasoning_summary_part.added":
                    summary_parts += 1
                    if summary_parts > 1:
                        yield {"type": "thinking_delta", "text": "\n\n"}
                elif etype == "response.output_item.added":
                    item = _get_attr(event, "item", None)
                    if _get_attr(item, "type", "") == "reasoning":
                        reasoning_started_at = time.monotonic()
                        yield {"type": "thinking_delta", "text": ""}
                    else:
                        reasoning_started_at = None
                elif etype in ("response.completed", "response.incomplete"):
                    final = _get_attr(event, "response", None)
                elif etype == "response.failed":
                    response = _get_attr(event, "response", None)
                    error = _get_attr(response, "error", None)
                    message = _get_attr(error, "message", "") or "OpenAI response failed"
                    code = str(_get_attr(error, "code", "") or "")
                    category = (
                        ErrorCategory.RATE_LIMITED
                        if "rate" in code
                        else ErrorCategory.SERVER_ERROR
                        if code in ("server_error", "")
                        else ErrorCategory.BAD_REQUEST
                    )
                    raise APIError(str(message), category=category)
                elif etype == "error":
                    message = _get_attr(event, "message", "") or "OpenAI stream error"
                    raise APIError(str(message), category=ErrorCategory.SERVER_ERROR)
        except APIError:
            raise
        except Exception as e:
            raise self._classify_error(e) from e
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
            aclose = getattr(stream, "close", None) or getattr(iterator, "aclose", None)
            if aclose is not None:
                try:
                    out = aclose()
                    if hasattr(out, "__await__"):
                        await out
                except Exception:  # noqa: BLE001 — 닫기 실패가 결과를 가리지 않는다
                    pass

        if final is None:
            raise APIError(
                "OpenAI Responses stream ended without a final response",
                category=ErrorCategory.SERVER_ERROR,
            )
        yield {
            "type": "message_complete",
            "response": self._parse_responses_response(final, request.model),
        }

    def _parse_response(self, raw: Any) -> APIResponse:
        choice = raw.choices[0]
        blocks: List[ContentBlock] = []

        if choice.message.content:
            blocks.append(
                ContentBlock(
                    type="text",
                    text=choice.message.content,
                    raw={"type": "text", "text": choice.message.content},
                )
            )

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_input = self._parse_tool_arguments(tc.function.arguments)
                blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=tc.id,
                        tool_name=tc.function.name,
                        tool_input=tool_input,
                        raw={
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tool_input,
                        },
                    )
                )

        stop_reason = normalize_stop_reason(choice.finish_reason or "", "openai")

        # ``raw`` is the provenance channel — see ``BaseClient._provenance``.
        provenance = self._provenance()
        provenance["response"] = raw

        return APIResponse(
            content=blocks,
            stop_reason=stop_reason,
            usage=self._parse_usage(getattr(raw, "usage", None)),
            model=raw.model,
            message_id=raw.id,
            raw=provenance,
        )

    def _parse_tool_arguments(self, raw: Any) -> Any:
        """Parse a tool-call ``arguments`` JSON string into Python.

        Default: strict ``json.loads`` with a ``{}`` fallback — the OpenAI
        SDK emits well-formed JSON, so nothing fancier is warranted. The
        OpenAI-compatible *local* clients (Ollama / LM Studio / custom)
        override this to repair the malformed JSON that local servers
        commonly emit (trailing commas, ``None``/``True`` literals,
        markdown fences) before giving up. Shared by the streaming and
        non-streaming parse paths so the two can't drift.
        """
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _parse_usage(self, usage_data: Any) -> TokenUsage:
        """OpenAI usage object → canonical :class:`TokenUsage`.

        Shared by the streaming and non-streaming paths so the cache
        accounting can't drift between them again. Maps
        ``prompt_tokens_details.cached_tokens`` (OpenAI's automatic
        prompt-cache hit counter) onto ``cache_read_input_tokens`` —
        same semantics as Anthropic's field. NOTE: unlike Anthropic,
        OpenAI's ``prompt_tokens`` already *includes* the cached
        portion; pricing code discounts cache reads, it must not add
        them on top. ``prompt_tokens_details`` may be an object (openai
        SDK) or a plain dict (vLLM and other compatible servers).
        """
        if usage_data is None:
            return TokenUsage()
        details = getattr(usage_data, "prompt_tokens_details", None)
        if isinstance(details, dict):
            cached = details.get("cached_tokens", 0) or 0
        else:
            cached = getattr(details, "cached_tokens", 0) or 0
        return TokenUsage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            cache_read_input_tokens=cached,
        )

    def _classify_error(self, e: Exception) -> APIError:
        try:
            import openai
        except ImportError:
            return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)

        if isinstance(e, openai.RateLimitError):
            return APIError(str(e), category=ErrorCategory.RATE_LIMITED, cause=e)
        if isinstance(e, openai.APITimeoutError):
            return APIError(str(e), category=ErrorCategory.TIMEOUT, cause=e)
        if isinstance(e, openai.APIConnectionError):
            return APIError(str(e), category=ErrorCategory.NETWORK, cause=e)
        if isinstance(e, openai.AuthenticationError):
            return APIError(str(e), category=ErrorCategory.AUTH, status_code=401, cause=e)
        if isinstance(e, openai.BadRequestError):
            return APIError(str(e), category=ErrorCategory.BAD_REQUEST, status_code=400, cause=e)
        if isinstance(e, openai.InternalServerError):
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, status_code=500, cause=e)
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
