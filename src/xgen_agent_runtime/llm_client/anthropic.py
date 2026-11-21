"""Anthropic Messages API client.

Near-verbatim port of the former :class:`AnthropicProvider` in
``stages/s06_api/artifact/default/providers.py``, restructured to
inherit from :class:`BaseClient` and expose a :class:`ClientCapabilities`
profile.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from xgen_agent_runtime.core.errors import APIError, ErrorCategory
from xgen_agent_runtime.core.state import TokenUsage
from xgen_agent_runtime.llm_client.base import (
    BaseClient,
    ClientCapabilities,
    prompt_tokens_with_cache,
)
from xgen_agent_runtime.llm_client.translators import canonical_messages_to_anthropic
from xgen_agent_runtime.llm_client.types import APIRequest, APIResponse, ContentBlock


logger = logging.getLogger(__name__)


# ── Alias resolution ────────────────────────────────────────────────
#
# The Anthropic Messages API only accepts canonical model IDs
# (``claude-opus-4-7``, ``claude-sonnet-4-6``, ``claude-haiku-4-5-…``);
# short aliases like ``opus`` / ``sonnet`` / ``haiku`` are only valid
# on the ``claude`` CLI binary surface, not on the HTTP API. Apps
# that share a model config between the CLI and HTTP paths (geny,
# anyone wrapping us) routinely tripped on this: the env stores
# ``opus`` from the CLI flow, the next session pins ``anthropic`` as
# its Stage 6 provider, and the API returns
# ``404 model: opus``.
#
# Resolve the well-known aliases to today's tier-leader canonical IDs
# right before the SDK call. Pinned to specific versions on purpose —
# silently floating an env's model id across releases would be a
# nasty surprise. Bump the right-hand side here when shipping a new
# default tier leader.
_ANTHROPIC_MODEL_ALIASES: Dict[str, str] = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


# ── Thinking-block binding (Claude 5.5 generation) ──────────────────
#
# A thinking block's signature records what came before it when the model wrote it: the system
# prompt, the tool set and every earlier message. Sonnet 5.5, Haiku 5.5, Opus 5.5 and the Fable /
# Mythos 5.1+ models check that record when the block is sent back; a request that changed any of
# it is rejected — 400 "Invalid `signature` in `thinking` block. The block is bound to a different
# conversation" (enforced by default for accounts created on or after 2026-08-31). Two things the
# runtime does on purpose change it mid-turn:
#
# * a gate tool opens its family, or ToolSearch activates a tool — the tool set (and the system
#   prompt's catalogue) grows. Every block written before is then bound to a different conversation.
#   The API allows removing a leading run of thinking blocks, so each block remembers the system
#   prompt and tool set it was written under (``_meta.binding`` — internal, never sent) and is left
#   out of later requests made under a different one (:func:`_without_unbound_thinking`). Blocks
#   written after the change stay valid, so only the reasoning from before the change is lost.
# * anything else (a history rewrite under context pressure) is the API's own safety net: with the
#   ``thinking-binding-controls-2026-08-01`` beta and ``block_binding.prefix_mismatch_behavior:
#   "drop_block"`` the API drops the stale blocks instead of failing the request. That field is
#   accepted only with ``thinking.type: "adaptive"``; for the other thinking types (Sonnet 5.5's
#   ``between_tools``, ``disabled``) a rejected request is retried once without thinking blocks
#   (:func:`_retry_kwargs_after_deprecation`).
_THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"
_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")


def _scrub_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _scrub_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_scrub_cache_control(v) for v in value]
    return value


def thinking_binding(system: Any, tools: Any) -> str:
    """The part of a thinking block's binding that is not the messages: system prompt + tool set.

    Compared the way the API compares them — ``cache_control`` markers, the tools' order and
    surrounding whitespace of system text do not count; tool names, descriptions and schemas do.
    """
    parts: List[Any] = []
    if isinstance(system, str):
        if system.strip():
            parts.append(system.strip())
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif block is not None:
                parts.append(_scrub_cache_control(block))
    tool_set = sorted(
        (_scrub_cache_control(t) for t in (tools or []) if isinstance(t, dict)),
        key=lambda t: str(t.get("name") or t.get("type") or ""),
    )
    raw = json.dumps([parts, tool_set], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:16]


def _block_binding(block: Any) -> Optional[str]:
    meta = block.get("_meta") if isinstance(block, dict) else None
    value = meta.get("binding") if isinstance(meta, dict) else None
    return value if isinstance(value, str) and value else None


def _without_unbound_thinking(
    messages: List[Dict[str, Any]], binding: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Leave out thinking blocks written under a different system prompt or tool set.

    Pure and deterministic, so every later request of the conversation leaves out the same blocks
    and the blocks written meanwhile keep a valid chain. Blocks without a recorded binding (older
    histories, another client) are kept as they are."""
    out: List[Dict[str, Any]] = []
    dropped = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if msg.get("role") != "assistant" or not isinstance(content, list):
            out.append(msg)
            continue
        kept = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") in _THINKING_BLOCK_TYPES
                and _block_binding(block) not in (None, binding)
            ):
                dropped += 1
                continue
            kept.append(block)
        if len(kept) == len(content):
            out.append(msg)
        elif kept:
            out.append({**msg, "content": kept})
        # An assistant message that held only stale thinking has nothing left to send.
    return out, dropped


def _without_any_thinking(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Every thinking block removed (text and tool calls stay) — the API's documented recovery."""
    out: List[Dict[str, Any]] = []
    dropped = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            out.append(msg)
            continue
        kept = [
            b
            for b in content
            if not (isinstance(b, dict) and b.get("type") in _THINKING_BLOCK_TYPES)
        ]
        dropped += len(content) - len(kept)
        if len(kept) == len(content):
            out.append(msg)
        elif kept:
            out.append({**msg, "content": kept})
    return out, dropped


def _is_thinking_binding_rejection(msg_lower: str) -> bool:
    return (
        "signature" in msg_lower
        and "thinking" in msg_lower
        and (
            "bound to a different conversation" in msg_lower
            or "invalid `signature`" in msg_lower
            or "invalid signature" in msg_lower
        )
    )


def _beta_values(headers: Optional[Dict[str, Any]]) -> List[str]:
    for key, value in (headers or {}).items():
        if str(key).lower() == "anthropic-beta" and value:
            return [b.strip() for b in str(value).split(",") if b.strip()]
    return []


def _with_beta(
    kwargs: Dict[str, Any], beta: str, default_headers: Optional[Dict[str, Any]]
) -> None:
    """Add ``beta`` to the request's ``anthropic-beta`` header, keeping the betas the client already
    sends (a per-request header replaces the client's default one, it does not merge)."""
    headers = {
        k: v
        for k, v in dict(kwargs.get("extra_headers") or {}).items()
        if str(k).lower() != "anthropic-beta"
    }
    betas = _beta_values(kwargs.get("extra_headers")) or _beta_values(default_headers)
    if beta not in betas:
        betas = [*betas, beta]
    headers["anthropic-beta"] = ",".join(betas)
    kwargs["extra_headers"] = headers


def _without_binding_controls(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    thinking = kwargs.get("thinking")
    if not (isinstance(thinking, dict) and "block_binding" in thinking):
        return None
    retry = dict(kwargs)
    retry["thinking"] = {k: v for k, v in thinking.items() if k != "block_binding"}
    headers = dict(kwargs.get("extra_headers") or {})
    betas = [b for b in _beta_values(headers) if b != _THINKING_BINDING_BETA]
    headers = {k: v for k, v in headers.items() if str(k).lower() != "anthropic-beta"}
    if betas:
        headers["anthropic-beta"] = ",".join(betas)
    if headers:
        retry["extra_headers"] = headers
    else:
        retry.pop("extra_headers", None)
    return retry


def _log_input_transformations(raw: Any) -> None:
    """Report thinking blocks the API dropped (``input_transformations``, present with the beta).

    ``prefix_binding_mismatch`` means something before the block changed after it was written — a
    history edit this runtime should not make, so it is a WARNING. A model switch
    (``model_binding_mismatch``) is expected and only INFO."""
    entries = getattr(raw, "input_transformations", None)
    if entries is None:
        extra = getattr(raw, "model_extra", None)
        entries = extra.get("input_transformations") if isinstance(extra, dict) else None
    if not isinstance(entries, list) or not entries:
        return
    reasons: Dict[str, List[str]] = {}
    for entry in entries:
        get = entry.get if isinstance(entry, dict) else (lambda k, e=entry: getattr(e, k, None))
        if get("type") != "thinking_dropped":
            continue
        reasons.setdefault(str(get("reason") or ""), []).append(str(get("path") or ""))
    stale = reasons.pop("prefix_binding_mismatch", [])
    if stale:
        logger.warning(
            "anthropic: the API dropped %d thinking block(s) whose earlier content changed "
            "(prefix_binding_mismatch, first at %s)",
            len(stale),
            stale[0],
        )
    for reason, paths in reasons.items():
        logger.info("anthropic: the API dropped %d thinking block(s) (%s)", len(paths), reason)


def _resolve_anthropic_model(model: str) -> str:
    """Return the canonical ID for a known short alias, otherwise the
    input unchanged. Pure function — easy to unit-test in isolation."""
    canonical = _ANTHROPIC_MODEL_ALIASES.get(model)
    if canonical is None:
        return model
    if canonical != model:
        logger.info(
            "anthropic: model alias %r resolved to canonical %r",
            model,
            canonical,
        )
    return canonical


# ── Extended-thinking sampling-param compatibility ──────────────────
#
# The Anthropic Messages API rejects ``temperature``, ``top_p`` and
# ``top_k`` when extended thinking is enabled — the sampler is fixed
# by the thinking machinery. The error reads
# ``temperature is deprecated for this model`` (despite being model-
# agnostic when ``thinking`` is set).
#
# Drop the offending fields at the boundary. Logged at INFO so an
# operator who explicitly chose a temperature can see why it was
# silently ignored.
_THINKING_INCOMPATIBLE_SAMPLING_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
)


# ── Models that reject sampling params unconditionally ──────────────
#
# Some models (currently the Opus 4.7 family — the only one verified
# against the live API in 2.1.2) reject ``temperature`` regardless of
# whether ``thinking`` is set. The error reads
# ``temperature is deprecated for this model.`` from
# ``api.anthropic.com``. The model is designed around fixed-sampler
# inference; the sampling kwargs become noise the API explicitly
# refuses.
#
# The set is keyed by the **resolved** canonical ID (so aliases get
# expanded first, see ``_resolve_anthropic_model``). Match is
# prefix-based — ``"claude-opus-4-7"`` covers any future
# ``claude-opus-4-7-20yyyymmdd`` pinned variant without needing an
# update here.
#
# AdaptiveModelRouter auto-promotes to Opus when ``thinking_enabled``
# is True (see ``stages/s06_api/artifact/default/router.py``), so an
# env that never sees Opus in its config can still hit this code
# path indirectly. The drop has to live at the boundary, not the
# router.
#
# 2026-09-28 live API 확인(temperature=0.7): opus-4-8·opus-5·sonnet-5·fable-5 도 400
# "`temperature` is deprecated for this model." — 목록에 없어서 매 호출이 400 을 한 번 맞고
# 치유 경로(_heal_request_kwargs)로 다시 보내졌다. sonnet-4-6·opus-4-6·haiku-4-5 는 받는다.
# "claude-opus-5" 는 opus-5-5, "claude-fable-5" 는 fable-5-1 까지 덮는다(같은 계열 문서 기준).
_TEMPERATURE_DEPRECATED_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos",
)

# ── 출력 한도에 걸려 쓸 수 있는 출력이 하나도 없을 때 ──────────────────
#
# Claude 5 계열은 기본으로 adaptive thinking 을 하고, thinking 도 max_tokens 안에서 쓴다.
# 노드 기본값(8192)으로는 생각만 하다 한도에 닿아 텍스트도 도구 호출도 없이 끝나는 경우가
# 있었다(로컬 벤치 claude-sonnet-5 70과제 중 8과제 — 출력 정확히 8192, 답 없음).
# 그 경우에만 한도를 올려 한 번 더 부른다. 텍스트가 이미 사용자에게 나갔으면 다시 부르지
# 않는다(중복). 다시 부른 호출이 실패하면 원래 응답을 그대로 돌려준다.
_TRUNCATION_RETRY_MAX_TOKENS_STREAM = 32000
#: 비스트리밍은 SDK HTTP 타임아웃 때문에 더 낮게.
_TRUNCATION_RETRY_MAX_TOKENS_SEND = 16000


#: thinking 파라미터 없이도 adaptive 로 생각하는 계열(Anthropic 문서: Sonnet 5·Opus 5·Opus 5.5·
#: Fable 5/5.1·Mythos·Haiku 5.5). Opus 4.7/4.8 은 생략하면 생각하지 않는다 — 넣지 않는다. Haiku 4.5 는
#: ``claude-haiku-4-5`` 라 걸리지 않는다. 여기 든 모델은 thinking 을 보내지 않아도 생각 블록을 돌려주므로
#: adaptive 를 명시해 묶임 안전망(block_binding)도 함께 싣는다 — Haiku 5.5 가 빠져 있어 홈 XGEN 에서 Haiku 5.5
#: 턴만 400 으로 끝났다(2026-11-21).
_THINKS_BY_DEFAULT_PREFIXES: tuple[str, ...] = (
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-mythos",
    "claude-haiku-5",
)


def _thinking_active(thinking: Any) -> bool:
    """이 thinking 값이 생각을 켜는가 — 비었거나 ``type: disabled`` 면 아니다."""
    if not thinking:
        return False
    # between_tools 는 Sonnet 5.5 에서 생각을 끄는 값이다(API 거절 문구가 그렇게 안내한다).
    return not (
        isinstance(thinking, dict) and thinking.get("type") in ("disabled", "between_tools")
    )


def _model_thinks_by_default(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _THINKS_BY_DEFAULT_PREFIXES)


def _truncated_without_output(message: Any) -> bool:
    """출력 한도로 멈췄고 쓸 수 있는 출력(텍스트)이 없다. 한도에서 끊긴 tool_use 는 입력이
    잘렸을 수 있어 실행하면 안 되므로 출력으로 치지 않는다."""
    if getattr(message, "stop_reason", "") != "max_tokens":
        return False
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", "") == "text" and (getattr(block, "text", "") or "").strip():
            return False
    return True


def _with_more_room(kwargs: Dict[str, Any], ceiling: int) -> Optional[Dict[str, Any]]:
    current = int(kwargs.get("max_tokens") or 0)
    if current >= ceiling:
        return None
    return {**kwargs, "max_tokens": ceiling}


def _model_rejects_sampling_params(model: str) -> bool:
    """True iff ``model`` (canonical ID) belongs to a family that
    unconditionally rejects ``temperature``/``top_p``/``top_k``."""
    return any(model.startswith(prefix) for prefix in _TEMPERATURE_DEPRECATED_PREFIXES)


# ── ``thinking.type=enabled`` → ``adaptive`` migration ──────────────
#
# Opus 4.7 (verified 2026-06-04) rejects ``thinking.type=enabled``
# and requires ``thinking.type=adaptive`` instead — the old
# enabled-with-fixed-budget shape isn't valid for the new generation
# of thinking-native models. The API error reads:
#
#   ``"thinking.type.enabled" is not supported for this model.
#     Use "thinking.type.adaptive" and "output_config.effort"``.
#
# Under ``adaptive``, the model picks its own budget; the legacy
# ``budget_tokens`` field is rejected as an extra input
# (``thinking.adaptive.budget_tokens: Extra inputs are not permitted``).
# Effort is *optional* — calls with bare ``{"type":"adaptive"}`` work.
# Translate at the boundary so callers that ship the v1 thinking
# shape continue to work against v2 models.
_THINKING_ADAPTIVE_ONLY_PREFIXES: tuple[str, ...] = ("claude-opus-4-7",)


def _model_requires_adaptive_thinking(model: str) -> bool:
    """True iff ``model`` only accepts ``thinking.type=adaptive``."""
    return any(model.startswith(prefix) for prefix in _THINKING_ADAPTIVE_ONLY_PREFIXES)


def _translate_thinking_to_adaptive(thinking: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a v1 (``type=enabled``) thinking dict to the v2
    (``type=adaptive``) shape Opus 4.7 demands.

      * ``type`` flips to ``"adaptive"``.
      * ``budget_tokens`` is dropped — the API rejects it under
        adaptive (``thinking.adaptive.budget_tokens: Extra inputs are
        not permitted``).
      * Any other unrelated keys (``display`` etc.) pass through.

    The kwarg ``output_config.effort`` is *not* added — bare
    ``{"type":"adaptive"}`` works against the live API; the API
    picks a default effort. Hosts that want to pin effort can do so
    explicitly by setting it on ``model_config`` and threading it
    through future plumbing.
    """
    out = dict(thinking)
    out["type"] = "adaptive"
    out.pop("budget_tokens", None)
    return out


# ── Last-line retry on a deprecation 400 ────────────────────────────
#
# Future Anthropic releases will deprecate more sampling params for
# more models; the static prefix list above will go stale. When the
# API surfaces the deprecation error we strip the offending field
# and retry once. Captures the same exact 400 strings Anthropic emits
# (sometimes wrapped in backticks, sometimes not).
_DEPRECATION_MSG_TO_KWARG_KEY: Dict[str, str] = {
    "temperature is deprecated": "temperature",
    "`temperature` is deprecated": "temperature",
    "top_p is deprecated": "top_p",
    "`top_p` is deprecated": "top_p",
    "top_k is deprecated": "top_k",
    "`top_k` is deprecated": "top_k",
}


def _retry_kwargs_after_deprecation(
    kwargs: Dict[str, Any],
    exc: BaseException,
) -> Optional[Dict[str, Any]]:
    """If ``exc`` is an Anthropic 400 we can self-heal, return a
    rebuilt kwargs. ``None`` means *don't retry* — let the caller
    re-raise.

    Two recognised classes today:

      1. **Sampling-param deprecation** — the API message names a
         specific field (``temperature``, ``top_p``, ``top_k``) as
         deprecated. Strip the field and retry.
      2. **Thinking v1 → v2 migration** — the API rejects
         ``thinking.type=enabled`` and asks for
         ``thinking.type.adaptive``. Translate via
         :func:`_translate_thinking_to_adaptive` and retry.

    Defends against future model rollouts our static prefix lists
    don't know about yet. Caller guarantees one retry per send (we
    never recurse); a retry that also 400s gets classified + raised
    by the outer handler.
    """
    msg = str(getattr(exc, "message", "") or exc)
    msg_lower = msg.lower()

    # Class 3 — the endpoint does not take the thinking-binding controls (a gateway or platform
    # without the beta): send the request as it was before them.
    if "block_binding" in msg_lower or _THINKING_BINDING_BETA in msg_lower:
        retry = _without_binding_controls(kwargs)
        if retry is not None:
            return retry

    # Class 4 — a thinking block is bound to a different conversation (history changed since it
    # was written, and the request could not carry ``drop_block``). The API's documented recovery:
    # the same request without thinking blocks; text and tool calls stay.
    if _is_thinking_binding_rejection(msg_lower) and isinstance(kwargs.get("messages"), list):
        messages, dropped = _without_any_thinking(kwargs["messages"])
        if dropped:
            logger.warning(
                "anthropic: thinking blocks rejected as bound to a different conversation — "
                "retrying once without the %d thinking block(s)",
                dropped,
            )
            retry = dict(kwargs)
            retry["messages"] = messages
            return retry

    # Class 2 — thinking v1→v2 migration. Run first because the
    # diagnostic is structural (the request shape, not just one
    # missing field).
    if (
        "thinking.type.enabled" in msg_lower or "thinking.type.adaptive" in msg_lower
    ) and isinstance(kwargs.get("thinking"), dict):
        thinking = kwargs["thinking"]
        if thinking.get("type") == "enabled":
            retry = dict(kwargs)
            retry["thinking"] = _translate_thinking_to_adaptive(thinking)
            return retry

    # Class 1 — sampling-param deprecation.
    for needle, key in _DEPRECATION_MSG_TO_KWARG_KEY.items():
        if needle in msg_lower and key in kwargs:
            retry = dict(kwargs)
            retry.pop(key, None)
            return retry
    return None


# ── 400 disambiguation: TOKEN_LIMIT vs BAD_REQUEST ──────────────────
#
# The 2.1.x heuristic was ``'token' in msg or 'context' in msg`` — which
# routed *param-shape* 400s into TOKEN_LIMIT. Concretely: the drift
# message this very module documents
# (``thinking.adaptive.budget_tokens: Extra inputs are not permitted``)
# contains the substring ``token``, so the next thinking-shape drift
# would have been diagnosed as "reduce your context" instead of "your
# request shape is stale" (audit §3.4). The two categories drive very
# different recovery: TOKEN_LIMIT makes Stage 6 compact the conversation and
# retry once (4.89.0: before that nothing acted on it and the turn failed);
# BAD_REQUEST is fatal and should bubble immediately.
#
# Classify TOKEN_LIMIT only on phrases the live API actually anchors
# overflow errors with, and short-circuit anything that *looks like a
# request-validation message* (a named param path or a deprecation
# notice) into BAD_REQUEST first.
_TOKEN_LIMIT_ANCHORS: tuple[str, ...] = (
    "maximum context length",
    "prompt is too long",
    "input length exceeds",
    "too many tokens",
    # The live API's combined-budget phrasing:
    # ``input length and `max_tokens` exceed context limit: …``
    "exceed context limit",
    # Bedrock's phrasing for the same overflow.
    "input is too long",
)

#: Substrings that mark a request-*validation* 400 (pydantic-style param
#: rejection or a deprecation notice) — never a context overflow.
_PARAM_SHAPE_NEEDLES: tuple[str, ...] = (
    ": extra inputs",
    "extra inputs are not permitted",
    "is deprecated",
)

#: A dotted parameter path (``thinking.adaptive.budget_tokens:``,
#: ``messages.0.content.0.image._meta:``) followed by a colon — the shape
#: the API uses to name the offending field in validation errors.
_PARAM_PATH_RE = re.compile(r"\b[a-z0-9_]+(?:\.[a-z0-9_\[\]]+)+\s*:")


def _classify_bad_request_message(msg_lower: str) -> ErrorCategory:
    """Pure classifier for an Anthropic ``BadRequestError`` message.

    Param-shape evidence wins over token-limit anchors on purpose: a
    validation message that happens to mention budget *tokens* must not
    be mistaken for an overflow (the misdiagnosis the audit flagged).
    """
    if any(needle in msg_lower for needle in _PARAM_SHAPE_NEEDLES):
        return ErrorCategory.BAD_REQUEST
    if _PARAM_PATH_RE.search(msg_lower):
        return ErrorCategory.BAD_REQUEST
    if any(anchor in msg_lower for anchor in _TOKEN_LIMIT_ANCHORS):
        return ErrorCategory.TOKEN_LIMIT
    return ErrorCategory.BAD_REQUEST


class AnthropicClient(BaseClient):
    """Real Anthropic API client using the official SDK."""

    def prompt_tokens_from_usage(self, usage: Any) -> int:
        """Anthropic (and Bedrock) report cache reads/writes OUTSIDE ``input_tokens``."""
        return prompt_tokens_with_cache(usage)

    provider = "anthropic"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=True,
        supports_system_prompt=True,
        supports_structured_output=False,
        supports_session_continuity=False,
        supports_mcp_passthrough=False,
        supports_budget_limit=False,
        supports_token_usage=True,
        supports_cost_usage=False,
        is_subprocess=False,
        requires_workspace=False,
        streaming_granularity="token",
    )

    _sdk_module = "anthropic"

    #: Send the thinking-binding controls (beta header + ``block_binding``) on adaptive-thinking
    #: requests. Off for platforms that take betas another way (Bedrock InvokeModel).
    _thinking_binding_controls = True

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
            import anthropic

            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            # 시간 상한·재시도는 한 곳(llm_client.timeouts)에서 — SDK 기본(600초·재시도 2)을
            # 그대로 두면 스테이지 재시도와 곱해져 호출 하나가 최악 2시간을 잡는다.
            from xgen_agent_runtime.llm_client.timeouts import sdk_client_kwargs

            kwargs.update(sdk_client_kwargs(anthropic))
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        """Establish the httpx pool before the first real call.

        A cheap ``GET /v1/models`` walks the full DNS + TCP + TLS
        handshake once, off the user's critical path; the SDK keeps the
        connection alive for the session's real requests.
        """
        import asyncio

        try:
            client = self._get_client()
            await asyncio.wait_for(client.models.list(limit=1), timeout=timeout_s)
            return True
        except Exception:  # noqa: BLE001 — warmup is best-effort by contract
            logger.debug("anthropic: warmup failed", exc_info=True)
            return False

    def _heal_request_kwargs(
        self, kwargs: Dict[str, Any], exc: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Route the 2.1.2/2.1.3 deprecation safety net through the
        :class:`BaseClient` heal hook.

        The mechanism (rebuild kwargs from the 400 message, retry once)
        was born here and got promoted to ``BaseClient`` in 2.2.0 so
        OpenAI/Google grow the same reflex; the Anthropic-specific
        needle table stays in :func:`_retry_kwargs_after_deprecation`
        (also kept module-level — the 2.1.x boundary tests pin it).
        """
        return _retry_kwargs_after_deprecation(kwargs, exc)

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        # Retry-on-heal safety net (``_heal_request_kwargs``). The static
        # prefix list in ``_TEMPERATURE_DEPRECATED_PREFIXES`` will go
        # stale as Anthropic deprecates more sampling params for more
        # models. When the API explicitly tells us a sampling param is
        # the problem, strip it and retry once. Beats a hard error on a
        # model whose prefix we don't know yet.
        raw_response = await self._invoke_with_heal(
            client.messages.create, kwargs, purpose=purpose or "messages.create"
        )
        roomier = (
            _with_more_room(kwargs, _TRUNCATION_RETRY_MAX_TOKENS_SEND)
            if _truncated_without_output(raw_response)
            else None
        )
        if roomier is not None:
            try:
                retried = await self._invoke_with_heal(
                    client.messages.create, roomier, purpose=purpose or "messages.create"
                )
                logger.warning(
                    "anthropic: output hit max_tokens=%s with nothing usable — retried once with %s",
                    kwargs.get("max_tokens"),
                    roomier["max_tokens"],
                )
                raw_response = retried
            except Exception:  # noqa: BLE001 — 더 부른 호출이 실패하면 원래 응답
                logger.warning(
                    "anthropic: max_tokens retry failed; keeping the truncated response",
                    exc_info=True,
                )
        return self._parse_response(raw_response, self._binding_of(request))

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
        """Streaming call via the SDK's high-level ``messages.stream()`` helper.

        NOTE: do not pass ``stream=True`` in kwargs — that helper handles it.
        """
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        client = self._get_client()
        kwargs = self._build_kwargs(request)
        binding = self._binding_of(request)

        # TTFT program (2.50.0, finding D1): iterate the FULL event
        # stream, not ``stream.text_stream``. The text-only iterator
        # silently dropped thinking deltas, so on a thinking-enabled
        # request the pipeline saw NOTHING until the model finished
        # reasoning and emitted its first text token — the entire
        # thinking budget was dead air. Raw ``content_block_delta``
        # events surface thinking (and tool input JSON) the moment they
        # arrive, matching what the google and CLI backends already do.
        def _canonical_chunk(event: Any) -> Optional[Dict[str, Any]]:
            etype = getattr(event, "type", "")
            if etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                dtype = getattr(delta, "type", "")
                if dtype == "text_delta":
                    text = getattr(delta, "text", "")
                    return {"type": "text_delta", "text": text} if text else None
                if dtype == "thinking_delta":
                    thinking = getattr(delta, "thinking", "")
                    return {"type": "thinking_delta", "text": thinking} if thinking else None
                if dtype == "input_json_delta":
                    partial = getattr(delta, "partial_json", "")
                    return {"type": "input_json_delta", "delta": partial} if partial else None
                return None
            if etype == "content_block_stop":
                return {"type": "content_block_stop"}
            return None

        try:
            text_emitted = False
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    chunk = _canonical_chunk(event)
                    if chunk is not None:
                        text_emitted = text_emitted or chunk.get("type") == "text_delta"
                        yield chunk

                final = await stream.get_final_message()
            roomier = (
                _with_more_room(kwargs, _TRUNCATION_RETRY_MAX_TOKENS_STREAM)
                if not text_emitted and _truncated_without_output(final)
                else None
            )
            if roomier is not None:
                logger.warning(
                    "anthropic: output hit max_tokens=%s with nothing usable — retrying once with %s",
                    kwargs.get("max_tokens"),
                    roomier["max_tokens"],
                )
                try:
                    async with client.messages.stream(**roomier) as stream:
                        async for event in stream:
                            chunk = _canonical_chunk(event)
                            if chunk is not None:
                                yield chunk
                        final = await stream.get_final_message()
                except Exception:  # noqa: BLE001 — 더 부른 호출이 실패하면 원래 응답
                    logger.warning(
                        "anthropic: max_tokens retry failed; keeping the truncated response",
                        exc_info=True,
                    )
            yield {
                "type": "message_complete",
                "response": self._parse_response(final, binding),
            }
        except Exception as e:
            # Same retry-on-heal safety net as ``_send``, routed through
            # the BaseClient hook. The SDK validates kwargs eagerly
            # inside the ``stream`` context manager, so the heal-able
            # 400 surfaces before any tokens reach the caller — safe to
            # retry once with the rebuilt kwargs. (Generators can't use
            # ``_invoke_with_heal`` directly; the structure stays
            # hand-rolled, the policy is shared.)
            retry_kwargs = self._heal_request_kwargs(kwargs, e)
            if retry_kwargs is not None:
                try:
                    async with client.messages.stream(**retry_kwargs) as stream:
                        async for event in stream:
                            chunk = _canonical_chunk(event)
                            if chunk is not None:
                                yield chunk
                        final = await stream.get_final_message()
                        self._report_drift_healed(
                            kwargs,
                            retry_kwargs,
                            e,
                            purpose=purpose or "messages.stream",
                        )
                        yield {
                            "type": "message_complete",
                            "response": self._parse_response(final, binding),
                        }
                    return
                except Exception as inner:
                    raise self._classify_error(inner) from inner
            raise self._classify_error(e) from e

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        # Strip executor-internal keys (e.g. ``_meta`` on image blocks added by
        # the s01 normalizer for downstream provenance) and lower unsupported
        # block types (``file``) into safe fallbacks. Without this the
        # Anthropic Messages API rejects requests with
        # ``messages.0.content.0.image._meta: Extra inputs are not permitted``.
        # Thinking written under another system prompt or tool set would fail the API's binding
        # check — leave it out (see the "Thinking-block binding" notes at module top).
        messages, unbound = _without_unbound_thinking(
            list(request.messages or []), self._binding_of(request)
        )
        if unbound:
            logger.info(
                "anthropic: left out %d thinking block(s) written under a different system prompt "
                "or tool set (a tool family opened mid-turn)",
                unbound,
            )
        sanitized_messages = canonical_messages_to_anthropic(messages)

        # Alias resolution — see ``_ANTHROPIC_MODEL_ALIASES`` docstring.
        # Pure function; no SDK call yet, so this is cheap.
        resolved_model = _resolve_anthropic_model(request.model)

        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": sanitized_messages,
            "max_tokens": request.max_tokens,
        }

        if request.system:
            kwargs["system"] = request.system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            kwargs["top_k"] = request.top_k
        if request.tools:
            kwargs["tools"] = request.tools
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences
        if request.thinking:
            kwargs["thinking"] = request.thinking
        if request.metadata:
            kwargs["metadata"] = request.metadata
        if request.thinking_level:
            # 생각의 표준 값 — 이 모델이 받는 모양(llm_client.thinking 표): 예산·adaptive+effort·끄는 값.
            # effort 는 SDK 판에 기대지 않게 본문에 싣는다(extra_body).
            from xgen_agent_runtime.llm_client.thinking import anthropic_request, thinking_spec

            shaped = anthropic_request(
                thinking_spec(self.thinking_provider(), request.model),
                request.thinking_level,
                request.max_tokens,
            )
            kwargs["thinking"] = shaped["thinking"]
            kwargs["max_tokens"] = shaped["max_tokens"]
            if "output_config" in shaped:
                extra = dict(kwargs.get("extra_body") or {})
                extra["output_config"] = shaped["output_config"]
                kwargs["extra_body"] = extra

        # Extended-thinking sampling-param compatibility — see the
        # ``_THINKING_INCOMPATIBLE_SAMPLING_KEYS`` block at module top.
        # Anthropic rejects ``temperature``/``top_p``/``top_k`` when
        # ``thinking`` is set; drop them silently at the boundary so
        # an env with both ``thinking_enabled=True`` and an explicit
        # ``temperature`` (the common combo Geny ships) still works.
        #
        # ``{"type": "disabled"}`` 은 생각을 **끄는** 값이다 — 그때는 표준 샘플러라 temperature 를
        # 받는다(2026-10-01 live: haiku-4-5·sonnet-4-6 모두 200). 예전에는 thinking 칸이 있기만
        # 하면 지워서, 생각을 끄고 temperature=0 으로 JSON 을 받으려던 호출(앱 LLM ask_json)이
        # 기본 temperature 1.0 으로 돌았다.
        if _thinking_active(kwargs.get("thinking")):
            for key in _THINKING_INCOMPATIBLE_SAMPLING_KEYS:
                if key in kwargs:
                    dropped = kwargs.pop(key)
                    logger.info(
                        "anthropic: dropped %r=%r — extended thinking "
                        "is enabled and the Messages API rejects this "
                        "sampling param",
                        key,
                        dropped,
                    )

        # Model-level unconditional rejection — see
        # ``_TEMPERATURE_DEPRECATED_PREFIXES`` at module top. Opus 4.7
        # refuses ``temperature`` regardless of whether ``thinking`` is
        # set; without this drop, ``AdaptiveModelRouter`` promoting a
        # thinking-enabled call to Opus 4.7 would still 400.
        if _model_rejects_sampling_params(resolved_model):
            for key in _THINKING_INCOMPATIBLE_SAMPLING_KEYS:
                if key in kwargs:
                    dropped = kwargs.pop(key)
                    logger.info(
                        "anthropic: dropped %r=%r — model %r refuses "
                        "this sampling param unconditionally",
                        key,
                        dropped,
                        resolved_model,
                    )

        # Thinking-shape migration — see
        # ``_THINKING_ADAPTIVE_ONLY_PREFIXES`` at module top. Opus 4.7
        # rejects ``thinking.type=enabled``; flip to ``adaptive`` (and
        # drop the now-invalid ``budget_tokens``) at the boundary so
        # ``ModelConfig`` callers that still emit the v1 shape
        # continue to work against v2 models.
        if (
            "thinking" in kwargs
            and isinstance(kwargs["thinking"], dict)
            and kwargs["thinking"].get("type") == "enabled"
            and _model_requires_adaptive_thinking(resolved_model)
        ):
            before = kwargs["thinking"]
            kwargs["thinking"] = _translate_thinking_to_adaptive(before)
            logger.info(
                "anthropic: translated thinking.type=enabled → adaptive "
                "(model=%r, dropped legacy budget_tokens=%r)",
                resolved_model,
                before.get("budget_tokens"),
            )

        # 기본으로 생각하는 모델(Claude 5 계열)은 thinking 을 안 보내도 adaptive 로 생각하고,
        # 표시 기본값이 "omitted" 라 생각하는 동안 빈 thinking 만 흘러온다. 그러면 사용자에게는
        # 긴 멈춤이고, 스테이지의 첫 내용 감시(180초)는 모델이 멈춘 줄 안다 — 출력 한도로 한 번
        # 더 부른 호출이 이 감시에 걸려 끝났다(로컬 벤치 5과제). 요약을 흘려보내게 한다(4.6 까지의
        # 기본 동작과 같다). 호출자가 display 를 정했으면 그대로 둔다.
        if _model_thinks_by_default(resolved_model):
            thinking = kwargs.get("thinking")
            if not thinking:
                kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            elif (
                isinstance(thinking, dict)
                and thinking.get("type") == "adaptive"
                and "display" not in thinking
            ):
                kwargs["thinking"] = {**thinking, "display": "summarized"}
            for key in _THINKING_INCOMPATIBLE_SAMPLING_KEYS:
                kwargs.pop(key, None)

        # The API's safety net for any other history change: drop stale thinking blocks instead of
        # failing the request. Only ``adaptive`` takes the field (``between_tools`` / ``disabled``
        # reject it — those rely on the retry in ``_retry_kwargs_after_deprecation``).
        thinking = kwargs.get("thinking")
        if (
            self._thinking_binding_controls
            and isinstance(thinking, dict)
            and thinking.get("type") == "adaptive"
            and "block_binding" not in thinking
        ):
            kwargs["thinking"] = {
                **thinking,
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
            _with_beta(kwargs, _THINKING_BINDING_BETA, self._default_headers)

        return kwargs

    @staticmethod
    def _binding_of(request: APIRequest) -> str:
        return thinking_binding(request.system, request.tools)

    def _parse_response(self, raw: Any, binding: Optional[str] = None) -> APIResponse:
        content_blocks: List[ContentBlock] = []
        # 같은 tool_use id 는 한 번만 받는다. 두 번 실리면 도구가 두 번 돌고, 그 응답을 담은 다음
        # 요청이 400 "tool_use ids must be unique" 로 거절된다(2026-11-09 claude-sonnet-5-5).
        tool_use_at: Dict[str, int] = {}

        for block in raw.content:
            if block.type == "tool_use" and block.id in tool_use_at:
                kept = content_blocks[tool_use_at[block.id]]
                # 비어 있는 쪽 대신 입력이 있는 쪽을 남긴다(스트림 조립이 블록을 한 번 더 붙이면
                # 덧붙은 쪽은 입력이 비어 있다).
                if not kept.tool_input and block.input:
                    kept.tool_input = block.input
                    kept.raw = {**(kept.raw or {}), "input": block.input}
                logger.warning(
                    "anthropic: response repeated tool_use id %s (model=%s, stop_reason=%s, blocks=%s)"
                    " — kept one",
                    block.id,
                    getattr(raw, "model", ""),
                    getattr(raw, "stop_reason", ""),
                    [getattr(b, "type", "") for b in raw.content],
                )
                continue
            if block.type == "tool_use":
                tool_use_at[block.id] = len(content_blocks)
            if block.type == "text":
                content_blocks.append(
                    ContentBlock(
                        type="text",
                        text=block.text,
                        raw={"type": "text", "text": block.text},
                    )
                )
            elif block.type == "tool_use":
                content_blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=block.id,
                        tool_name=block.name,
                        tool_input=block.input,
                        raw={
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        },
                    )
                )
            elif block.type == "thinking":
                # 서명까지 그대로 되돌려 보내야 한다 — Anthropic 은 도구 루프에서 직전 assistant 의
                # thinking 블록을 **수정 없이**(signature 포함) 요구한다. 빠뜨리면 다음 호출이
                # 400 "thinking.signature: Field required" 로 거절돼 턴이 첫 도구 호출 뒤에 끝났다
                # (dev 09-08~ claude-sonnet-5·opus-5 도구 턴 전부 모델 호출 1회·답 없음).
                raw_thinking = {"type": "thinking", "thinking": block.thinking}
                signature = getattr(block, "signature", None)
                if signature:
                    raw_thinking["signature"] = signature
                if binding:
                    # 어떤 system · 도구 목록 아래에서 쓴 생각인가(보내지 않는 내부 키) — 목록이
                    # 바뀐 뒤의 요청에서는 이 블록을 뺀다(_without_unbound_thinking).
                    raw_thinking["_meta"] = {"binding": binding}
                content_blocks.append(
                    ContentBlock(
                        type="thinking",
                        thinking_text=block.thinking,
                        raw=raw_thinking,
                    )
                )
            elif block.type == "redacted_thinking":
                # 안전상 가려진 추론 — 내용은 없지만 같은 이유로 그대로 되돌려 보내야 한다.
                raw_redacted = {"type": "redacted_thinking", "data": getattr(block, "data", "")}
                if binding:
                    raw_redacted["_meta"] = {"binding": binding}
                content_blocks.append(ContentBlock(type="redacted_thinking", raw=raw_redacted))

        _log_input_transformations(raw)

        if raw.usage:
            # ⚠ ``or 0`` 가 핵심 — SDK Usage 모델은 응답에 없는 필드를 **None**
            # 으로 준다 (getattr 기본값은 속성이 존재하면 무력). Bedrock 응답은
            # 캐시 필드를 생략하므로 None 이 그대로 흐르면 토큰 회계(int 가산·
            # 비교)가 매 턴 TypeError 로 죽는다.
            usage = TokenUsage(
                input_tokens=getattr(raw.usage, "input_tokens", 0) or 0,
                output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(raw.usage, "cache_creation_input_tokens", 0)
                or 0,
                cache_read_input_tokens=getattr(raw.usage, "cache_read_input_tokens", 0) or 0,
            )
        else:
            usage = TokenUsage()

        # ``raw`` is the provenance channel (the CLI client already ships
        # a dict here with ``cli_version``). ``response`` carries the SDK
        # object for callers that need vendor-specific fields.
        provenance = self._provenance()
        provenance["response"] = raw

        return APIResponse(
            content=content_blocks,
            stop_reason=raw.stop_reason or "",
            usage=usage,
            model=raw.model,
            message_id=raw.id,
            raw=provenance,
        )

    def _classify_error(self, e: Exception) -> APIError:
        import anthropic

        if isinstance(e, anthropic.RateLimitError):
            return APIError(str(e), category=ErrorCategory.RATE_LIMITED, cause=e)
        if isinstance(e, anthropic.APITimeoutError):
            return APIError(str(e), category=ErrorCategory.TIMEOUT, cause=e)
        if isinstance(e, anthropic.APIConnectionError):
            return APIError(str(e), category=ErrorCategory.NETWORK, cause=e)
        if isinstance(e, anthropic.AuthenticationError):
            return APIError(str(e), category=ErrorCategory.AUTH, status_code=401, cause=e)
        if isinstance(e, anthropic.BadRequestError):
            # TOKEN_LIMIT only on anchored overflow phrases; param-shape
            # validation messages (which routinely mention "tokens") are
            # BAD_REQUEST — see ``_classify_bad_request_message``.
            category = _classify_bad_request_message(str(e).lower())
            return APIError(str(e), category=category, status_code=400, cause=e)
        if isinstance(e, anthropic.InternalServerError):
            return APIError(str(e), category=ErrorCategory.SERVER_ERROR, status_code=500, cause=e)
        if isinstance(e, APIError):
            return e
        return APIError(str(e), category=ErrorCategory.UNKNOWN, cause=e)
