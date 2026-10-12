"""Single-source estimate of the NEXT request's input size, for every context-budget decision.

The Stage 2 proactive compaction trigger, the Stage 4 budget guard, the Stage 6 overflow recovery
and the Stage 16 token dimension all ask one question, "how big is the next API call?", and must
get ONE answer, otherwise compaction (which shrinks ``state.messages``) cannot relieve a check that
measures something else.

How the answer is built (4.89.0)
--------------------------------
1. **What the provider reported last.** After every successful call Stage 6 records the prompt
   size the provider *counted* (:func:`record_prompt_usage`) together with an identity anchor of
   the message list it sent. While that anchor still describes the live list (same head/tail
   objects), the estimate is ``reported + estimate(everything that changed since)``. This is the
   Codex CLI scheme: real tokenizer numbers for the bulk, an estimate only for the delta.
2. **Otherwise, a script-aware estimate.** :func:`xgen_agent_runtime.core.text_tokens.estimate_tokens`
   counts character-class runs (Hangul per syllable, Latin per ~7 chars …), multiplied by the
   provider factor and a safety margin. The old ``len(text) // 4`` under-counted Korean 2-3x, so the
   80% trigger and the guard fired after the provider had already rejected the request.
3. **Calibration.** Each reported/estimated pair yields a ratio; later pure estimates in the same
   turn (e.g. right after a compaction, when no anchor applies) are scaled by it.

Images count a flat 1,600 tokens (a full-size screenshot); a base64 PDF counts per page. Counting
base64 characters would trip compaction on the first screenshot.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Dict, List, Optional, Union

from xgen_agent_runtime.core.text_tokens import (
    SAFETY_MULTIPLIER,
    estimate_tokens_cached,
    provider_token_factor,
)

# Anthropic vision tokens land roughly here for a typical full-size image.
_IMAGE_TOKEN_ESTIMATE = 1_600
#: One PDF page (its text plus the page image the provider renders): roughly 1.5k-3k tokens.
_PDF_PAGE_TOKEN_ESTIMATE = 2_500
#: A document whose pages cannot be counted is never cheaper than one image.
_DOCUMENT_TOKEN_FLOOR = 1_600
_PDF_PAGE_RE = re.compile(rb"/Type\s*/Page(?![A-Za-z])")
#: Image-ish block types across canonical / OpenAI-shaped content.
_IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})

#: ``state.shared`` keys. Strings are the contract (hosts and tests may read them).
_ESTIMATE_MEMO_KEY = "_prompt_tokens_memo"
USAGE_ANCHOR_KEY = "_prompt_usage_anchor"
CALIBRATION_KEY = "_prompt_tokens_calibration"
#: Calibration is clamped: one odd report must not make every later estimate absurd.
_CALIBRATION_MIN = 0.9
_CALIBRATION_MAX = 2.5

_pdf_page_cache: Dict[tuple, int] = {}


def _estimate_text(text: Any) -> int:
    """Raw (uncorrected) token estimate of a text."""
    return estimate_tokens_cached(text)


def _pdf_pages(data: str) -> int:
    """Pages in a base64 PDF (0 when it cannot be decoded). Cached by content."""
    key = (hash(data), len(data))
    cached = _pdf_page_cache.get(key)
    if cached is not None:
        return cached
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        raw = b""
    pages = len(_PDF_PAGE_RE.findall(raw)) if raw else 0
    if len(_pdf_page_cache) > 256:
        _pdf_page_cache.clear()
    _pdf_page_cache[key] = pages
    return pages


def _estimate_document(block: Dict[str, Any]) -> int:
    source = block.get("source")
    if isinstance(source, dict):
        stype = source.get("type")
        data = source.get("data")
        if stype == "text" and isinstance(data, str):
            return _estimate_text(data)
        if stype == "content" and isinstance(source.get("content"), list):
            return sum(_estimate_block(b) for b in source["content"])
        if stype == "base64" and isinstance(data, str):
            pages = _pdf_pages(data)
            if pages:
                return pages * _PDF_PAGE_TOKEN_ESTIMATE
            # Unknown layout: about 3 KB of PDF per page is typical; never below one image.
            return max(
                _DOCUMENT_TOKEN_FLOOR, (len(data) * 3 // 4) // 3_000 * _PDF_PAGE_TOKEN_ESTIMATE
            )
    text = block.get("text") or block.get("content")
    if isinstance(text, str):
        return _estimate_text(text)
    return _DOCUMENT_TOKEN_FLOOR


def _estimate_block(block: Any) -> int:
    """Raw estimate of one content block (text / image / document / tool_use / tool_result …)."""
    if not isinstance(block, dict):
        return _estimate_text(str(block))
    btype = block.get("type")
    if btype in _IMAGE_TYPES:
        return _IMAGE_TOKEN_ESTIMATE
    if btype == "document":
        return _estimate_document(block)
    if btype == "reasoning":
        # OpenAI reasoning items: the encrypted payload's length is not tokens. Summary + small fixed.
        return _estimate_text(str(block.get("summary") or "")) + 64
    total = 0
    if btype == "tool_use":
        total += _estimate_text(str(block.get("name") or "")) + 8
    for key in ("text", "content", "input", "thinking"):
        val = block.get(key)
        if isinstance(val, str):
            total += _estimate_text(val)
        elif isinstance(val, list):
            total += sum(_estimate_block(sub) for sub in val)
        elif val is not None:
            total += _estimate_text(str(val))
    return total or _estimate_text(str(block))


def _estimate_content(content: Union[str, List[Any], Any]) -> int:
    if isinstance(content, str):
        return _estimate_text(content)
    if isinstance(content, list):
        return sum(_estimate_block(b) for b in content)
    if content is None:
        return 0
    return _estimate_text(str(content))


def _raw_message(message: Any) -> int:
    if not isinstance(message, dict):
        return 0
    # +4: role/turn framing every provider adds per message.
    return _estimate_content(message.get("content", "")) + 4


def estimate_message_tokens(messages: List[Any]) -> int:
    """Raw token estimate of a message list (content only, no provider factor)."""
    return sum(_raw_message(m) for m in (messages or []))


def _raw_system_and_tools(state: Any) -> int:
    total = _estimate_content(getattr(state, "system", "") or "")
    tools = getattr(state, "tools", None)
    if tools:
        total += sum(_estimate_text(str(tool)) for tool in tools)
    return total


def _provider(state: Any) -> str:
    client = getattr(state, "llm_client", None)
    return str(getattr(client, "provider", "") or "")


def _scale(state: Any) -> float:
    """Raw → provider-corrected tokens."""
    return provider_token_factor(_provider(state)) * SAFETY_MULTIPLIER


def _calibration(state: Any) -> float:
    shared = getattr(state, "shared", None)
    value = shared.get(CALIBRATION_KEY) if isinstance(shared, dict) else None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return 1.0


def _estimate_fingerprint(state: Any) -> tuple:
    """Cheap identity of the inputs the estimate depends on (memo key).

    Appends change the count, compaction replaces the head message (new id), Stage 3 rebuilding the
    system prompt changes its length, a tool-registry rebuild changes the tools count, and a new
    provider report replaces the anchor object. In-place edits that keep every length (the
    deterministic prune) pop the memo explicitly.
    """
    messages = getattr(state, "messages", []) or []
    system = getattr(state, "system", "") or ""
    tools = getattr(state, "tools", None) or []
    shared = getattr(state, "shared", None)
    anchor = shared.get(USAGE_ANCHOR_KEY) if isinstance(shared, dict) else None
    if isinstance(system, str):
        sys_len = len(system)
    else:
        sys_len = sum(len(str(b)) for b in system)
    return (
        len(messages),
        id(messages[0]) if messages else 0,
        id(messages[-1]) if messages else 0,
        sys_len,
        len(tools),
        id(anchor) if anchor is not None else 0,
        _calibration(state),
    )


def _anchored(state: Any, anchor: Dict[str, Any]) -> Optional[int]:
    """``reported + estimate(delta)`` while the anchor still describes the live message list."""
    messages = getattr(state, "messages", []) or []
    n = int(anchor.get("n") or 0)
    if n <= 0 or len(messages) < n:
        return None
    if id(messages[0]) != anchor.get("head") or id(messages[n - 1]) != anchor.get("tail"):
        return None  # compaction / preload rewrote the list the provider counted
    prefix_raw = estimate_message_tokens(messages[:n])
    tail_raw = estimate_message_tokens(messages[n:])
    sys_tools_raw = _raw_system_and_tools(state)
    # In-place edits to the counted prefix (the deterministic prune) and system/tool changes show up
    # as raw deltas: negative when something shrank.
    delta_raw = (
        tail_raw
        + (prefix_raw - int(anchor.get("prefix_raw") or 0))
        + (sys_tools_raw - int(anchor.get("sys_tools_raw") or 0))
    )
    delta = int(delta_raw * _scale(state) * _calibration(state))
    return max(0, int(anchor.get("tokens") or 0) + delta)


def estimate_messages_scaled(state: Any, messages: List[Any]) -> int:
    """Provider-corrected estimate of ``messages`` alone (no anchor): for planning a cut."""
    return int(estimate_message_tokens(messages) * _scale(state) * _calibration(state))


def estimate_overhead_tokens(state: Any) -> int:
    """Provider-corrected estimate of the system prompt and tool schemas."""
    return int(_raw_system_and_tools(state) * _scale(state) * _calibration(state))


def estimate_text_scaled(state: Any, text: Any) -> int:
    """Provider-corrected estimate of a text (prompts the runtime builds itself)."""
    return int(_estimate_text(text) * _scale(state) * _calibration(state))


def estimate_prompt_tokens(state: Any) -> int:
    """INPUT-token estimate for the next API call (system + messages + tools).

    Used identically by the Stage 2 compaction trigger, the Stage 4 budget guard, the Stage 6
    overflow recovery and the Stage 16 token dimension, so compacting ``state.messages`` measurably
    lowers the number every one of them checks. Memoized per state via a fingerprint.
    """
    shared = getattr(state, "shared", None)
    fingerprint = _estimate_fingerprint(state)
    if isinstance(shared, dict):
        memo = shared.get(_ESTIMATE_MEMO_KEY)
        if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == fingerprint:
            return memo[1]

    total: Optional[int] = None
    anchor = shared.get(USAGE_ANCHOR_KEY) if isinstance(shared, dict) else None
    if isinstance(anchor, dict):
        total = _anchored(state, anchor)
    if total is None:
        raw = _raw_system_and_tools(state) + estimate_message_tokens(
            getattr(state, "messages", []) or []
        )
        total = int(raw * _scale(state) * _calibration(state))

    if isinstance(shared, dict):
        shared[_ESTIMATE_MEMO_KEY] = (fingerprint, total)
    return total


def record_prompt_usage(state: Any, prompt_tokens: int) -> None:
    """Stage 6, after a successful call: remember what the provider counted for ``state.messages``.

    ``prompt_tokens`` is the FULL prompt the provider billed (uncached + cache read + cache write);
    see ``BaseClient.prompt_tokens_from_usage``. A zero/absent report (CLI backends, mocks) leaves
    the previous anchor alone.
    """
    shared = getattr(state, "shared", None)
    messages = getattr(state, "messages", None) or []
    try:
        reported = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        return
    if not isinstance(shared, dict) or reported <= 0 or not messages:
        return
    prefix_raw = estimate_message_tokens(messages)
    sys_tools_raw = _raw_system_and_tools(state)
    pure = int((prefix_raw + sys_tools_raw) * _scale(state))
    if pure > 0:
        ratio = reported / pure
        shared[CALIBRATION_KEY] = min(_CALIBRATION_MAX, max(_CALIBRATION_MIN, ratio))
    shared[USAGE_ANCHOR_KEY] = {
        "tokens": reported,
        "n": len(messages),
        "head": id(messages[0]),
        "tail": id(messages[-1]),
        "prefix_raw": prefix_raw,
        "sys_tools_raw": sys_tools_raw,
    }
    shared.pop(_ESTIMATE_MEMO_KEY, None)


def calibrate_from_overflow(state: Any, actual_tokens: int) -> None:
    """A provider rejected the request and said how big it really was: trust that number.

    Drops the anchor (the rejected list is about to be compacted) and raises the calibration so the
    compaction target is measured against the provider's count, not our under-estimate.
    """
    shared = getattr(state, "shared", None)
    if not isinstance(shared, dict):
        return
    try:
        actual = int(actual_tokens or 0)
    except (TypeError, ValueError):
        return
    # The anchor goes first: the ratio must correct the estimate that is used from here on (raw ×
    # scale × calibration), not the anchored one the rejected request was measured with.
    shared.pop(USAGE_ANCHOR_KEY, None)
    shared.pop(_ESTIMATE_MEMO_KEY, None)
    if actual <= 0:
        return
    current = estimate_prompt_tokens(state)
    if current > 0 and actual > current:
        ratio = _calibration(state) * (actual / current)
        shared[CALIBRATION_KEY] = min(_CALIBRATION_MAX, max(_CALIBRATION_MIN, ratio))
    shared.pop(_ESTIMATE_MEMO_KEY, None)


__all__ = [
    "estimate_message_tokens",
    "estimate_messages_scaled",
    "estimate_overhead_tokens",
    "estimate_text_scaled",
    "estimate_prompt_tokens",
    "record_prompt_usage",
    "calibrate_from_overflow",
    "USAGE_ANCHOR_KEY",
    "CALIBRATION_KEY",
]
