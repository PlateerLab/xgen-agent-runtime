"""Deterministic tool-output pruning — the no-LLM pass that runs BEFORE
summary compaction.

LLM summarization is expensive and lossy; a large share of context pressure
is mechanical redundancy that a deterministic pass can reclaim first:

* **Duplicate tool outputs** — the same file read five times keeps five full
  copies in history. Only the NEWEST copy carries information; older ones are
  rewritten to a one-line back-reference (hash-matched, exact).
* **Stale image payloads** — a base64 screenshot in an old tool result rides
  every subsequent request forever (and survives summary compaction whenever
  the compactor keeps that message verbatim).
* **Oversized stale outputs** — a 10k-char terminal dump from twenty turns ago
  is almost never load-bearing; the head plus an explicit trim marker is.

Invariants (why this is safe to run on any history):

* Message COUNT and ORDER never change — STM watermarks and compaction
  bookkeeping stay valid.
* ``tool_use`` blocks and every ``tool_use_id`` are untouched — pruning can
  never orphan a tool call (pair repair stays a no-op).
* Blocks are rewritten in place, never removed; only ``tool_result`` content
  and image blocks shrink.
* The newest ``protect_last`` messages are never touched — recent context is
  exactly what the model is working from.

Everything here is pure function + stdlib; no model, no network.

When this runs (4.35.0 — the cost trigger)
------------------------------------------

Until 4.35.0 this pass only ran *inside* compaction, and compaction only fires
at 80% of the model's context window. Measured on dev over 28 days: the window
is 200,000 (claude-sonnet-4-6) or 524,288 (qwen3.8-27b `max_model_len`), so the
thresholds were 160k / 419k — while the largest prompt any real turn ever built
was 135,487 tokens. **The prune therefore never ran once.**

That is a trigger bug, not a tuning one: the only trigger was *capacity* (don't
overflow the window), and there was none for *cost* (don't pay to resend the
same stale dump on every call). The bigger the window, the less it ever runs.

So Stage 2 also runs this pass on an absolute token threshold
(:data:`DEFAULT_PRUNE_OVER_TOKENS`), independent of the window. Evidence for the
default, same 28 days (turns with usage records, bench excluded):

* Turn input splits into prefix (per-call system+tools) and *history resent on
  every call*. History is 22% of input at 2–5 calls, 45% at 6–15, and **71–72%
  at 16+** — 66% overall. History grows ~2,000 tokens per call.
* At a 30,000-token threshold, **no turn with 5 or fewer calls is touched at
  all**, while 13 of the 14 turns with 8+ calls are. The number separates short
  turns from long ones without encoding anything about what the turns *did*.

The pass stays conservative by construction — only results older than the last
``protect_last`` messages, only ones over ``trim_over_chars``, head kept plus an
explicit marker — so a model that still needs the detail can re-read the source.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

__all__ = ["DEFAULT_PRUNE_OVER_TOKENS", "AGGRESSIVE", "prune_messages", "PruneMetrics"]

#: Stage 2 runs the prune when the projected prompt exceeds this many tokens,
#: regardless of the context window. See the module docstring for the evidence.
DEFAULT_PRUNE_OVER_TOKENS = 30_000
#: Messages within this tail window are never modified.
DEFAULT_PROTECT_LAST = 6
#: Tool-result text shorter than this is never considered for dedup — tiny
#: results ("ok", exit codes) legitimately repeat and cost nothing.
DEFAULT_MIN_DUP_CHARS = 512
#: Stale tool-result text longer than this is trimmed to its head.
DEFAULT_TRIM_OVER_CHARS = 4000
#: How much head text a trimmed output keeps.
DEFAULT_TRIM_KEEP_CHARS = 600

#: Stale tool-call INPUTS (Write/Edit file bodies, long commands) longer than this are trimmed too:
#: a 40 KB file body in an old ``Write`` call is resent on every request otherwise.
DEFAULT_INPUT_TRIM_OVER_CHARS = 4000
DEFAULT_INPUT_TRIM_KEEP_CHARS = 400

#: Settings for HARD pressure (the guard or a provider overflow: the request cannot go out as is):
#: protect only the last exchange and trim harder. Still deterministic, still pair-safe.
AGGRESSIVE = {
    "protect_last": 2,
    "trim_over_chars": 1000,
    "trim_keep_chars": 300,
    "input_trim_over_chars": 1000,
    "input_trim_keep_chars": 200,
}

_DUP_NOTE = "[duplicate tool output — identical to a more recent result]"
_IMAGE_NOTE = "[image removed during context compaction]"
_DOCUMENT_NOTE = "[document removed during context compaction]"


class PruneMetrics(Dict[str, int]):
    """Plain dict of counters; a class only so callers can type it."""


def _iter_tool_results(message: Dict[str, Any]):
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            yield block


def _result_text_parts(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The text sub-blocks of a tool_result whose content is a block list."""
    content = block.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "text"]


def _content_hash(block: Dict[str, Any]) -> str:
    """Stable digest of a tool_result's content (str or block list)."""
    content = block.get("content")
    try:
        payload = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 — unhashable exotic content: skip dedup
        return ""
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()


def _content_size(block: Dict[str, Any]) -> int:
    content = block.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return 0


def _strip_images_in(blocks: List[Any], *, documents: bool = False) -> int:
    """Replace base64 image blocks (and, with ``documents``, PDF document blocks) with a small text
    marker, recursing into tool_result content lists. Returns the number removed."""
    removed = 0
    kinds = ("image", "document") if documents else ("image",)
    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if (
            btype in kinds
            and isinstance(b.get("source"), dict)
            and b["source"].get("type") == "base64"
        ):
            blocks[i] = {
                "type": "text",
                "text": _IMAGE_NOTE if btype == "image" else _DOCUMENT_NOTE,
            }
            removed += 1
        elif btype == "tool_result" and isinstance(b.get("content"), list):
            removed += _strip_images_in(b["content"], documents=documents)
    return removed


def _trim_tool_inputs(message: Dict[str, Any], over: int, keep: int) -> int:
    """Shorten long string values inside stale ``tool_use`` inputs. Keeps the input a valid
    object of the same keys (providers validate it), so pairing and schema shape survive.
    Returns chars removed."""
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    saved = 0
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        params = block.get("input")
        if not isinstance(params, dict):
            continue
        trimmed: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, str) and len(value) > over:
                dropped = len(value) - keep
                trimmed[key] = (
                    value[:keep] + f"\n… [{dropped} chars trimmed during context compaction]"
                )
                saved += dropped
        if trimmed:
            # A new dict, not an in-place edit: the same input object may still be held by the
            # tool-call record or a rollout recorder that must keep what was actually sent.
            block["input"] = {**params, **trimmed}
    return saved


def _trim_result(block: Dict[str, Any], keep: int) -> int:
    """Trim an oversized tool_result to its head + explicit marker. Returns
    chars removed."""
    content = block.get("content")
    if isinstance(content, str):
        dropped = len(content) - keep
        block["content"] = (
            content[:keep] + f"\n… [{dropped} chars trimmed during context compaction]"
        )
        return dropped
    if isinstance(content, list):
        dropped = 0
        budget = keep
        for sub in _result_text_parts(block):
            text = sub.get("text", "")
            if budget <= 0:
                dropped += len(text)
                sub["text"] = ""
            elif len(text) > budget:
                dropped += len(text) - budget
                sub["text"] = text[:budget]
                budget = 0
            else:
                budget -= len(text)
        if dropped > 0:
            # Drop the text parts the trim emptied (providers reject empty text blocks), keeping
            # every non-text part and at least one text part for the marker.
            emptied = [p for p in _result_text_parts(block) if not p.get("text")]
            if emptied:
                keep_one = len(emptied) == len(_result_text_parts(block))
                drop = {id(e) for e in (emptied[1:] if keep_one else emptied)}
                block["content"] = [b for b in content if id(b) not in drop]
            parts = _result_text_parts(block)
            if parts:
                parts[-1]["text"] += f"\n… [{dropped} chars trimmed during context compaction]"
        return dropped
    return 0


def shrink_tool_results(
    messages: List[Dict[str, Any]],
    factor: float,
    *,
    min_keep: int = 300,
    strip_images: bool = False,
) -> Dict[str, int]:
    """Shorten every tool_result in *messages* to ``factor`` of its text, in place (head kept, with
    the usual marker); with ``strip_images`` also replace the images inside results with a marker.
    The last resort under hard pressure: one round of parallel tool calls can return more than the
    window even when each result is within its own cap, and the protected tail is exactly where
    those results sit. Returns ``{"trimmed", "chars_saved", "images_stripped"}``."""
    trimmed = saved = images = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for block in _iter_tool_results(msg):
            if strip_images and isinstance(block.get("content"), list):
                images += _strip_images_in(block["content"])
            size = _content_size(block)
            keep = max(min_keep, int(size * max(0.0, factor)))
            if size > keep:
                removed = _trim_result(block, keep)
                if removed > 0:
                    trimmed += 1
                    saved += removed
    return {"trimmed": trimmed, "chars_saved": saved, "images_stripped": images}


#: User-role messages the runtime itself adds (Stage 16's deliverable check, the turn input budget
#: note): instructions to the model, never the person's request. Shared with the summary compactor.
RUNTIME_NOTE_PREFIXES = ("[Deliverable check", "[Turn budget")
#: Content blocks that are attachments rather than words.
ATTACHMENT_TYPES = ("image", "document", "image_url", "input_image", "file")


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text") or "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def is_runtime_note(message: Any) -> bool:
    return isinstance(message, dict) and _message_text(message).lstrip().startswith(
        RUNTIME_NOTE_PREFIXES
    )


def _is_request_message(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user" or is_runtime_note(message):
        return False
    content = message.get("content")
    return not (
        isinstance(content, list)
        and content
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def current_request_index(messages: List[Dict[str, Any]]) -> int:
    """Index of the newest user message that is the person's request (not only tool results, not a
    runtime note; a checkpoint carrying the request counts), or -1."""
    for i in range(len(messages or []) - 1, -1, -1):
        if _is_request_message(messages[i]):
            return i
    return -1


def current_request_span(messages: List[Dict[str, Any]]) -> List[int]:
    """The request being answered: :func:`current_request_index` and the user messages right before
    it with no reply in between (a file sent first, the question in the next message)."""
    idx = current_request_index(messages)
    if idx < 0:
        return []
    span = [idx]
    j = idx - 1
    while j >= 0 and _is_request_message(messages[j]):
        span.insert(0, j)
        j -= 1
    return span


def strip_current_attachments(messages: List[Dict[str, Any]], *, limit: int = 0) -> int:
    """Remove images and documents of the request being answered, at most ``limit`` (0 = all). The
    very last resort under hard pressure; returns how many were removed."""
    removed = 0
    for idx in reversed(current_request_span(messages)):
        content = messages[idx].get("content")
        if not isinstance(content, list):
            continue
        for i, b in enumerate(content):
            if limit and removed >= limit:
                return removed
            if not isinstance(b, dict) or b.get("type") not in ("image", "document"):
                continue
            if not (isinstance(b.get("source"), dict) and b["source"].get("type") == "base64"):
                continue
            content[i] = {
                "type": "text",
                "text": _IMAGE_NOTE if b["type"] == "image" else _DOCUMENT_NOTE,
            }
            removed += 1
    return removed


def prune_messages(
    messages: List[Dict[str, Any]],
    *,
    protect_last: int = DEFAULT_PROTECT_LAST,
    min_dup_chars: int = DEFAULT_MIN_DUP_CHARS,
    trim_over_chars: int = DEFAULT_TRIM_OVER_CHARS,
    trim_keep_chars: int = DEFAULT_TRIM_KEEP_CHARS,
    input_trim_over_chars: int = DEFAULT_INPUT_TRIM_OVER_CHARS,
    input_trim_keep_chars: int = DEFAULT_INPUT_TRIM_KEEP_CHARS,
    strip_documents: bool = False,
    protect_current_request: bool = True,
) -> PruneMetrics:
    """Run the deterministic prune over *messages* in place.

    Returns metrics: ``deduped`` (results back-referenced), ``images_stripped`` (images, and PDF
    documents when ``strip_documents``), ``trimmed`` (oversized results shortened),
    ``inputs_trimmed`` (stale tool calls whose long input strings were shortened), ``chars_saved``
    (text chars removed, images excluded).

    ``strip_documents``: attached PDFs are often what the request is about, so they are removed only
    under hard pressure (``core.compaction`` passes it with the aggressive profile).

    ``protect_current_request``: the images and documents of the request being answered
    (:func:`current_request_index`) are never stripped here, however many tool rounds followed it.
    ``core.compaction`` removes them only as the very last resort (:func:`strip_current_attachments`).
    """
    metrics = PruneMetrics(deduped=0, images_stripped=0, trimmed=0, inputs_trimmed=0, chars_saved=0)
    if not messages:
        return metrics
    cutoff = max(0, len(messages) - max(0, protect_last))

    # Pass 1 — hashes of every PROTECTED/newer occurrence win. Walk from the
    # end so the newest copy of any repeated output is the one kept.
    seen: set = set()
    for mi in range(len(messages) - 1, -1, -1):
        msg = messages[mi]
        if not isinstance(msg, dict):
            continue
        protected = mi >= cutoff
        for block in _iter_tool_results(msg):
            if _content_size(block) < min_dup_chars:
                continue
            digest = _content_hash(block)
            if not digest:
                continue
            if digest in seen and not protected:
                old = _content_size(block)
                block["content"] = _DUP_NOTE
                metrics["deduped"] += 1
                metrics["chars_saved"] += max(0, old - len(_DUP_NOTE))
            else:
                seen.add(digest)

    # Pass 2 — stale-region cleanup: image stripping + oversize trimming.
    current = set(current_request_span(messages)) if protect_current_request else set()
    for mi in range(cutoff):
        msg = messages[mi]
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list) and mi not in current:
            metrics["images_stripped"] += _strip_images_in(content, documents=strip_documents)
        if msg.get("role") == "assistant":
            saved_inputs = _trim_tool_inputs(msg, input_trim_over_chars, input_trim_keep_chars)
            if saved_inputs > 0:
                metrics["inputs_trimmed"] += 1
                metrics["chars_saved"] += saved_inputs
        for block in _iter_tool_results(msg):
            size = _content_size(block)
            if size > trim_over_chars:
                saved = _trim_result(block, trim_keep_chars)
                if saved > 0:
                    metrics["trimmed"] += 1
                    metrics["chars_saved"] += saved
    return metrics
