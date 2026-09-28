"""Tool-call/result pairing repairs for ``state.messages`` (audit D4).

The Anthropic Messages API (and OpenAI's tool protocol) require that
every assistant ``tool_use`` block is answered by a following
``tool_result`` and, conversely, that a ``tool_result`` names a
``tool_use`` that precedes it. A turn that dies between appending the
assistant ``tool_use`` and appending the tool results — a user stop, a
tool-access denial, a crash inside the dispatch ``gather`` — leaves
``state.messages`` ending in a dangling ``tool_use``, and every
subsequent request 400s ("tool_use ids without tool_result"). Compaction
can independently orphan the OTHER side, slicing a kept window that
begins with a ``tool_result`` whose ``tool_use`` was dropped.

Both repairs are pure list surgery — no message content is invented
beyond a synthetic "[interrupted]" result — and are backend-agnostic
(the invariant they enforce is shared by every tool-using provider).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Sequence


_INTERRUPTED_RESULT_CONTENT = "[interrupted — the previous turn ended before this tool finished]"


def _tool_use_ids(msg: Dict[str, Any]) -> List[str]:
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [
        b["id"]
        for b in content
        if (
            isinstance(b, dict)
            and b.get("type") == "tool_use"
            and isinstance(b.get("id"), str)
            and b.get("id")
        )
    ]


def _tool_result_ids(msg: Dict[str, Any]) -> set[str]:
    content = msg.get("content")
    if not isinstance(content, list):
        return set()
    result_ids: set[str] = set()
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        result_id = block.get("tool_use_id")
        if isinstance(result_id, str) and result_id:
            result_ids.add(result_id)
    return result_ids


def _is_tool_result_only(msg: Dict[str, Any]) -> bool:
    """True when the message is a user turn made ENTIRELY of tool_result
    blocks — the shape that is orphaned when its tool_use is dropped."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _synthetic_tool_result(tool_use_id: str) -> Dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": _INTERRUPTED_RESULT_CONTENT,
        "is_error": True,
    }


def normalize_messages_for_request(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a provider-safe copy of an entire canonical history.

    This is the request-boundary counterpart to the state-mutating recovery
    helpers below.  It adapts Codex's history-normalization invariant to this
    runtime's role/content message shape:

    * every assistant ``tool_use`` has a later ``tool_result``;
    * every ``tool_result`` refers to an earlier assistant ``tool_use``;
    * repairs are request-only and never mutate the persisted history.

    Missing outputs are inserted immediately after their assistant call.  An
    orphan result block is removed wherever it occurs; non-result blocks in
    the same message are preserved.  The operation is idempotent, so retries,
    resumed sessions, and compacted histories all produce the same prompt.
    """
    normalized: List[Dict[str, Any]] = []
    seen_call_ids: set[str] = set()

    # Remove outputs that cannot be paired with an earlier call.  Copy only a
    # message whose content changes; clean histories retain their nested
    # objects and avoid a full deep-copy on every model request.
    for message in messages:
        if not isinstance(message, dict):
            # The public annotation is canonical dictionaries, but preserving
            # an unexpected item is safer than losing host data during repair.
            normalized.append(message)
            continue

        current_call_ids = _tool_use_ids(message) if message.get("role") == "assistant" else []

        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            seen_call_ids.update(current_call_ids)
            continue

        filtered: List[Any] = []
        removed_orphan = False
        for block in content:
            is_result = isinstance(block, dict) and block.get("type") == "tool_result"
            if is_result:
                result_id = block.get("tool_use_id")
                if (
                    not isinstance(result_id, str)
                    or not result_id
                    or result_id not in seen_call_ids
                ):
                    removed_orphan = True
                    continue
            filtered.append(block)

        if not removed_orphan:
            normalized.append(message)
        elif filtered:
            normalized.append({**message, "content": filtered})
        # A message made entirely of orphan results has no model-visible
        # content left and is intentionally omitted.
        seen_call_ids.update(current_call_ids)

    answered_ids: set[str] = set()
    for message in normalized:
        if isinstance(message, dict):
            answered_ids.update(
                result_id
                for result_id in _tool_result_ids(message)
                if isinstance(result_id, str) and result_id
            )

    missing_after_message: List[tuple[int, List[str]]] = []
    for index, message in enumerate(normalized):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        missing = [tool_id for tool_id in _tool_use_ids(message) if tool_id not in answered_ids]
        if missing:
            missing_after_message.append((index, missing))

    # Reverse insertion keeps the collected indices valid and makes repeated
    # normalization byte-stable for prompt-cache reuse.
    for index, missing in reversed(missing_after_message):
        normalized.insert(
            index + 1,
            {
                "role": "user",
                "content": [_synthetic_tool_result(tool_id) for tool_id in missing],
            },
        )

    return normalized


def repair_dangling_tool_calls(messages: List[Dict[str, Any]]) -> int:
    """Append synthetic error tool_results for any unanswered tool_use.

    Scans for the last assistant message that issued ``tool_use`` blocks;
    if any of its ids are not answered by a following ``tool_result``,
    inserts a user ``tool_result`` (``is_error``, ``"[interrupted]"``)
    for each unmatched id immediately after that assistant message — the
    exact position the API requires. Returns the number of results
    injected (0 when history is already consistent). Mutates ``messages``
    in place.
    """
    if not messages:
        return 0

    last_use_idx = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant" and _tool_use_ids(msg):
            last_use_idx = i
            break
    if last_use_idx is None:
        return 0

    use_ids = _tool_use_ids(messages[last_use_idx])
    answered: set = set()
    for j in range(last_use_idx + 1, len(messages)):
        if isinstance(messages[j], dict):
            answered |= _tool_result_ids(messages[j])
    missing = [uid for uid in use_ids if uid not in answered]
    if not missing:
        return 0

    synthetic = {
        "role": "user",
        "content": [_synthetic_tool_result(uid) for uid in missing],
    }
    messages.insert(last_use_idx + 1, synthetic)
    return len(missing)


def strip_leading_orphan_tool_results(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop leading messages that are orphaned ``tool_result`` turns.

    A compaction that keeps ``messages[-keep:]`` can begin the window
    with a user ``tool_result`` message whose ``tool_use`` was in the
    dropped prefix — invalid. Returns a copy with such leading orphans
    removed so the kept window always opens on a real turn. Only drops a
    leading tool_result message when its ids are NOT answered by a
    ``tool_use`` still present earlier in the (already-sliced) window,
    which for a leading message is always — so it simply skips any
    tool_result-only message at the head.
    """
    i = 0
    while i < len(messages) and _is_tool_result_only(messages[i]):
        i += 1
    return messages[i:] if i else messages


# ── Retiring calls to tools that are gone ─────────────────────────────

_RETIRED_INPUT_CHARS = 200
_RETIRED_RESULT_CHARS = 400
_BRIDGE_PREFIX = re.compile(r"^mcp__[A-Za-z0-9-]+__")


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                pieces.append(str(block.get("text", "")))
            elif block.get("type") == "image":
                pieces.append("[image]")
        return " ".join(p for p in pieces if p)
    return "" if content is None else str(content)


def retire_tool_calls(
    messages: Sequence[Dict[str, Any]],
    *,
    is_retired: Callable[[str], bool],
    reason: str = "tool not available now",
) -> List[Dict[str, Any]]:
    """Rewrite history calls to tools that are gone into one plain-text line each.

    A tool that existed in an earlier turn may be absent now (the host
    stopped offering it). Its ``tool_use``/``tool_result`` pair stays in
    history verbatim, so the model tends to call it again or trust what it
    returned as still current. This returns a COPY where each such call
    and its result become a short text block marked with ``reason``.

    Both sides of a pair are rewritten together — rewriting only one would
    orphan the other. Message count and role order are unchanged.
    """
    retired_ids = set()
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") != "assistant" or not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and is_retired(str(block.get("name", "")))
            ):
                retired_ids.add(block.get("id"))
    if not retired_ids:
        return list(messages)

    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        changed = False
        blocks: List[Any] = []
        for block in content:
            if not isinstance(block, dict):
                blocks.append(block)
                continue
            kind = block.get("type")
            if kind == "tool_use" and block.get("id") in retired_ids:
                args = json.dumps(block.get("input") or {}, ensure_ascii=False)
                blocks.append(
                    {
                        "type": "text",
                        "text": (
                            f"[earlier call, {reason}] "
                            f"{block.get('name')}({_clip(args, _RETIRED_INPUT_CHARS)})"
                        ),
                    }
                )
                changed = True
            elif kind == "tool_result" and block.get("tool_use_id") in retired_ids:
                blocks.append(
                    {
                        "type": "text",
                        "text": (
                            "[result of that earlier call] "
                            f"{_clip(_result_text(block.get('content')), _RETIRED_RESULT_CHARS)}"
                        ),
                    }
                )
                changed = True
            else:
                blocks.append(block)
        out.append({**msg, "content": blocks} if changed else msg)
    return out


def retire_tool_calls_by_name(
    messages: Sequence[Dict[str, Any]], spec: Any
) -> List[Dict[str, Any]]:
    """``retire_tool_calls`` driven by a plain-data spec from ``state.shared``.

    ``spec`` is ``{"names": [...], "reason": str}``. Names match with or
    without a CLI bridge server prefix (``mcp__connector__<name>``). A
    malformed spec leaves the messages untouched.
    """
    if not isinstance(spec, dict):
        return list(messages)
    names = {str(n) for n in (spec.get("names") or []) if n}
    if not names:
        return list(messages)
    reason = str(spec.get("reason") or "tool not available now")
    return retire_tool_calls(
        messages,
        is_retired=lambda name: _BRIDGE_PREFIX.sub("", str(name or "")) in names,
        reason=reason,
    )
