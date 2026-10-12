"""Tool result persistence — send large payloads to disk, short summary to LLM.

Cycle 20260424 executor uplift — Phase 2 Week 4 Checkpoint 2.

When a tool returns a payload larger than
``ToolCapabilities.max_result_chars``, ``maybe_persist_large_result`` writes
the full body to ``{storage_path}/tool-results/{tool_use_id}.json`` and
returns a new ``ToolResult`` with a short ``display_text`` pointing at the
persisted path. The LLM then sees the summary, while the host retains the
full payload for audit / replay / Stage 15 memory.

Fallback policy (fail-open to preserve correctness):

* If ``storage_path`` is not set, the result is returned untouched. The
  orchestrator will still inline the full payload this turn; the caller
  is responsible for attaching storage before running in production.
* If writing the file fails for any reason, the error is logged via the
  standard ``logging`` module and the original result is returned — the
  turn must not die because the on-disk sink is unavailable.
* ``max_result_chars=0`` disables the cap (matches the ``ToolCapabilities``
  contract: ``0`` = infinite).

See ``executor_uplift/06_design_tool_system.md`` §6 and
``executor_uplift/12_detailed_plan.md`` §2 (E-2.4).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from typing import Any, Dict, Optional

from xgen_agent_runtime.tools.base import ToolCapabilities, ToolContext, ToolResult

_LOG = logging.getLogger(__name__)

# Subdirectory under ``ToolContext.storage_path`` where full tool result
# bodies are written. Kept public so hosts (Geny, Stage 15 memory) can
# enumerate or garbage-collect the sink.
TOOL_RESULTS_DIRNAME = "tool-results"


def _render_content(content: Any) -> str:
    """Render a tool result ``content`` payload to a string for size check.

    Mirrors the branching that ``ToolResult.to_api_format`` uses so the
    threshold we compare against matches the bytes the LLM would see.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list) and any(_is_image(b) for b in content):
        # 이미지 블록은 글자가 아니다 — base64 를 글자 수로 세면 보통 크기 렌더(PNG 75KB 이상)가
        # 전부 10만 자를 넘어 "잘렸다" 미리보기로 바뀌고, 모델은 그림 대신 base64 앞머리를 받는다
        # (2026-10-11 dev: 도면 렌더 Read 5회 모두 잘림). 그림 크기는 Read 의 bound_image 가
        # 이미 묶는다. context_prune._content_size 처럼 이미지 밖의 내용만 센다.
        return "\n".join(_render_block(b) for b in content)
    if isinstance(content, (list, dict)):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _is_image(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in ("image", "image_url")


def _render_block(block: Any) -> str:
    if _is_image(block):
        return "[image]"
    if (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ):
        return block["text"]
    if isinstance(block, str):
        return block
    return json.dumps(block, ensure_ascii=False, default=str)


def _reachable(path: str, context: Any) -> bool:
    """에이전트의 파일 도구가 이 절대 경로를 실제로 열 수 있는가.

    열 수 없는 경로를 "여기 저장했다" 고 말하면 두 가지를 한꺼번에 잃는다: 그 턴에 파일을
    못 읽고, 모델은 **읽을 수 없는 뿌리를 정상 경로로 학습한다**(2026-09-21 실측 — 내부
    저장소는 세션 루트의 형제라 가드가 늘 거절하는데, 모델은 그 접두사를 첨부 경로에 다시
    썼다). 경로를 말할 자격은 도달 가능할 때만 생긴다.
    """
    target = str(path or "")
    if not target:
        return False
    roots = []
    sandbox = getattr(context, "sandbox", None)
    if sandbox is not None:
        for attr in ("workdir", "extra_roots"):
            value = getattr(sandbox, attr, None)
            if isinstance(value, str):
                roots.append(value)
            elif isinstance(value, (list, tuple)):
                roots.extend(str(v) for v in value)
    else:
        working = getattr(context, "working_dir", "")
        if working:
            roots.append(str(working))
        allowed = getattr(context, "allowed_paths", None) or []
        roots.extend(str(p) for p in allowed)
    for root in roots:
        root = root.rstrip("/")
        if root and (target == root or target.startswith(root + "/")):
            return True
    return False


def _summarise(body: str, *, limit: int, path: str, reachable: bool) -> str:
    """Short LLM-facing replacement for an oversized body.

    Shows the first ~480 characters of the persisted body so the model
    still has a peek at the content, followed by where the full body went
    and the total length. Callers may override by pre-setting
    ``display_text`` on the returned ``ToolResult``.
    """
    peek = body[:480].rstrip()
    if len(body) > 480:
        peek = peek + "…"
    where = (
        f"Full body saved at: {path}"
        if reachable
        else (
            "Full body kept on the server; it is NOT in your workspace and your "
            "file tools cannot open it. Re-run the tool with a narrower query if "
            "you need more than the preview."
        )
    )
    return f"[tool result truncated: {len(body)} chars > {limit} limit]\n{where}\nPreview:\n{peek}"


#: Share of the context window one tool result may take in the request (4.89.0).
_WINDOW_SHARE = 0.25
#: Never cap below this many tokens (a 4k-window model still gets a usable result).
_WINDOW_CAP_FLOOR_TOKENS = 1_000


def _window_cap_tokens(context: ToolContext) -> Optional[int]:
    """Tokens a single result may take in this turn's request, or ``None`` if the window is unknown."""
    state = getattr(context, "state_view", None)
    try:
        window = int(getattr(state, "context_window_budget", 0) or 0)
    except (TypeError, ValueError):
        window = 0
    if window <= 0:
        return None
    return max(_WINDOW_CAP_FLOOR_TOKENS, int(window * _WINDOW_SHARE))


def _view_tokens(text: str) -> int:
    """Script-aware token estimate of what the model will read, with the conservative factor (the
    provider is not known here). Characters are not tokens: 100k chars of English is ~25k tokens,
    of Korean ~100k."""
    from xgen_agent_runtime.core.text_tokens import (
        DEFAULT_PROVIDER_FACTOR,
        SAFETY_MULTIPLIER,
        estimate_tokens_cached,
    )

    return int(estimate_tokens_cached(text) * DEFAULT_PROVIDER_FACTOR * SAFETY_MULTIPLIER)


def _clip_inline(body: str, keep_chars: int, *, saved_at: str = "") -> str:
    """Head and tail of ``body`` within about ``keep_chars`` chars, with an explicit marker."""
    tail = keep_chars // 4
    head = max(0, keep_chars - tail)
    where = (
        f"The full result is saved at: {saved_at}"
        if saved_at
        else "Re-run with a narrower query (a line range, a filter) if you need the omitted part"
    )
    return (
        f"[tool result clipped to fit this model's context window: {len(body)} chars, the middle "
        f"is omitted. {where}.]\n"
        f"{body[:head]}\n…\n{body[-tail:] if tail else ''}"
    )


def _fit_to_window(result: ToolResult, *, tool_name: str, context: ToolContext) -> ToolResult:
    """Clip what the model will read (``display_text`` when set, else the content) to a quarter of
    the context window. The tool's own character cap ignores the model: 100k chars is 25k-100k
    tokens, so on a 32k-window model one result could overflow the request by itself."""
    cap = _window_cap_tokens(context)
    if cap is None:
        return result
    view = (
        result.display_text if result.display_text is not None else _render_content(result.content)
    )
    tokens = _view_tokens(view)
    if tokens <= cap:
        return result
    keep = max(400, int(len(view) * cap / tokens * 0.95))
    saved_at = ""
    if result.persist_full and _reachable(result.persist_full, context):
        saved_at = result.persist_full
    _LOG.info(
        "tool %s result ~%d tokens > window cap %d; clipping to %d chars",
        tool_name,
        tokens,
        cap,
        keep,
    )
    return replace(result, display_text=_clip_inline(view, keep, saved_at=saved_at))


def maybe_persist_large_result(
    result: ToolResult,
    *,
    tool_use_id: str,
    tool_name: str,
    capabilities: ToolCapabilities,
    context: ToolContext,
) -> ToolResult:
    """Apply the tool's character cap (:func:`_persist_over_tool_cap`), then bound what the model
    reads by the context window (:func:`_fit_to_window`, 4.89.0). ``max_result_chars`` of 0 or
    below opts out of both."""
    if capabilities.max_result_chars <= 0:
        return result
    out = _persist_over_tool_cap(
        result,
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        capabilities=capabilities,
        context=context,
    )
    return _fit_to_window(out, tool_name=tool_name, context=context)


def _persist_over_tool_cap(
    result: ToolResult,
    *,
    tool_use_id: str,
    tool_name: str,
    capabilities: ToolCapabilities,
    context: ToolContext,
) -> ToolResult:
    """Return a possibly-rewritten ``ToolResult`` with large bodies on disk.

    Arguments:
        result: The raw tool output.
        tool_use_id: Stable call identifier used to name the persisted file.
        tool_name: Tool name (embedded in the persisted envelope for audit).
        capabilities: Resolved capabilities for this invocation. The
            ``max_result_chars`` field gates persistence.
        context: Tool execution context; its ``storage_path`` determines
            the sink directory.

    Returns:
        A new ``ToolResult`` when persistence happened, or the input
        ``result`` unchanged. ``display_text`` and ``persist_full`` on the
        returned value reflect the persisted-path + summary. When the
        tool already set ``display_text``, it is preserved — tools may
        choose their own summary, and the helper only needs to store the
        full body and populate ``persist_full``.
    """
    # Respect explicit opt-outs before even rendering the body.
    if capabilities.max_result_chars == 0:
        return result
    if capabilities.max_result_chars < 0:
        return result
    if result.persist_full:
        # Tool has already persisted externally; nothing to do.
        return result

    rendered = _render_content(result.content)
    if len(rendered) <= capabilities.max_result_chars:
        return result

    storage_path = context.storage_path
    if not storage_path:
        # No sink configured — log once per call so ops can spot misconfig,
        # but return the original result so correctness is preserved.
        _LOG.warning(
            "tool %s returned %d chars (> %d) but no storage_path; inlining the full payload",
            tool_name,
            len(rendered),
            capabilities.max_result_chars,
        )
        return result

    try:
        target_dir = os.path.join(storage_path, TOOL_RESULTS_DIRNAME)
        os.makedirs(target_dir, exist_ok=True)
        # Defensive: tool_use_ids from the API are safe; still sanitize.
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_use_id)
        target_path = os.path.join(target_dir, f"{safe_id}.json")

        envelope: Dict[str, Any] = {
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "is_error": result.is_error,
            "content": result.content,
            "metadata": result.metadata,
            "artifacts": result.artifacts,
        }
        with open(target_path, "w", encoding="utf-8") as fh:
            json.dump(envelope, fh, ensure_ascii=False, default=str, indent=2)
    except OSError as exc:
        _LOG.warning(
            "failed to persist tool result for %s (%s): %s — inlining payload",
            tool_name,
            tool_use_id,
            exc,
        )
        return result

    summary: Optional[str] = result.display_text
    if summary is None:
        summary = _summarise(
            rendered,
            limit=capabilities.max_result_chars,
            path=target_path,
            reachable=_reachable(target_path, context),
        )

    return replace(
        result,
        display_text=summary,
        persist_full=target_path,
    )
