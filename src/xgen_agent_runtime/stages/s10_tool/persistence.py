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
    """Tokens a single result may take in this turn's request, or ``None`` if the window is unknown.

    A quarter of the INPUT budget (the window minus the room kept for the answer,
    ``core.compaction.input_budget_of``): on an 8k model with ``max_tokens`` 8192 a quarter of the
    window was half of what the request could hold.
    """
    state = getattr(context, "state_view", None)
    try:
        window = int(getattr(state, "context_window_budget", 0) or 0)
    except (TypeError, ValueError):
        window = 0
    if window <= 0:
        return None
    from xgen_agent_runtime.core.compaction import input_budget_of

    return max(_WINDOW_CAP_FLOOR_TOKENS, int(input_budget_of(state) * _WINDOW_SHARE))


#: Above this many characters the token estimate is taken from a sample (head and tail) and
#: scaled: a per-character count of a multi-megabyte result blocked the event loop for seconds.
_SAMPLE_OVER_CHARS = 200_000
_SAMPLE_CHARS = 100_000


def _view_tokens(text: str) -> int:
    """Script-aware token estimate of what the model will read, with the conservative factor (the
    provider is not known here). Characters are not tokens: 100k chars of English is ~25k tokens,
    of Korean ~100k."""
    from xgen_agent_runtime.core.text_tokens import (
        DEFAULT_PROVIDER_FACTOR,
        SAFETY_MULTIPLIER,
        estimate_tokens_cached,
    )

    if len(text) > _SAMPLE_OVER_CHARS:
        sample = text[:_SAMPLE_CHARS] + text[-_SAMPLE_CHARS:]
        raw = estimate_tokens_cached(sample) * len(text) / len(sample)
    else:
        raw = estimate_tokens_cached(text)
    return int(raw * DEFAULT_PROVIDER_FACTOR * SAFETY_MULTIPLIER)


def _clip_marker(total_chars: int, saved_at: str, read_id: str = "") -> str:
    if read_id:
        where = (
            f'Read any part of the full result with ToolResultRead(id="{read_id}", offset=<char '
            "offset>) before relying on what is omitted"
        )
        if saved_at:
            where += f" (it is also saved at: {saved_at})"
    elif saved_at:
        where = f"The full result is saved at: {saved_at}"
    else:
        where = "Re-run with a narrower query (a line range, a filter) if you need the omitted part"
    return (
        f"[tool result clipped to fit this model's context window: {total_chars} chars, the middle "
        f"is omitted. {where}.]"
    )


def _safe_id(tool_use_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(tool_use_id or ""))


def persisted_path(storage_path: str, tool_use_id: str) -> str:
    """Where :func:`_persist_full` keeps a result's full body (``ToolResultRead`` reads it)."""
    return os.path.join(storage_path, TOOL_RESULTS_DIRNAME, f"{_safe_id(tool_use_id)}.json")


def _persist_full(
    result: ToolResult, *, tool_use_id: str, tool_name: str, context: ToolContext
) -> str:
    """Write the full result to the host's tool-result store. Returns the path, or "" when there is
    no store (or writing failed). Every clip keeps the original this way: the model gets a cut copy
    and a way to read the rest (``ToolResultRead``), never only the cut (2026-10-23 audit)."""
    if result.persist_full:
        return str(result.persist_full)
    storage_path = getattr(context, "storage_path", None)
    if not storage_path or not tool_use_id:
        return ""
    try:
        target_path = persisted_path(storage_path, tool_use_id)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        envelope: Dict[str, Any] = {
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "is_error": result.is_error,
            "content": result.content,
            "display_text": result.display_text,
            "metadata": result.metadata,
            "artifacts": result.artifacts,
        }
        with open(target_path, "w", encoding="utf-8") as fh:
            json.dump(envelope, fh, ensure_ascii=False, default=str)
        return target_path
    except (OSError, TypeError, ValueError) as exc:
        _LOG.warning("failed to persist tool result for %s (%s): %s", tool_name, tool_use_id, exc)
        return ""


def _clip_to_tokens(text: str, cap: int) -> str:
    """Head and tail of ``text`` whose estimate is within ``cap`` tokens (re-measured: the head and
    tail can be denser than the average)."""
    tokens = _view_tokens(text)
    if tokens <= cap:
        return text
    keep = max(200, int(len(text) * cap / tokens * 0.95))
    clipped = text
    for _ in range(6):
        tail = keep // 4
        head = max(0, keep - tail)
        clipped = f"{text[:head]}\n…\n{text[-tail:] if tail else ''}"
        measured = _view_tokens(clipped)
        if measured <= cap or keep <= 200:
            break
        keep = max(200, int(keep * min(0.9, cap / measured * 0.95)))
    return clipped


def _error_header(content: Any) -> str:
    """The first line ``ToolResult.to_api_format`` gives a structured error, kept through a clip."""
    if isinstance(content, dict):
        err = content.get("error")
        if (
            isinstance(err, dict)
            and isinstance(err.get("code"), str)
            and isinstance(err.get("message"), str)
        ):
            return f"ERROR {err['code']}: {err['message']}"
    return ""


def _fit_to_window(
    result: ToolResult, *, tool_name: str, context: ToolContext, tool_use_id: str = ""
) -> ToolResult:
    """Clip what the model will read (``display_text`` when set, else the content) to a quarter of
    the input budget. The tool's own character cap ignores the model: 100k chars is 25k-100k tokens,
    so on a 32k-window model one result could overflow the request by itself.

    Text is clipped head-and-tail; image blocks in a block-list result are kept as they are (they
    are counted separately). A structured error keeps its ``ERROR <code>: <message>`` line.
    """
    cap = _window_cap_tokens(context)
    if cap is None:
        return result
    saved_at = ""
    if result.persist_full and _reachable(result.persist_full, context):
        saved_at = result.persist_full
    read_id = ""

    def _keep_original() -> None:
        # Only when a clip is about to happen: keep the full body and name it for ToolResultRead.
        nonlocal read_id, result
        path = _persist_full(result, tool_use_id=tool_use_id, tool_name=tool_name, context=context)
        if path:
            read_id = tool_use_id
            result = replace(result, persist_full=path)

    if result.display_text is None and isinstance(result.content, list):
        texts = [b for b in result.content if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n".join(str(b.get("text") or "") for b in texts)
        if not texts or _view_tokens(joined) <= cap:
            return result
        _keep_original()
        clipped = _clip_to_tokens(joined, cap)
        kept = [b for b in result.content if not (isinstance(b, dict) and b.get("type") == "text")]
        _LOG.info("tool %s result text over window cap %d; clipping (images kept)", tool_name, cap)
        marker = _clip_marker(len(joined), saved_at, read_id)
        return replace(result, content=[{"type": "text", "text": f"{marker}\n{clipped}"}] + kept)

    view = (
        result.display_text if result.display_text is not None else _render_content(result.content)
    )
    if _view_tokens(view) <= cap:
        return result
    header = _error_header(result.content) if result.display_text is None else ""
    _keep_original()
    clipped = _clip_to_tokens(view, cap)
    _LOG.info(
        "tool %s result over window cap %d; clipping to %d chars", tool_name, cap, len(clipped)
    )
    lines = [header] if header else []
    lines += [_clip_marker(len(view), saved_at, read_id), clipped]
    return replace(result, display_text="\n".join(lines))


def maybe_persist_large_result(
    result: ToolResult,
    *,
    tool_use_id: str,
    tool_name: str,
    capabilities: ToolCapabilities,
    context: ToolContext,
) -> ToolResult:
    """Apply the tool's character cap (:func:`_persist_over_tool_cap`), then bound what the model
    reads by the context window (:func:`_fit_to_window`, 4.89.0).

    ``max_result_chars`` of 0 (ToolBatch) or below opts out of the tool's own cap only: the window
    bound protects the request and applies to every tool. Never raises: a failure here returns the
    result as it was (the turn must not die in result bookkeeping).
    """
    if context.cancellation_scope is not None:
        context.cancellation_scope.check()
    out = result
    if capabilities.max_result_chars > 0:
        out = _persist_over_tool_cap(
            result,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            capabilities=capabilities,
            context=context,
        )
    try:
        return _fit_to_window(out, tool_name=tool_name, context=context, tool_use_id=tool_use_id)
    except Exception:  # noqa: BLE001 (fail-open)
        _LOG.warning("tool %s: window fit failed; result left as is", tool_name, exc_info=True)
        return out


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

    target_path = _persist_full(
        result, tool_use_id=tool_use_id, tool_name=tool_name, context=context
    )
    if not target_path:
        # No sink configured (or it failed) — log so ops can spot misconfig, and inline the full
        # payload: correctness over size (the window fit below still bounds what the model reads).
        _LOG.warning(
            "tool %s returned %d chars (> %d) but %s; inlining the full payload",
            tool_name,
            len(rendered),
            capabilities.max_result_chars,
            "no storage_path"
            if not getattr(context, "storage_path", None)
            else "it could not be kept",
        )
        return result
    if result.display_text is not None:
        # The tool chose its own summary; the full body is kept for ToolResultRead.
        return replace(result, persist_full=target_path)
    if _window_cap_tokens(context) is not None:
        # The model's window decides how much it reads (:func:`_fit_to_window`: head and tail plus
        # the id to read the rest). The fixed cap used to cut every result over 100,000 chars to a
        # 480-char preview, even on 1M-token models, and the exit code and STDERR at the end of a
        # Bash result were the first thing lost (2026-10-23 audit).
        return replace(result, persist_full=target_path)
    # Window unknown: bound by the tool's cap, head and tail, with the id to read the rest.
    limit = capabilities.max_result_chars
    tail = limit // 4
    head = max(0, limit - tail)
    clipped = f"{rendered[:head]}\n…\n{rendered[-tail:] if tail else ''}"
    saved_at = target_path if _reachable(target_path, context) else ""
    where = (
        f'Read any part of the full result with ToolResultRead(id="{tool_use_id}", offset=<char '
        "offset>) before relying on what is omitted"
    )
    if saved_at:
        where += f" (it is also saved at: {saved_at})"
    marker = (
        f"[tool result clipped: {len(rendered)} chars, over this tool's {limit}-char limit; the "
        f"middle is omitted. {where}.]"
    )
    return replace(result, display_text=f"{marker}\n{clipped}", persist_full=target_path)
