"""History compactors — concrete implementations for history compression."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.core.message_repair import strip_leading_orphan_tool_results
from xgen_agent_runtime.core.schema import ConfigField, ConfigSchema
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.stages.s02_context.interface import HistoryCompactor


def _safe_recent(messages: list, keep: int) -> list:
    """Last ``keep`` messages, with the window snapped so it never opens
    on a tool_result whose tool_use was dropped (audit D4 / C3)."""
    return strip_leading_orphan_tool_results(messages[-keep:]) if keep > 0 else []


class TruncateCompactor(HistoryCompactor):
    """Truncate oldest messages."""

    def __init__(self, keep_last: int = 20):
        self._keep_last = keep_last

    @property
    def name(self) -> str:
        return "truncate"

    @property
    def description(self) -> str:
        return f"Keep last {self._keep_last} messages, drop older"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="truncate",
            fields=[
                ConfigField(
                    name="keep_last",
                    type="integer",
                    label="Keep last (messages)",
                    description="Drop everything older than the last N messages once history exceeds the threshold.",
                    default=20,
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("keep_last")
        if isinstance(n, int) and n > 0:
            self._keep_last = n

    def get_config(self) -> Dict[str, Any]:
        return {"keep_last": self._keep_last}

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) > self._keep_last:
            state.messages = _safe_recent(state.messages, self._keep_last)


class SummaryCompactor(HistoryCompactor):
    """Replace old messages with a summary placeholder.

    Non-LLM fallback: replaces dropped messages with a static placeholder.
    See :class:`LLMSummaryCompactor` for the real summarization path that
    calls ``state.llm_client`` when the hosting stage has a model override.
    """

    def __init__(self, keep_recent: int = 10, summary_text: str = ""):
        self._keep_recent = keep_recent
        self._summary_text = summary_text

    @property
    def name(self) -> str:
        return "summary"

    @property
    def description(self) -> str:
        return "Replace old messages with summary"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="summary",
            fields=[
                ConfigField(
                    name="keep_recent",
                    type="integer",
                    label="Keep recent (messages)",
                    description="Number of most recent messages to keep verbatim.",
                    default=10,
                    min_value=1,
                ),
                ConfigField(
                    name="summary_text",
                    type="string",
                    label="Summary placeholder",
                    description="Optional static summary text. If empty, a generic placeholder is used.",
                    default="",
                    ui_widget="textarea",
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        keep = config.get("keep_recent")
        if isinstance(keep, int) and keep > 0:
            self._keep_recent = keep
        text = config.get("summary_text")
        if isinstance(text, str):
            self._summary_text = text

    def get_config(self) -> Dict[str, Any]:
        return {
            "keep_recent": self._keep_recent,
            "summary_text": self._summary_text,
        }

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) <= self._keep_recent:
            return

        old_count = len(state.messages) - self._keep_recent
        recent = _safe_recent(state.messages, self._keep_recent)

        summary = self._summary_text or (
            f"[Summary of {old_count} previous messages. "
            "Conversation history has been compacted to save context window.]"
        )

        state.messages = [
            {"role": "user", "content": summary},
            {
                "role": "assistant",
                "content": "Understood, I have the context from our previous conversation.",
            },
        ] + recent


#: Opens every synthetic checkpoint message: how a later compaction (and a reader of the transcript)
#: tells a checkpoint from something the user said.
CHECKPOINT_MARKER = "[Context checkpoint]"
_CHECKPOINT_INTRO = (
    f"{CHECKPOINT_MARKER} The conversation before this point no longer fit the context window, so it "
    "was compacted. Below is a summary of it, written so you can continue the work without the "
    "earlier messages."
)
#: The assistant turn that keeps roles alternating after the checkpoint (only when the next kept
#: message is a user message).
CHECKPOINT_ACK = "Understood. I have the checkpoint summary and will continue from it."
#: Acks written by runtimes before 4.89.0: recognised so a later compaction folds them in too.
_LEGACY_ACKS = ("Understood, I have the context from our previous conversation.",)

#: ``state.shared`` slot carrying verbatim user messages across compactions within a turn.
PINNED_USERS_KEY = "_compaction_pinned_users"

_SUMMARY_PROMPT = """You are compacting the history of an ongoing agent conversation because it no longer fits the model's context window. Write a handoff summary that lets the agent continue the work seamlessly WITHOUT the older messages.

Rules:
- Write in the language the conversation uses.
- Plain text only. Do not call tools.
- Facts found by tools must survive: the raw tool output will be gone. Keep concrete file paths, identifiers, commands, numbers, error messages and URLs.
- If the transcript starts with an earlier checkpoint summary, fold everything it says into this one: this summary replaces it.
- Be complete on facts and terse on prose. Stay under about {words} words.

Use these sections (skip one only if it would be empty):
1. User requests and intent: every request, constraint and preference the user stated; quote the wording where it matters.
2. Work done: what was tried, which tools ran on what, and what came back.
3. Decisions: what was decided and why.
4. Problems: errors met and how they were resolved, or that they are still open.
5. Open items: what remains to be done.
6. Current state: what was being worked on right before this point, and the immediate next step.

<transcript>
{transcript}
</transcript>"""

#: Per-message clipping inside the summarizer's transcript (chars).
_CLIP_USER = 6_000
_CLIP_ASSISTANT = 4_000
_CLIP_TOOL_INPUT = 400
_CLIP_TOOL_RESULT_HEAD = 1_500
_CLIP_TOOL_RESULT_TAIL = 500
#: Verbatim user messages kept beside the summary: share of the window, with an absolute cap.
_PINNED_RATIO = 0.08
_PINNED_MAX_TOKENS = 20_000
#: Summary length (output tokens): window/16, between these bounds.
_SUMMARY_MIN_TOKENS = 1_024
_SUMMARY_MAX_TOKENS = 4_096
#: OpenAI reasoning models spend completion tokens on reasoning before the text.
_REASONING_HEADROOM = 4_096
_OPENAI_FAMILY = ("openai", "azure", "azure_foundry")


def _content_blocks(content: Any) -> list:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return []


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        str(b.get("text", "")) for b in _content_blocks(content) if b.get("type") == "text"
    )


def _is_tool_result_only(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    blocks = _content_blocks(message.get("content"))
    return bool(blocks) and all(b.get("type") == "tool_result" for b in blocks)


def _is_checkpoint(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "user"
        and _message_text(message).lstrip().startswith(CHECKPOINT_MARKER)
    )


def _is_ack(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and _message_text(message).strip() in (CHECKPOINT_ACK,) + _LEGACY_ACKS
    )


def _is_real_user(message: Any) -> bool:
    """A user message a person wrote (not a tool result, not a checkpoint)."""
    return (
        isinstance(message, dict)
        and message.get("role") == "user"
        and not _is_tool_result_only(message)
        and not _is_checkpoint(message)
        and bool(_message_text(message).strip())
    )


def _clip(text: str, limit: int, tail: int = 0) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    if tail <= 0:
        tail = limit // 4
    head = max(0, limit - tail)
    return f"{text[:head]}\n… [{len(text) - head - tail} chars omitted] …\n{text[-tail:]}"


def _render_block(role: str, block: Dict[str, Any]) -> List[str]:
    btype = block.get("type")
    if btype == "text":
        limit = _CLIP_USER if role == "user" else _CLIP_ASSISTANT
        text = str(block.get("text") or "").strip()
        return [f"{role}: {_clip(text, limit)}"] if text else []
    if btype == "tool_use":
        try:
            args = json.dumps(block.get("input"), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            args = str(block.get("input"))
        return [f"assistant → tool {block.get('name', '?')}({_clip(args, _CLIP_TOOL_INPUT)})"]
    if btype == "tool_result":
        inner = block.get("content")
        if isinstance(inner, list):
            parts = []
            for sub in inner:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    parts.append(str(sub.get("text") or ""))
                elif isinstance(sub, dict) and sub.get("type") in ("image", "document"):
                    parts.append(f"[{sub.get('type')}]")
            inner = "\n".join(parts)
        text = str(inner or "").strip()
        label = "tool error" if block.get("is_error") else "tool result"
        body = _clip(
            text, _CLIP_TOOL_RESULT_HEAD + _CLIP_TOOL_RESULT_TAIL, tail=_CLIP_TOOL_RESULT_TAIL
        )
        return [f"← {label}: {body or '(empty)'}"]
    if btype in ("image", "image_url", "input_image"):
        return [f"{role}: [image]"]
    if btype == "document":
        return [f"{role}: [document]"]
    return []


def _render_message(message: Any) -> List[str]:
    if not isinstance(message, dict):
        return []
    role = str(message.get("role") or "")
    if _is_checkpoint(message):
        return ["(earlier checkpoint summary)\n" + _message_text(message).strip()]
    if _is_ack(message):
        return []
    lines: List[str] = []
    for block in _content_blocks(message.get("content")):
        lines.extend(_render_block(role, block))
    return lines


@dataclass
class CompactionPlan:
    """What :meth:`LLMSummaryCompactor.compact` would do: also what the runner's gate reads."""

    #: ``messages[:cut]`` are summarized, ``messages[cut:]`` are kept verbatim.
    cut: int
    #: Estimated tokens of the NEW material in the summarized region (an earlier checkpoint excluded).
    region_tokens: int
    #: Earlier user messages kept verbatim beside the summary, oldest first.
    pinned: List[str]
    #: The request this turn is working on, when it falls inside the summarized region.
    current_request: str = ""


class LLMSummaryCompactor(SummaryCompactor):
    """Summarizes the older part of the conversation with the agent's own model.

    What survives a compaction (4.89.0)
    -----------------------------------
    * **Recent messages verbatim**: the longest recent suffix that fits the target, at most
      ``keep_recent`` messages, never opening on a tool result whose call was dropped, and always
      at least the last exchange.
    * **What the user asked, verbatim**: earlier user messages within a budget (8% of the window,
      at most 20k tokens), newest first, and always the request the turn is working on even when it
      is in the summarized part. The summary is lossy; the user's words should not be.
    * **A structured summary** of the rest, built from user and assistant text AND tool calls and
      their (clipped) results: facts found by tools used to vanish because only text blocks were
      summarized. When the transcript is too long for the summarizer it keeps the NEWEST part (and
      any earlier checkpoint): it used to keep the oldest 12,000 characters, dropping exactly the
      context closest to the current work.

    Failure
    -------
    The summary call is retried once. If there is still no summary: under soft pressure (the
    proactive 80% trigger) the history is left as it is: the guard still protects the request;
    under hard pressure (guard, provider overflow, explicit request) a deterministic digest of the
    summarized part replaces it. Before 4.89.0 any failure replaced the older history with a
    one-line placeholder.

    No model wired (no ``state.model``/client, or a manifest without an override) counts as a
    failure without a call: the no-cost-by-default rule still holds.
    """

    uses_llm = True

    def __init__(
        self,
        keep_recent: int = 10,
        summary_text: str = "",
        *,
        resolve_cfg: Optional[Callable[[PipelineState], ModelConfig]] = None,
        has_override: Optional[Callable[[], bool]] = None,
        client_getter: Optional[Callable[[PipelineState], Any]] = None,
    ):
        super().__init__(keep_recent=keep_recent, summary_text=summary_text)
        self._resolve_cfg = resolve_cfg
        self._has_override = has_override or (lambda: False)
        self._client_getter = client_getter or (lambda s: getattr(s, "llm_client", None))

    @property
    def name(self) -> str:
        return "llm_summary"

    @property
    def description(self) -> str:
        return "LLM-backed summary compactor (falls back to a deterministic digest when no model is wired)"

    # ── planning ─────────────────────────────────────────────────────

    @staticmethod
    def _summary_tokens(window: int, provider: str = "") -> int:
        tokens = min(_SUMMARY_MAX_TOKENS, max(_SUMMARY_MIN_TOKENS, window // 16))
        if provider in _OPENAI_FAMILY:
            tokens = min(tokens + _REASONING_HEADROOM, max(_SUMMARY_MIN_TOKENS, window // 8))
        return tokens

    def plan(
        self, state: PipelineState, target_tokens: Optional[int] = None
    ) -> Optional[CompactionPlan]:
        """Where to cut, what to pin. ``None`` when there is nothing to summarize."""
        from xgen_agent_runtime.core.compaction import TARGET_RATIO, window_of
        from xgen_agent_runtime.core.token_estimate import (
            estimate_messages_scaled,
            estimate_overhead_tokens,
            estimate_text_scaled,
        )

        msgs = list(state.messages or [])
        n = len(msgs)
        if n <= 1:
            return None
        window = window_of(state)
        target = int(target_tokens or window * TARGET_RATIO)
        provider = str(getattr(self._client_getter(state), "provider", "") or "")

        # Minimal tail: the last message; when it is a tool result, also the call it answers.
        cut = n - 1
        if _is_tool_result_only(msgs[cut]) and cut > 0 and msgs[cut - 1].get("role") == "assistant":
            cut -= 1

        pinned_budget = min(int(window * _PINNED_RATIO), _PINNED_MAX_TOKENS)
        budget = (
            target
            - estimate_overhead_tokens(state)
            - self._summary_tokens(window, provider)
            - 400
            - pinned_budget
        )
        kept_tokens = estimate_messages_scaled(state, msgs[cut:])
        while cut > 0 and (n - (cut - 1)) <= max(1, self._keep_recent):
            t = estimate_messages_scaled(state, [msgs[cut - 1]])
            if kept_tokens + t > budget:
                break
            kept_tokens += t
            cut -= 1
        # Never open the kept part on a tool result whose call is being summarized.
        while cut < n - 1 and _is_tool_result_only(msgs[cut]):
            cut += 1
        if cut <= 0:
            return None

        region = msgs[:cut]
        start = 0
        if region and _is_checkpoint(region[0]):
            start = 1
            if len(region) > 1 and _is_ack(region[1]):
                start = 2
        new_material = region[start:]
        if not new_material:
            return None  # only an earlier checkpoint would be summarized again

        current = ""
        for m in reversed(msgs):
            if _is_real_user(m):
                current = _message_text(m).strip() if any(m is r for r in region) else ""
                break

        carried = list(state.shared.get(PINNED_USERS_KEY) or [])
        candidates = carried + [_message_text(m).strip() for m in new_material if _is_real_user(m)]
        if current and candidates and candidates[-1] == current:
            candidates = candidates[:-1]  # shown separately as the request being worked on
        pinned: List[str] = []
        used = estimate_text_scaled(state, current) if current else 0
        for text in reversed(candidates):  # newest first
            cost = estimate_text_scaled(state, text)
            if used + cost > pinned_budget:
                break
            pinned.insert(0, text)
            used += cost
        if current and estimate_text_scaled(state, current) > pinned_budget:
            current = _clip(current, max(400, int(pinned_budget)))

        return CompactionPlan(
            cut=cut,
            region_tokens=estimate_messages_scaled(state, new_material),
            pinned=pinned,
            current_request=current,
        )

    # ── transcript ───────────────────────────────────────────────────

    def _transcript(self, state: PipelineState, region: List[Any], budget_tokens: int) -> str:
        """Render the summarized region, keeping the NEWEST lines (and an earlier checkpoint)."""
        from xgen_agent_runtime.core.token_estimate import estimate_text_scaled

        rendered: List[List[str]] = [_render_message(m) for m in region]
        head: List[str] = []
        body = rendered
        if region and _is_checkpoint(region[0]):
            head = [_clip("\n".join(rendered[0]), max(2_000, int(budget_tokens * 0.3)))]
            body = rendered[1:]
        lines = [line for chunk in body for line in chunk]
        used = sum(estimate_text_scaled(state, h) for h in head)
        kept: List[str] = []
        for line in reversed(lines):
            cost = estimate_text_scaled(state, line) + 1
            if used + cost > budget_tokens:
                break
            kept.insert(0, line)
            used += cost
        omitted = len(lines) - len(kept)
        if omitted:
            kept.insert(0, f"[… {omitted} earlier lines omitted to fit the summarizer …]")
        return "\n".join(head + kept)

    # ── fallbacks ────────────────────────────────────────────────────

    def _digest(self, region: List[Any], old_count: int) -> str:
        """Deterministic stand-in when no model summary could be made: no call, no invention."""
        lines = [
            self._summary_text
            or (
                f"[Summary of {old_count} previous messages. The model summary was unavailable, "
                "so this is a mechanical digest of what happened.]"
            )
        ]
        calls: List[str] = []
        last_text = ""
        for m in region:
            if not isinstance(m, dict):
                continue
            if _is_checkpoint(m):
                lines.append("Earlier checkpoint:\n" + _clip(_message_text(m).strip(), 3_000))
                continue
            for block in _content_blocks(m.get("content")):
                if block.get("type") == "tool_use":
                    try:
                        args = json.dumps(block.get("input"), ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        args = str(block.get("input"))
                    calls.append(f"- {block.get('name', '?')}({_clip(args, 160)})")
            if m.get("role") == "assistant" and not _is_ack(m):
                text = _message_text(m).strip()
                if text:
                    last_text = text
        if calls:
            lines.append("Tool calls made (most recent last):\n" + "\n".join(calls[-20:]))
        if last_text:
            lines.append("Last assistant note:\n" + _clip(last_text, 1_500))
        return "\n\n".join(lines)

    # ── compaction ───────────────────────────────────────────────────

    def _resolve(self, state: PipelineState) -> Tuple[Optional[Callable], bool]:
        resolve = self._resolve_cfg
        has_override = self._has_override()
        if resolve is None:
            state_model = getattr(state, "model", None)
            if state_model:
                resolve = lambda s: ModelConfig(  # noqa: E731
                    model=getattr(s, "model", state_model) or state_model,
                    max_tokens=_SUMMARY_MAX_TOKENS,
                    temperature=0.0,
                    thinking_enabled=False,
                )
                has_override = True
        return resolve, has_override

    async def _summarize(
        self, state: PipelineState, region: List[Any], resolve: Callable, client: Any
    ) -> Tuple[Optional[str], Any, int]:
        """(summary or None, model config, attempts)."""
        from dataclasses import replace as _replace

        from xgen_agent_runtime.core.compaction import window_of

        window = window_of(state)
        provider = str(getattr(client, "provider", "") or "")
        summary_tokens = self._summary_tokens(window, provider)
        transcript_budget = max(2_000, min(int(window * 0.6), window - summary_tokens - 3_000))
        transcript = self._transcript(state, region, transcript_budget)
        words = max(300, min(1_500, summary_tokens // 3))
        prompt = _SUMMARY_PROMPT.format(words=words, transcript=transcript)

        cfg = resolve(state)
        if getattr(cfg, "thinking_level", None):
            # The user's thinking level is for answers: the summary uses the model default.
            cfg = _replace(cfg, thinking_level=None)
        if self._resolve_cfg is None and getattr(cfg, "max_tokens", 0) != summary_tokens:
            cfg = _replace(cfg, max_tokens=summary_tokens)

        attempts = 0
        for attempt in range(2):
            attempts = attempt + 1
            try:
                resp = await client.create_message(
                    model_config=cfg,
                    messages=[{"role": "user", "content": prompt}],
                    purpose="s02.compact",
                )
            except Exception as exc:  # noqa: BLE001 (retried once, then the caller decides)
                state.add_event(
                    "memory.compaction.llm_failed",
                    {"error": str(exc), "compactor": self.name, "attempt": attempts},
                )
                if attempt == 0:
                    await asyncio.sleep(1.0)
                continue
            # The summary is a model call outside the main loop: report it as auxiliary usage.
            # (Background compaction gets a shadow instead of the state; recording must not break it.)
            _record_aux = getattr(state, "add_aux_usage", None)
            if callable(_record_aux):
                try:
                    _record_aux("compact", getattr(resp, "usage", None))
                except Exception:  # noqa: BLE001
                    pass
            text = (resp.text or "").strip()
            if text:
                return text, cfg, attempts
            state.add_event(
                "memory.compaction.llm_failed",
                {"error": "empty summary", "compactor": self.name, "attempt": attempts},
            )
        return None, cfg, attempts

    async def compact(self, state: PipelineState) -> None:
        from xgen_agent_runtime.core.compaction import COMPACTION_CONTEXT_KEY

        request = (
            state.shared.get(COMPACTION_CONTEXT_KEY) if isinstance(state.shared, dict) else None
        )
        request = request if isinstance(request, dict) else {}
        # A direct call (no runner request) is an explicit ask to compact.
        hard = bool(request.get("hard", True))
        outline = self.plan(state, request.get("target_tokens"))
        if outline is None:
            return
        msgs = list(state.messages)
        region = msgs[: outline.cut]
        kept = msgs[outline.cut :]
        old_count = len(region)

        resolve, has_override = self._resolve(state)
        client = self._client_getter(state) if (has_override and resolve is not None) else None
        summary: Optional[str] = None
        cfg: Any = None
        attempts = 0
        # digest_only: the turn already used its model summaries (core.compaction) but the request
        # cannot go out as it is: relieve it without another model call.
        if client is not None and resolve is not None and not request.get("digest_only"):
            summary, cfg, attempts = await self._summarize(state, region, resolve, client)
        if summary is None:
            if not hard:
                state.add_event(
                    "context.compaction_deferred",
                    {"compactor": self.name, "reason": "no summary", "attempts": attempts},
                )
                return  # soft pressure: losing history is worse than waiting for the guard
            summary = self._digest(region, old_count)
            source = "digest"
        else:
            source = "model"

        sections = [_CHECKPOINT_INTRO, summary]
        if outline.pinned:
            sections.append(
                "Earlier user messages, verbatim (oldest first):\n"
                + "\n".join(f"- {text}" for text in outline.pinned)
            )
        if outline.current_request:
            sections.append("The request being worked on, verbatim:\n" + outline.current_request)
        head: List[Dict[str, Any]] = [{"role": "user", "content": "\n\n".join(sections)}]
        if kept and isinstance(kept[0], dict) and kept[0].get("role") == "user":
            head.append({"role": "assistant", "content": CHECKPOINT_ACK})
        state.messages = head + strip_leading_orphan_tool_results(kept)
        state.shared[PINNED_USERS_KEY] = list(outline.pinned) + (
            [outline.current_request] if outline.current_request else []
        )
        state.add_event(
            "memory.compaction.summarized",
            {
                "model": getattr(cfg, "model", "") if cfg is not None else "",
                "provider": getattr(client, "provider", "") if client is not None else "",
                "old_count": old_count,
                "kept_count": len(kept),
                "pinned_user_messages": len(outline.pinned) + (1 if outline.current_request else 0),
                "summary_chars": len(summary),
                "source": source,
                "attempts": attempts,
            },
        )


class SlidingWindowCompactor(HistoryCompactor):
    """Sliding window — maintains a fixed message window, summarizes overflow."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size

    @property
    def name(self) -> str:
        return "sliding_window"

    @property
    def description(self) -> str:
        return f"Fixed window of {self._window_size} messages"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="sliding_window",
            fields=[
                ConfigField(
                    name="window_size",
                    type="integer",
                    label="Window size (messages)",
                    description="Fixed size of the rolling window. Older messages collapse into a single summary marker.",
                    default=30,
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("window_size")
        if isinstance(n, int) and n > 0:
            self._window_size = n

    def get_config(self) -> Dict[str, Any]:
        return {"window_size": self._window_size}

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) <= self._window_size:
            return

        overflow = len(state.messages) - self._window_size
        summary = {
            "role": "user",
            "content": f"[{overflow} earlier messages summarized and compacted.]",
        }
        state.messages = [summary] + _safe_recent(state.messages, self._window_size)
