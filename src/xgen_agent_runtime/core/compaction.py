"""Shared compaction runner and pressure policy: the ONE place that decides whether, and how, the
conversation is shrunk.

Every trigger goes through :func:`run_compaction`:

* Stage 2 *proactive* (start of an iteration, estimate above :data:`PROACTIVE_RATIO` of the input
  budget: the window minus the room kept for the answer, see :func:`input_budget_of`),
* Stage 4 *proactive* (after Stage 3 built the system prompt and tool list: the only point where the
  first iteration's estimate is complete) and the Stage 4 *guard* (the next request would not leave
  room for the answer),
* Stage 6 *overflow* (the provider rejected the request as too long: compact and retry once),
* Stage 16 *requested* and the ``/compact`` command.

What the runner guarantees (4.89.0)
-----------------------------------
* **Cheap first.** The deterministic prune (dedupe repeated tool output, drop stale images, trim
  stale large results and tool inputs) runs before any model call. Under hard pressure the
  compactor runs next, and only if the request still does not fit come the losses: the aggressive
  prune, the newest tool results shortened, and last the attachments of the request itself.
* **One summary per request.** Stage 2, Stage 4 and the guard can all see pressure before the same
  API call; only the first may summarize. A second summary of ``[summary, ack, two messages]``
  removes nothing and costs a model call (2026-10-12: observed re-summarizing every iteration).
* **No pointless summaries.** A compactor that can say what it would summarize
  (``plan(state, target_tokens)``) is skipped when that region is too small to matter, unless the
  pressure is hard.
* **A ceiling per turn.** At most :data:`MAX_LLM_COMPACTIONS_PER_TURN` summary attempts per turn, and
  after :data:`MAX_SUMMARY_FAILURES` failed ones soft pressure stops calling the model. Past either
  limit hard pressure still relieves the request with the compactor's deterministic digest.
* **Nothing summarized is forgotten by memory.** Messages that leave ``state.messages`` before the
  turn's recorders ran (STM, conversation archive) are kept in ``state.metadata`` and recorded
  first, in order. Synthetic summary messages are never recorded as if the user had said them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.core.token_estimate import estimate_prompt_tokens

logger = logging.getLogger(__name__)

#: Proactive trigger: compact when the next request is estimated above this share of the input
#: budget (:func:`input_budget_of`).
PROACTIVE_RATIO = 0.8
#: Where compaction aims, as a share of the input budget. Low enough that the next compaction is many
#: tool rounds away: compacting to just under the trigger made every few calls compact again, and
#: every compaction loses detail.
TARGET_RATIO = 0.5
#: Failed summary attempts after which soft pressure stops calling the model for the rest of the turn.
MAX_SUMMARY_FAILURES = 2
#: Summaries allowed in one turn. A turn that needs more is refilling as fast as it is compacted;
#: further summaries only pile summaries on summaries.
MAX_LLM_COMPACTIONS_PER_TURN = 6
#: A summarizable region smaller than this share of the window is not worth a model call.
MIN_SUMMARY_REGION_RATIO = 0.05
#: ...nor one below this many tokens: on a small window at most :data:`_MIN_REGION_WINDOW_CAP` of
#: it, or an 8k model could never summarize at all and every relief would wait for the guard.
MIN_SUMMARY_REGION_TOKENS = 2_000
_MIN_REGION_WINDOW_CAP = 0.15


def min_summary_region(window: int) -> int:
    """Smallest summarizable region (tokens) worth a model call for this window."""
    window = max(0, int(window))
    return max(
        int(window * MIN_SUMMARY_REGION_RATIO),
        min(MIN_SUMMARY_REGION_TOKENS, int(window * _MIN_REGION_WINDOW_CAP)),
    )


#: ``state.shared`` slots (strings are the contract).
#: Number of API calls Stage 6 has completed in this turn (advanced by :func:`note_api_call`).
API_CALLS_KEY = "_context_api_calls"
#: Value of :data:`API_CALLS_KEY` when the last model summary ran: "already summarized for this call".
SUMMARIZED_AT_KEY = "_context_summarized_at_call"
#: Model summary attempts in this turn (successful or not).
LLM_COMPACTIONS_KEY = "_context_llm_compactions"
#: Failed model summary attempts in a row in this turn.
LLM_FAILURES_KEY = "_context_llm_failures"
#: The compactor reads the current request here (``{"trigger", "target_tokens", "hard",
#: "digest_only"}``) and reports back into the same dict: ``model_calls`` (summary calls it made)
#: and ``source`` (``"model"`` or ``"digest"`` when it rewrote the history).
COMPACTION_CONTEXT_KEY = "_context_compaction_request"

#: Stage-18's STM-recording watermark (index into ``state.messages``).
#: Duplicated here (not imported from s18) so the core has no stage dep;
#: the string is the contract.
_STATE_LAST_RECORDED = "memory.last_recorded_idx"
_STRATEGY_RECORDED = "memory.provider_strategy_recorded_idx"
_ARCHIVED = "geny_bridge.conversation_archived_idx"
_WINDOW_LEN = "memory.short_term_window_len"

#: EVERY metadata key that holds an index into ``state.messages`` and therefore
#: stops meaning what it says the moment compaction shrinks that list. One key
#: was translated before 4.38.0 and the other three were not, so a compaction
#: inside the turn left them pointing past the end of the list: Stage 18 then
#: recorded ``messages[stale:]`` == nothing and the turn vanished from STM and
#: from the conversation rollup — the agent forgot the turn it had just taken.
#: Strings, not imports: the core keeps no stage/host dependency.
RECORDED_INDEX_KEYS = (
    _STATE_LAST_RECORDED,
    # ProviderDrivenStrategy / host history preload / Stage 2 short-term window.
    _STRATEGY_RECORDED,
    # Host conversation rollup (``ConversationArchivingStrategy``).
    _ARCHIVED,
    # Window length — the rollup reads it as its watermark default, so it is an
    # index in every way that matters here.
    _WINDOW_LEN,
)

#: ``state.metadata`` slot: messages compaction removed before they were recorded,
#: per recorder (``"stm"`` / ``"archive"``), oldest first.
PENDING_RECORDS_KEY = "memory.compacted_unrecorded"
RECORDER_STM = "stm"
RECORDER_ARCHIVE = "archive"
#: Newest messages kept per recorder. A host with no recorder (memory off) never takes them, and a
#: state reused across many compactions must not grow without bound.
PENDING_RECORDS_CAP = 300


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def reconcile_recorded_index(before: List[Any], after: List[Any], metadata: dict) -> None:
    """Translate every message-index watermark across a compaction (audit D3), and keep what the
    recorders have not seen yet.

    Stage 18 records ``state.messages[last_idx:]`` as STM turns and the host archive records
    ``messages[archived_idx:]``. Compactors keep a SUFFIX of the real messages (the same dict
    objects) and prepend synthetic summary messages. We find that kept suffix by identity, then for
    each recorder:

    * messages between its watermark and the start of the kept suffix were summarized away before
      it recorded them: they go to :data:`PENDING_RECORDS_KEY` and are recorded first;
    * its watermark moves past the synthetic head (a summary is not something the user said) and
      keeps pointing at the same kept message otherwise.

    A recorder that has not recorded anything yet (no watermark) is treated as watermark 0.
    Before 4.89.0 such a recorder was left at 0 and later recorded the synthetic summary as a
    user turn, while the summarized messages were never recorded at all.
    """
    before_ids = [id(m) for m in before]
    after_ids = [id(m) for m in after]
    kept = 0
    bi, ai = len(before_ids) - 1, len(after_ids) - 1
    while bi >= 0 and ai >= 0 and before_ids[bi] == after_ids[ai]:
        kept += 1
        bi -= 1
        ai -= 1
    start = len(before) - kept  # first before-index that survived
    n_synthetic = len(after) - kept  # summary messages prepended
    if start == 0 and n_synthetic == 0:
        return  # nothing changed

    def _translate(old_idx: int) -> int:
        if old_idx <= start:
            return n_synthetic
        return n_synthetic + (old_idx - start)

    pending: Dict[str, List[Any]] = metadata.get(PENDING_RECORDS_KEY) or {}
    if not isinstance(pending, dict):
        pending = {}

    stm_old = max(_int(metadata.get(_STATE_LAST_RECORDED)), _int(metadata.get(_STRATEGY_RECORDED)))
    archive_old = _int(metadata.get(_ARCHIVED, metadata.get(_WINDOW_LEN, 0)))
    for recorder, old_idx in ((RECORDER_STM, stm_old), (RECORDER_ARCHIVE, archive_old)):
        if old_idx < start:
            lost = [m for m in before[old_idx:start] if isinstance(m, dict)]
            if lost:
                pending[recorder] = (list(pending.get(recorder) or []) + lost)[
                    -PENDING_RECORDS_CAP:
                ]
    if pending:
        metadata[PENDING_RECORDS_KEY] = pending

    stm_new = max(0, min(_translate(stm_old), len(after)))
    metadata[_STATE_LAST_RECORDED] = stm_new
    metadata[_STRATEGY_RECORDED] = stm_new
    metadata[_ARCHIVED] = max(0, min(_translate(archive_old), len(after)))
    window_len = _int(metadata.get(_WINDOW_LEN))
    if window_len > 0:
        metadata[_WINDOW_LEN] = max(0, min(_translate(window_len), len(after)))


def take_pending_records(state: Any, recorder: str) -> List[Any]:
    """Messages compaction removed before ``recorder`` saw them: returned once, then forgotten."""
    metadata = getattr(state, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    pending = metadata.get(PENDING_RECORDS_KEY)
    if not isinstance(pending, dict):
        return []
    out = pending.pop(recorder, None) or []
    if not pending:
        metadata.pop(PENDING_RECORDS_KEY, None)
    return [m for m in out if isinstance(m, dict)]


def peek_pending_records(state: Any, recorder: str) -> List[Any]:
    """Like :func:`take_pending_records` but leaves them in place."""
    metadata = getattr(state, "metadata", None)
    pending = metadata.get(PENDING_RECORDS_KEY) if isinstance(metadata, dict) else None
    if not isinstance(pending, dict):
        return []
    return [m for m in (pending.get(recorder) or []) if isinstance(m, dict)]


def note_api_call(state: Any) -> None:
    """Stage 6: one more request went out. Advances the per-call compaction marker."""
    shared = getattr(state, "shared", None)
    if isinstance(shared, dict):
        shared[API_CALLS_KEY] = _int(shared.get(API_CALLS_KEY)) + 1


def summarized_for_this_call(state: Any) -> bool:
    """True when a model summary already ran before the upcoming request."""
    shared = getattr(state, "shared", None)
    if not isinstance(shared, dict) or SUMMARIZED_AT_KEY not in shared:
        return False
    return shared.get(SUMMARIZED_AT_KEY) == _int(shared.get(API_CALLS_KEY))


def window_of(state: Any) -> int:
    return _int(getattr(state, "context_window_budget", 0)) or 200_000


#: ``state.metadata`` key: the output cap the client actually sends this turn (thinking raises it above
#: ``max_tokens``; :func:`llm_client.thinking.output_tokens_sent`). Set by the host.
ANSWER_ROOM_KEY = "answer_room_tokens"


def answer_room_of(state: Any) -> int:
    """Output tokens this turn's requests ask for: ``max_tokens``, or more when thinking raises it."""
    base = _int(getattr(state, "max_tokens", 0))
    meta = getattr(state, "metadata", None)
    sent = _int(meta.get(ANSWER_ROOM_KEY)) if isinstance(meta, dict) else 0
    return max(base, sent)


def output_reserve(window: int, max_tokens: int) -> int:
    """Room kept free for the answer: ``max(4096, max_tokens + 2048)``, at most half the window.

    The Stage 4 ``TokenBudgetGuard`` the host wires (``host.runner``) reserves exactly this, so the
    proactive trigger (80% of what is left) always comes before the guard. Before 4.89.0 the
    trigger was 80% of the whole window while the guard fired at ``window - max_tokens - 2048``:
    on a 32k model with ``max_tokens=8192`` that is 68% of the window, so every compaction was a
    guard emergency.
    """
    window = max(0, int(window or 0))
    reserve = max(4096, max(0, int(max_tokens or 0)) + 2048)
    if window > 0:
        # Never more than half the window, so a max_tokens set close to the window (a config error)
        # still leaves room for the input instead of failing every turn.
        reserve = min(reserve, window // 2)
    return reserve


def input_budget_of(state: Any) -> int:
    """Tokens the next request may use: the window minus :func:`output_reserve`."""
    window = window_of(state)
    return max(1, window - output_reserve(window, answer_room_of(state)))


def fixed_tokens(state: Any) -> int:
    """What no compaction can remove from the next request: the system prompt, tool schemas and turn
    context, and the attachments of the request being answered (a checkpoint carries those too)."""
    from xgen_agent_runtime.core.context_prune import ATTACHMENT_TYPES, current_request_span
    from xgen_agent_runtime.core.token_estimate import (
        estimate_messages_scaled,
        estimate_overhead_tokens,
    )

    messages = state.messages or []
    blocks = [
        b
        for i in current_request_span(messages)
        if isinstance(messages[i].get("content"), list)
        for b in messages[i]["content"]
        if isinstance(b, dict) and b.get("type") in ATTACHMENT_TYPES
    ]
    attached = (
        estimate_messages_scaled(state, [{"role": "user", "content": blocks}]) if blocks else 0
    )
    return estimate_overhead_tokens(state) + attached


def soft_level(state: Any, ratio: float) -> int:
    """``ratio`` of the room compaction can actually work with: the fixed part (:func:`fixed_tokens`)
    plus ``ratio`` of what the input budget leaves after it.

    The proactive trigger is ``soft_level(PROACTIVE_RATIO)`` and the target
    ``soft_level(TARGET_RATIO)``. Measured on the whole budget, a large attached PDF kept every
    request above the trigger right after a summary, so the turn summarized on every request
    (2026-10-12 review: 6 summaries on 6 requests). With a small fixed part this is about the plain
    share of the budget.
    """
    budget = input_budget_of(state)
    fixed = min(budget, fixed_tokens(state))
    return fixed + int((budget - fixed) * ratio)


def _compactor_name(compactor: Any) -> str:
    return str(getattr(compactor, "name", None) or type(compactor).__name__)


def _summary_text(state: PipelineState) -> str:
    """Best-effort extraction of the summary a compactor placed at the head."""
    msgs = state.messages or []
    if not msgs:
        return ""
    head = msgs[0]
    if not isinstance(head, dict):
        return ""
    content = head.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _prune(state: PipelineState, *, aggressive: bool, trigger: str) -> Dict[str, int]:
    """Deterministic pre-pass. In-place, count/order-preserving, pair-safe.

    Attached documents (PDFs) are removed only under hard pressure: the request may still be about
    them (2026-10-12 review: a 12-page PDF alone passed the cost threshold and was stripped from the
    request being answered).
    """
    try:
        from xgen_agent_runtime.core.context_prune import AGGRESSIVE, prune_messages

        from xgen_agent_runtime.core.originals import keep_original

        options: Dict[str, Any] = dict(AGGRESSIVE) if aggressive else {}
        options["strip_documents"] = aggressive
        metrics = prune_messages(
            state.messages or [], before_change=lambda m: keep_original(state, m), **options
        )
    except Exception:  # noqa: BLE001 (relief, never a gate)
        logger.debug("deterministic prune pass failed", exc_info=True)
        return {}
    if any(metrics.get(k) for k in ("deduped", "images_stripped", "trimmed", "inputs_trimmed")):
        # The memo keys on list length + head/tail ids, which the prune preserves.
        state.shared.pop("_prompt_tokens_memo", None)
        state.add_event("context.pruned", dict(metrics, aggressive=aggressive, trigger=trigger))
    return metrics


#: ``state.shared`` slot: wall-clock time until which a background summary scheduled by Stage 2
#: (hosts that keep a state across turns) counts as running. A plain number, so the state stays
#: copyable and picklable; Stage 2 clears it when it collects or cancels the summary, and it expires
#: on its own if the stage never comes back.
BACKGROUND_UNTIL_KEY = "_context_background_until"
#: How long a background summary may count as running at most.
BACKGROUND_MAX_SECONDS = 300.0


def background_summary_running(state: Any) -> bool:
    """True while a background summary scheduled by Stage 2 has not been collected."""
    import time

    shared = getattr(state, "shared", None)
    until = shared.get(BACKGROUND_UNTIL_KEY) if isinstance(shared, dict) else None
    return isinstance(until, (int, float)) and until > time.time()


def _plan_reason(
    state: PipelineState, compactor: Any, target_tokens: Optional[int]
) -> Optional[str]:
    """Why the compactor's own plan is not worth a model call, or ``None``."""
    plan = getattr(compactor, "plan", None)
    if not callable(plan):
        return None
    try:
        outline = plan(state, target_tokens)
    except Exception:  # noqa: BLE001 (a planning bug must not stop compaction)
        logger.debug("compactor plan failed", exc_info=True)
        return None
    if outline is None:
        return "nothing_to_summarize"
    region = _int(getattr(outline, "region_tokens", 0))
    if region < min_summary_region(window_of(state)):
        return "summary_region_too_small"
    return None


def _llm_gate(
    state: PipelineState, compactor: Any, *, trigger: str, target_tokens: Optional[int], hard: bool
) -> Optional[str]:
    """Why a model-backed compactor must NOT run now (``None`` = go ahead).

    Only a person asking (``/compact``) bypasses the policy. A Stage 16 "requested" compaction is
    the loop's own pressure signal and obeys the same limits (it used to bypass them all and could
    summarize on every iteration of a turn that compaction could not bring under the threshold).
    Under hard pressure the limits do not stop relief (the runner falls back to the digest), but a
    region too small to matter is still not summarized: summarizing one question made the request
    bigger (2026-10-12 review).
    """
    if not getattr(compactor, "uses_llm", False) or trigger == "manual":
        return None
    shared = state.shared
    if not hard:
        if _int(shared.get(LLM_COMPACTIONS_KEY)) >= MAX_LLM_COMPACTIONS_PER_TURN:
            return "turn_compaction_limit"
        if _int(shared.get(LLM_FAILURES_KEY)) >= MAX_SUMMARY_FAILURES:
            return "summary_model_failing"
        if summarized_for_this_call(state):
            return "already_summarized_for_this_request"
        if background_summary_running(state):
            # The summary of this history is already being made; a synchronous one now would be
            # thrown away with it (the prefix check) and cost a call.
            return "background_summary_running"
    reason = _plan_reason(state, compactor, target_tokens)
    if reason is not None:
        return reason
    if hard and _int(shared.get(LLM_COMPACTIONS_KEY)) >= MAX_LLM_COMPACTIONS_PER_TURN:
        return "turn_compaction_limit"
    if hard and _int(shared.get(LLM_FAILURES_KEY)) >= MAX_SUMMARY_FAILURES:
        return "summary_model_failing"
    return None


async def run_compaction(
    state: PipelineState,
    compactor: Any,
    *,
    trigger: str,
    provider: Optional[Any] = None,
    target_tokens: Optional[int] = None,
    hard: bool = False,
) -> dict:
    """Relieve context pressure with ``compactor`` and return a result summary dict.

    ``target_tokens``: where the relief should land. With a target, a deterministic prune that
    already reaches it ends the call (no model summary); ``None`` means the caller has decided
    compaction is needed, so the compactor always runs (aiming at :data:`TARGET_RATIO` of the
    input budget). A ``manual`` trigger (``/compact``) always runs the compactor too.

    ``hard`` marks pressure the request cannot survive (guard, provider overflow). The compactor
    runs first, on the history as it is (the summarizer reads tool results before they are cut);
    a compactor that cannot reach its model falls back to a deterministic digest. Only if the
    request is still over the input budget does the last-resort relief run, cheapest loss first:
    the aggressive prune (older images and documents, stale results cut short), then the remaining
    tool results shortened in proportion, then the attachments of the request being answered.

    Emits ``context.compacted`` (and ``context.compaction_skipped`` when nothing changed or the
    policy declined) and, when a provider exposes ``record_compaction`` and the compactor does not
    persist its own snapshot, records the snapshot. Never raises: compaction is best-effort relief,
    not a correctness gate; failures are logged as events and swallowed.
    """
    before_list = list(state.messages or [])
    before_msgs = len(before_list)
    before_tokens = estimate_prompt_tokens(state)
    prune_may_suffice = target_tokens is not None and trigger != "manual"
    if target_tokens is None:
        target_tokens = soft_level(state, TARGET_RATIO)

    _prune(state, aggressive=False, trigger=trigger)
    after_prune = estimate_prompt_tokens(state)
    if prune_may_suffice and after_prune <= target_tokens:
        return _report_prune(state, trigger, before_msgs, before_tokens, after_prune, target_tokens)

    skip = _llm_gate(state, compactor, trigger=trigger, target_tokens=target_tokens, hard=hard)
    digest_only = False
    if hard and skip in ("turn_compaction_limit", "summary_model_failing"):
        # Out of model summaries for this turn (or the summary model keeps failing), yet the request
        # cannot go out as it is: the compactor relieves it with its deterministic digest.
        skip, digest_only = None, True
    if skip is not None:
        state.add_event(
            "context.compaction_skipped",
            {
                "strategy": _compactor_name(compactor),
                "trigger": trigger,
                "reason": skip,
                "tokens_estimate": after_prune,
                "target_tokens": target_tokens,
            },
        )
        if not hard:
            return {
                "ok": True,
                "skipped": skip,
                "before_messages": before_msgs,
                "after_messages": len(state.messages or []),
                "replaced": 0,
                "saved_tokens_estimate": max(0, before_tokens - after_prune),
                "after_tokens_estimate": after_prune,
                "target_tokens": target_tokens,
                "target_met": after_prune <= target_tokens,
                "summarized": False,
            }

    changed = False
    after_list = before_list
    if skip is None:
        request: Dict[str, Any] = {
            "trigger": trigger,
            "target_tokens": target_tokens,
            "hard": bool(hard),
            "digest_only": digest_only,
        }
        state.shared[COMPACTION_CONTEXT_KEY] = request
        try:
            await compactor.compact(state)
        except Exception as exc:  # noqa: BLE001 (best effort)
            _count_attempt(state, compactor, request)
            state.add_event(
                "context.compaction_failed",
                {"compactor": _compactor_name(compactor), "trigger": trigger, "error": str(exc)},
            )
            logger.warning("Compaction (%s) failed: %s", trigger, exc)
            if hard:
                _relieve_hard(state, trigger)
            now = estimate_prompt_tokens(state)
            # The prune and the hard relief may still have made the request fit: report that, so the
            # guard re-checks and Stage 6 retries instead of failing a request that now fits.
            return {
                "ok": now < before_tokens,
                "before_messages": before_msgs,
                "after_messages": len(state.messages or []),
                "replaced": 0,
                "saved_tokens_estimate": max(0, before_tokens - now),
                "after_tokens_estimate": now,
                "target_tokens": target_tokens,
                "target_met": now <= target_tokens,
                "summarized": False,
            }
        finally:
            state.shared.pop(COMPACTION_CONTEXT_KEY, None)
        _count_attempt(state, compactor, request)

        after_list = list(state.messages or [])
        changed = len(after_list) != before_msgs or any(
            a is not b for a, b in zip(after_list, before_list)
        )
        if changed:
            # Keep every recorder's watermark valid across the shrink (audit D3) and remember what
            # they have not recorded yet.
            reconcile_recorded_index(before_list, after_list, state.metadata)
            state.shared.pop("_prompt_tokens_memo", None)

    if hard:
        _relieve_hard(state, trigger)

    if not changed:
        now = estimate_prompt_tokens(state)
        if now < before_tokens:
            # The compactor left the list alone but the prune (or the hard relief) shrank it in
            # place: report that relief.
            return _report_prune(state, trigger, before_msgs, before_tokens, now, target_tokens)
        if skip is None:
            # Nothing changed (a soft summary that did not come back, nothing to cut): not a
            # compaction. Reporting it as ``context.compacted`` made hosts over-count compactions.
            state.add_event(
                "context.compaction_skipped",
                {
                    "strategy": _compactor_name(compactor),
                    "trigger": trigger,
                    "reason": "no_change",
                    "tokens_estimate": now,
                    "target_tokens": target_tokens,
                },
            )
        return {
            "ok": True,
            "skipped": skip or "no_change",
            "before_messages": before_msgs,
            "after_messages": before_msgs,
            "replaced": 0,
            "saved_tokens_estimate": max(0, before_tokens - now),
            "after_tokens_estimate": now,
            "target_tokens": target_tokens,
            "target_met": now <= target_tokens,
            "summarized": False,
        }

    after_list = list(state.messages or [])
    after_msgs = len(after_list)
    after_tokens = estimate_prompt_tokens(state)
    replaced = max(0, before_msgs - after_msgs)
    saved_tokens = max(0, before_tokens - after_tokens)

    target_met = after_tokens <= target_tokens
    state.add_event(
        "context.compacted",
        {
            "strategy": _compactor_name(compactor),
            "trigger": trigger,
            "messages_before": before_msgs,
            "messages_after": after_msgs,
            "saved_tokens_estimate": saved_tokens,
            "tokens_after_estimate": after_tokens,
            "target_tokens": target_tokens,
            "target_met": target_met,
        },
    )
    if not target_met:
        state.add_event(
            "context.compaction_target_missed",
            {
                "strategy": _compactor_name(compactor),
                "trigger": trigger,
                "tokens_after_estimate": after_tokens,
                "target_tokens": target_tokens,
            },
        )

    # Persist the snapshot unless the compactor already does it itself.
    if (
        replaced > 0
        and provider is not None
        and not getattr(compactor, "persists_own_compaction", False)
        and hasattr(provider, "record_compaction")
    ):
        try:
            await provider.record_compaction(
                _summary_text(state),
                replaced_count=replaced,
                strategy=_compactor_name(compactor),
                saved_tokens=saved_tokens,
                session_id=getattr(state, "session_id", "") or "",
                trigger=trigger,
            )
        except Exception as exc:  # noqa: BLE001 (best effort)
            state.add_event(
                "context.compaction_record_failed",
                {"compactor": _compactor_name(compactor), "error": str(exc)},
            )
            logger.debug("record_compaction failed: %s", exc)

    return {
        "ok": True,
        "before_messages": before_msgs,
        "after_messages": after_msgs,
        "replaced": replaced,
        "saved_tokens_estimate": saved_tokens,
        "after_tokens_estimate": after_tokens,
        "target_tokens": target_tokens,
        "target_met": target_met,
        "summarized": True,
    }


def _report_prune(
    state: PipelineState,
    trigger: str,
    before_msgs: int,
    before_tokens: int,
    now: int,
    target_tokens: int,
) -> dict:
    """``context.compacted`` (strategy ``prune``) for relief made in place, without the compactor."""
    state.add_event(
        "context.compacted",
        {
            "strategy": "prune",
            "trigger": trigger,
            "messages_before": before_msgs,
            "messages_after": len(state.messages or []),
            "saved_tokens_estimate": max(0, before_tokens - now),
            "tokens_after_estimate": now,
            "target_tokens": target_tokens,
            "target_met": now <= target_tokens,
        },
    )
    return {
        "ok": True,
        "before_messages": before_msgs,
        "after_messages": len(state.messages or []),
        "replaced": 0,
        "saved_tokens_estimate": max(0, before_tokens - now),
        "after_tokens_estimate": now,
        "target_tokens": target_tokens,
        "target_met": now <= target_tokens,
        "summarized": False,
    }


#: Under hard pressure the request must end up below this share of the input budget.
_HARD_FIT_RATIO = 0.95


def _relieve_hard(state: PipelineState, trigger: str) -> None:
    """Last-resort relief under hard pressure, cheapest loss first, each step only while the
    request is still over the input budget: the aggressive prune, then the remaining tool results
    in proportion (:func:`_fit_tool_results`), then the attachments of the request being answered."""
    limit = int(input_budget_of(state) * _HARD_FIT_RATIO)
    if estimate_prompt_tokens(state) <= limit:
        return
    _prune(state, aggressive=True, trigger=trigger)
    if estimate_prompt_tokens(state) <= limit:
        return
    _fit_tool_results(state, trigger, limit)
    estimate = estimate_prompt_tokens(state)
    if estimate <= limit:
        return
    from xgen_agent_runtime.core.context_prune import strip_current_attachments

    # The request's own attachments, one at a time and only if removing them can close the gap: a
    # request over by a few hundred tokens lost every image, and one that could not fit anyway lost
    # them before the guard rejected it.
    attached = fixed_tokens(state) - _overhead_only(state)
    if attached <= 0 or estimate - attached > limit:
        return
    removed = 0
    from xgen_agent_runtime.core.originals import keep_original

    while estimate > limit and strip_current_attachments(
        state.messages or [], limit=1, before_change=lambda m: keep_original(state, m)
    ):
        removed += 1
        state.shared.pop("_prompt_tokens_memo", None)
        estimate = estimate_prompt_tokens(state)
    if removed:
        state.add_event(
            "context.pruned",
            {
                "trigger": trigger,
                "aggressive": True,
                "deduped": 0,
                "images_stripped": removed,
                "trimmed": 0,
                "inputs_trimmed": 0,
                "chars_saved": 0,
                "current_attachments_stripped": removed,
            },
        )


def _overhead_only(state: PipelineState) -> int:
    from xgen_agent_runtime.core.token_estimate import estimate_overhead_tokens

    return estimate_overhead_tokens(state)


def _fit_tool_results(state: PipelineState, trigger: str, limit: int) -> int:
    """Shorten the tool results that are left (the newest ones, in the protected tail) so the
    request fits ``limit``: images inside results first, then text in proportion. One round of
    parallel tool calls can return more than the window even when each result is within its own
    cap; before this the guard rejected the turn. Does nothing when even cutting every result to a
    stub could not close the gap (the guard then rejects with the results intact). Returns results
    changed.
    """
    from xgen_agent_runtime.core.context_prune import shrink_tool_results
    from xgen_agent_runtime.core.token_estimate import estimate_messages_scaled

    estimate = estimate_prompt_tokens(state)
    if estimate <= limit:
        return 0

    def _results() -> List[Any]:
        return [
            block
            for message in state.messages or []
            if isinstance(message, dict) and isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]

    result_tokens = estimate_messages_scaled(state, [{"role": "user", "content": _results()}])
    if result_tokens <= 0 or result_tokens * 0.95 < estimate - limit:
        return 0
    totals = {"trimmed": 0, "chars_saved": 0, "images_stripped": 0}
    from xgen_agent_runtime.core.originals import keep_original

    def _keep(message: Any) -> None:
        keep_original(state, message)

    metrics = shrink_tool_results(state.messages or [], 1.0, strip_images=True, before_change=_keep)
    for key in totals:
        totals[key] += metrics.get(key, 0)
    if metrics["images_stripped"]:
        state.shared.pop("_prompt_tokens_memo", None)
    # Text in proportion, re-measured: the cut is by characters, and a result whose head is denser
    # than its tail (a Korean body with an ASCII footer) needs another pass.
    for _ in range(3):
        estimate_mid = estimate_prompt_tokens(state)
        if estimate_mid <= limit:
            break
        result_tokens = estimate_messages_scaled(state, [{"role": "user", "content": _results()}])
        if result_tokens <= 0:
            break
        keep = max(0.02, (result_tokens - (estimate_mid - limit)) / result_tokens * 0.9)
        if keep >= 1.0:
            break
        metrics = shrink_tool_results(state.messages or [], keep, before_change=_keep)
        for key in totals:
            totals[key] += metrics.get(key, 0)
        state.shared.pop("_prompt_tokens_memo", None)
        if not metrics.get("trimmed"):
            break
    changed = totals["trimmed"] + totals["images_stripped"]
    if changed:
        state.shared.pop("_prompt_tokens_memo", None)
        state.add_event(
            "context.pruned",
            {
                "trigger": trigger,
                "aggressive": True,
                "deduped": 0,
                "images_stripped": totals["images_stripped"],
                "trimmed": totals["trimmed"],
                "inputs_trimmed": 0,
                "chars_saved": totals["chars_saved"],
                "results_fitted": totals["trimmed"],
                "tokens_before": estimate,
                "tokens_after": estimate_prompt_tokens(state),
            },
        )
    return changed


def _count_attempt(state: PipelineState, compactor: Any, request: Dict[str, Any]) -> None:
    """Book a compactor run against the per-request and per-turn limits.

    A model-backed compactor reports the summary calls it made (``model_calls``) and what replaced
    the history (``source``: ``"model"``, ``"digest"``, or ``"unavailable"`` when no model is wired).
    Any call (successful or not) uses up this request's one soft summary and one of the turn's
    attempts; a call that did not produce a model summary, or no model at all, counts as a failure.
    Before this, only a run that changed the history was counted, so a failing summary model was
    retried by Stage 2 and Stage 4 on every request (2026-10-12 review: 20 failed calls in a turn).
    """
    if not getattr(compactor, "uses_llm", False):
        return
    shared = state.shared
    calls = _int(request.get("model_calls"))
    source = request.get("source")
    if calls <= 0 and source is None:
        return
    shared[SUMMARIZED_AT_KEY] = _int(shared.get(API_CALLS_KEY))
    if calls > 0:
        shared[LLM_COMPACTIONS_KEY] = _int(shared.get(LLM_COMPACTIONS_KEY)) + 1
    if source == "model":
        shared[LLM_FAILURES_KEY] = 0
    elif calls > 0 or source == "unavailable":
        shared[LLM_FAILURES_KEY] = _int(shared.get(LLM_FAILURES_KEY)) + 1


async def relieve_if_over(
    state: PipelineState,
    compactor: Any,
    *,
    trigger: str,
    provider: Optional[Any] = None,
    ratio: float = PROACTIVE_RATIO,
) -> Optional[dict]:
    """Proactive check shared by Stage 2, Stage 4 and the internal tool loop: compact when the
    estimate crosses :func:`soft_level` at ``ratio``, down to the target level."""
    if compactor is None:
        return None
    if estimate_prompt_tokens(state) <= soft_level(state, ratio):
        return None
    return await run_compaction(
        state,
        compactor,
        trigger=trigger,
        provider=provider,
        target_tokens=soft_level(state, TARGET_RATIO),
    )


__all__ = [
    "PROACTIVE_RATIO",
    "TARGET_RATIO",
    "MAX_LLM_COMPACTIONS_PER_TURN",
    "MAX_SUMMARY_FAILURES",
    "LLM_FAILURES_KEY",
    "output_reserve",
    "input_budget_of",
    "BACKGROUND_UNTIL_KEY",
    "background_summary_running",
    "fixed_tokens",
    "soft_level",
    "RECORDED_INDEX_KEYS",
    "PENDING_RECORDS_KEY",
    "RECORDER_STM",
    "RECORDER_ARCHIVE",
    "PENDING_RECORDS_CAP",
    "min_summary_region",
    "reconcile_recorded_index",
    "take_pending_records",
    "peek_pending_records",
    "note_api_call",
    "summarized_for_this_call",
    "run_compaction",
    "relieve_if_over",
    "window_of",
]
