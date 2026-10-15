"""Stage-18 STM watermark survives compaction (audit D3, 2.51.0).

Compaction shrinks ``state.messages``; the recorded-watermark index must
be remapped so recording doesn't silently stop until the list regrows.
"""

from __future__ import annotations

from xgen_agent_runtime.core.compaction import (
    _STATE_LAST_RECORDED,
    reconcile_recorded_index,
)


def _msgs(n):
    # Distinct dict objects so identity matching is meaningful.
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def test_watermark_remaps_when_boundary_inside_kept_suffix():
    before = _msgs(61)  # indices 0..60
    # LLMSummary-style: 2 synthetic + last 4 real messages (same objects).
    kept = before[-4:]
    after = [{"role": "user", "content": "S1"}, {"role": "assistant", "content": "S2"}] + kept
    meta = {_STATE_LAST_RECORDED: 60}  # msgs 0..59 recorded, msg60 pending

    reconcile_recorded_index(before, after, meta)

    # kept suffix starts at before-index 57; synthetic=2 → new = 2 + (60-57) = 5.
    assert meta[_STATE_LAST_RECORDED] == 5
    # after[5:] == [msg60] → the pending message is still recordable, and
    # the already-recorded msg57/58/59 (after indices 2,3,4) are not re-run.
    assert after[5:] == [before[60]]


def test_watermark_clamps_when_boundary_before_kept():
    before = _msgs(20)
    kept = before[-3:]  # start=17
    after = [{"role": "assistant", "content": "S"}] + kept  # synthetic=1
    meta = {_STATE_LAST_RECORDED: 10}  # boundary well before the kept region

    reconcile_recorded_index(before, after, meta)

    # old_idx(10) <= start(17) → new = n_synthetic = 1; kept suffix all recordable.
    assert meta[_STATE_LAST_RECORDED] == 1
    assert after[1:] == kept


def test_watermark_never_exceeds_new_length():
    before = _msgs(30)
    after = _msgs(3)  # truncate, no shared identity (worst case)
    meta = {_STATE_LAST_RECORDED: 30}
    reconcile_recorded_index(before, after, meta)
    assert 0 <= meta[_STATE_LAST_RECORDED] <= 3


def test_absent_watermark_counts_as_nothing_recorded():
    """4.89.0: no watermark means nothing was recorded yet (a compaction inside the first turn,
    before Stage 18 ran). The summarized messages are kept for the recorders, and the watermark
    moves past the synthetic head so the summary is not recorded as something the user said."""
    from xgen_agent_runtime.core.compaction import PENDING_RECORDS_KEY

    for meta in ({}, {_STATE_LAST_RECORDED: 0}):
        before = _msgs(10)
        kept = before[-2:]
        after = [{"role": "user", "content": "SUMMARY"}] + kept
        reconcile_recorded_index(before, after, meta)
        assert meta[_STATE_LAST_RECORDED] == 1
        assert meta[PENDING_RECORDS_KEY]["stm"] == before[:8]
        assert meta[PENDING_RECORDS_KEY]["archive"] == before[:8]


def test_pending_records_are_capped():
    from xgen_agent_runtime.core.compaction import PENDING_RECORDS_CAP, PENDING_RECORDS_KEY

    meta = {}
    for _ in range(3):
        before = _msgs(PENDING_RECORDS_CAP)
        reconcile_recorded_index(before, [{"role": "user", "content": "S"}] + before[-1:], meta)
        meta[_STATE_LAST_RECORDED] = 0
        meta["geny_bridge.conversation_archived_idx"] = 0
    assert len(meta[PENDING_RECORDS_KEY]["stm"]) == PENDING_RECORDS_CAP


def test_truncate_compactor_keeps_recording_after_shrink():
    """End-to-end: after a real TruncateCompactor shrink, the watermark
    points such that the newest message is still recordable."""
    import asyncio

    from xgen_agent_runtime.core.compaction import run_compaction
    from xgen_agent_runtime.core.state import PipelineState
    from xgen_agent_runtime.stages.s02_context.artifact.default.compactors import (
        TruncateCompactor,
    )

    state = PipelineState()
    state.messages = _msgs(40)
    state.metadata[_STATE_LAST_RECORDED] = 40  # everything recorded so far
    newest = state.messages[-1]

    asyncio.run(run_compaction(state, TruncateCompactor(keep_last=5), trigger="proactive"))

    idx = state.metadata[_STATE_LAST_RECORDED]
    assert 0 <= idx <= len(state.messages)
    # The newest message survived the truncation and must NOT be behind
    # the watermark forever (the pre-2.51 bug left idx=40 > len=5).
    assert newest in state.messages
    assert idx <= len(state.messages)
