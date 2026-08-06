"""Stage 19: Summarize — interface definitions (S9b.4)."""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from xgen_agent_runtime.core.stage import Strategy
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.memory.provider import Importance
from xgen_agent_runtime.stages.s19_summarize.types import SummaryRecord


# state.shared keys.
TURN_SUMMARY_KEY = "turn_summary"
SUMMARY_HISTORY_KEY = "summary_history"


class Summarizer(Strategy):
    """Produce a :class:`SummaryRecord` for the current turn.

    Returning ``None`` (the :class:`NoSummarizer` default) signals
    "skip this turn"; the stage then doesn't publish anything.
    """

    @abstractmethod
    async def summarize(self, state: PipelineState) -> Optional[SummaryRecord]: ...


class ImportanceScorer(Strategy):
    """Assign an :class:`Importance` grade to a freshly-built summary.

    Receives the produced record and the live state so heuristic
    scorers can read tool flags, cost, etc. Returns the grade; the
    stage applies it back onto the record.
    """

    @abstractmethod
    async def score(self, record: SummaryRecord, state: PipelineState) -> Importance: ...


__all__ = [
    "ImportanceScorer",
    "SUMMARY_HISTORY_KEY",
    "Summarizer",
    "TURN_SUMMARY_KEY",
]
