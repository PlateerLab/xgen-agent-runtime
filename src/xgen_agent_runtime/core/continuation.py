"""Explicit input sentinel for continuing a suspended execution slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuationInput:
    """Continue from existing state without appending a synthetic user turn."""


CONTINUE_RUN = ContinuationInput()


__all__ = ["CONTINUE_RUN", "ContinuationInput"]
