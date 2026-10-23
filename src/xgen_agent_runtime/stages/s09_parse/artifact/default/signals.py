"""Completion signal detection — concrete Level 2 strategies."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from xgen_agent_runtime.stages.s09_parse.interface import CompletionSignal, CompletionSignalDetector


class RegexDetector(CompletionSignalDetector):
    """Regex-based signal detection — the ``[SIGNAL: detail]`` token that **ends** a response.

    The prompt protocol (``memory/presets.py``) tells the model to *end* its response with
    ``[TASK_COMPLETE]``, ``[CONTINUE: next action]`` or ``[BLOCKED: reason]``. Only that
    trailing token, in exact upper case, is a signal.

    Before 2026-10-23 any ``[signal…]`` anywhere in the text matched, case-insensitively: a
    correct answer containing ``[Error codes](…)``, ``[Errors]`` or ``[Blocked users]`` was
    marked failed or blocked, and the non-streaming host replaced the answer with an error.
    ``[ERROR]`` is not part of the protocol at all, so model text never fails a turn — real
    failures (exceptions, provider errors) do.
    """

    # The token must be the last thing in the response (trailing whitespace allowed).
    PATTERNS = {
        CompletionSignal.CONTINUE: re.compile(r"\[CONTINUE(?::[ \t]*([^\]\n]*))?\]\s*\Z"),
        CompletionSignal.COMPLETE: re.compile(r"\[(?:TASK_)?COMPLETE(?::[ \t]*([^\]\n]*))?\]\s*\Z"),
        CompletionSignal.BLOCKED: re.compile(r"\[BLOCKED(?::[ \t]*([^\]\n]*))?\]\s*\Z"),
    }

    @property
    def name(self) -> str:
        return "regex"

    @property
    def description(self) -> str:
        return "Trailing [SIGNAL: detail] token (exact upper case) at the end of the response"

    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        tail = (text or "")[-1000:]
        for signal, pattern in self.PATTERNS.items():
            match = pattern.search(tail)
            if match:
                detail = (match.group(1) or "").strip() or None
                return signal, detail
        return CompletionSignal.NONE, None


class StructuredDetector(CompletionSignalDetector):
    """JSON-based signal detection for structured output."""

    @property
    def name(self) -> str:
        return "structured"

    @property
    def description(self) -> str:
        return "JSON-based structured signal detection"

    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        import json

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Only an explicit ``signal`` key. A ``status`` field is ordinary data in a
                # structured answer ({"status": "error", ...} describing a record) — reading it
                # as a signal turned such answers into failed turns.
                signal_str = data.get("signal")
                detail = data.get("detail", data.get("reason", None))
                if isinstance(signal_str, str) and signal_str.lower() != "error":
                    try:
                        signal = CompletionSignal(signal_str.lower())
                        return signal, detail
                    except ValueError:
                        pass
        except (json.JSONDecodeError, TypeError):
            pass

        return CompletionSignal.NONE, None


class HybridDetector(CompletionSignalDetector):
    """Tries regex first, then structured."""

    def __init__(self):
        self._regex = RegexDetector()
        self._structured = StructuredDetector()

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def description(self) -> str:
        return "Regex + JSON hybrid detection"

    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        signal, detail = self._regex.detect(text)
        if signal != CompletionSignal.NONE:
            return signal, detail
        return self._structured.detect(text)
