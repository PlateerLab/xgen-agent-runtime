"""CLI subscription limits (4.90.0): keep the signal, classify the exhaustion.

Two things changed and both are pinned here.

1. ``rate_limit_event`` used to be dropped. It is the only per-account view of a
   Claude subscription the host can get (the usage API does not accept a
   setup-token), so the accumulator now keeps the last value per window and
   hands it to the host on ``APIResponse.raw["rate_limits"]``.
2. A hit limit used to fall through to CLI_PROTOCOL_ERROR, which Stage 6 retried
   three times on the same account, and the Claude notice streamed out as the
   assistant's answer. It is now CLI_RATE_LIMITED (not retried) and carries
   ``details["reset_at"]`` so the host can rest that account and use another.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.core.errors import APIError, ErrorCategory
from xgen_agent_runtime.llm_client._cli_limits import (
    is_usage_limit_text,
    reset_at_from_text,
)
from xgen_agent_runtime.llm_client.translators._cli import (
    StreamJsonAccumulator,
    limit_error_from_line,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
FAKE_CLAUDE = str(Path(__file__).resolve().parents[2] / "_fixtures" / "fake_claude.py")


def _feed_golden(name: str) -> StreamJsonAccumulator:
    accum = StreamJsonAccumulator(model="sonnet")
    for raw in (GOLDEN / name).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            accum.feed(json.loads(raw))
    return accum


def test_golden_rate_limit_events_are_kept_per_window() -> None:
    warning = _feed_golden("cli-2.1.149-stream.jsonl")
    five = warning.rate_limits["five_hour"]
    assert five["status"] == "allowed_warning"
    assert five["utilization"] == pytest.approx(0.91)
    assert five["resets_at"] == 1781060400
    assert warning.finalize().raw["rate_limits"]["five_hour"]["status"] == "allowed_warning"

    normal = _feed_golden("cli-2.1.162-stream.jsonl")
    # Below the warning threshold the CLI sends no utilization: say so, do not invent 0.
    assert normal.rate_limits["five_hour"]["utilization"] is None
    assert normal.rate_limits["five_hour"]["status"] == "allowed"


def test_rejected_window_gives_the_reset_time() -> None:
    accum = StreamJsonAccumulator(model="sonnet")
    accum.feed({
        "type": "rate_limit_event",
        "rate_limit_info": {"status": "rejected", "resetsAt": 1781000400, "rateLimitType": "seven_day"},
    })
    err = limit_error_from_line(
        {"type": "assistant", "error": "rate_limit",
         "message": {"content": [{"type": "text", "text": "You've hit your limit"}]}},
        accum.rate_limits,
    )
    assert err.category is ErrorCategory.CLI_RATE_LIMITED
    assert err.details["reset_at"] == 1781000400
    assert err.details["rate_limits"]["seven_day"]["status"] == "rejected"


def test_limit_texts_and_their_reset_times() -> None:
    now = datetime(2026, 10, 13, 10, 0, tzinfo=ZoneInfo("Asia/Seoul")).timestamp()
    # Old Claude form: epoch after a bar.
    assert reset_at_from_text("Claude AI usage limit reached|1781000400", now=now) == 1781000400
    # Current Claude form: local clock time with its zone, later today.
    three_pm = datetime(2026, 10, 13, 15, 0, tzinfo=ZoneInfo("Asia/Seoul")).timestamp()
    assert reset_at_from_text("You've hit your limit · resets 3pm (Asia/Seoul)", now=now) == three_pm
    # Already past today: tomorrow.
    nine_am = datetime(2026, 10, 14, 9, 0, tzinfo=ZoneInfo("Asia/Seoul")).timestamp()
    assert reset_at_from_text("resets 9am (Asia/Seoul)", now=now) == nine_am
    # Codex: relative duration.
    assert reset_at_from_text(
        "You've hit your usage limit. Try again in 2 days 3 hours 5 minutes.", now=now
    ) == now + 2 * 86400 + 3 * 3600 + 5 * 60
    assert reset_at_from_text('{"resets_in_seconds": 120}', now=now) == now + 120
    # Unknown zone or no time: unknown, not guessed.
    assert reset_at_from_text("resets 3pm (Mars/Olympus)", now=now) is None
    assert reset_at_from_text("usage limit reached", now=now) is None


def test_strict_mode_ignores_loose_phrases_from_stderr_noise() -> None:
    noisy = "mcp connector: log shipper hit a rate limit, retrying"
    assert is_usage_limit_text(noisy)
    assert not is_usage_limit_text(noisy, strict=True)
    assert is_usage_limit_text("stream error: exceeded retry limit, last status: 429 Too Many Requests", strict=True)
    assert is_usage_limit_text('API Error: {"type":"error","error":{"type":"rate_limit_error"}}', strict=True)
    # A token count is not a status code.
    assert not is_usage_limit_text("input 429 tokens, output 12", strict=True)


def test_rate_limited_is_not_retried_on_the_same_account() -> None:
    from xgen_agent_runtime.stages.s06_api.artifact.default.retry import ExponentialBackoffRetry

    assert not ErrorCategory.CLI_RATE_LIMITED.is_recoverable
    assert not ExponentialBackoffRetry().should_retry(ErrorCategory.CLI_RATE_LIMITED, 0)


@pytest.mark.asyncio
async def test_claude_stream_raises_rate_limited_and_does_not_stream_the_notice() -> None:
    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient

    if not os.access(FAKE_CLAUDE, os.X_OK):
        os.chmod(FAKE_CLAUDE, 0o755)
    client = ClaudeCodeCLIClient(
        binary_path=FAKE_CLAUDE,
        workspace_dir=os.getcwd(),
        api_key="sk-fake",
        bare_mode=True,
        timeout_s=10.0,
        env_extras={"FAKE_CLAUDE_SCENARIO": "message_form_rate_limited"},
    )
    events = []
    with pytest.raises(APIError) as got:
        async for evt in client.create_message_stream(
            model_config=ModelConfig(model="sonnet"),
            messages=[{"role": "user", "content": "hi"}],
        ):
            events.append(evt)
    assert got.value.category is ErrorCategory.CLI_RATE_LIMITED
    assert got.value.details["reset_at"] == 1781000400
    assert not [e for e in events if e.get("type") == "text_delta"]


def test_codex_reported_limit_is_rate_limited() -> None:
    from xgen_agent_runtime.llm_client.codex import CodexCLIClient

    client = CodexCLIClient.__new__(CodexCLIClient)
    client._cli_version_value = "0.162.1"
    err = client._reported_error("You've hit your usage limit. Try again in 3 hours.")
    assert err.category is ErrorCategory.CLI_RATE_LIMITED
    assert err.details["reset_at"] is not None
    other = client._reported_error("model metadata not found")
    assert other.category is ErrorCategory.CLI_PROTOCOL_ERROR
