"""Bounded head/tail output retention adapted from the Codex harness."""

from __future__ import annotations

import pytest

from xgen_agent_runtime.core._head_tail_buffer import HeadTailBuffer


def test_keeps_prefix_and_suffix_when_over_budget() -> None:
    buffer = HeadTailBuffer(10)

    buffer.push_chunk(b"0123456789")
    buffer.push_chunk(b"ab")

    assert buffer.retained_bytes == 10
    assert buffer.omitted_bytes == 2
    assert buffer.total_bytes == 12
    assert buffer.to_bytes() == b"01234789ab"
    assert buffer.to_bytes_with_omission_marker() == (
        b"01234\n... 2 bytes omitted ...\n789ab"
    )


def test_fills_head_and_tail_across_multiple_chunks() -> None:
    buffer = HeadTailBuffer(10)

    for chunk in (b"01", b"234", b"567", b"89"):
        buffer.push_chunk(chunk)

    assert buffer.to_bytes() == b"0123456789"
    assert buffer.omitted_bytes == 0
    buffer.push_chunk(b"a")
    assert buffer.to_bytes() == b"012346789a"
    assert buffer.omitted_bytes == 1


def test_chunk_larger_than_tail_budget_keeps_its_end() -> None:
    buffer = HeadTailBuffer(10)
    buffer.push_chunk(b"0123456789")

    buffer.push_chunk(b"ABCDEFGHIJK")

    assert buffer.to_bytes() == b"01234GHIJK"
    assert buffer.omitted_bytes == 11


@pytest.mark.parametrize(
    ("budget", "expected", "omitted"),
    [
        (0, b"", 3),
        (1, b"c", 2),
        (2, b"ac", 1),
    ],
)
def test_tiny_budgets_are_bounded(budget: int, expected: bytes, omitted: int) -> None:
    buffer = HeadTailBuffer(budget)
    buffer.push_chunk(b"abc")

    assert buffer.to_bytes() == expected
    assert buffer.retained_bytes <= budget
    assert buffer.omitted_bytes == omitted


def test_empty_chunks_do_not_change_accounting() -> None:
    buffer = HeadTailBuffer(4)
    buffer.push_chunk(b"ab")
    buffer.push_chunk(b"")
    buffer.push_chunk(memoryview(b"cd"))

    assert buffer.to_bytes_with_omission_marker() == b"abcd"
    assert buffer.total_bytes == 4


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "10"])
def test_invalid_budget_is_rejected(invalid: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        HeadTailBuffer(invalid)  # type: ignore[arg-type]
