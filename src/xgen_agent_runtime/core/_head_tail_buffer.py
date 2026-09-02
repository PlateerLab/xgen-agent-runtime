"""Bounded byte buffer that preserves the start and end of output.

Long command output usually carries setup context at the beginning and the
decisive error or summary at the end.  A prefix-only slice loses the latter;
an unbounded collector risks exhausting the host.  This request-internal
primitive keeps both under a fixed retained-byte budget.
"""

from __future__ import annotations

from typing import Union


BytesLike = Union[bytes, bytearray, memoryview]


class HeadTailBuffer:
    """Collect bytes with symmetric, bounded head/tail retention."""

    __slots__ = ("_head", "_head_budget", "_max_bytes", "_omitted_bytes", "_tail")

    def __init__(self, max_bytes: int) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        self._max_bytes = max_bytes
        self._head_budget = max_bytes // 2
        self._head = bytearray()
        self._tail = bytearray()
        self._omitted_bytes = 0

    @property
    def retained_bytes(self) -> int:
        return len(self._head) + len(self._tail)

    @property
    def omitted_bytes(self) -> int:
        return self._omitted_bytes

    @property
    def total_bytes(self) -> int:
        return self.retained_bytes + self.omitted_bytes

    @property
    def tail_budget(self) -> int:
        return self._max_bytes - self._head_budget

    def push_chunk(self, chunk: BytesLike) -> None:
        """Append a chunk while retaining at most ``max_bytes`` bytes."""

        incoming = bytes(chunk)
        if not incoming:
            return

        head_room = self._head_budget - len(self._head)
        if head_room > 0:
            head_part = incoming[:head_room]
            self._head.extend(head_part)
            incoming = incoming[len(head_part) :]
        self._push_tail(incoming)

    def _push_tail(self, chunk: bytes) -> None:
        if not chunk:
            return
        budget = self.tail_budget
        if budget == 0:
            self._omitted_bytes += len(chunk)
            return

        remaining = budget - len(self._tail)
        excess = max(0, len(chunk) - remaining)
        self._omitted_bytes += excess

        if excess <= len(self._tail):
            if excess:
                del self._tail[:excess]
            self._tail.extend(chunk)
            return

        skip = excess - len(self._tail)
        self._tail.clear()
        self._tail.extend(chunk[skip:])

    def to_bytes(self) -> bytes:
        """Return retained bytes without an omission marker."""

        return bytes(self._head + self._tail)

    def to_bytes_with_omission_marker(self) -> bytes:
        """Return retained output with an explicit middle marker."""

        if self.omitted_bytes == 0:
            return self.to_bytes()
        marker = f"... {self.omitted_bytes} bytes omitted ...".encode()
        return bytes(self._head) + b"\n" + marker + b"\n" + bytes(self._tail)

