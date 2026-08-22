"""
Process-wide cancellation registry for streaming executions.

Used to cooperatively stop background agent execution when the client disconnects (SSE stop).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

_lock = threading.Lock()
_cancelled_until: dict[str, float] = {}

# Safety: in case we only have interaction_id (no io_id yet), don't poison future runs forever.
_DEFAULT_TTL_SECONDS = 60.0


def _key(interaction_id: Optional[str], response_io_id: Optional[int] = None) -> Optional[str]:
    if not interaction_id:
        return None
    if response_io_id is None:
        return str(interaction_id)
    return f"{interaction_id}:{response_io_id}"


def _purge_expired(now: float) -> None:
    expired = [k for k, until in _cancelled_until.items() if until <= now]
    for k in expired:
        _cancelled_until.pop(k, None)


def request_cancel(interaction_id: Optional[str], response_io_id: Optional[int] = None, *, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
    """
    Mark execution as cancelled for a short TTL.
    If response_io_id is available, only the scoped key is set (future messages won't be affected).
    """
    key = _key(interaction_id, response_io_id)
    if not key:
        return
    now = time.monotonic()
    until = now + max(1.0, float(ttl_seconds))
    with _lock:
        _purge_expired(now)
        _cancelled_until[key] = until


def is_cancelled(interaction_id: Optional[str], response_io_id: Optional[int] = None) -> bool:
    """
    Check if execution is cancelled.
    If response_io_id is present, checks scoped key first, then falls back to interaction key.
    """
    now = time.monotonic()
    with _lock:
        _purge_expired(now)
        scoped = _key(interaction_id, response_io_id)
        if scoped and scoped in _cancelled_until:
            return True
        base = _key(interaction_id, None)
        return bool(base and base in _cancelled_until)


def clear_cancel(interaction_id: Optional[str], response_io_id: Optional[int] = None) -> None:
    """
    Clear cancellation flags. If response_io_id is provided, clears only scoped key.
    """
    with _lock:
        scoped = _key(interaction_id, response_io_id)
        if scoped:
            _cancelled_until.pop(scoped, None)
        elif interaction_id:
            base = _key(interaction_id, None)
            if base:
                _cancelled_until.pop(base, None)
