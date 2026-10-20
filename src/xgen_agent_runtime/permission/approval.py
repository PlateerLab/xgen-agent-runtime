"""Host-authenticated, final-argument approvals with atomic durable consumption.

The runtime never authenticates an actor. Hosts supply verified identifiers and
call ``decide`` only after authorizing the human answering the request.
"""
from __future__ import annotations

import hashlib
import json
import math
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol


class ApprovalError(RuntimeError):
    def __init__(self, code: str = "approval_unavailable") -> None:
        super().__init__(code)
        self.code = code


def canonical_digest(value: Any, *, max_bytes: int = 1_048_576) -> str:
    """Reject ambiguous/non-JSON values and bound work before hashing."""
    nodes = 0

    def check(item: Any, depth: int = 0) -> None:
        nonlocal nodes
        nodes += 1
        if depth > 32 or nodes > 100_000:
            raise ApprovalError("approval_input_invalid")
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, dict) and all(isinstance(k, str) for k in item):
            for child in item.values():
                check(child, depth + 1)
            return
        if isinstance(item, list):
            for child in item:
                check(child, depth + 1)
            return
        raise ApprovalError("approval_input_invalid")

    check(value)
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError):
        raise ApprovalError("approval_input_invalid") from None
    if len(raw) > max_bytes:
        raise ApprovalError("approval_input_invalid")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ApprovalContext:
    principal_id: str
    workflow_id: str
    run_id: str
    session_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v or len(v) > 256 for v in asdict(self).values()):
            raise ApprovalError("approval_identity_required")


@dataclass(frozen=True)
class ApprovalEnvelope:
    token: str
    context: ApprovalContext
    tool_name: str
    args_digest: str
    expires_at: float

    @classmethod
    def issue(cls, context: ApprovalContext, tool_name: str, tool_input: dict, *, ttl_s: float = 120) -> "ApprovalEnvelope":
        if not isinstance(context, ApprovalContext) or not tool_name:
            raise ApprovalError("approval_identity_required")
        if not math.isfinite(ttl_s) or not 0 < ttl_s <= 300:
            raise ApprovalError("approval_expiry_invalid")
        return cls(secrets.token_urlsafe(32), context, tool_name, canonical_digest(tool_input), time.time() + ttl_s)

    def to_dict(self) -> dict:
        return {"version": 1, **asdict(self)}

    def canonical(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


class ApprovalStore(Protocol):
    def issue(self, envelope: ApprovalEnvelope) -> None: ...
    def decide(self, envelope: ApprovalEnvelope, actor: ApprovalContext, decision: str) -> bool: ...
    def consume(self, envelope: ApprovalEnvelope) -> bool: ...


class RedisApprovalStore:
    """Synchronous Redis port; every decision/consume is a single Lua operation.

    Supply the host's bounded-timeout Redis client. No local-memory fallback is
    allowed when this store is unavailable.
    """

    durable = True

    _TRANSITION = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local row = cjson.decode(raw)
if row.envelope ~= ARGV[1] or row.status ~= ARGV[2] then return 0 end
local now = redis.call('TIME')
if tonumber(row.expires_at) <= tonumber(now[1]) + tonumber(now[2]) / 1000000 then return 0 end
row.status = ARGV[3]
redis.call('SET', KEYS[1], cjson.encode(row), 'KEEPTTL')
return 1
"""

    def __init__(self, client: Any, *, prefix: str = "runtime:approval:") -> None:
        self.client, self.prefix = client, prefix

    def issue(self, envelope: ApprovalEnvelope) -> None:
        ttl = math.ceil(envelope.expires_at - time.time())
        if ttl <= 0 or ttl > 300:
            raise ApprovalError("approval_expired")
        row = json.dumps({"envelope": envelope.canonical(), "status": "pending", "expires_at": envelope.expires_at})
        try:
            if not self.client.set(self.prefix + envelope.token, row, ex=ttl, nx=True):
                raise ApprovalError("approval_replay")
        except ApprovalError:
            raise
        except Exception:
            raise ApprovalError() from None

    def _transition(self, envelope: ApprovalEnvelope, before: str, after: str) -> bool:
        try:
            return bool(self.client.eval(self._TRANSITION, 1, self.prefix + envelope.token, envelope.canonical(), before, after))
        except Exception:
            raise ApprovalError() from None

    def decide(self, envelope: ApprovalEnvelope, actor: ApprovalContext, decision: str) -> bool:
        if actor != envelope.context or decision not in ("approve", "reject", "cancel"):
            return False
        return self._transition(envelope, "pending", "approved" if decision == "approve" else "rejected")

    def consume(self, envelope: ApprovalEnvelope) -> bool:
        return self._transition(envelope, "approved", "consumed")


class MemoryApprovalStore:
    """Explicit test/local-only store. Serving hosts must use a durable port."""

    durable = False

    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def issue(self, envelope: ApprovalEnvelope) -> None:
        with self._lock:
            if envelope.token in self._rows:
                raise ApprovalError("approval_replay")
            self._rows[envelope.token] = (envelope.canonical(), "pending")

    def _transition(self, envelope: ApprovalEnvelope, before: str, after: str) -> bool:
        with self._lock:
            if envelope.expires_at <= time.time() or self._rows.get(envelope.token) != (envelope.canonical(), before):
                return False
            self._rows[envelope.token] = (envelope.canonical(), after)
            return True

    def decide(self, envelope: ApprovalEnvelope, actor: ApprovalContext, decision: str) -> bool:
        return actor == envelope.context and decision in ("approve", "reject", "cancel") and self._transition(
            envelope, "pending", "approved" if decision == "approve" else "rejected",
        )

    def consume(self, envelope: ApprovalEnvelope) -> bool:
        return self._transition(envelope, "approved", "consumed")
