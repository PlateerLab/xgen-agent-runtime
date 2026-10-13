"""CLI 구독 한도: 신호 읽기와 한도 소진 판정.

Claude Code·Codex CLI 는 구독 한도가 차면 **같은 계정으로 다시 보내도 소용이 없다.**
그런데 예전에는 한도 소진이 ``CLI_PROTOCOL_ERROR``(재시도 대상)로 떨어져 Stage 6 이
같은 계정으로 세 번을 더 보냈고, 호스트는 다른 계정으로 넘길 근거(초기화 시각)를
받지 못했다. 이 모듈은 그 두 가지를 한 곳에서 정한다.

* :func:`normalize_rate_limit_event`: Claude CLI 가 stream-json 에 내는
  ``rate_limit_event`` 를 창(rateLimitType)별 값으로 바꾼다. 평상시에는
  ``utilization`` 이 없고 경고(90% 넘음)일 때만 실린다(골든 캡처 2.1.149·2.1.162).
* :func:`is_usage_limit_text`: 오류 자리(stderr, error 프레임, is_error 결과)의
  글이 한도 소진인지. 정상 답변 본문에는 쓰지 않는다: 답변이 "rate limit" 을
  설명할 수도 있다.
* :func:`reset_at_from_text`: 문구에서 초기화 시각(유닉스 초)을 뽑는다.
* :func:`usage_limit_error`: ``CLI_RATE_LIMITED`` APIError 를 만든다. 호스트는
  ``error.details`` 의 ``reset_at``·``rate_limits`` 로 그 계정을 쉬게 하고 다음
  계정으로 넘긴다.

모르면 모른다고 둔다: 시각을 못 뽑으면 ``reset_at`` 은 None 이고, 쉬는 시간은
호스트가 정한다.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

from xgen_agent_runtime.core.errors import APIError, ErrorCategory

__all__ = [
    "LIMIT_ASSISTANT_ERRORS",
    "is_usage_limit_text",
    "normalize_rate_limit_event",
    "rejected_reset_at",
    "reset_at_from_text",
    "usage_limit_error",
]

#: Claude assistant 프레임의 ``error`` 값 중 "이 계정으로는 지금 안 된다" 인 것.
#: (``authentication_failed`` 는 인증 실패로 따로 다룬다.)
LIMIT_ASSISTANT_ERRORS = frozenset({"rate_limit", "billing_error"})

#: 오류 자리의 글에서 한도 소진을 가리키는 표현(소문자 비교).
#: 숫자 상태 코드는 아래 정규식으로만 본다: "429" 한 글자는 토큰 수나 id 에도 나온다.
_LIMIT_PHRASES = (
    "usage limit",
    "usage_limit",
    "limit reached",
    "hit your limit",
    "hit your usage limit",
    "rate_limit_error",
    "rate_limit_exceeded",
    "too many requests",
    "spend limit",
    "credit balance",
    "credits_required",
    "out of credits",
    "insufficient_quota",
    "quota exceeded",
    "exceeded your current quota",
    "payment required",
)

#: 넓은 표현. CLI 가 직접 낸 오류 프레임·결과에서만 본다. stderr 에는 MCP 셔틀이나 도구
#: 로그가 섞이므로(엄격 모드) 이것만으로 한도라고 하면 멀쩡한 계정을 쉬게 만든다.
_LOOSE_PHRASES = ("rate limit", "rate_limit", "ratelimit")

_STATUS_CODE = re.compile(
    r"(?:status|code|http)[^0-9a-z]{0,8}(?:429|402)\b|\b(?:429|402)\s+(?:too many requests|payment required)",
    re.IGNORECASE,
)

# "Claude AI usage limit reached|1781000400" (옛 Claude 형식)
_EPOCH_AFTER_BAR = re.compile(r"limit reached\|(\d{9,11})", re.IGNORECASE)
# Codex 429 본문 resets_in_seconds, HTTP retry-after
_SECONDS_FIELD = re.compile(
    r"(?:resets?_in_seconds|retry[-_ ]after)\"?\s*[:=]?\s*\"?(\d{1,7})", re.IGNORECASE
)
# "try again in 2 days 3 hours 5 minutes", "resets in 2h 5m", "available again in 37m"
_RELATIVE = re.compile(
    r"(?:try again|resets?|available again|retry)\s+in\s+"
    r"((?:\d+\s*(?:days?|d|hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)\b[\s,]*(?:and\s+)?)+)",
    re.IGNORECASE,
)
_RELATIVE_PART = re.compile(
    r"(\d+)\s*(days?|d|hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)\b", re.IGNORECASE
)
# "resets 3pm (Asia/Seoul)", "resets at 3:30pm (America/New_York)", "resets Oct 20, 9am (UTC)"
_CLOCK = re.compile(
    r"resets?\s+(?:at\s+)?(?:([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(?:at\s+)?)?"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(([^)]+)\)",
    re.IGNORECASE,
)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def is_usage_limit_text(text: Any, *, strict: bool = False) -> bool:
    """오류 자리의 글이 구독 한도·지출 한도 소진을 말하는가.

    ``strict`` 는 stderr·종료 코드처럼 다른 출력이 섞이는 자리에서 쓴다: 넓은
    표현("rate limit")은 보지 않는다.
    """
    if not text:
        return False
    s = str(text)
    low = s.lower()
    if any(p in low for p in _LIMIT_PHRASES):
        return True
    if not strict and any(p in low for p in _LOOSE_PHRASES):
        return True
    return bool(_STATUS_CODE.search(s))


def _unit_seconds(unit: str) -> int:
    u = unit.lower()
    if u.startswith("d"):
        return 86400
    if u.startswith("h"):
        return 3600
    if u.startswith("m"):
        return 60
    return 1


def _clock_reset(match: "re.Match[str]", now: float) -> Optional[float]:
    mon, day, hour, minute, ampm, tz_name = match.groups()
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name.strip())
    except Exception:  # noqa: BLE001 (모르는 시간대면 시각을 모른다)
        return None
    h = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
    m = int(minute or 0)
    if h > 23 or m > 59:
        return None
    current = datetime.fromtimestamp(now, tz)
    if mon:
        key = mon.lower()[:3]
        if key not in _MONTHS:
            return None
        try:
            target = current.replace(
                month=_MONTHS.index(key) + 1,
                day=int(day),
                hour=h,
                minute=m,
                second=0,
                microsecond=0,
            )
        except ValueError:
            return None
        if target.timestamp() <= now:
            try:
                target = target.replace(year=target.year + 1)
            except ValueError:
                return None
        return target.timestamp()
    target = current.replace(hour=h, minute=m, second=0, microsecond=0)
    if target.timestamp() <= now:
        target = target + timedelta(days=1)
    return target.timestamp()


def reset_at_from_text(text: Any, *, now: Optional[float] = None) -> Optional[float]:
    """한도 문구에서 초기화 시각(유닉스 초). 모르면 None."""
    if not text:
        return None
    s = str(text)
    t = time.time() if now is None else float(now)
    m = _EPOCH_AFTER_BAR.search(s)
    if m:
        return float(m.group(1))
    m = _SECONDS_FIELD.search(s)
    if m:
        return t + int(m.group(1))
    m = _RELATIVE.search(s)
    if m:
        total = sum(int(n) * _unit_seconds(u) for n, u in _RELATIVE_PART.findall(m.group(1)))
        if total > 0:
            return t + total
    m = _CLOCK.search(s)
    if m:
        return _clock_reset(m, t)
    return None


def _num(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_rate_limit_event(line: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """``rate_limit_event`` 줄 하나 → ``{type, status, resets_at, utilization, ...}``.

    wire 는 camelCase(``rateLimitType``, ``resetsAt``)이고 SDK 문서는 snake_case 다.
    둘 다 읽는다. 창 이름이 없으면 None(어느 창인지 모르는 값은 쌓지 않는다).
    """
    info = line.get("rate_limit_info") if isinstance(line, Mapping) else None
    if not isinstance(info, Mapping):
        return None
    kind = str(info.get("rateLimitType") or info.get("rate_limit_type") or "").strip()
    if not kind:
        return None
    resets = _num(info.get("resetsAt", info.get("resets_at")))
    out: Dict[str, Any] = {
        "type": kind,
        "status": str(info.get("status") or ""),
        "resets_at": int(resets) if resets else None,
        "utilization": _num(info.get("utilization")),
    }
    for src, dst in (
        ("overageStatus", "overage_status"),
        ("isUsingOverage", "is_using_overage"),
        ("surpassedThreshold", "surpassed_threshold"),
    ):
        value = info.get(src, info.get(dst))
        if value is not None:
            out[dst] = value
    return out


def rejected_reset_at(rate_limits: Optional[Mapping[str, Mapping[str, Any]]]) -> Optional[float]:
    """거부된 창들 중 가장 늦은 초기화 시각. 거부가 없거나 시각이 없으면 None.

    5시간 창과 주간 창이 둘 다 막혔으면 둘 다 풀려야 쓸 수 있으므로 늦은 쪽이다.
    """
    if not rate_limits:
        return None
    resets = [
        float(v["resets_at"])
        for v in rate_limits.values()
        if isinstance(v, Mapping) and str(v.get("status")) == "rejected" and v.get("resets_at")
    ]
    return max(resets) if resets else None


def usage_limit_error(
    message: str,
    *,
    rate_limits: Optional[Mapping[str, Mapping[str, Any]]] = None,
    kind: str = "limit",
    now: Optional[float] = None,
) -> APIError:
    """``CLI_RATE_LIMITED`` 오류. ``details``: reset_at·rate_limits·kind.

    ``kind`` 는 limit(구독 창·속도) 또는 billing(결제·크레딧). 호스트는 billing 을
    더 길게 쉬게 할 수 있다.
    """
    reset_at = rejected_reset_at(rate_limits) or reset_at_from_text(message, now=now)
    details: Dict[str, Any] = {"reset_at": reset_at, "kind": kind}
    if rate_limits:
        details["rate_limits"] = {k: dict(v) for k, v in rate_limits.items()}
    if reset_at:
        stamp = datetime.fromtimestamp(reset_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        message = f"{message} [reset_at={stamp}]"
    return APIError(message, category=ErrorCategory.CLI_RATE_LIMITED, details=details)
