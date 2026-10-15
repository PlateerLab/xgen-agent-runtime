"""사람이 거부한 동작을 같은 턴에서 다시 묻지 않는다.

실측 (2026-09-23 dev, gpt-4.1, 시연 준비 "업무 비서", trace 46745)
-----------------------------------------------------------------
사용자 PC 에서 ``rm -rf 임시_삭제테스트`` 를 실행하려 하자 커넥터가 확인 창을 띄웠고 사용자가
**거부**했다. 모델은 같은 명령을 다시 불렀다 — **세 번**, 확인 창도 세 번 떴다. 반복 가드는
같은 입력·같은 오류를 3번까지는 경고만 하므로(BLOCK_AT 4) 막지 못했다.

그 확인 창은 "되돌릴 수 없는 일은 사람이 멈춘다" 의 전부다. 같은 질문을 거듭 받으면 사람은
결국 잘못 누른다 — **"이번만 허용" 을 한 번이라도 누르면 그대로 실행된다.** 거부는 그 턴의
답이다. 다시 물을 일이 아니다.

규칙
----
* 결과가 ``ERROR user_denied:`` / ``ERROR access_denied:`` 로 시작하면 거부다 (런타임 자체 권한
  매트릭스·HITL 거부, 그리고 어댑터가 커넥터 거부를 이 머리말로 바꿔 준 것).
* 거부된 호출의 서명을 이 턴 동안 기억한다. 같은 동작이 다시 오면 **실행하지 않는다** — 확인
  창도 다시 뜨지 않는다. 거절 결과는 "거부 누적" 으로 세어 쌓이면 반복 종료(RepeatStop)가 턴을
  끝낸다.
* "같은 동작" 은 넓게 본다: 문자열의 따옴표·연속 공백을 무시한다. 거부된 일을 비슷하게 다시
  시도하는 쪽으로 잘못 맞히는 것은 안전한 실수이고, 반대쪽은 아니다.
* 턴 단위다(``PipelineState.begin_turn`` 이 비운다). 다음 턴에 사용자가 "해도 돼" 라고 하면 다시
  물을 수 있다.

도메인 규칙 없음 — 어떤 도구의 어떤 거부든 같다.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

#: state.shared 키 — 이 턴에 거부된 호출 서명 → 거부 사유.
DENIED_KEY = "tool.denied_calls"

_DENIAL_PREFIXES = ("ERROR user_denied", "ERROR access_denied")
_QUOTES = re.compile(r"[\"'`]")
_SPACES = re.compile(r"\s+")


def _text(result: Dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        return "\n".join(p for p in parts if isinstance(p, str))
    return str(content or "")


def is_denial(result: Dict[str, Any]) -> bool:
    return bool(result.get("is_error")) and _text(result).lstrip().startswith(_DENIAL_PREFIXES)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return _SPACES.sub(" ", _QUOTES.sub("", value)).strip()
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def signature(tool_call: Dict[str, Any]) -> str:
    """도구 이름 + 정규화한 입력. 따옴표·공백만 다른 재시도는 같은 동작이다."""
    name = str(tool_call.get("tool_name") or "")
    try:
        raw = json.dumps(
            _normalize(tool_call.get("tool_input") or {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        raw = str(tool_call.get("tool_input"))
    return name + ":" + hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def refused_result(tool_call: Dict[str, Any], shared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """이 턴에 이미 거부된 동작이면 **실행 대신** 돌려줄 결과."""
    denied = shared.get(DENIED_KEY) or {}
    reason = denied.get(signature(tool_call))
    if reason is None:
        return None
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.get("tool_use_id", ""),
        "is_error": True,
        "content": (
            "ERROR user_denied_repeat: 사용자가 이번 요청에서 이미 거부한 동작이다 — 실행하지 않았고 "
            "확인 창도 다시 띄우지 않았다.\n"
            f"거부 사유: {reason}\n"
            "같은 동작을 다시 시도하지 마라(따옴표·공백만 바꾼 것도 같다). 같은 결과를 내는 다른 "
            "명령으로 우회하지도 마라. 사용자에게 그 일은 실행되지 않았다고 알리고, 어떻게 할지 물어라."
        ),
    }


def observe(
    tool_calls: List[Dict[str, Any]], results: List[Dict[str, Any]], shared: Dict[str, Any]
) -> List[str]:
    """이번 라운드에서 새로 거부된 호출을 기억한다. 새로 기억한 도구 이름을 돌려준다."""
    by_id = {str(r.get("tool_use_id") or ""): r for r in results}
    denied = dict(shared.get(DENIED_KEY) or {})
    added: List[str] = []
    for tc in tool_calls:
        r = by_id.get(str(tc.get("tool_use_id") or ""))
        if r is None or not is_denial(r):
            continue
        sig = signature(tc)
        if sig not in denied:
            first_line = _text(r).strip().splitlines()[0] if _text(r).strip() else "denied"
            denied[sig] = first_line[:300]
            added.append(str(tc.get("tool_name") or ""))
    if added:
        shared[DENIED_KEY] = denied
    return added
