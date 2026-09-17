"""같은 도구가 같은 오류로 되풀이 실패하는 루프를 끊는다.

실측 (2026-09-16 dev, claude-sonnet-4-6): 에이전트가 ``lotteimall_search`` 에
``max_results: "3"`` (문자열)을 넘겨 ``'3' is not of type 'integer'`` 가 났는데,
인자를 고치지 않고 검색어만 바꿔 **15번** 같은 호출을 반복했다. 매 호출마다
대화 전체가 모델에 다시 들어가므로 실패 하나가 수십만 토큰이 됐다.

판정 키는 **(도구 이름, 정규화한 오류 문구)** 다. 인자까지 키에 넣으면 위 사고처럼
검색어만 바뀌는 반복을 놓친다. 오류 문구가 같다는 것은 모델이 원인을 고치지
않았다는 뜻이다.

* ``WARN_AT`` 번째 같은 실패 — 결과에 "같은 방식으로 다시 부르지 말라" 는 안내를
  덧붙인다. 모델에게 고칠 기회를 준다.
* ``BLOCK_AT`` 번째부터 — 이번 턴(연속 슬라이스 포함) 동안 그 도구를 실행하지 않고
  차단 결과를 돌려준다. 다른 도구는 계속 쓸 수 있고, 성공이 한 번 나오면 그 도구의
  카운트는 비워진다.

상태는 ``state.shared`` 에 둔다 — 턴마다 새 ``PipelineState`` 이므로 턴을 넘어가지
않고, 같은 턴의 연속 슬라이스(CONTINUE_RUN)에는 이어진다.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

WARN_AT = 3
BLOCK_AT = 5

_COUNTS_KEY = "tool.repeat_error_counts"
_BLOCKED_KEY = "tool.repeat_error_blocked"

#: 오류 문구에서 호출마다 달라지는 조각 — 이것 때문에 같은 원인이 다른 키가 되면 안 된다.
_VOLATILE = [
    (re.compile(r"\{.*", re.S), ""),  # 구조화 오류의 JSON 본문(요청 id·경로 등)
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "#"),  # id·해시
    (re.compile(r"\b\d{2,}\b"), "#"),  # 시각·포트·길이 같은 긴 숫자
    (re.compile(r"\s+"), " "),
]
_KEY_TEXT_CAP = 240


def _error_text(result: Dict[str, Any]) -> Optional[str]:
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


def normalize_error(text: str) -> str:
    out = str(text or "").strip()
    for pattern, repl in _VOLATILE:
        out = pattern.sub(repl, out)
    return out.strip()[:_KEY_TEXT_CAP]


def _key(tool_name: str, error: str) -> str:
    return f"{tool_name}␟{normalize_error(error)}"


def blocked_result(
    tool_call: Dict[str, Any], state_shared: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """이 호출이 차단 대상이면 실행 대신 돌려줄 tool_result, 아니면 ``None``."""
    blocked = state_shared.get(_BLOCKED_KEY) or {}
    name = str(tool_call.get("tool_name") or "")
    reason = blocked.get(name)
    if not reason:
        return None
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.get("tool_use_id", ""),
        "is_error": True,
        "content": (
            f"ERROR repeated_failure_blocked: '{name}' 는 이번 요청에서 같은 오류로 "
            f"{BLOCK_AT}번 이상 실패해 더 실행하지 않는다.\n"
            f"마지막 오류: {reason}\n"
            "같은 도구를 다시 부르지 마라. 오류 원인(인자 타입·필수 값·경로 등)을 사용자에게 "
            "설명하고, 다른 방법이 있으면 그것을 쓰거나 사용자에게 확인을 요청하라."
        ),
    }


def observe(
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    state_shared: Dict[str, Any],
) -> List[Tuple[str, int]]:
    """실행 결과를 보고 카운트를 갱신하고, 필요하면 결과에 안내를 덧붙인다.

    ``results`` 는 제자리에서 고친다. 반환값은 (도구, 누적 횟수) 중 안내를
    붙이거나 차단으로 넘어간 것 — 이벤트 기록용.
    """
    counts: Dict[str, int] = state_shared.setdefault(_COUNTS_KEY, {})
    blocked: Dict[str, str] = state_shared.setdefault(_BLOCKED_KEY, {})
    names = {str(tc.get("tool_use_id") or ""): str(tc.get("tool_name") or "") for tc in tool_calls}
    flagged: List[Tuple[str, int]] = []

    for result in results:
        name = names.get(str(result.get("tool_use_id") or ""), "")
        if not name:
            continue
        if not result.get("is_error"):
            # 성공하면 그 도구의 실패 이력은 끝난 일이다.
            for key in [k for k in counts if k.startswith(f"{name}␟")]:
                counts.pop(key, None)
            continue
        text = _error_text(result)
        if not text or text.startswith("ERROR repeated_failure_blocked"):
            continue
        key = _key(name, text)
        counts[key] = counts.get(key, 0) + 1
        n = counts[key]
        if n >= BLOCK_AT:
            blocked[name] = normalize_error(text)
            flagged.append((name, n))
        elif n >= WARN_AT:
            flagged.append((name, n))
        if n >= WARN_AT and isinstance(result.get("content"), str):
            result["content"] = (
                f"{result['content']}\n\n[반복 실패 {n}회] '{name}' 가 같은 오류로 {n}번 실패했다. "
                "같은 방식으로 다시 부르지 마라 — 오류 문구대로 인자를 고치거나, 고칠 수 없으면 "
                f"사용자에게 알려라. {BLOCK_AT}번째부터는 이 요청에서 이 도구가 차단된다."
            )
    return flagged
