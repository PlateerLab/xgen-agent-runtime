"""같은 도구가 같은 오류로 되풀이 실패하는 루프를 끊는다.

실측 (2026-09-16 dev, claude-sonnet-4-6): 에이전트가 ``lotteimall_search`` 에
``max_results: "3"`` (문자열)을 넘겨 ``'3' is not of type 'integer'`` 가 났는데,
인자를 고치지 않고 검색어만 바꿔 **15번** 같은 호출을 반복했다. 매 호출마다
대화 전체가 모델에 다시 들어가므로 실패 하나가 수십만 토큰이 됐다.

판정은 오류 종류에 따라 다르다.

* **입력 오류** (``ERROR invalid_input``) — 키는 **(도구, 정규화한 오류 문구)**.
  인자까지 키에 넣으면 위 사고처럼 검색어만 바뀌는 반복을 놓친다. 입력 오류 문구가
  같다는 것은 모델이 원인을 고치지 않았다는 뜻이다.
* **그 밖의 실행 오류** ("not found", 업스트림 5xx 등) — 서로 다른 항목이 각자 정당하게
  같은 문구로 실패할 수 있다(상품 5개가 각각 없음). 그래서 두 가지로 센다.
  **같은 인자**로 같은 오류면 입력 오류와 같은 문턱(``WARN_AT``/``BLOCK_AT``),
  **인자가 매번 달라도** 같은 오류면 더 넓은 문턱(``ANY_INPUT_WARN_AT``/
  ``ANY_INPUT_BLOCK_AT``) — 인증 실패처럼 무엇을 넣어도 안 되는 루프는 결국 끊는다.
  실행 오류 카운트는 **다른 도구가 성공하면** 비운다. 코딩의 "고치고 → 다시 빌드" 는
  같은 명령이 여러 번 실패하는 게 정상이다.

* 경고 문턱 — 결과에 "같은 방식으로 다시 부르지 말라" 는 안내를 덧붙인다.
  모델에게 고칠 기회를 준다.
* 차단 문턱부터 — 이번 턴(연속 슬라이스 포함) 동안 그 도구를 실행하지 않고
  차단 결과를 돌려준다. 다른 도구는 계속 쓸 수 있고, 성공이 한 번 나오면 그 도구의
  카운트는 비워진다.

상태는 ``state.shared`` 에 둔다 — 턴마다 새 ``PipelineState`` 이므로 턴을 넘어가지
않고, 같은 턴의 연속 슬라이스(CONTINUE_RUN)에는 이어진다.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

WARN_AT = 3
BLOCK_AT = 5
ANY_INPUT_WARN_AT = 5
ANY_INPUT_BLOCK_AT = 8

#: 같은 도구·같은 입력이 **같은 결과**를 낸 횟수 — 성공도 센다. 같은 입력에 같은 결과면
#: 새 정보가 없다는 것이 정의상 확실하다. 실측 (2026-08~09 dev 기록, 오프라인 평가):
#: 도구 12회+ 턴 73건 중 낭비 루프 14건에서 13건 적중·정상 55건 중 오탐 1건(게임 버튼 4회
#: 클릭 — 최대 4회), 설계에 안 쓴 기간 41건에서 3/3 적중·오탐 0. 반복은 gpt-4.1·qwen 에
#: 몰려 있었다(pwd 21회, 안내 도구 100회, 같은 메모리 저장 62회).
SAME_RESULT_WARN_AT = 4
#: 이 횟수째 같은 호출은 실행하지 않고 직전 결과를 돌려준다 — 같은 부작용(저장·전송)의
#: 반복도 막는다. 결과가 같았으니 모델이 받는 정보는 실행했을 때와 같다.
SAME_RESULT_SKIP_AT = 8

_COUNTS_KEY = "tool.repeat_error_counts"
_BLOCKED_KEY = "tool.repeat_error_blocked"
_SAME_KEY = "tool.same_result_counts"

#: 오류 문구에서 호출마다 달라지는 조각 — 이것 때문에 같은 원인이 다른 키가 되면 안 된다.
_VOLATILE = [
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "#"),  # id·해시
    (re.compile(r"\b\d{2,}\b"), "#"),  # 시각·포트·길이 같은 긴 숫자
    (re.compile(r"\s+"), " "),
]
#: 구조화 오류(``ERROR <code>: <message>`` 헤더)의 JSON 본문 — 요청 id·경로 등. 헤더가 있을 때만 뗀다.
#: 빌드 로그·스택 트레이스 같은 일반 출력의 중괄호까지 잘라 내면 실제 오류가 사라진다.
_STRUCTURED_BODY = re.compile(r"\{.*", re.S)
_KEY_HEAD = 80
_KEY_TAIL = 200
_KEY_TEXT_CAP = _KEY_HEAD + _KEY_TAIL


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
    """판정 키용 오류 문구.

    **앞과 끝을 함께 본다.** 명령 출력은 앞부분(빌드 배너·명령 에코)이 매번 같고 실제
    오류는 끝에 나온다 — 앞 240자만 보면 ``npm run build`` 가 서로 다른 오류로 실패해도
    같은 실패로 세어, 고치고 다시 빌드하는 정상 루프가 차단됐다.
    """
    out = str(text or "").strip()
    if out.startswith("ERROR "):
        out = _STRUCTURED_BODY.sub("", out)
    for pattern, repl in _VOLATILE:
        out = pattern.sub(repl, out)
    out = out.strip()
    if len(out) > _KEY_TEXT_CAP:
        out = f"{out[:_KEY_HEAD]}…{out[-_KEY_TAIL:]}"
    return out


def is_input_error(text: str) -> bool:
    return str(text or "").lstrip().startswith("ERROR invalid_input")


def _input_sig(tool_input: Any) -> str:
    try:
        raw = json.dumps(tool_input, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        raw = str(tool_input)
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


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
    *,
    count_across_inputs: bool = True,
) -> List[Tuple[str, int]]:
    """실행 결과를 보고 카운트를 갱신하고, 필요하면 결과에 안내를 덧붙인다.

    ``results`` 는 제자리에서 고친다. 반환값은 (도구, 누적 횟수) 중 안내를
    붙이거나 차단으로 넘어간 것 — 이벤트 기록용.

    ``count_across_inputs=False`` 는 한 번에 여러 항목을 도는 호출(ToolBatch)용이다 —
    항목마다 입력이 다른 게 정상이라, 실행 오류는 같은 인자 반복만 센다.
    """
    counts: Dict[str, int] = state_shared.setdefault(_COUNTS_KEY, {})
    blocked: Dict[str, str] = state_shared.setdefault(_BLOCKED_KEY, {})
    calls = {str(tc.get("tool_use_id") or ""): tc for tc in tool_calls}
    flagged: List[Tuple[str, int]] = []

    def _bump(key: str) -> int:
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    for result in results:
        tc = calls.get(str(result.get("tool_use_id") or "")) or {}
        name = str(tc.get("tool_name") or "")
        if not name:
            continue
        if not result.get("is_error"):
            # 성공하면 그 도구의 실패 이력은 끝난 일이다. 실행 오류는 **다른 도구의 성공**
            # 으로도 비운다 — 파일을 고치고(Edit/Write) 다시 빌드하는 루프는 원인을 안 고친
            # 반복이 아니다. 입력 오류는 그대로 둔다: 다른 일을 해도 인자는 여전히 틀렸다.
            for key in [k for k in counts if k.startswith(f"{name}␟") or "␟R␟" in k]:
                counts.pop(key, None)
            continue
        text = _error_text(result)
        if not text or text.startswith("ERROR repeated_failure_blocked"):
            continue
        err = normalize_error(text)
        if is_input_error(text):
            n = _bump(f"{name}␟I␟{err}")
            warn, block = n >= WARN_AT, n >= BLOCK_AT
            block_at = BLOCK_AT
        else:
            same = _bump(f"{name}␟R␟{_input_sig(tc.get('tool_input'))}␟{err}")
            any_ = _bump(f"{name}␟R␟*␟{err}") if count_across_inputs else 0
            warn = same >= WARN_AT or any_ >= ANY_INPUT_WARN_AT
            block = same >= BLOCK_AT or any_ >= ANY_INPUT_BLOCK_AT
            n = max(same, any_)
            block_at = BLOCK_AT if same >= WARN_AT else ANY_INPUT_BLOCK_AT
        if block:
            blocked[name] = err
        if warn:
            flagged.append((name, n))
            if isinstance(result.get("content"), str):
                result["content"] = (
                    f"{result['content']}\n\n[반복 실패 {n}회] '{name}' 가 같은 오류로 {n}번 실패했다. "
                    "같은 방식으로 다시 부르지 마라 — 오류 문구대로 인자를 고치거나, 고칠 수 없으면 "
                    f"사용자에게 알려라. {block_at}번째부터는 이 요청에서 이 도구가 차단된다."
                )
    return flagged


# ── 같은 호출·같은 결과 (성공 포함) ─────────────────────────────────────


def _result_sig(result: Dict[str, Any]) -> Optional[str]:
    text = _error_text(result)
    if text is None:
        return None
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def _call_key(tool_call: Dict[str, Any]) -> str:
    return f"{tool_call.get('tool_name') or ''}␟{_input_sig(tool_call.get('tool_input'))}"


def skip_identical(
    tool_call: Dict[str, Any], state_shared: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """같은 호출이 같은 결과를 ``SAME_RESULT_SKIP_AT - 1`` 번 냈다면 실행 대신 돌려줄 결과."""
    entry = (state_shared.get(_SAME_KEY) or {}).get(_call_key(tool_call))
    if not entry or entry.get("n", 0) < SAME_RESULT_SKIP_AT - 1:
        return None
    name = str(tool_call.get("tool_name") or "")
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.get("tool_use_id", ""),
        "content": (
            f"{entry.get('last', '')}\n\n[같은 호출 {entry['n'] + 1}회째 — 실행하지 않음] "
            f"'{name}' 를 같은 입력으로 {entry['n']}번 불러 매번 위와 같은 결과를 받았다. "
            "다시 불러도 새 정보는 없다. 이 결과로 다음 단계를 진행하거나, 막혔다면 무엇이 "
            "막혔는지 사용자에게 알려라."
        ),
    }


def observe_same(
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    state_shared: Dict[str, Any],
) -> List[Tuple[str, int]]:
    """같은 호출·같은 결과 횟수를 갱신하고, 경고 문턱부터 결과에 안내를 붙인다."""
    same: Dict[str, Dict[str, Any]] = state_shared.setdefault(_SAME_KEY, {})
    calls = {str(tc.get("tool_use_id") or ""): tc for tc in tool_calls}
    flagged: List[Tuple[str, int]] = []
    for result in results:
        tc = calls.get(str(result.get("tool_use_id") or ""))
        if not tc:
            continue
        if result.get("is_error"):
            continue  # 실패의 반복은 위의 반복 실패 차단 몫이다
        content = result.get("content")
        if isinstance(content, str) and "실행하지 않음]" in content:
            continue  # 이미 건너뛴 호출 — 다시 세지 않는다
        sig = _result_sig(result)
        if sig is None:
            continue
        key = _call_key(tc)
        entry = same.get(key)
        if entry and entry.get("sig") == sig:
            entry["n"] += 1
        else:
            entry = same[key] = {"sig": sig, "n": 1, "last": ""}
        if isinstance(content, str):
            entry["last"] = content[:4000]
        n = entry["n"]
        if n >= SAME_RESULT_WARN_AT:
            name = str(tc.get("tool_name") or "")
            flagged.append((name, n))
            if isinstance(content, str):
                result["content"] = (
                    f"{content}\n\n[같은 호출·같은 결과 {n}회] '{name}' 를 같은 입력으로 "
                    f"{n}번 불러 매번 같은 결과를 받았다. 새 정보가 없으니 같은 호출을 반복하지 "
                    "말고 접근을 바꾸거나, 막혔다면 사용자에게 알려라. "
                    f"{SAME_RESULT_SKIP_AT}번째부터는 실행하지 않고 이 결과를 그대로 돌려준다."
                )
    return flagged
