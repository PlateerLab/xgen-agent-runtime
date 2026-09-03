"""최근 대화를 **논리 턴** 단위로 묶어 렌더한다 — L0 (recent_turns) 의 정본.

## 왜 필요했나

STM 은 파이프라인이 만든 메시지를 **하나도 빠짐없이** 적는다. 도구를 한 번 쓰면
그 턴은 메시지 4개가 된다: 사용자 지시 · assistant tool_use · user tool_result ·
최종 답변. 그런데 L0 는 "최근 N개" 를 **행 수**로 세어 가져왔고, 가져온 것 중
텍스트 블록이 아닌 것은 렌더에서 버렸다. 결과:

    3턴(각 도구 1회) → STM 12행 → recent(6) 은 1.5턴만 → 필터 후 3줄만 남고
    첫 턴은 통째로 사라진다 (실측)

도구 행이 **자리는 차지하면서 모델에게는 보이지 않는** 것이 문제의 핵심이었다.
그래서 에이전트는 두 턴 전에 자기가 무엇을 했는지 모르고, 같은 조회를 반복한다.

## 이 모듈이 정하는 것

* **논리 턴을 센다.** 한 턴은 "사용자의 새 지시" 에서 시작해 다음 지시 직전까지다.
  도구를 몇 번 쓰든 한 턴은 한 턴이다. 그래서 ``limit=3`` 이면 사용자 입력 3개가
  **반드시** 들어간다 — 그 사이의 도구 호출 개수와 무관하게.
* **도구는 호출 기준으로 한 줄씩 요약한다.** 무엇을 어떤 인자로 불렀고 어떻게
  됐는지를 남긴다. 결과 전문을 넣으면 예산이 바로 터지고, 반대로 지워 버리면
  같은 호출을 다시 한다 — 둘 사이의 한 줄이 이 모듈의 답이다.
* **같은 호출이 이어지면 묶는다** (``×3``). 반복은 그 자체가 신호이고, 세 줄로
  늘어놓으면 예산만 먹는다.
* **턴을 버리지 않는다.** 예산이 모자라면 줄을 짧게 만들지, 턴을 없애지 않는다.
  "최근 3턴" 이 조건부라면 그건 계약이 아니다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 한 도구 줄의 기본 상한. 인자와 결과를 합쳐 이 길이를 넘지 않는다.
DEFAULT_TOOL_LINE_CHARS = 200
# 사용자/어시스턴트 한 발화의 기본 상한.
DEFAULT_MESSAGE_CHARS = 1200
# 압축을 해도 이보다 짧게는 줄이지 않는다 — 이 아래로 가면 남은 글자가
# 문장이 아니라 조각이라 모델이 오히려 잘못 읽는다.
MIN_MESSAGE_CHARS = 200


def _blocks(content: Any) -> List[Dict[str, Any]]:
    """content 를 블록 리스트로 정규화. 문자열이면 text 블록 하나로 본다."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _is_tool_result_only(content: Any) -> bool:
    """이 user 메시지가 도구 결과뿐인가 — 그렇다면 새 턴의 시작이 아니다.

    턴 경계를 role 만으로 정하면 도구를 쓸 때마다 턴이 하나 더 세어진다
    (tool_result 도 user 메시지다). 그게 "3턴" 이 1.5턴이 되던 이유다.
    """
    blocks = _blocks(content)
    if not blocks:
        return False
    return all(b.get("type") == "tool_result" for b in blocks)


def _clip(s: str, limit: int) -> str:
    """길이를 줄이되 **줄였다는 사실을 남긴다.**

    조용히 자르면 모델은 잘린 것을 전부로 읽는다 — 도구 결과에서는 그게 곧
    "빈 디렉터리였다" 같은 오독이 된다.
    """
    s = " ".join(str(s).split())
    if limit <= 0 or len(s) <= limit:
        return s
    keep = max(1, limit - 8)
    return f"{s[:keep]}…[+{len(s) - keep}]"


def _args_text(tool_input: Any, limit: int) -> str:
    """호출 인자를 ``k=v, k=v`` 로. 무엇을 불렀는지 구분되는 것이 목적이다.

    같은 도구를 **다른 인자로** 부른 것과 **같은 인자로** 부른 것을 구분하지 못하면
    반복 호출을 막을 수 없다 — 그래서 이름만 남기지 않는다.
    """
    if not isinstance(tool_input, dict) or not tool_input:
        return ""
    parts: List[str] = []
    per = max(20, limit // max(1, len(tool_input)))
    for k, v in tool_input.items():
        if isinstance(v, (dict, list)):
            try:
                v = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                v = str(v)
        parts.append(f"{k}={_clip(str(v), per)}")
    return _clip(", ".join(parts), limit)


def _result_text(result: Optional[Dict[str, Any]], limit: int) -> str:
    """결과를 한 조각으로. 성공/실패와 **무엇이 돌아왔는지**를 함께 남긴다.

    성공을 ``ok`` 로만 적으면 답을 잊은 에이전트가 같은 것을 또 부른다.
    """
    if result is None:
        return "(결과 없음)"
    body = result.get("content")
    if isinstance(body, list):
        body = "\n".join(
            str(b.get("text", "")) for b in body if isinstance(b, dict) and b.get("type") == "text"
        )
    body = _clip(str(body or ""), limit)
    if result.get("is_error"):
        return f"error: {body}" if body else "error"
    return f"ok: {body}" if body else "ok"


class LogicalTurn:
    """사용자 지시 하나와 그 뒤에 붙은 모든 것."""

    __slots__ = ("messages",)

    def __init__(self, messages: List[Any]) -> None:
        self.messages = messages


def group_logical_turns(turns: Sequence[Any], limit: int) -> List[LogicalTurn]:
    """시간순 turn 목록을 논리 턴으로 묶어 **마지막 ``limit`` 개**를 돌려준다.

    경계는 "도구 결과가 아닌 user 메시지". 첫 경계 앞에 남은 꼬리(이전 턴의
    잔여분)는 버린다 — 사용자 지시가 없는 조각은 맥락 없이 답만 떠 있는 꼴이다.
    """
    if limit <= 0:
        return []
    groups: List[List[Any]] = []
    for t in turns:
        role = str(getattr(t, "role", "") or "")
        content = getattr(t, "content", "")
        starts_turn = role == "user" and not _is_tool_result_only(content)
        if starts_turn or not groups:
            if not starts_turn and not groups:
                continue  # 첫 사용자 지시 이전의 잔여분은 버린다
            groups.append([t])
        else:
            groups[-1].append(t)
    return [LogicalTurn(g) for g in groups[-limit:]]


def _render_turn(
    turn: LogicalTurn, *, message_chars: int, tool_line_chars: int
) -> Tuple[List[str], List[int]]:
    """한 턴 → (줄 목록, 도구 줄 인덱스). 도구 줄 인덱스는 압축 순서에 쓰인다."""
    # 결과를 먼저 모은다 — tool_use 와 tool_result 는 서로 다른 메시지에 있고,
    # 짝을 지어야 "무엇을 불렀고 어떻게 됐는지" 가 한 줄이 된다.
    results: Dict[str, Dict[str, Any]] = {}
    for m in turn.messages:
        for b in _blocks(getattr(m, "content", "")):
            if b.get("type") == "tool_result":
                results[str(b.get("tool_use_id") or "")] = b

    lines: List[str] = []
    tool_idx: List[int] = []
    last_sig: Optional[str] = None  # 연속 중복 묶기용

    for m in turn.messages:
        role = str(getattr(m, "role", "") or "user")
        content = getattr(m, "content", "")
        if role == "user" and _is_tool_result_only(content):
            continue  # 이미 tool_use 줄에 붙었다
        for b in _blocks(content):
            btype = b.get("type")
            if btype == "text":
                text = str(b.get("text", "")).strip()
                if text:
                    lines.append(f"[{role}] {_clip(text, message_chars)}")
                    last_sig = None
            elif btype == "tool_use":
                name = str(b.get("name") or "tool")
                args = _args_text(b.get("input"), max(40, tool_line_chars // 2))
                sig = f"{name}({args})"
                if sig == last_sig and tool_idx:
                    # 같은 호출이 이어졌다 — 줄을 늘리지 않고 횟수만 올린다.
                    prev = lines[tool_idx[-1]]
                    lines[tool_idx[-1]] = _bump_repeat(prev, sig)
                    continue
                res = _result_text(results.get(str(b.get("id") or "")), tool_line_chars // 2)
                tool_idx.append(len(lines))
                lines.append(f"[tool] {sig} → {res}")
                last_sig = sig
    return lines, tool_idx


def _bump_repeat(line: str, sig: str) -> str:
    """``[tool] Sig → ok`` → ``[tool] Sig ×2 → ok``."""
    marker = f"[tool] {sig}"
    rest = line[len(marker):]
    if rest.startswith(" ×"):
        num, _, tail = rest[2:].partition(" ")
        try:
            return f"{marker} ×{int(num) + 1} {tail}"
        except ValueError:
            pass
    return f"{marker} ×2{rest}"


def render_recent_turns(
    turns: Sequence[Any],
    *,
    limit: int,
    max_chars: int,
    message_chars: int = DEFAULT_MESSAGE_CHARS,
    tool_line_chars: int = DEFAULT_TOOL_LINE_CHARS,
) -> str:
    """최근 ``limit`` 개 논리 턴을 ``max_chars`` 안에 렌더한다.

    예산이 모자라면 **줄을 짧게 만들지 턴을 버리지 않는다.** 세 단계로 조인다:

      1. 발화 상한을 필요한 만큼 낮춘다 (하한 ``MIN_MESSAGE_CHARS``).
      2. 그래도 넘치면 **오래된 턴의 도구 줄부터** 뺀다 — 도구 줄은 요약이라
         복구 가능한 손실이고, 사용자 지시와 답변은 그렇지 않다.
      3. 그래도 넘치면 앞에서부터 잘라 낸다(최신이 살아남는다).
    """
    groups = group_logical_turns(turns, limit)
    if not groups:
        return ""

    def _compose(msg_cap: int, tool_cap: int) -> Tuple[List[List[str]], List[List[int]]]:
        rendered, idxs = [], []
        for g in groups:
            ls, ti = _render_turn(g, message_chars=msg_cap, tool_line_chars=tool_cap)
            rendered.append(ls)
            idxs.append(ti)
        return rendered, idxs

    def _size(rendered: List[List[str]]) -> int:
        return sum(len(line) + 1 for lines_ in rendered for line in lines_)

    rendered, idxs = _compose(message_chars, tool_line_chars)
    if max_chars > 0 and _size(rendered) > max_chars:
        # 1) 발화 상한을 실제 초과분에 비례해 낮춘다.
        over = _size(rendered)
        ratio = max_chars / float(over)
        msg_cap = max(MIN_MESSAGE_CHARS, int(message_chars * ratio))
        tool_cap = max(80, int(tool_line_chars * ratio))
        rendered, idxs = _compose(msg_cap, tool_cap)

        # 2) 오래된 턴의 도구 줄부터 뺀다.
        gi = 0
        while _size(rendered) > max_chars and gi < len(rendered):
            if idxs[gi]:
                drop = set(idxs[gi])
                rendered[gi] = [ln for i, ln in enumerate(rendered[gi]) if i not in drop]
                idxs[gi] = []
            else:
                gi += 1

    body = "\n".join(line for lines_ in rendered for line in lines_)
    if max_chars > 0 and len(body) > max_chars:
        body = body[-max_chars:]  # 3) 최신이 살아남는다
    return body
