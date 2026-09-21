"""단기 기억 창 — 최근 논리 턴을 **messages 로** 되살린다 (L0 ``recent_turns`` 의 후계).

## 왜 바꿨나

이력을 preload 하지 않는 호스트(XGEN 의 DB Memory 노드 없는 XGeny)는 지난 턴을 Stage 2 의
``# Relevant Knowledge`` 불릿(``[short_term] recent_turns: …``)으로만 봤다. 그러면 모델은 자기가
지난 턴에 한 말을 **대화 기록이 아니라 확인해야 할 지식**으로 읽고, 매 단계 "지난 대화를
보니까…" 를 다시 쓴다(2026-09-21 관측: 같은 서술이 24단계 내내 반복). 레퍼런스(OpenAI Agents
Sessions·Hermes·Claude Code·LangChain trim_messages)는 모두 이력을 messages 로 재생한다.

## 정책 (5턴 보장)

논리 턴 = 사용자의 새 지시부터 다음 지시 직전까지 (:func:`group_logical_turns`).

* **T-1, T-2 — 가까운 2턴은 도구까지 그대로.** 사용자 텍스트 · assistant 텍스트 · ``tool_use`` ·
  ``tool_result`` 블록을 순서·id 그대로. 큰 결과만 앞머리를 남기고 절단 표식(내용만 줄인다,
  블록은 남긴다 — Claude Context Editing 이 결과만 지우고 호출은 두는 것과 같다).
* **T-3 ~ T-5 — 먼 3턴은 대화만.** 사용자 텍스트 + assistant **최종 답변** 텍스트. 도구 블록은
  ``tool_use``/``tool_result`` **양쪽 다** 뺀다(쌍을 가르지 않는다). 답변 끝에 한 줄
  ``[used tools: ForgeTool ×4 (3 failed), Bash ×2]`` — 두 턴 전에 무엇을 했는지 모르면 같은
  조회를 또 한다.
* **예산이 모자라면 깎는 순서**: ① 가까운 턴의 tool_result 를 더 짧게 ② 가장 먼 대화 턴부터 제거
  ③ T-2 의 도구 블록 제거(대화만) ④ T-1 은 마지막까지 남긴다(결과를 최소로 줄여서라도).
* **쌍 불변**: 결과 없는 ``tool_use`` 에는 합성 결과를 붙이고(Hermes stub), 앞머리의 고아
  ``tool_result`` 는 버린다 — :mod:`xgen_agent_runtime.core.message_repair` 재사용.
* 지난 턴의 ``thinking`` 블록은 재생하지 않는다(서명이 있는 블록은 다른 요청에서 거부될 수 있다).
* **역할은 교대한다.** 답변 없이 끝난 턴은 user 가 둘 연속이 되고, 도구 실행 중 끊긴 턴은 합성
  tool_result(=user)로 끝나 이번 턴의 지시와 맞붙는다. 순수 발화는 합치고, 창이 user 로 끝나면
  중단 표식 한 줄로 닫는다 — 연속 role 허용치는 백엔드마다 다르다.

## 알려진 한계 (고치지 않고 적는다)

* **CLI 백엔드(claude_code·codex)에서는 창이 다시 텍스트가 된다.** 그 와이어 규격은 전체 이력을
  ``type:user`` 봉투 하나 안의 마크다운 프리앰블로 접는다. "대화를 대화 자리에" 가 완전히
  성립하는 것은 API 백엔드다. 그래도 시스템 프롬프트의 "지식" 불릿보다는 낫다(대화로 읽힌다).
* **창은 매 턴 미끄러지므로 messages 접두부가 고정되지 않는다.** 프롬프트 캐시는 접두부 일치라
  메시지 층 캐시 이득은 크지 않다. 시스템 프롬프트에서 지난 턴을 걷어내 그쪽이 안정해진 만큼은
  벌었다. 접두부를 고정하려면 미끄러지는 창 대신 누적 요약(v2)이 필요하다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from xgen_agent_runtime.core.message_repair import (
    repair_dangling_tool_calls,
    strip_leading_orphan_tool_results,
)
from xgen_agent_runtime.memory.transcript import (
    LogicalTurn,
    _blocks,
    _is_tool_result_only,
    group_logical_turns,
)

#: state.metadata 표식 — 창이 이 턴의 messages 앞에 들어갔다. {"turns","full","dialogue","chars"}.
WINDOW_KEY = "memory.short_term_window"
#: 창의 메시지 수 — 대화 아카이브(호스트)가 워터마크 기본값으로 읽는다.
WINDOW_LEN_KEY = "memory.short_term_window_len"

DEFAULT_FULL_TURNS = 2
DEFAULT_DIALOGUE_TURNS = 3
#: 창 전체 상한(문자). 200K 모델의 Hermes tail(≈20K 토큰)의 절반쯤 — 이 창은 "최근 5턴" 이지
#: "맞는 만큼 전부" 가 아니다.
DEFAULT_MAX_CHARS = 40_000
#: 가까운 턴의 tool_result 본문이 이보다 길면 앞머리만 남긴다.
DEFAULT_RESULT_TRIM_OVER = 4_000
DEFAULT_RESULT_KEEP = 1_200
#: 예산 압박 때 결과를 이보다 짧게는 줄이지 않는다.
MIN_RESULT_KEEP = 300
#: 먼 턴의 발화 상한(사용자·답변 각각).
DEFAULT_DIALOGUE_MESSAGE_CHARS = 4_000
MIN_DIALOGUE_MESSAGE_CHARS = 400
#: STM 에서 읽어 오는 행 상한 — 한 턴이 도구 40행일 수 있다.
SCAN_ROWS = 400
#: 첫 시도 행 수. STM 행은 도구 결과 **원문** 을 들고 있어 400행을 늘 읽으면 버릴 데이터를
#: 매 턴 다 읽는다(턴 시작 지연에 그대로 얹힌다). 작게 읽어 보고, 필요한 턴 수를 못 덮었을
#: 때만 상한까지 한 번 더 간다.
SCAN_FIRST = 96

_TRIM_NOTE = "…[+{n} chars trimmed from an earlier turn]"
_IMAGE_NOTE = "[image removed from an earlier turn]"
#: 창이 user 메시지로 끝나면 이번 턴의 사용자 지시와 맞붙어 user 가 둘 연속이 된다. 끊긴 턴
#: (도구 호출 중 중단)이 딱 그 모양을 만든다. 합성 tool_result 와 같은 계열의 표식 한 줄로
#: 창을 닫아 역할이 교대하게 둔다 — 백엔드마다 연속 역할 허용치가 다르다.
_INTERRUPTED_TAIL = "[The previous turn was interrupted before it finished.]"


@dataclass
class WindowConfig:
    full_turns: int = DEFAULT_FULL_TURNS
    dialogue_turns: int = DEFAULT_DIALOGUE_TURNS
    max_chars: int = DEFAULT_MAX_CHARS
    result_trim_over: int = DEFAULT_RESULT_TRIM_OVER
    result_keep: int = DEFAULT_RESULT_KEEP
    dialogue_message_chars: int = DEFAULT_DIALOGUE_MESSAGE_CHARS
    used_tools_line: bool = True

    @classmethod
    def from_hooks(cls, hooks: Any) -> "WindowConfig":
        g = lambda name, default: int(getattr(hooks, name, default) or 0)  # noqa: E731
        return cls(
            full_turns=max(0, g("window_full_turns", DEFAULT_FULL_TURNS)),
            dialogue_turns=max(0, g("window_dialogue_turns", DEFAULT_DIALOGUE_TURNS)),
            max_chars=max(1_000, g("window_max_chars", DEFAULT_MAX_CHARS)),
            result_trim_over=max(200, g("window_result_trim_over", DEFAULT_RESULT_TRIM_OVER)),
            result_keep=max(MIN_RESULT_KEEP, g("window_result_keep", DEFAULT_RESULT_KEEP)),
            dialogue_message_chars=max(
                MIN_DIALOGUE_MESSAGE_CHARS,
                g("window_dialogue_message_chars", DEFAULT_DIALOGUE_MESSAGE_CHARS),
            ),
            used_tools_line=bool(getattr(hooks, "window_used_tools_line", True)),
        )

    @property
    def enabled(self) -> bool:
        return self.full_turns + self.dialogue_turns > 0


@dataclass
class WindowReport:
    turns: int = 0
    full: int = 0
    dialogue: int = 0
    chars: int = 0
    degraded: List[str] = field(default_factory=list)

    def as_event(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "full": self.full,
            "dialogue": self.dialogue,
            "chars": self.chars,
            "degraded": list(self.degraded),
        }


# ── 텍스트 도우미 ────────────────────────────────────────────────────────────


def _text_of(content: Any) -> str:
    return "\n".join(
        str(b.get("text", "")).strip() for b in _blocks(content) if b.get("type") == "text"
    ).strip()


def _clip_text(s: str, limit: int) -> str:
    s = str(s or "")
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + _TRIM_NOTE.format(n=len(s) - limit)


def _msg_role(m: Any) -> str:
    return str(getattr(m, "role", "") or (m.get("role") if isinstance(m, dict) else "") or "")


def _msg_content(m: Any) -> Any:
    if isinstance(m, dict):
        return m.get("content", "")
    return getattr(m, "content", "")


def _size(messages: Sequence[Dict[str, Any]]) -> int:
    try:
        return len(json.dumps(messages, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return sum(len(str(m)) for m in messages)


# ── 턴 하나 → 메시지들 ─────────────────────────────────────────────────────────


def _tool_stats(turn: LogicalTurn) -> List[Tuple[str, int, int]]:
    """턴 안의 도구 사용을 (이름, 호출 수, 실패 수) 로 — 호출 순서 유지."""
    results: Dict[str, bool] = {}
    for m in turn.messages:
        for b in _blocks(_msg_content(m)):
            if b.get("type") == "tool_result":
                results[str(b.get("tool_use_id") or "")] = bool(b.get("is_error"))
    order: List[str] = []
    calls: Dict[str, int] = {}
    fails: Dict[str, int] = {}
    for m in turn.messages:
        if _msg_role(m) != "assistant":
            continue
        for b in _blocks(_msg_content(m)):
            if b.get("type") != "tool_use":
                continue
            name = str(b.get("name") or "tool")
            if name not in calls:
                order.append(name)
                calls[name] = 0
                fails[name] = 0
            calls[name] += 1
            if results.get(str(b.get("id") or ""), False):
                fails[name] += 1
    return [(n, calls[n], fails[n]) for n in order]


def used_tools_line(turn: LogicalTurn) -> str:
    """``[used tools: ForgeTool ×4 (3 failed), Bash ×2]`` — 도구를 안 썼으면 빈 문자열."""
    stats = _tool_stats(turn)
    if not stats:
        return ""
    parts = []
    for name, n, f in stats:
        part = f"{name} ×{n}" if n > 1 else name
        if f:
            part += f" ({f} failed)"
        parts.append(part)
    return "[used tools: " + ", ".join(parts) + "]"


def _dialogue_messages(
    turn: LogicalTurn, cfg: WindowConfig, *, message_chars: int
) -> List[Dict[str, Any]]:
    """먼 턴: 사용자 텍스트 + 최종 답변 텍스트(+ used tools 한 줄). 도구 블록은 양쪽 다 뺀다."""
    user_text = ""
    final_text = ""
    last_nonempty = ""
    for m in turn.messages:
        role = _msg_role(m)
        content = _msg_content(m)
        if role == "user":
            if _is_tool_result_only(content):
                continue
            if not user_text:
                user_text = _text_of(content)
        elif role == "assistant":
            t = _text_of(content)
            if t:
                last_nonempty = t
            final_text = t  # 마지막 assistant 메시지의 텍스트가 최종 답변이다(비어 있을 수 있다)
    if not user_text:
        return []
    answer = final_text or last_nonempty
    line = used_tools_line(turn) if cfg.used_tools_line else ""
    answer = _clip_text(answer, message_chars)
    if line:
        answer = f"{answer}\n{line}" if answer else line
    out: List[Dict[str, Any]] = [{"role": "user", "content": _clip_text(user_text, message_chars)}]
    if answer:
        out.append({"role": "assistant", "content": answer})
    return out


def _shrink_result_block(block: Dict[str, Any], *, trim_over: int, keep: int) -> Dict[str, Any]:
    """tool_result 블록의 내용만 줄인다. 블록·id 는 그대로."""
    content = block.get("content")
    if isinstance(content, str):
        if len(content) > trim_over:
            return {**block, "content": _clip_text(content, keep)}
        return block
    if isinstance(content, list):
        new_parts: List[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                new_parts.append({"type": "text", "text": _IMAGE_NOTE})
                continue
            if isinstance(part, dict) and part.get("type") == "text":
                text = str(part.get("text", ""))
                if len(text) > trim_over:
                    part = {**part, "text": _clip_text(text, keep)}
            new_parts.append(part)
        return {**block, "content": new_parts}
    return block


def _full_messages(
    turn: LogicalTurn, cfg: WindowConfig, *, result_keep: int
) -> List[Dict[str, Any]]:
    """가까운 턴: 순서·블록 그대로. 큰 결과와 이미지만 줄이고, thinking 은 뺀다."""
    out: List[Dict[str, Any]] = []
    for m in turn.messages:
        role = _msg_role(m)
        if role not in ("user", "assistant"):
            continue
        content = _msg_content(m)
        if isinstance(content, str):
            if content.strip():
                out.append({"role": role, "content": content})
            continue
        blocks: List[Dict[str, Any]] = []
        for b in _blocks(content):
            btype = b.get("type")
            if btype in ("thinking", "redacted_thinking"):
                continue
            if btype == "tool_result":
                b = _shrink_result_block(b, trim_over=cfg.result_trim_over, keep=result_keep)
            elif btype == "image":
                b = {"type": "text", "text": _IMAGE_NOTE}
            blocks.append(b)
        if blocks:
            out.append({"role": role, "content": blocks})
    return out


def _is_plain(msg: Dict[str, Any]) -> bool:
    """도구 블록이 없는 순수 발화인가 — 합쳐도 짝이 깨지지 않는 메시지."""
    content = msg.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return all(isinstance(b, dict) and b.get("type") == "text" for b in content)
    return False


def _coalesce_plain_roles(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """같은 role 이 연속된 **순수 발화** 를 하나로 합친다.

    답변 없이 끝난 턴(사용자가 중간에 멈춘 턴)이 창에 들어오면 user 메시지가 연달아 놓인다.
    도구 블록이 낀 메시지는 건드리지 않는다 — tool_use/tool_result 짝과 블록 순서 규칙을
    건드리는 것이 연속 role 보다 위험하다.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if out and out[-1].get("role") == msg.get("role") and _is_plain(out[-1]) and _is_plain(msg):
            prev_text = _text_of(out[-1].get("content"))
            next_text = _text_of(msg.get("content"))
            merged = "\n\n".join(t for t in (prev_text, next_text) if t)
            out[-1] = {"role": msg.get("role"), "content": merged}
            continue
        out.append(msg)
    return out


def _sanitize_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs = strip_leading_orphan_tool_results(list(messages))
    repair_dangling_tool_calls(msgs)
    msgs = _coalesce_plain_roles(msgs)
    # 창은 assistant 로 끝나야 이번 턴의 사용자 지시와 역할이 교대한다. 끊긴 턴을 복구하면
    # 마지막이 합성 tool_result(= user)가 된다.
    if msgs and str(msgs[-1].get("role") or "") == "user":
        msgs.append({"role": "assistant", "content": _INTERRUPTED_TAIL})
    return msgs


# ── 창 조립 ────────────────────────────────────────────────────────────────


def build_window(
    turns: Sequence[Any], cfg: Optional[WindowConfig] = None
) -> Tuple[List[Dict[str, Any]], WindowReport]:
    """시간순 STM turn 목록 → (messages, 보고). 현재 턴은 STM 에 아직 없으므로 들어오지 않는다."""
    cfg = cfg or WindowConfig()
    report = WindowReport()
    if not cfg.enabled:
        return [], report
    groups = group_logical_turns(turns, cfg.full_turns + cfg.dialogue_turns)
    if not groups:
        return [], report
    n_full = min(cfg.full_turns, len(groups))
    full = groups[len(groups) - n_full :] if n_full else []
    dialogue = groups[: len(groups) - n_full]

    result_keep = cfg.result_keep
    message_chars = cfg.dialogue_message_chars

    def _trimmable(keep: int) -> bool:
        """더 깎을 tool_result 가 남았는가 — 없으면 result_keep 을 줄여도 크기가 안 준다."""
        for turn in full:
            for m in turn.messages:
                for b in _blocks(_msg_content(m)):
                    if b.get("type") != "tool_result":
                        continue
                    body = b.get("content")
                    if isinstance(body, str) and len(body) > keep:
                        return True
                    if isinstance(body, list):
                        for part in body:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "text"
                                and len(str(part.get("text", ""))) > keep
                            ):
                                return True
        return False

    def _assemble() -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = []
        for t in dialogue:
            msgs.extend(_dialogue_messages(t, cfg, message_chars=message_chars))
        for t in full:
            msgs.extend(_full_messages(t, cfg, result_keep=result_keep))
        return _sanitize_pairs(msgs)

    msgs = _assemble()
    # 예산 강등 — 순서가 계약이다(모듈 docstring).
    guard = 0
    while _size(msgs) > cfg.max_chars and guard < 40:
        guard += 1
        if result_keep > MIN_RESULT_KEEP and _trimmable(result_keep):
            result_keep = max(MIN_RESULT_KEEP, result_keep // 2)
            report.degraded.append(f"result_keep={result_keep}")
        elif dialogue:
            dialogue = dialogue[1:]
            report.degraded.append("drop_oldest_dialogue")
        elif len(full) > 1:
            demoted = full[0]
            full = full[1:]
            dialogue = [demoted]
            report.degraded.append("demote_oldest_full")
        elif message_chars > MIN_DIALOGUE_MESSAGE_CHARS:
            message_chars = max(MIN_DIALOGUE_MESSAGE_CHARS, message_chars // 2)
            report.degraded.append(f"message_chars={message_chars}")
        else:
            break  # T-1 하나만 남았고 최소까지 줄였다 — 그대로 둔다(턴을 버리지 않는다)
        msgs = _assemble()
    report.turns = len(full) + len(dialogue)
    report.full = len(full)
    report.dialogue = len(dialogue)
    report.chars = _size(msgs)
    return msgs, report


def _turn_starts(rows: Sequence[Any]) -> int:
    """논리 턴을 여는 행(도구 결과가 아닌 user)의 수."""
    count = 0
    for row in rows:
        role = str(getattr(row, "role", "") or "")
        if role == "user" and not _is_tool_result_only(getattr(row, "content", "")):
            count += 1
    return count


async def load_window(
    provider: Any, cfg: Optional[WindowConfig] = None
) -> Tuple[List[Dict[str, Any]], WindowReport]:
    """provider 의 STM 에서 창을 읽어 조립한다. STM 이 없거나 실패하면 빈 창.

    적재는 **필요한 만큼만**: STM 한 행은 도구 결과 원문을 들고 있어서(수백 KB 가 될 수 있다)
    상한을 늘 읽으면 버릴 데이터를 매 턴 다 읽는다. 작게 읽어 필요한 턴 수를 덮었는지 보고,
    못 덮었을 때만 상한까지 한 번 더 읽는다.
    """
    cfg = cfg or WindowConfig()
    if not cfg.enabled or provider is None:
        return [], WindowReport()
    stm_fn = getattr(provider, "stm", None)
    if not callable(stm_fn):
        return [], WindowReport()
    needed = cfg.full_turns + cfg.dialogue_turns
    try:
        stm = stm_fn()
        rows = list(await stm.recent(n=SCAN_FIRST) or [])
        # 행이 요청보다 적으면 STM 전체를 읽은 것이고, 턴 시작이 needed 보다 많으면 가장
        # 오래된 턴까지 온전히 덮었다. 둘 다 아니면 상한까지 한 번 더.
        if len(rows) >= SCAN_FIRST and _turn_starts(rows) <= needed:
            rows = list(await stm.recent(n=SCAN_ROWS) or [])
    except Exception:  # noqa: BLE001 — 창이 없어도 턴은 돈다
        return [], WindowReport()
    return build_window(rows, cfg)


__all__ = [
    "WINDOW_KEY",
    "WINDOW_LEN_KEY",
    "WindowConfig",
    "WindowReport",
    "build_window",
    "load_window",
    "used_tools_line",
]
