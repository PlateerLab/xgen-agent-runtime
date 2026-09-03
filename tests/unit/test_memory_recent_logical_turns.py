"""L0(recent_turns) 는 **논리 턴** 을 세고, 그 사이 도구는 한 줄로 요약한다.

이 파일이 막으려는 회귀는 하나다: **에이전트가 두 턴 전 자기 행동을 잊고 같은
호출을 반복하는 것.**

예전에는 L0 가 STM **행 수** 를 셌다. 도구를 한 번 쓰면 그 턴은 행 4개(지시 ·
tool_use · tool_result · 답변)가 되므로 "최근 6개" 는 1.5턴이었고, 가져온 6개 중
도구 행은 렌더에서 버려져 **자리만 차지했다**. 3번째 턴에서 첫 지시가 통째로
사라졌다 — 실측으로 3턴 12행 중 살아남은 줄이 3개였다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from xgen_agent_runtime.memory.provider import MemoryHooks, Turn
from xgen_agent_runtime.memory.retriever import _scan_rows
from xgen_agent_runtime.memory.transcript import (
    group_logical_turns,
    render_recent_turns,
)


def _t(role: str, content: Any) -> Turn:
    return Turn(role=role, content=content, timestamp=datetime.now(timezone.utc))


def _tool_use(tid: str, name: str, **inp: Any) -> Dict[str, Any]:
    return {"type": "tool_use", "id": tid, "name": name, "input": dict(inp)}


def _tool_result(tid: str, body: str, *, error: bool = False) -> Dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tid, "content": body, "is_error": error}


def _turn(i: int, calls: List[tuple]) -> List[Turn]:
    """한 논리 턴 = 사용자 지시 + 도구 왕복들 + 최종 답변."""
    out = [_t("user", f"지시{i}")]
    for j, (name, inp, res, err) in enumerate(calls):
        tid = f"t{i}_{j}"
        out.append(_t("assistant", [_tool_use(tid, name, **inp)]))
        out.append(_t("user", [_tool_result(tid, res, error=err)]))
    out.append(_t("assistant", [{"type": "text", "text": f"답변{i}"}]))
    return out


# ── 턴 세기 ─────────────────────────────────────────────────────────────


def test_a_turn_is_one_turn_no_matter_how_many_tools_it_used() -> None:
    rows = _turn(1, [("Shell", {"command": "ls"}, "a", False)] * 5)
    assert len(rows) == 12  # 지시 + (tool_use+tool_result)*5 + 답변
    assert len(group_logical_turns(rows, 3)) == 1


def test_tool_results_do_not_start_a_new_turn() -> None:
    """tool_result 도 user 메시지다 — role 만 보면 턴이 부풀어 오른다."""
    rows = _turn(1, [("Shell", {"command": "ls"}, "a", False)])
    groups = group_logical_turns(rows, 5)
    assert len(groups) == 1


def test_three_user_inputs_always_survive() -> None:
    rows = _turn(1, [("Shell", {"command": "ls"}, "a", False)])
    rows += _turn(2, [("Shell", {"command": "cat a"}, "b", False)])
    rows += _turn(3, [("Shell", {"command": "cat c"}, "d", False)])

    body = render_recent_turns(rows, limit=3, max_chars=4000)

    for i in (1, 2, 3):
        assert f"지시{i}" in body, f"턴{i} 의 사용자 지시가 사라졌다"
        assert f"답변{i}" in body


def test_older_turns_beyond_the_limit_are_dropped() -> None:
    rows = _turn(1, []) + _turn(2, []) + _turn(3, []) + _turn(4, [])
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert "지시1" not in body
    assert "지시4" in body


def test_a_dangling_tail_without_a_user_instruction_is_dropped() -> None:
    """사용자 지시 없이 답만 떠 있는 조각은 맥락이 아니다."""
    rows = [_t("assistant", [{"type": "text", "text": "이전 턴 잔여"}])] + _turn(1, [])
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert "이전 턴 잔여" not in body
    assert "지시1" in body


# ── 도구 요약 ───────────────────────────────────────────────────────────


def test_tool_calls_survive_as_one_line_each() -> None:
    """예전에는 도구 행이 자리만 차지하고 렌더에서 통째로 버려졌다."""
    rows = _turn(1, [("Shell", {"command": "ls -al /work"}, "a.txt b.txt", False)])
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert "[tool] Shell(" in body
    assert "ls -al /work" in body, "무엇을 불렀는지 없으면 같은 호출을 또 한다"
    assert "a.txt" in body, "결과가 없으면 답을 잊고 다시 부른다"


def test_arguments_distinguish_two_calls_of_the_same_tool() -> None:
    """도구 이름만 남기면 '같은 인자로 또 불렀는지' 를 알 수 없다."""
    rows = _turn(1, [
        ("Shell", {"command": "cat a"}, "A", False),
        ("Shell", {"command": "cat b"}, "B", False),
    ])
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert "cat a" in body and "cat b" in body


def test_errors_are_marked_so_the_agent_does_not_retry_blindly() -> None:
    rows = _turn(1, [("Shell", {"command": "cat nope"}, "No such file", False or True)])
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert "error:" in body
    assert "No such file" in body


def test_repeated_identical_calls_collapse_with_a_count() -> None:
    """반복은 그 자체가 신호다 — 세 줄로 늘어놓으면 예산만 먹는다."""
    rows = _turn(1, [("Read", {"file_path": "/a"}, "x", False)] * 3)
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert body.count("[tool] Read(") == 1
    assert "×3" in body


def test_different_calls_are_not_collapsed() -> None:
    rows = _turn(1, [
        ("Read", {"file_path": "/a"}, "x", False),
        ("Read", {"file_path": "/b"}, "y", False),
    ])
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert body.count("[tool] Read(") == 2
    assert "×" not in body


def test_long_values_say_how_much_was_cut() -> None:
    """조용히 자르면 잘린 것을 전부로 읽는다 — 도구 결과에서 특히 위험하다."""
    rows = _turn(1, [("Shell", {"command": "cat big"}, "x" * 5000, False)])
    body = render_recent_turns(rows, limit=3, max_chars=4000)
    assert "…[+" in body


# ── 예산 ────────────────────────────────────────────────────────────────


def test_budget_shrinks_lines_but_never_drops_a_turn() -> None:
    """'최근 3턴' 이 예산에 따라 조건부라면 그건 계약이 아니다."""
    rows = _turn(1, [("Shell", {"command": "a" * 400}, "r" * 2000, False)])
    rows += _turn(2, [("Shell", {"command": "b" * 400}, "r" * 2000, False)])
    rows += _turn(3, [("Shell", {"command": "c" * 400}, "r" * 2000, False)])

    body = render_recent_turns(rows, limit=3, max_chars=600)

    assert len(body) <= 600
    for i in (1, 2, 3):
        assert f"지시{i}" in body, f"예산 때문에 턴{i} 을 버렸다"


def test_tool_lines_are_sacrificed_before_user_and_assistant_text() -> None:
    """도구 줄은 요약이라 복구 가능한 손실이고, 지시와 답변은 그렇지 않다."""
    rows = _turn(1, [("Shell", {"command": "x" * 300}, "y" * 300, False)])
    rows += _turn(2, [("Shell", {"command": "x" * 300}, "y" * 300, False)])
    rows += _turn(3, [("Shell", {"command": "keep-me"}, "kept", False)])

    body = render_recent_turns(rows, limit=3, max_chars=260)

    for i in (1, 2, 3):
        assert f"지시{i}" in body
    # 가장 오래된 턴의 도구 줄부터 사라진다.
    assert body.count("[tool]") < 3


# ── 훑는 행 수 ──────────────────────────────────────────────────────────


def test_scan_rows_fetches_far_more_rows_than_turns() -> None:
    """한 턴이 몇 행인지는 미리 알 수 없다 — 행 수를 턴 수로 쓰면 안 된다."""
    h = MemoryHooks()
    assert h.recent_turns == 3
    assert _scan_rows(h) >= 80
    assert _scan_rows(h) > h.recent_turns


def test_scan_rows_scales_when_a_host_asks_for_many_turns() -> None:
    h = MemoryHooks(recent_turns=40, recent_scan_rows=10)
    assert _scan_rows(h) >= 160


def test_zero_turns_renders_nothing() -> None:
    rows = _turn(1, [])
    assert render_recent_turns(rows, limit=0, max_chars=4000) == ""
