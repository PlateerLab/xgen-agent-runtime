"""단기 기억 창 — 가까운 2턴은 도구까지, 먼 3턴은 대화만, 합쳐 5턴을 messages 로."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

from xgen_agent_runtime.memory.short_term_window import (  # noqa: F401
    SCAN_MAX,
    SCAN_ROWS,
    WindowConfig,
    build_window,
    load_window,
    used_tools_line,
)


def _t(role: str, content: Any) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


def _turn(n: int, *, tools: int = 1, fail: int = 0, big_result: int = 0, thinking: bool = False) -> List[Any]:
    """논리 턴 하나: 사용자 지시 → (tool_use → tool_result)×tools → 최종 답변."""
    msgs: List[Any] = [_t("user", f"request {n}")]
    for i in range(tools):
        tid = f"toolu_{n}_{i}"
        blocks: List[Dict[str, Any]] = []
        if thinking:
            blocks.append({"type": "thinking", "thinking": "hmm", "signature": "sig"})
        blocks.append({"type": "text", "text": f"working {n}.{i}"})
        blocks.append({"type": "tool_use", "id": tid, "name": "ForgeTool" if i == 0 else "Bash", "input": {"k": i}})
        msgs.append(_t("assistant", blocks))
        body = ("x" * big_result) if big_result else f"result {n}.{i}"
        msgs.append(
            _t("user", [{"type": "tool_result", "tool_use_id": tid, "content": body, "is_error": i < fail}])
        )
    msgs.append(_t("assistant", [{"type": "text", "text": f"answer {n}"}]))
    return msgs


def _turns(n: int, **kw) -> List[Any]:
    out: List[Any] = []
    for i in range(1, n + 1):
        out.extend(_turn(i, **kw))
    return out


def _tool_use_ids(msgs: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            ids += [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
    return ids


def _tool_result_ids(msgs: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            ids += [b["tool_use_id"] for b in m["content"] if b.get("type") == "tool_result"]
    return ids


class TestShape:
    def test_five_turns_two_full_three_dialogue(self):
        msgs, report = build_window(_turns(6, tools=2, fail=1))
        assert (report.turns, report.full, report.dialogue) == (5, 2, 3)
        # 먼 3턴: user 텍스트 + assistant 텍스트(도구 한 줄 포함), 도구 블록 없음
        dialogue = msgs[:6]
        assert [m["role"] for m in dialogue] == ["user", "assistant"] * 3
        assert all(isinstance(m["content"], str) for m in dialogue)
        assert dialogue[0]["content"] == "request 2", "T-6 은 빠지고 T-5 부터"
        assert dialogue[1]["content"] == "answer 2\n[used tools: ForgeTool (1 failed), Bash]"
        # 가까운 2턴: 순서·블록·id 그대로
        full = msgs[6:]
        assert full[0]["content"] == "request 5"
        assert _tool_use_ids(full) == ["toolu_5_0", "toolu_5_1", "toolu_6_0", "toolu_6_1"]
        assert _tool_result_ids(full) == _tool_use_ids(full)
        assert full[-1] == {"role": "assistant", "content": [{"type": "text", "text": "answer 6"}]}

    def test_fewer_turns_than_the_window_are_all_kept(self):
        msgs, report = build_window(_turns(3))
        assert (report.turns, report.full, report.dialogue) == (3, 2, 1)
        assert msgs[0]["content"] == "request 1"

    def test_no_turns_is_an_empty_window(self):
        assert build_window([]) == ([], build_window([])[1])

    def test_thinking_blocks_are_not_replayed(self):
        msgs, _ = build_window(_turns(2, thinking=True))
        for m in msgs:
            if isinstance(m["content"], list):
                assert all(b.get("type") not in ("thinking", "redacted_thinking") for b in m["content"])

    def test_the_current_turn_is_not_in_stm_so_only_previous_turns_appear(self):
        """STM 은 지난 턴만 담는다 — 창은 현재 사용자 메시지를 만들지 않는다."""
        msgs, _ = build_window(_turns(1))
        assert msgs[0]["content"] == "request 1" and msgs[-1]["content"][0]["text"] == "answer 1"


class TestUsedToolsLine:
    def test_counts_and_failures(self):
        from xgen_agent_runtime.memory.transcript import group_logical_turns

        turn = group_logical_turns(_turn(1, tools=3, fail=2), 1)[0]
        assert used_tools_line(turn) == "[used tools: ForgeTool (1 failed), Bash ×2 (1 failed)]"

    def test_no_tools_no_line(self):
        from xgen_agent_runtime.memory.transcript import group_logical_turns

        turn = group_logical_turns([_t("user", "q"), _t("assistant", "a")], 1)[0]
        assert used_tools_line(turn) == ""

    def test_line_can_be_turned_off(self):
        msgs, _ = build_window(_turns(4), WindowConfig(used_tools_line=False))
        assert msgs[1]["content"] == "answer 1"


class TestResultTrimming:
    def test_big_results_in_full_turns_keep_the_head_with_a_marker(self):
        msgs, _ = build_window(_turns(1, big_result=10_000), WindowConfig(result_trim_over=4000, result_keep=1200))
        res = [b for m in msgs if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"][0]
        assert res["content"].startswith("x" * 1200) and "chars trimmed" in res["content"]
        assert len(res["content"]) < 1300

    def test_small_results_are_verbatim(self):
        msgs, _ = build_window(_turns(1))
        res = [b for m in msgs if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"][0]
        assert res["content"] == "result 1.0"


def _dialogue_turn(n: int, *, answer_chars: int) -> List[Any]:
    return [_t("user", f"request {n}"), _t("assistant", [{"type": "text", "text": "y" * answer_chars}])]


def _heavy_turn(n: int, *, tools: int, input_chars: int, result_chars: int) -> List[Any]:
    """도구를 많이 쓴 턴: 긴 명령(입력)과 긴 결과. 그것만으로 예산을 넘긴다."""
    msgs: List[Any] = [_t("user", f"request {n}")]
    for i in range(tools):
        tid = f"toolu_{n}_{i}"
        msgs.append(_t("assistant", [
            {"type": "text", "text": f"step {i}"},
            {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": "c" * input_chars}},
        ]))
        msgs.append(_t("user", [{"type": "tool_result", "tool_use_id": tid, "content": "r" * result_chars}]))
    msgs.append(_t("assistant", [{"type": "text", "text": f"answer {n}: 1. a 2. b 3. c"}]))
    return msgs


def _user_texts(msgs: List[Dict[str, Any]]) -> List[str]:
    out = []
    for m in msgs:
        if m["role"] != "user":
            continue
        c = m["content"]
        if isinstance(c, str):
            out.append(c)
        else:
            out += [b["text"] for b in c if b.get("type") == "text"]
    return out


class TestBudgetDegradation:
    def test_order_results_inputs_demote_then_dialogue(self):
        cfg = WindowConfig(max_chars=2_500, result_trim_over=200, result_keep=1200)
        msgs, report = build_window(_turns(6, tools=2, big_result=3_000), cfg)
        assert report.degraded and report.degraded[0].startswith("result_keep=")
        assert "demote_oldest_full" in report.degraded
        # 결과를 다 줄이기 전에는 턴을 강등하지 않는다
        first_demote = report.degraded.index("demote_oldest_full")
        assert all(d.startswith("result_keep=") for d in report.degraded[:first_demote])
        assert msgs[0]["content"].startswith("request")
        # T-1 은 끝까지 남는다
        assert any(m["content"] == "request 6" for m in msgs)

    def test_t1_is_never_dropped(self):
        cfg = WindowConfig(max_chars=1_000)
        msgs, report = build_window(_turns(6, tools=3, big_result=2_000), cfg)
        assert report.full == 1
        assert "request 6" in _user_texts(msgs)

    def test_dialogue_survives_when_t1_alone_overflows(self):
        """2026-10-20 dev: 도구 49번 턴(줄인 뒤 11.8만 자) 뒤에 창이 T-1 하나만 남겼다.
        몇백 자짜리 지난 질문과 답은 버려도 예산이 맞지 않는다. 버리지 않는다."""
        turns = [
            *_dialogue_turn(1, answer_chars=300),
            *_dialogue_turn(2, answer_chars=300),
            *_dialogue_turn(3, answer_chars=300),
            *_dialogue_turn(4, answer_chars=300),
            *_heavy_turn(5, tools=40, input_chars=3_000, result_chars=5_000),
        ]
        msgs, report = build_window(turns, WindowConfig(max_chars=20_000))
        assert report.turns == 5 and report.full == 1 and report.dialogue == 4
        assert "drop_oldest_dialogue" not in report.degraded
        assert _user_texts(msgs)[:5] == [f"request {i}" for i in range(1, 6)]
        # T-1 의 최종 답변은 줄이지 않는다
        assert msgs[-1]["content"][0]["text"] == "answer 5: 1. a 2. b 3. c"

    def test_tool_use_inputs_shrink_but_pairs_hold(self):
        turns = _heavy_turn(1, tools=10, input_chars=6_000, result_chars=100)
        msgs, report = build_window(turns, WindowConfig(max_chars=8_000))
        assert any(d.startswith("input_keep=") for d in report.degraded)
        uses = [b for m in msgs if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use"]
        assert len(uses) == 10
        assert all(len(b["input"]["command"]) < 6_000 and "chars trimmed" in b["input"]["command"] for b in uses)
        assert _tool_use_ids(msgs) == _tool_result_ids(msgs)

    def test_inputs_are_verbatim_without_budget_pressure(self):
        turns = _heavy_turn(1, tools=2, input_chars=3_000, result_chars=10)
        msgs, report = build_window(turns)
        assert not report.degraded
        uses = [b for m in msgs if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use"]
        assert all(b["input"]["command"] == "c" * 3_000 for b in uses)

    def test_oversized_dialogue_is_shortened_first(self):
        turns = [*(_dialogue_turn(i, answer_chars=4_000) for i in range(1, 5))]
        turns = [m for t in turns for m in t] + _turn(5) + _turn(6)
        msgs, report = build_window(turns, WindowConfig(max_chars=3_000))
        assert any(d.startswith("message_chars=") for d in report.degraded)
        assert "drop_oldest_dialogue" not in report.degraded  # 줄여서 맞으면 턴은 버리지 않는다
        assert report.turns == 5 and report.chars <= 3_000

    def test_dialogue_is_dropped_oldest_first_when_shortening_is_not_enough(self):
        turns = [*(_dialogue_turn(i, answer_chars=4_000) for i in range(1, 5))]
        turns = [m for t in turns for m in t] + _turn(5) + _turn(6)
        msgs, report = build_window(turns, WindowConfig(max_chars=1_500))
        assert f"message_chars={400}" in report.degraded
        assert "drop_oldest_dialogue" in report.degraded
        assert "request 6" in _user_texts(msgs) and "request 1" not in _user_texts(msgs)


class TestPairInvariant:
    def test_a_tool_use_without_a_result_gets_a_synthetic_result(self):
        turns = _turns(1)
        # 결과 없이 끊긴 tool_use 를 마지막 턴 끝에 붙인다
        turns.append(_t("assistant", [{"type": "tool_use", "id": "toolu_cut", "name": "Bash", "input": {}}]))
        msgs, _ = build_window(turns)
        assert "toolu_cut" in _tool_result_ids(msgs)

    def test_no_orphan_result_leads_the_window(self):
        turns = [_t("user", [{"type": "tool_result", "tool_use_id": "ghost", "content": "x"}])] + _turns(1)
        msgs, _ = build_window(turns)
        assert msgs[0]["role"] == "user" and msgs[0]["content"] == "request 1"


class TestLoadWindow:
    def test_reads_stm_recent_and_builds(self):
        class _STM:
            async def recent(self, n=20):
                # 첫 적재는 작은 쪽(SCAN_FIRST) — 도구 결과 원문을 매 턴 다 읽지 않는다.
                assert 0 < n <= SCAN_ROWS
                return _turns(2)

        class _P:
            def stm(self):
                return _STM()

        msgs, report = asyncio.run(load_window(_P()))
        assert report.turns == 2 and msgs[0]["content"] == "request 1"

    def test_a_turn_longer_than_the_scan_still_comes_back_whole(self):
        """반복 200번짜리 턴은 400행을 넘는다. 그 턴의 사용자 지시가 범위 밖이면 창에서 통째로 빠졌다."""
        rows = _turns(1) + _turn(2, tools=300)  # T-1 = 1 + 600 + 1 행
        asked: List[int] = []

        class _STM:
            async def recent(self, n=20):
                asked.append(n)
                return rows[-n:]

        class _P:
            def stm(self):
                return _STM()

        msgs, report = asyncio.run(load_window(_P()))
        assert asked[-1] == SCAN_MAX and len(asked) == 3
        assert "request 2" in _user_texts(msgs)

    def test_ordinary_turns_do_not_read_the_wide_scan(self):
        asked: List[int] = []

        class _STM:
            async def recent(self, n=20):
                asked.append(n)
                return _turns(8, tools=30)[-n:]

        class _P:
            def stm(self):
                return _STM()

        asyncio.run(load_window(_P()))
        assert SCAN_MAX not in asked

    def test_provider_without_stm_or_failing_stm_gives_nothing(self):
        assert asyncio.run(load_window(object())) == ([], asyncio.run(load_window(object()))[1])

        class _Bad:
            def stm(self):
                raise RuntimeError("down")

        assert asyncio.run(load_window(_Bad()))[0] == []

    def test_disabled_by_hooks(self):
        cfg = WindowConfig.from_hooks(SimpleNamespace(window_full_turns=0, window_dialogue_turns=0))
        assert not cfg.enabled and build_window(_turns(3), cfg) == ([], build_window(_turns(3), cfg)[1])

    def test_json_serialisable(self):
        msgs, _ = build_window(_turns(3, thinking=True, big_result=6000))
        json.dumps(msgs)
