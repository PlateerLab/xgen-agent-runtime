"""압축·이중 기록·역할 교대 — 2026-09-21 검토에서 나온 결함들의 회귀 시험.

셋 다 "기억이 조용히 사라지거나 두 배가 된다" 계열이라, 증상이 다음 턴에야 보인다.
그래서 단언은 전부 **행 수** 와 **역할 배열** 같은 눈에 보이는 것으로 둔다.
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.core.compaction import (
    RECORDED_INDEX_KEYS,
    reconcile_recorded_index,
)
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.memory.short_term_window import WindowConfig, build_window
from xgen_agent_runtime.memory.strategy import ProviderDrivenStrategy, _RECORDED_KEY
from xgen_agent_runtime.stages.s18_memory.artifact.default.stage import MemoryStage


class _Row:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class _STM:
    def __init__(self):
        self.rows = []

    async def append(self, turn):
        self.rows.append(turn)

    async def recent(self, n=20):
        return list(self.rows)[-n:]


class _Provider:
    def __init__(self):
        self.recorded = []
        self._stm = _STM()

    def stm(self):
        return self._stm

    async def record_turn(self, turn):
        self.recorded.append((turn.role, str(turn.content)[:40]))

    async def record_execution(self, summary):  # pragma: no cover - 종료 경로 아님
        raise AssertionError("이 시험은 터미널 경로를 타지 않는다")

    async def reflect(self, ctx):
        return []

    def set_hooks(self, hooks):
        return None


def _state(messages, **meta):
    state = PipelineState(session_id="s1")
    state.messages = list(messages)
    state.metadata.update(meta)
    return state


class TestCompactionKeepsEveryWatermark:
    """압축이 턴 도중에 돌아도 그 턴은 STM·아카이브에 남아야 한다."""

    def test_every_message_index_key_is_translated(self):
        window = [{"role": "user", "content": f"과거{i}"} for i in range(12)]
        current = [
            {"role": "user", "content": "현재 지시"},
            {"role": "assistant", "content": "현재 답변"},
        ]
        before = window + current
        metadata = {key: len(window) for key in RECORDED_INDEX_KEYS}

        after = [{"role": "user", "content": "[summary]"}] + before[-4:]
        reconcile_recorded_index(before, after, metadata)

        for key in RECORDED_INDEX_KEYS:
            idx = metadata[key]
            assert idx <= len(after), f"{key} 가 리스트 밖을 가리킨다"
            # 이번 턴의 두 메시지는 여전히 "아직 안 적은 것" 으로 남아야 한다.
            assert len(after[idx:]) == 2, f"{key} 뒤로 이번 턴이 사라졌다"

    def test_untouched_metadata_stays_untouched(self):
        before = [{"role": "user", "content": "a"}]
        after = list(before)
        metadata = {"unrelated": 7}
        reconcile_recorded_index(before, after, metadata)
        assert metadata == {"unrelated": 7}


class TestNoDoubleRecording:
    """전략과 스테이지가 같은 접두부를 두 번 적으면 다음 턴 창이 같은 지시를 두 번 보여 준다."""

    @pytest.mark.asyncio
    async def test_a_turn_is_recorded_once(self):
        provider = _Provider()
        stage = MemoryStage(strategy=ProviderDrivenStrategy(provider))
        stage.provider = provider
        state = _state(
            [
                {"role": "user", "content": "현재 지시"},
                {"role": "assistant", "content": "답변"},
            ]
        )

        await stage.execute(None, state)

        assert len(provider.recorded) == 2, provider.recorded

    @pytest.mark.asyncio
    async def test_the_preloaded_window_is_never_re_recorded(self):
        provider = _Provider()
        stage = MemoryStage(strategy=ProviderDrivenStrategy(provider))
        stage.provider = provider
        window = [
            {"role": "user", "content": "지난 지시"},
            {"role": "assistant", "content": "지난 답변"},
        ]
        state = _state(
            window + [{"role": "user", "content": "현재 지시"}],
            **{_RECORDED_KEY: len(window)},
        )

        await stage.execute(None, state)

        assert [r[1] for r in provider.recorded] == ["현재 지시"]


class TestWindowRolesAlternate:
    def test_a_cancelled_turn_does_not_leave_two_user_messages(self):
        rows = [
            _Row("user", "답변 없이 끝난 지시"),
            _Row("user", "다음 지시"),
            _Row("assistant", "답변"),
        ]
        messages, _ = build_window(rows, WindowConfig())
        roles = [m["role"] for m in messages]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles

    def test_an_interrupted_tool_turn_closes_with_an_assistant_message(self):
        rows = [
            _Row("user", "도구 지시"),
            _Row("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]),
        ]
        messages, _ = build_window(rows, WindowConfig())
        assert messages[-1]["role"] == "assistant"
        roles = [m["role"] for m in messages]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles

    def test_tool_blocks_are_never_merged_away(self):
        rows = [
            _Row("user", "도구 지시"),
            _Row("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]),
            _Row("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]),
            _Row("assistant", "답변"),
        ]
        messages, _ = build_window(rows, WindowConfig())
        kinds = [
            b.get("type")
            for m in messages
            if isinstance(m["content"], list)
            for b in m["content"]
        ]
        assert "tool_use" in kinds and "tool_result" in kinds


class _CountingSTM(_STM):
    def __init__(self, rows):
        super().__init__()
        self.rows = list(rows)
        self.requested = []

    async def recent(self, n=20):
        self.requested.append(n)
        return list(self.rows)[-n:]


class _FetchProvider:
    def __init__(self, rows):
        self._stm = _CountingSTM(rows)

    def stm(self):
        return self._stm


class TestAdaptiveScan:
    """STM 한 행은 도구 결과 원문을 든다 — 필요한 만큼만 읽는다."""

    @pytest.mark.asyncio
    async def test_a_short_conversation_reads_one_small_page(self):
        from xgen_agent_runtime.memory.short_term_window import SCAN_FIRST, load_window

        rows = []
        for i in range(8):
            rows += [_Row("user", f"지시{i}"), _Row("assistant", f"답변{i}")]
        provider = _FetchProvider(rows)

        await load_window(provider, WindowConfig())

        assert provider._stm.requested == [SCAN_FIRST]

    @pytest.mark.asyncio
    async def test_a_tool_heavy_conversation_escalates_and_still_gets_five_turns(self):
        from xgen_agent_runtime.memory.short_term_window import (
            SCAN_FIRST,
            SCAN_ROWS,
            load_window,
        )

        rows = []
        for i in range(6):
            rows.append(_Row("user", f"지시{i}"))
            for j in range(30):
                rows.append(
                    _Row(
                        "assistant",
                        [{"type": "tool_use", "id": f"t{i}{j}", "name": "Bash", "input": {}}],
                    )
                )
                rows.append(
                    _Row("user", [{"type": "tool_result", "tool_use_id": f"t{i}{j}", "content": "ok"}])
                )
            rows.append(_Row("assistant", f"답변{i}"))
        provider = _FetchProvider(rows)

        _, report = await load_window(provider, WindowConfig())

        assert provider._stm.requested == [SCAN_FIRST, SCAN_ROWS]
        assert report.turns == 5


class TestCurrentSessionFilter:
    """이 세션의 대화 기록은 지식 층에서 빼되, 애먼 노트까지 지우지는 않는다."""

    def test_the_session_archive_note_is_excluded(self):
        from xgen_agent_runtime.memory.retriever import _is_current_session_record

        class _Hit:
            key = "conversations/sess-1__user__hello.md"
            content = "## turn-abc"
            metadata = {"filename": "conversations/sess-1__user__hello.md"}

        assert _is_current_session_record(_Hit(), "sess-1") is True

    def test_a_note_that_merely_mentions_the_session_id_survives(self):
        from xgen_agent_runtime.memory.retriever import _is_current_session_record

        class _Hit:
            key = "notes/deploy-runbook.md"
            content = "배포 사고 기록. 문제의 대화는 sess-1 이었다."
            metadata = {"filename": "notes/deploy-runbook.md"}

        assert _is_current_session_record(_Hit(), "sess-1") is False

    def test_a_record_whose_header_names_the_session_is_excluded(self):
        from xgen_agent_runtime.memory.retriever import _is_current_session_record

        class _Hit:
            key = "daily/exec-0001-abc.md"
            content = "---\nsession_id: sess-1\n---\n\n실행 카드"
            metadata = {}

        assert _is_current_session_record(_Hit(), "sess-1") is True


class TestDegradationLog:
    def test_result_keep_is_not_reported_when_there_is_nothing_to_trim(self):
        rows = []
        for i in range(5):
            rows += [_Row("user", "지" * 9_000), _Row("assistant", "답" * 9_000)]

        _, report = build_window(rows, WindowConfig(max_chars=4_000))

        assert not any(step.startswith("result_keep") for step in report.degraded), report.degraded
