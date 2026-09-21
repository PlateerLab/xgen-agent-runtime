"""단기 기억 창의 배선 — Stage 2 preload · L0 게이트 · 검색에서 현재 세션 기록 제외 · 아카이브 워터마크 ·
실행 카드 3상태 · 러너 도구 통계."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.memory.provider import MemoryHooks, RetrievalResult
from xgen_agent_runtime.memory.short_term_window import WINDOW_KEY, WINDOW_LEN_KEY
from xgen_agent_runtime.memory.strategy import _RECORDED_KEY


def _t(role, content):
    return SimpleNamespace(role=role, content=content)


def _prev_turns(n: int) -> List[Any]:
    out: List[Any] = []
    for i in range(1, n + 1):
        out += [
            _t("user", f"q{i}"),
            _t("assistant", [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {}}]),
            _t("user", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}]),
            _t("assistant", [{"type": "text", "text": f"a{i}"}]),
        ]
    return out


class _STM:
    def __init__(self, turns):
        self._turns = turns

    async def recent(self, n=20):
        return list(self._turns)


class _Provider:
    def __init__(self, turns, hooks=None):
        self._stm = _STM(turns)
        self.hooks = hooks

    def stm(self):
        return self._stm

    async def retrieve(self, query):
        return RetrievalResult(chunks=[], layer_breakdown={})


def _stage(provider):
    from xgen_agent_runtime.stages.s02_context.artifact.default.stage import ContextStage

    return ContextStage(provider=provider, compaction_enabled=False, background_compaction=False)


def _state(**meta) -> PipelineState:
    st = PipelineState(session_id="sess-1")
    st.messages = [{"role": "user", "content": "current question"}]
    st.metadata.update(meta)
    return st


class TestStage2Preload:
    def test_window_is_prepended_with_watermarks_and_event(self):
        st = _state()
        asyncio.run(_stage(_Provider(_prev_turns(6))).execute(None, st))
        assert st.messages[-1]["content"] == "current question"
        window_len = len(st.messages) - 1
        assert window_len > 0
        assert st.metadata[_RECORDED_KEY] == window_len == st.metadata[WINDOW_LEN_KEY]
        assert st.metadata[WINDOW_KEY]["turns"] == 5 and st.metadata[WINDOW_KEY]["full"] == 2
        assert any(e["type"] == "context.short_term_window" for e in st.events)
        # 먼 턴은 대화만(문자열), 가까운 턴은 도구 블록
        assert isinstance(st.messages[0]["content"], str)
        assert any(isinstance(m["content"], list) for m in st.messages[:-1])

    def test_host_preloaded_history_means_no_window(self):
        st = _state(**{_RECORDED_KEY: 1})
        asyncio.run(_stage(_Provider(_prev_turns(3))).execute(None, st))
        assert len(st.messages) == 1 and WINDOW_KEY not in st.metadata

    def test_only_at_iteration_zero_and_not_twice(self):
        st = _state()
        stage = _stage(_Provider(_prev_turns(2)))
        asyncio.run(stage.execute(None, st))
        n = len(st.messages)
        st.iteration = 1
        asyncio.run(stage.execute(None, st))
        assert len(st.messages) == n

    def test_hooks_can_disable_the_window(self):
        hooks = MemoryHooks(window_full_turns=0, window_dialogue_turns=0)
        st = _state()
        asyncio.run(_stage(_Provider(_prev_turns(3), hooks=hooks)).execute(None, st))
        assert len(st.messages) == 1

    def test_stm_failure_does_not_break_the_turn(self):
        class _Bad(_Provider):
            def stm(self):
                raise RuntimeError("down")

        st = _state()
        asyncio.run(_stage(_Bad(_prev_turns(2))).execute(None, st))
        assert len(st.messages) == 1


class TestRetrieverGate:
    def _retriever(self, hooks):
        from xgen_agent_runtime.memory.retriever import MemoryAwareRetriever

        class _Notes:
            async def load_pinned(self, **k):
                return ""

            async def list(self, **k):
                return []

            async def search(self, *a, **k):
                return []

        class _P(_Provider):
            def notes(self):
                return _Notes()

            def ltm(self):
                return SimpleNamespace(read_main=self._none, search=self._empty)

            def vector(self):
                return None

            def index(self):
                return SimpleNamespace(render_vault_map=self._none, build_vault_map=self._none)

            async def _none(self, *a, **k):
                return None

            async def _empty(self, *a, **k):
                return []

        return MemoryAwareRetriever(_P(_prev_turns(3)), hooks=hooks)

    def test_recent_turns_layer_is_skipped_when_the_window_is_in_messages(self):
        hooks = MemoryHooks(recent_turns=3, slim_mode=True, always_render_vault_map=False)
        r = self._retriever(hooks)
        with_window = asyncio.run(r.retrieve("q", _state(**{WINDOW_KEY: {"turns": 3}})))
        assert all(c.metadata.get("layer") != "recent_turns" for c in with_window)
        without = asyncio.run(r.retrieve("q", _state()))
        assert any(c.metadata.get("layer") == "recent_turns" for c in without)

    def test_current_session_records_are_excluded(self):
        from xgen_agent_runtime.memory.retriever import _is_current_session_record

        archive = SimpleNamespace(metadata={"filename": "sess-1__user__hello.md"}, key="x", content="…")
        card = SimpleNamespace(metadata={}, key="exec-0002-abc.md", content="### [✅] Execution #2\n> **Session:** sess-1")
        other = SimpleNamespace(metadata={"filename": "sess-9__user__hi.md"}, key="y", content="Session: sess-9")
        assert _is_current_session_record(archive, "sess-1") and _is_current_session_record(card, "sess-1")
        assert not _is_current_session_record(other, "sess-1") and not _is_current_session_record(card, "")


class TestArchiveWatermark:
    def test_window_length_is_the_default_watermark(self):
        from xgen_agent_runtime.host.conversation_archive import ConversationArchivingStrategy

        st = _state(**{WINDOW_LEN_KEY: 2})
        st.messages = [{"role": "user", "content": "old q"}, {"role": "assistant", "content": "old a"},
                       {"role": "user", "content": "new q"}, {"role": "assistant", "content": "new a"}]
        fresh, mark = ConversationArchivingStrategy._new_utterances(st)
        assert fresh == [("user", "new q"), ("assistant", "new a")] and mark == 4


class TestExecutionOutcome:
    def test_classify(self):
        from xgen_agent_runtime.host.execution_record import classify_outcome

        assert classify_outcome(True) == "ok"
        assert classify_outcome(True, tool_calls=5) == "ok"
        assert classify_outcome(True, tool_calls=8, tool_failures=3) == "partial"
        assert classify_outcome(True, blocked=1) == "partial"
        assert classify_outcome(False, tool_calls=0) == "failed"

    def test_card_body_says_partial_with_tool_counts(self):
        from xgen_agent_runtime.host.execution_record import _build_card_body

        body = _build_card_body(number=2, input_text="삼성전자 도구", output_text="done", success=True, duration_ms=1000,
                                session_id="s", provider_name="", model="", error="", tool_calls=8, tool_failures=8, blocked=1)
        assert body.startswith("### [⚠️] Execution #2")
        assert "**Tools:** 8 calls · 8 failed · 1 blocked" in body and "partial" in body

    def test_record_writes_partial_tag_and_journal_mark(self):
        from xgen_agent_runtime.host.execution_record import record_turn_execution

        written: List[Any] = []

        class _Notes:
            async def list(self, **k):
                return []

            async def write(self, draft):
                written.append(draft)
                return None

            async def read(self, fn):
                return None

            async def update(self, fn, patch):
                written.append(patch)

        class _P:
            def notes(self):
                return _Notes()

        asyncio.run(record_turn_execution(_P(), input_text="q", output_text="a", success=True, duration_ms=10,
                                          session_id="s", tool_calls=4, tool_failures=4))
        card = written[0]
        assert "partial" in card.tags and card.body.startswith("### [⚠️]")
        assert card.frontmatter["outcome"] == "partial" and card.frontmatter["tool_failures"] == 4
        journal = written[1]
        assert "⚠️" in journal.body


class TestRunnerToolStats:
    def test_counts_from_events(self):
        from xgen_agent_runtime.host.runner import _tool_stats_from_events

        st = PipelineState(session_id="s")
        st.add_event("tool.execute_complete", {"count": 3, "errors": 2})
        st.add_event("tool.execute_complete", {"count": 1, "errors": 0})
        st.add_event("tool.repeat_blocked", {"tools": ["ForgeTool"]})
        assert _tool_stats_from_events(st) == (4, 2, 1)
        assert _tool_stats_from_events(SimpleNamespace(events=None)) == (0, 0, 0)


class TestGuardThresholds:
    def test_values(self):
        from xgen_agent_runtime.stages.s10_tool import repeat_guard as g

        assert (g.WARN_AT, g.BLOCK_AT, g.SAME_RESULT_WARN_AT, g.SAME_RESULT_SKIP_AT) == (3, 4, 4, 5)


class TestPromptSentence:
    def test_memory_blocks_frame_history_as_history(self):
        from xgen_agent_runtime.host._constants import MEMORY_PROMPT_BLOCK, MEMORY_READONLY_PROMPT_BLOCK

        for block in (MEMORY_PROMPT_BLOCK, MEMORY_READONLY_PROMPT_BLOCK):
            assert "treat them as history, not as facts to re-verify" in block
