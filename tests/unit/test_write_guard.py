"""읽지 않은 파일을 말없이 덮어쓰지 않는다 (4.51.0).

참고 하네스의 기본 계약이다 — 파일을 바꾸려면 먼저 읽는다. 우리에겐 없어서
`Write` 가 이미 있는 파일을 내용도 모른 채 잘라내고 새로 쓸 수 있었다. 사용자가
올린 문서·이전 세션 산출물이 그렇게 사라져도 **로그에는 흔적이 없다**(성공한
쓰기다). 그래서 이 구멍은 기록으로 셀 수 없고 계약으로만 막는다.

마찰 크기는 실측했다 (dev 30일): Write 1,016건 중 **841건이 그 대화에서 처음
건드리는 경로** — 대부분 새 파일이라 그대로 통과한다. Edit 은 372건 중 371건이
이미 내용을 알고 있어(Read 또는 자기가 Write) 손대지 않았다.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace



from xgen_agent_runtime.stages.s10_tool.state_mutation import apply_state_mutations
from xgen_agent_runtime.tools.base import ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in._file_witness import (
    MAX_ENTRIES,
    WITNESSED_KEY,
    is_witnessed,
    witnessed_mutation,
)
from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
from xgen_agent_runtime.tools.built_in.write_tool import WriteTool


def _ctx(tmp_path, shared=None):
    return ToolContext(
        session_id="t",
        working_dir=str(tmp_path),
        state_view=SimpleNamespace(shared=shared if shared is not None else {}),
    )


def _run(tool, payload, ctx):
    return asyncio.run(tool.execute(payload, ctx))


class TestLedger:
    def test_uses_an_allowed_executor_namespace(self):
        assert WITNESSED_KEY == "executor.file_witnessed"

    def test_records_both_spellings(self):
        view = SimpleNamespace(shared={})
        mut = witnessed_mutation(view, "a.txt", "/abs/a.txt")
        assert mut[WITNESSED_KEY] == ["a.txt", "/abs/a.txt"]

    def test_reread_does_not_duplicate(self):
        view = SimpleNamespace(shared={WITNESSED_KEY: ["a", "b"]})
        assert witnessed_mutation(view, "a")[WITNESSED_KEY] == ["b", "a"]

    def test_forgets_the_oldest_beyond_the_cap(self):
        view = SimpleNamespace(shared={WITNESSED_KEY: [str(i) for i in range(MAX_ENTRIES)]})
        book = witnessed_mutation(view, "new")[WITNESSED_KEY]
        assert len(book) == MAX_ENTRIES
        assert book[-1] == "new" and "0" not in book

    def test_unknown_state_view_is_not_witnessed(self):
        assert is_witnessed(None, "/a") is False

    def test_reads_the_legacy_key_but_migrates_on_write(self):
        view = SimpleNamespace(shared={"file.witnessed": ["old.txt"]})
        assert is_witnessed(view, "old.txt") is True
        mutation = witnessed_mutation(view, "new.txt")
        assert mutation == {WITNESSED_KEY: ["old.txt", "new.txt"]}

    def test_stage_ten_accepts_the_witness_mutation(self):
        shared: dict = {}
        mutation = witnessed_mutation(SimpleNamespace(shared=shared), "a.txt")
        applied = apply_state_mutations(
            ToolResult(content="read", state_mutations=mutation),
            shared,
            tool_name="Read",
        )
        assert applied == {WITNESSED_KEY: ["a.txt"]}
        assert shared == applied


class TestWriteGuard:
    def test_new_file_is_written_without_reading(self, tmp_path):
        """마찰을 만들지 않는 자리 — 실측상 Write 의 대부분이다."""
        target = tmp_path / "new.txt"
        r = _run(WriteTool(), {"file_path": str(target), "content": "hi"}, _ctx(tmp_path))
        assert r.is_error is False
        assert target.read_text() == "hi"

    def test_existing_file_is_not_clobbered_blindly(self, tmp_path):
        target = tmp_path / "기능요구사항.docx"
        target.write_text("사용자가 올린 내용")
        r = _run(WriteTool(), {"file_path": str(target), "content": "덮어씀"}, _ctx(tmp_path))
        assert r.is_error is True
        assert "have not read it" in r.content
        assert target.read_text() == "사용자가 올린 내용"  # 그대로다

    def test_reading_first_allows_the_write(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("before")
        shared: dict = {}
        ctx = _ctx(tmp_path, shared)
        read = _run(ReadTool(), {"file_path": str(target)}, ctx)
        shared.update(read.state_mutations)  # 파이프라인이 하는 일
        r = _run(WriteTool(), {"file_path": str(target), "content": "after"}, ctx)
        assert r.is_error is False
        assert target.read_text() == "after"

    def test_an_empty_file_is_not_protected(self, tmp_path):
        """잃을 내용이 없다 — 막을 이유도 없다."""
        target = tmp_path / "empty.txt"
        target.write_text("")
        r = _run(WriteTool(), {"file_path": str(target), "content": "x"}, _ctx(tmp_path))
        assert r.is_error is False

    def test_the_refusal_names_the_next_move(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("x")
        r = _run(WriteTool(), {"file_path": str(target), "content": "y"}, _ctx(tmp_path))
        assert "Read it first" in r.content
        assert "Edit" in r.content
        assert "Nothing was written" in r.content

    def test_a_relative_path_read_covers_the_resolved_write(self, tmp_path):
        target = tmp_path / "rel.txt"
        target.write_text("before")
        shared: dict = {}
        ctx = _ctx(tmp_path, shared)
        read = _run(ReadTool(), {"file_path": "rel.txt"}, ctx)
        assert read.is_error is False
        shared.update(read.state_mutations)
        r = _run(WriteTool(), {"file_path": str(target), "content": "after"}, ctx)
        assert r.is_error is False


# ── 실제 반영 경로 (4.60.0) ───────────────────────────────────────────
# 위 테스트들은 장부를 손으로 shared 에 넣었다. 실제로는 Stage 10 이 state_mutations 를
# **허용 이름공간만** 반영한다 — 4.51~4.59 의 키(file.witnessed)는 조용히 버려져 장부가 비어 있었다.


class TestWitnessThroughTheRealApplyPath:
    def test_the_ledger_key_survives_state_mutation_filtering(self):
        from xgen_agent_runtime.stages.s10_tool.state_mutation import apply_state_mutations
        from xgen_agent_runtime.tools.base import ToolResult

        shared: dict = {}
        applied = apply_state_mutations(
            ToolResult(content="ok", state_mutations=witnessed_mutation(SimpleNamespace(shared=shared), "a.txt")),
            shared,
            tool_name="Read",
        )
        assert WITNESSED_KEY in applied and is_witnessed(SimpleNamespace(shared=shared), "a.txt")

    def test_write_then_rewrite_and_read_then_write_in_a_real_turn(self, tmp_path):
        """파이프라인 한 턴: 새 파일 Write → 같은 파일 다시 Write, 남의 파일 Read → Write. 전부 성공해야 한다."""
        from xgen_agent_runtime.core.state import PipelineState, TokenUsage
        from xgen_agent_runtime.host import runner
        from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
        from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
        from xgen_agent_runtime.tools import ToolRegistry

        existing = tmp_path / "given.txt"
        existing.write_text("original", encoding="utf-8")
        mine = tmp_path / "out.txt"
        script = [
            ("Write", {"file_path": str(mine), "content": "v1"}),
            ("Write", {"file_path": str(mine), "content": "v2"}),
            ("Read", {"file_path": str(existing)}),
            ("Write", {"file_path": str(existing), "content": "updated"}),
        ]

        class _Scripted(BaseClient):
            provider = "fake"
            capabilities = ClientCapabilities()

            def __init__(self, **kw):
                super().__init__(**kw)
                self.n = 0

            async def _send(self, request, *, purpose=""):
                usage = TokenUsage(input_tokens=10, output_tokens=2)
                if self.n >= len(script):
                    return APIResponse(content=[ContentBlock(type="text", text="done")], stop_reason="end_turn",
                                       usage=usage, model="fake")
                name, inp = script[self.n]
                self.n += 1
                return APIResponse(content=[ContentBlock(type="tool_use", tool_use_id=f"t{self.n}", tool_name=name,
                                                         tool_input=inp)], stop_reason="tool_use", usage=usage, model="fake")

        reg = ToolRegistry()
        reg.register(WriteTool())
        reg.register(ReadTool())
        pipe = runner.build_pipeline(
            name="t", provider="openai", model="m", api_key="k", llm_client=_Scripted(api_key="k"), stream=False,
            enable_compaction=False, registry=reg, max_iterations=20, turn_input_budget_tokens=None,
            tool_context=ToolContext(session_id="t", working_dir=str(tmp_path)),
        )
        runner.run_turn(pipe, "go", PipelineState(session_id="t", model="m"))
        assert mine.read_text(encoding="utf-8") == "v2"
        assert existing.read_text(encoding="utf-8") == "updated"
