"""완료 직전 요건 대조 (stages/s16_loop/completion_review.py RequirementReviewer, 4.36.0).

근거: Harness-Bench 홀드아웃 31과제 × 4회차에서 4회 모두 실패한 체크 37개의 대부분이
"스펙을 읽고도 출력을 스펙과 필드 단위로 대조하지 않음" — 라우팅 계약의 키를 자기 말로
바꿔 쓰고(071), 매니페스트 필수 필드를 빠뜨리고(077), 리네임 로그의 정렬·정확 집합을
어긴다(021). 산출물 대조는 형식만 봐서 여기엔 한 번도 개입하지 않았다. 하네스는 요구사항을
모른 채 모델에게 "요청에서 뽑아 실제 출력을 읽고 ✓/✗ 하라" 고만 한다.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock, TokenUsage
from xgen_agent_runtime.stages.s16_loop.completion_review import (
    REQUIREMENT_KEY,
    RequirementReviewer,
)
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.registry import ToolRegistry
from tests.test_completion_review import _Write, _assistant, _tool_result, _tool_use


def _state(*tool_uses: Dict[str, Any], final: str = "done") -> PipelineState:
    state = PipelineState(session_id="s", model="m")
    state.messages = [{"role": "user", "content": "계약대로 라우팅 표를 만들어"}]
    for tu in tool_uses:
        state.messages += [_assistant(tu), _tool_result()]
    state.messages.append(_assistant({"type": "text", "text": final}))
    state.final_text = final
    return state


# ── 검토자 단위 ────────────────────────────────────────────────────────


def test_fires_once_per_turn_when_files_were_written() -> None:
    state = _state(_tool_use("Write", file_path="out/routing.csv", content="x"),
                   _tool_use("Edit", file_path="out/notes.md", old_string="a", new_string="b"))
    rv = RequirementReviewer()

    note = asyncio.run(rv.review(state))
    assert note and note.startswith("[Requirement check")
    assert "read the file or run the command" in note  # 기억이 아니라 실제 출력을 보라
    assert "exact identifiers, never a paraphrase" in note  # 071 형 실패의 정곡
    assert note.rstrip().endswith("Files written this turn: out/routing.csv, out/notes.md")
    assert state.shared[REQUIREMENT_KEY] == {"done": True, "files": 2}
    assert state.events[-1]["type"] == "loop.requirement_review"
    assert state.events[-1]["data"]["paths"] == ["out/routing.csv", "out/notes.md"]

    assert asyncio.run(rv.review(state)) is None  # 턴당 한 번


def test_silent_on_turns_that_wrote_nothing() -> None:
    """잡담·읽기만 한 턴엔 비용 0 — 검증할 산출물이 없다."""
    state = _state(_tool_use("Read", file_path="in/spec.md"), final="스펙을 읽었습니다.")
    assert asyncio.run(RequirementReviewer().review(state)) is None
    assert REQUIREMENT_KEY not in state.shared


def test_does_not_count_files_from_a_previous_turn() -> None:
    state = PipelineState(session_id="s", model="m")
    state.messages = [
        {"role": "user", "content": "지난 턴"},
        _assistant(_tool_use("Write", file_path="old/prev.csv", content="x")), _tool_result(),
        {"role": "user", "content": "이번 턴은 설명만"},
        _assistant({"type": "text", "text": "설명입니다."}),
    ]
    assert asyncio.run(RequirementReviewer().review(state)) is None


def test_long_file_lists_are_truncated_with_a_count() -> None:
    uses = [_tool_use("Write", file_path=f"out/f{i}.json", content="{}") for i in range(25)]
    note = asyncio.run(RequirementReviewer(max_files=20).review(_state(*uses)))
    assert note and "out/f19.json (+5 more)" in note and "out/f20.json" not in note


def test_begin_turn_clears_the_marker() -> None:
    state = PipelineState(session_id="s", model="m")
    state.shared[REQUIREMENT_KEY] = {"done": True}
    state.begin_turn()
    assert REQUIREMENT_KEY not in state.shared


# ── 파이프라인 끝까지 ──────────────────────────────────────────────────


class _Client(BaseClient):
    """1: 계약과 다른 키로 파일을 쓴다. 2: 완료 선언. (요건 대조 뒤) 3: 계약의 키로 고친다. 4: 완료."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        n = len(self.requests)
        if n == 1:
            blocks = [ContentBlock(type="tool_use", tool_use_id="t1", tool_name="Write",
                                   tool_input={"file_path": "out/routing.csv", "content": "ticket,reply_key\nT-1,damaged_reship_with_evidence\n"})]
            return APIResponse(content=blocks, stop_reason="tool_use", usage=usage, model="fake")
        if n == 2:
            return APIResponse(content=[ContentBlock(type="text", text="out/routing.csv 완료. [COMPLETE]")],
                               stop_reason="end_turn", usage=usage, model="fake")
        if n == 3:
            blocks = [ContentBlock(type="tool_use", tool_use_id="t2", tool_name="Write",
                                   tool_input={"file_path": "out/routing.csv", "content": "ticket,reply_key\nT-1,tpl_reship_damage_photo\n"})]
            return APIResponse(content=blocks, stop_reason="tool_use", usage=usage, model="fake")
        return APIResponse(content=[ContentBlock(type="text", text="계약의 키로 고쳤습니다. [COMPLETE]")],
                           stop_reason="end_turn", usage=usage, model="fake")


def _last_user_text(request: Any) -> Optional[str]:
    for m in reversed(getattr(request, "messages", None) or []):
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return None


def _pipe(tmp_path: Path, client: BaseClient, **kw: Any):
    reg = ToolRegistry()
    reg.register(_Write(tmp_path), core=True)
    ctx = ToolContext(working_dir=str(tmp_path), allowed_paths=[str(tmp_path)])
    return runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client,
        stream=False, enable_compaction=False, registry=reg, tool_context=ctx, **kw,
    )


def test_pipeline_defers_completion_once_with_the_requirement_note(tmp_path: Path) -> None:
    client = _Client(api_key="k")
    text = runner.run_turn(_pipe(tmp_path, client), "계약(tpl_*)대로 out/routing.csv 를 내라",
                           PipelineState(session_id="s", model="m"))

    assert len(client.requests) == 4, "완료를 한 번 미루고(요건 대조) 고친 뒤 끝나야 한다"
    note = _last_user_text(client.requests[2])
    assert note and note.startswith("[Requirement check") and "out/routing.csv" in note
    assert "계약의 키로 고쳤습니다" in text
    assert "tpl_reship_damage_photo" in (tmp_path / "out" / "routing.csv").read_text()


def test_deliverable_check_runs_first_then_requirement_check_on_the_next_completion(tmp_path: Path) -> None:
    """형식 문제(ragged CSV)가 있으면 산출물 대조가 먼저, 고친 뒤 완료 선언에 요건 대조가 한 번 — 둘 다 턴당 1회."""

    class _Ragged(_Client):
        async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
            self.requests.append(request)
            usage = TokenUsage(input_tokens=10, output_tokens=5)
            n = len(self.requests)
            if n == 1:
                return APIResponse(content=[ContentBlock(type="tool_use", tool_use_id="t1", tool_name="Write",
                                                         tool_input={"file_path": "out/a.csv", "content": "a,b\n1\n2,3,4\n"})],
                                   stop_reason="tool_use", usage=usage, model="fake")
            if n == 3:
                return APIResponse(content=[ContentBlock(type="tool_use", tool_use_id="t2", tool_name="Write",
                                                         tool_input={"file_path": "out/a.csv", "content": "a,b\n1,2\n"})],
                                   stop_reason="tool_use", usage=usage, model="fake")
            return APIResponse(content=[ContentBlock(type="text", text="out/a.csv 완료. [COMPLETE]")],
                               stop_reason="end_turn", usage=usage, model="fake")

    client = _Ragged(api_key="k")
    runner.run_turn(_pipe(tmp_path, client), "out/a.csv 를 내라", PipelineState(session_id="s", model="m"))
    # 1 write, 2 complete→산출물 대조, 3 fix, 4 complete→요건 대조, 5 complete
    assert len(client.requests) == 5, len(client.requests)
    assert (_last_user_text(client.requests[2]) or "").startswith("[Deliverable check")
    assert (_last_user_text(client.requests[4]) or "").startswith("[Requirement check")


def test_review_off_or_no_files_adds_no_round_trip(tmp_path: Path) -> None:
    class _Plain(BaseClient):
        provider = "fake"
        capabilities = ClientCapabilities()

        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.n = 0

        async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
            self.n += 1
            return APIResponse(content=[ContentBlock(type="text", text="파일 없이 답합니다.")],
                               stop_reason="end_turn", usage=TokenUsage(input_tokens=1, output_tokens=1), model="fake")

    plain = _Plain(api_key="k")
    runner.run_turn(_pipe(tmp_path, plain), "안녕", PipelineState(session_id="s", model="m"))
    assert plain.n == 1

    client = _Client(api_key="k")
    runner.run_turn(_pipe(tmp_path, client, enable_requirement_review=False), "out/routing.csv 를 내라",
                    PipelineState(session_id="s", model="m"))
    assert len(client.requests) == 2  # 끄면 산출물 대조(형식 정상)만 — 왕복 없음
