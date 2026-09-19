"""완료 직전 산출물 대조 (stages/s16_loop/completion_review.py).

근거: Harness-Bench 68과제(2026-09-19) 에서 잃은 점수의 23% 가 "요구 조건을 다시
대조하지 않고 완료 선언" — 정확히 3행이어야 하는 CSV 에 5행, 없는 파일을 있다고 보고.
하네스는 조건을 모른다. 모델이 주장한 파일을 실제로 읽어 요약을 한 번 보여 주고,
요청과 대조하라고 할 뿐이다.
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
    REVIEW_KEY,
    DeliverableReviewer,
    claimed_paths,
    describe_file,
)
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry

# ── 주장한 경로 ────────────────────────────────────────────────────────


def _assistant(*blocks: Dict[str, Any]) -> Dict[str, Any]:
    return {"role": "assistant", "content": list(blocks)}


def _tool_use(name: str, **inp: Any) -> Dict[str, Any]:
    return {"type": "tool_use", "id": "t", "name": name, "input": inp}


def _tool_result() -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}


def test_claimed_paths_written_first_then_mentioned() -> None:
    msgs = [
        {"role": "user", "content": "지난 턴"},
        _assistant(_tool_use("Write", file_path="old/prev.csv", content="x")),
        _tool_result(),
        {"role": "user", "content": "감사 결과를 out/audit.csv 로 내라"},
        _assistant(_tool_use("Write", file_path="out/audit.csv", content="a,b")),
        _tool_result(),
        _assistant(_tool_use("ToolBatch", tool="Write",
                             inputs=[{"file_path": "out/notes.md", "content": "n"},
                                     {"file_path": "out/audit.csv", "content": "dup"}])),
        _tool_result(),
        _assistant({"type": "text", "text": "done"}),
    ]
    final = "out/audit.csv 와 `out/summary.json` 을 만들었습니다. 원본은 in/data.csv. 참고 https://x.com/a.b"
    got = claimed_paths(msgs, final)
    assert got[:2] == ["out/audit.csv", "out/notes.md"], got  # 쓴 것 먼저, 중복 없음
    assert "out/summary.json" in got and "in/data.csv" in got
    assert "old/prev.csv" not in got  # 지난 턴은 아니다
    assert not any("x.com" in p for p in got)  # URL 은 경로가 아니다


def test_claimed_paths_ignores_versions_and_bare_domains() -> None:
    assert claimed_paths([], "python 3.12.1 로 v1.2.3 을 example.com 에 올렸다") == []


# ── 파일 요약 ──────────────────────────────────────────────────────────


def test_describe_file_states_missing_csv_json_and_text() -> None:
    assert describe_file("out/x.csv", None) == "MISSING"
    assert describe_file("out/x.csv", b"") == "EMPTY (0 bytes)"
    csv_desc = describe_file("out/x.csv", b"id,issue,fix\n1,a,b\n2,c,d\n3,e,f\n4,g,h\n5,i,j\n")
    assert '"id,issue,fix" (3 cols)' in csv_desc and "5 data rows" in csv_desc and "ragged" not in csv_desc
    assert "ragged rows" in describe_file("r.csv", b"a,b\n1\n2,3,4\n")
    assert "valid JSON object, 2 keys: threads, count" == describe_file("s.json", b'{"threads": [], "count": 0}')
    assert describe_file("s.json", b'{"a": ').startswith("INVALID JSON")
    assert "3 JSON lines, all valid" == describe_file("e.jsonl", b'{"a":1}\n{"b":2}\n{"c":3}\n')
    assert "1 INVALID JSON (first at line 2)" in describe_file("e.jsonl", b'{"a":1}\nnope\n')
    assert describe_file("bin.png", b"\x89PNG\x00\x00").startswith("binary")
    assert 'starts "# Report"' in describe_file("r.md", b"# Report\n\nbody\n")


# ── 검토자: 로컬 파일 경로 ─────────────────────────────────────────────


def _local_reviewer(tmp: Path) -> DeliverableReviewer:
    ctx = ToolContext(working_dir=str(tmp), allowed_paths=[str(tmp)])
    return DeliverableReviewer(lambda: ctx)


def test_reviewer_summarises_claimed_files_once_per_turn(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "audit.csv").write_text("a,b\n1,2\n3,4\n")
    state = PipelineState(session_id="s", model="m")
    state.messages = [
        {"role": "user", "content": "감사해"},
        _assistant(_tool_use("Write", file_path="out/audit.csv", content="a,b\n1,2\n3,4\n")),
        _tool_result(),
        _assistant({"type": "text", "text": "out/audit.csv 와 out/report.md 를 만들었습니다."}),
    ]
    state.final_text = "out/audit.csv 와 out/report.md 를 만들었습니다."
    rv = _local_reviewer(tmp_path)

    note = asyncio.run(rv.review(state))
    assert note and note.startswith("[Deliverable check")
    assert "- out/audit.csv — 3 lines: header \"a,b\" (2 cols), 2 data rows" in note
    assert "- out/report.md — MISSING" in note
    assert state.shared[REVIEW_KEY] == {"done": True, "files": 2, "missing": 1}
    assert state.events[-1]["type"] == "loop.completion_review"

    assert asyncio.run(rv.review(state)) is None  # 턴당 한 번


def test_reviewer_skips_paths_outside_the_allowed_tree_and_turns_without_files(tmp_path: Path) -> None:
    state = PipelineState(session_id="s", model="m")
    state.messages = [{"role": "user", "content": "q"}, _assistant({"type": "text", "text": "봐라 /etc/passwd"})]
    state.final_text = "봐라 /etc/passwd"
    assert asyncio.run(_local_reviewer(tmp_path).review(state)) is None
    assert REVIEW_KEY not in state.shared

    state.final_text = "파일 없이 답만"
    state.messages[-1] = _assistant({"type": "text", "text": "파일 없이 답만"})
    assert asyncio.run(_local_reviewer(tmp_path).review(state)) is None


def test_begin_turn_clears_the_review_marker() -> None:
    state = PipelineState(session_id="s", model="m")
    state.shared[REVIEW_KEY] = {"done": True}
    state.shared["tool.same_result_counts"] = {"x": 3}
    state.begin_turn()
    assert REVIEW_KEY not in state.shared and "tool.same_result_counts" not in state.shared


# ── 샌드박스 경로 ──────────────────────────────────────────────────────


class _FakeSandbox:
    workdir = "/xgeny/workspace/workflow/w1/workspace"
    extra_roots: List[str] = []
    readonly_roots: List[str] = []

    def __init__(self) -> None:
        self.files: Dict[str, bytes] = {}

    async def ensure(self) -> None:
        return None

    async def read_bytes(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_bytes(self, path: str, data: bytes) -> int:
        self.files[path] = data
        return len(data)


def test_reviewer_reads_through_the_sandbox() -> None:
    sb = _FakeSandbox()
    sb.files[f"{sb.workdir}/out/summary.json"] = b'{"n": 1}'
    ctx = ToolContext(working_dir=sb.workdir, sandbox=sb)
    state = PipelineState(session_id="s", model="m")
    state.messages = [{"role": "user", "content": "q"},
                      _assistant({"type": "text", "text": "out/summary.json 완료"})]
    state.final_text = "out/summary.json 완료"
    note = asyncio.run(DeliverableReviewer(lambda: ctx).review(state))
    assert note and "- out/summary.json — valid JSON object, 1 keys: n" in note


# ── 파이프라인 끝까지: 완료를 한 번 미루고, 고친 뒤 끝난다 ──────────────


class _Write(Tool):
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return "write"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        p = self.root / input["file_path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(input["content"])
        return ToolResult(content=f"wrote {input['file_path']}")


class _Client(BaseClient):
    """1: 5행 CSV 를 쓴다. 2: 완료 선언. (검토 뒤) 3: 3행으로 고친다. 4: 완료."""

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
                                   tool_input={"file_path": "out/audit.csv", "content": "id,issue\n" + "".join(f"{i},x\n" for i in range(5))})]
            return APIResponse(content=blocks, stop_reason="tool_use", usage=usage, model="fake")
        if n == 2:
            return APIResponse(content=[ContentBlock(type="text", text="out/audit.csv 에 감사 결과를 썼습니다. [COMPLETE]")],
                               stop_reason="end_turn", usage=usage, model="fake")
        if n == 3:
            blocks = [ContentBlock(type="tool_use", tool_use_id="t2", tool_name="Write",
                                   tool_input={"file_path": "out/audit.csv", "content": "id,issue\n1,x\n2,x\n3,x\n"})]
            return APIResponse(content=blocks, stop_reason="tool_use", usage=usage, model="fake")
        return APIResponse(content=[ContentBlock(type="text", text="정확히 3행으로 고쳤습니다. [COMPLETE]")],
                           stop_reason="end_turn", usage=usage, model="fake")


def _last_user_text(request: Any) -> Optional[str]:
    msgs = getattr(request, "messages", None) or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return None


def test_pipeline_defers_completion_once_and_shows_the_digest(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.register(_Write(tmp_path), core=True)
    client = _Client(api_key="k")
    ctx = ToolContext(working_dir=str(tmp_path), allowed_paths=[str(tmp_path)])
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client,
        stream=False, enable_compaction=False, registry=reg, tool_context=ctx,
    )
    text = runner.run_turn(pipe, "정확히 3행짜리 out/audit.csv 를 내라", PipelineState(session_id="s", model="m"))

    assert len(client.requests) == 4, "완료를 한 번 미루고(검토) 고친 뒤 끝나야 한다"
    digest = _last_user_text(client.requests[2])
    assert digest and digest.startswith("[Deliverable check") and "5 data rows" in digest
    assert "3행으로 고쳤습니다" in text
    assert (tmp_path / "out" / "audit.csv").read_text().count("\n") == 4


def test_pipeline_without_claimed_files_or_with_review_off_does_not_add_a_round_trip(tmp_path: Path) -> None:
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

    ctx = ToolContext(working_dir=str(tmp_path), allowed_paths=[str(tmp_path)])
    for review in (True, False):
        client = _Plain(api_key="k")
        pipe = runner.build_pipeline(
            name="t", provider="openai", model="m", api_key="k", llm_client=client,
            stream=False, enable_compaction=False, tool_context=ctx, enable_deliverable_review=review,
        )
        runner.run_turn(pipe, "안녕", PipelineState(session_id="s", model="m"))
        assert client.n == 1
