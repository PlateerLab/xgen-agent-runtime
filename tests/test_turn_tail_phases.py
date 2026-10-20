"""답이 다 나온 뒤의 기다림을 호스트에 알린다 (host.note_turn_phase).

Agent Geny 의 턴은 마지막 글자 뒤에도 한동안 끝나지 않는다. 답에 나온 파일을 작업 공간에서 읽어
점검하고(Stage 16 완료 점검), 이번 턴을 대화 기억에 쓰고(Stage 18), 실행 기록을 남긴다. 그동안
화면에는 커서만 깜박여 무엇을 하는지 알 수 없었다. 런타임은 이 구간을 호스트에 알리고, 서버
호스트는 화면에 보낸다(turn_phase). 알림이 없는 호스트(커넥터 등)에는 아무 일도 없다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock, TokenUsage
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult
from xgen_agent_runtime.tools.registry import ToolRegistry


class _Write(Tool):
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return "write a file"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities()

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        p = self.root / input["file_path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(input["content"])
        return ToolResult(content=f"wrote {input['file_path']}")


class _Client(BaseClient):
    """1: 파일을 쓴다. 2: 그 파일을 말하며 끝낸다(완료 점검이 그 파일을 읽는다)."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        if len(self.requests) == 1:
            return APIResponse(
                content=[ContentBlock(type="tool_use", tool_use_id="t1", tool_name="Write",
                                      tool_input={"file_path": "out/a.csv", "content": "id\n1\n"})],
                stop_reason="tool_use", usage=usage, model="fake",
            )
        return APIResponse(content=[ContentBlock(type="text", text="out/a.csv 완료. [COMPLETE]")],
                           stop_reason="end_turn", usage=usage, model="fake")


class _Host:
    def __init__(self) -> None:
        self.notes: List[tuple] = []

    def note_turn_phase(self, phase: str, state: str, **extra: Any) -> None:
        self.notes.append((phase, state, extra))


def _pipeline(tmp_path: Path) -> Any:
    reg = ToolRegistry()
    reg.register(_Write(tmp_path), core=True)
    ctx = ToolContext(working_dir=str(tmp_path), allowed_paths=[str(tmp_path)])
    return runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=_Client(api_key="k"),
        stream=False, enable_compaction=False, registry=reg, tool_context=ctx,
    )


def test_the_completion_review_after_the_answer_is_announced(tmp_path: Path) -> None:
    host = _Host()
    out = list(runner.stream_turn(_pipeline(tmp_path), "out/a.csv 를 내라",
                                  PipelineState(session_id="s", model="m"), host=host))

    assert any(isinstance(c, str) and "완료" in c for c in out)
    review = [(p, s) for p, s, _ in host.notes if p == "review"]
    assert review == [("review", "start"), ("review", "done")], host.notes
    done = next(extra for p, s, extra in host.notes if p == "review" and s == "done")
    assert isinstance(done.get("elapsed_ms"), int)


def test_hosts_without_the_hook_are_left_alone(tmp_path: Path) -> None:
    assert runner._TurnPhaseNotifier.for_host(None) is None
    assert runner._TurnPhaseNotifier.for_host(object()) is None
    # 알림 없는 호스트로도 턴은 그대로 돈다.
    out = list(runner.stream_turn(_pipeline(tmp_path), "out/a.csv 를 내라",
                                  PipelineState(session_id="s", model="m"), host=object()))
    assert any(isinstance(c, str) and "완료" in c for c in out)


def test_a_broken_host_hook_does_not_break_the_turn(tmp_path: Path) -> None:
    class _Boom:
        def note_turn_phase(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("화면 끊김")

    out = list(runner.stream_turn(_pipeline(tmp_path), "out/a.csv 를 내라",
                                  PipelineState(session_id="s", model="m"), host=_Boom()))
    assert any(isinstance(c, str) and "완료" in c for c in out)


def test_writing_the_turn_to_memory_and_the_execution_record_are_announced() -> None:
    """기억이 켜진 에이전트는 답 뒤에 대화 기억(Stage 18)과 실행 기록을 쓴다 — 둘 다 알린다."""
    from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider

    class _Say(BaseClient):
        provider = "fake"
        capabilities = ClientCapabilities()

        async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
            return APIResponse(content=[ContentBlock(type="text", text="안녕하세요")], stop_reason="end_turn",
                               usage=TokenUsage(input_tokens=1, output_tokens=1), model="fake")

    pipe = runner.build_pipeline(
        name="t", provider="anthropic", model="m", api_key="k", llm_client=_Say(api_key="k"), stream=False,
        enable_compaction=False, memory_provider=EphemeralMemoryProvider(), turn_input_budget_tokens=None,
    )
    host = _Host()
    list(runner.stream_turn(pipe, "안녕", PipelineState(session_id="s", model="m"), host=host))

    seen = [(p, s) for p, s, _ in host.notes]
    assert ("remember", "start") in seen and ("remember", "done") in seen, seen
    assert seen.index(("record", "start")) < seen.index(("record", "done")), seen
    assert seen.index(("remember", "done")) < seen.index(("record", "start")), "답 → 기억 → 실행 기록 순서"
