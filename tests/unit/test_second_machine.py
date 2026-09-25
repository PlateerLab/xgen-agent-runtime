"""기계가 둘인 대화에서 sandbox 의 '없음' 은 '여기엔 없음' 이다 (stages/s10_tool/second_machine.py).

재현: 2026-09-26 dev(gpt-4.1, trace 46792) — "작업 폴더의 dex_note_0926.md 첫 줄" 을 sandbox Read 로 한 번
찾고 "없다" 고 답했다. 파일은 사용자 PC 에 있었다.
"""

from xgen_agent_runtime.stages.s10_tool import second_machine as sm

PC = ["Read", "Bash", "Glob", "mcp_local_LocalControl", "mcp_local_ReadFile"]
WEB = ["Read", "Bash", "Glob", "WebFetch"]


def _run(name, text, names=PC, shared=None, is_error=True):
    shared = {} if shared is None else shared
    r = {"tool_use_id": "t1", "content": text, "is_error": is_error}
    n = sm.annotate([{"tool_use_id": "t1", "tool_name": name}], [r], names, shared)
    return n, r["content"], shared


def test_sandbox_read_miss_points_to_the_users_computer():
    n, out, _ = _run("Read", "File not found: /xgeny/workspace/workflow/wf/workspace/dex_note_0926.md")
    assert n == 1 and "[Not in your sandbox]" in out and "mcp_local_LocalControl" in out


def test_glob_and_bash_misses_too():
    assert _run("Glob", "No files matching '**/*.md' in /ws", is_error=False)[0] == 1
    assert _run("Bash", "ls: cannot access 'x': No such file or directory\nExit code: 2")[0] == 1


def test_web_conversation_has_one_machine_and_gets_no_note():
    n, out, _ = _run("Read", "File not found: /ws/a.md", names=WEB)
    assert n == 0 and "[Not in your sandbox]" not in out


def test_the_users_own_tools_are_not_annotated():
    assert _run("mcp_local_ReadFile", "파일을 찾을 수 없습니다: /home/u/a.md")[0] == 0


def test_successful_results_are_untouched():
    assert _run("Read", "1\thello", is_error=False)[0] == 0


def test_at_most_max_notes_per_turn():
    shared = {}
    counts = [_run("Read", "File not found: /ws/x", shared=shared)[0] for _ in range(4)]
    assert sum(counts) == sm.MAX_NOTES


def test_unprefixed_local_control_is_not_a_second_machine():
    """접두 없는 LocalControl(CLI 경로 등)은 PC 연결 신호로 보지 않는다 — 게이트 규칙과 같다."""
    assert sm.local_gate(["Read", "LocalControl"]) is None


def test_the_model_actually_receives_the_note_in_a_real_turn(tmp_path):
    """파이프라인 한 턴: sandbox Read 가 없음 → 다음 모델 요청의 tool_result 에 안내가 들어 있다.

    (스트림의 tool.call_complete 사건은 안내를 붙이기 전에 나가므로 사건만 보고 판단하면 안 된다.)
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    from xgen_agent_runtime.core.state import PipelineState, TokenUsage
    from xgen_agent_runtime.host import runner
    from xgen_agent_runtime.host.tools import adapt_tools
    from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
    from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
    from xgen_agent_runtime.tools.base import ToolContext
    from xgen_agent_runtime.tools.built_in.read_tool import ReadTool

    class _NoArgs(BaseModel):
        pass

    def _lc(**_):
        return "{}"

    gate = StructuredTool.from_function(func=_lc, name="mcp_local_LocalControl", description="PC", args_schema=_NoArgs)
    reg = adapt_tools([gate], core=True)
    reg.register(ReadTool())

    class _Client(BaseClient):
        provider = "fake"
        capabilities = ClientCapabilities()

        def __init__(self, **kw):
            super().__init__(**kw)
            self.requests = []

        async def _send(self, request, *, purpose=""):
            self.requests.append(request)
            usage = TokenUsage(input_tokens=10, output_tokens=2)
            if len(self.requests) == 1:
                return APIResponse(content=[ContentBlock(type="tool_use", tool_use_id="t1", tool_name="Read",
                                                         tool_input={"file_path": "dex_note_0926.md"})],
                                   stop_reason="tool_use", usage=usage, model="fake")
            return APIResponse(content=[ContentBlock(type="text", text="ok")], stop_reason="end_turn", usage=usage, model="fake")

    client = _Client(api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, registry=reg, max_iterations=5, turn_input_budget_tokens=None,
        tool_context=ToolContext(session_id="t", working_dir=str(tmp_path)),
    )
    runner.run_turn(pipe, "dex_note_0926.md 첫 줄?", PipelineState(session_id="t", model="m"))
    seen = str(client.requests[1].messages[-1])
    assert "File not found" in seen and "[Not in your sandbox]" in seen and "mcp_local_LocalControl" in seen
