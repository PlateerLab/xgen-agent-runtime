"""사람이 거부한 동작을 같은 턴에서 다시 묻지 않는다 (stages/s10_tool/denial_guard.py).

재현: 2026-09-23 dev(gpt-4.1, trace 46745) — 사용자 PC 에서 ``rm -rf`` 를 거부했는데 모델이 같은
명령을 세 번 불러 확인 창이 세 번 떴다. 같은 질문을 거듭 받으면 사람은 결국 잘못 누르고,
"이번만 허용" 을 한 번이라도 누르면 그대로 실행된다.
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.host.tools import adapt_tools
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.stages.s10_tool import denial_guard

DEX_DENIAL = "사용자가 이 명령의 실행을 거부했습니다 (위험할 수 있는 명령)."


class _ShellArgs(BaseModel):
    command: str


def _connector_shell(dialogs: List[str], *, error: Exception | None = None):
    """커넥터 Shell 흉내 — 부를 때마다 확인 창이 한 번 뜨고 사용자는 거부한다."""

    async def _run(command: str) -> str:
        dialogs.append(command)
        raise error or RuntimeError(DEX_DENIAL)

    return StructuredTool.from_function(
        coroutine=_run, name="mcp_local_Shell", description="run on the user's PC", args_schema=_ShellArgs
    )


class _Stubborn(BaseClient):
    """거부당해도 같은 rm -rf 를 계속 부른다 — 따옴표를 바꿔 가며(46745 의 모양)."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        n = len(self.requests)
        usage = TokenUsage(input_tokens=100, output_tokens=5)
        if "[Stopped:" in str(request.messages[-1].get("content")):
            return APIResponse(content=[ContentBlock(type="text", text="삭제하지 못했습니다 — 거부되었습니다.")],
                               stop_reason="end_turn", usage=usage, model="fake")
        cmd = "rm -rf 임시_삭제테스트" if n % 2 else "rm -rf '임시_삭제테스트'"
        return APIResponse(
            content=[ContentBlock(type="tool_use", tool_use_id=f"t{n}", tool_name="mcp_local_Shell",
                                  tool_input={"command": cmd})],
            stop_reason="tool_use", usage=usage, model="fake",
        )


def _run(dialogs: List[str], **tool_kw: Any):
    reg = adapt_tools([_connector_shell(dialogs, **tool_kw)], core=True)
    client = _Stubborn(api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, registry=reg, max_iterations=50, turn_input_budget_tokens=None,
    )
    return runner.run_turn(pipe, "임시 폴더 지워줘", PipelineState(session_id="s", model="m"))


def test_the_confirmation_dialog_appears_only_once_per_turn():
    """고치기 전: 거부 뒤에도 같은 명령이 계속 실행을 시도해 확인 창이 거듭 떴다."""
    dialogs: List[str] = []
    text = _run(dialogs)
    assert len(dialogs) == 1, dialogs
    assert "[안내: 같은 작업이 반복되어" in text  # 거부 누적으로 반복 종료가 턴을 끝냄


def test_a_structured_denial_code_works_without_the_legacy_phrase():
    """DeX 가 구조화 코드를 보내기 시작하면(권장 계약) 한국어 문구 없이도 알아본다."""

    class Denied(Exception):
        code = "user_denied"

    dialogs: List[str] = []
    _run(dialogs, error=Denied("blocked"))
    assert len(dialogs) == 1


def test_an_ordinary_failure_is_not_a_denial():
    """평범한 실패(명령 오류)는 거부가 아니다 — 반복 가드의 기존 규칙을 따른다."""
    dialogs: List[str] = []
    _run(dialogs, error=RuntimeError("bash: line 1: rmx: command not found"))
    assert len(dialogs) > 1


# ── 서명 ─────────────────────────────────────────────────────────────


def _tc(cmd: str, name: str = "mcp_local_Shell"):
    return {"tool_name": name, "tool_use_id": "x", "tool_input": {"command": cmd}}


def test_quoting_and_spacing_do_not_make_a_different_action():
    base = denial_guard.signature(_tc("rm -rf 임시_삭제테스트"))
    assert denial_guard.signature(_tc("rm -rf '임시_삭제테스트'")) == base
    assert denial_guard.signature(_tc('rm  -rf  "임시_삭제테스트"')) == base


def test_a_different_target_is_a_different_action():
    assert denial_guard.signature(_tc("rm -rf a")) != denial_guard.signature(_tc("rm -rf b"))
    assert denial_guard.signature(_tc("rm -rf a", "Bash")) != denial_guard.signature(_tc("rm -rf a"))


def test_denials_are_forgotten_at_the_next_turn():
    """다음 턴에 사용자가 허락하면 다시 물을 수 있어야 한다."""
    state = PipelineState(session_id="s", model="m")
    state.shared[denial_guard.DENIED_KEY] = {"x": "denied"}
    state.begin_turn()
    assert denial_guard.DENIED_KEY not in state.shared
