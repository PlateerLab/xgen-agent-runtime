"""앞 턴에 연 도구는 다음 턴에도 열려 있다 (tools.gates.restore_from_history).

재현: 2026-09-23 dev(gpt-4.1, "업무 비서"). 모델이 말로 "정말 삭제할까요?" 를 묻고(도구 0회)
사용자가 "정말 삭제해도 좋다" 고 답한 다음 턴 — 모델은 문(LocalControl)을 다시 열지 않고
브라우저 안내만 5번 부르다 반복 종료로 끝났다. 호스트가 턴마다 레지스트리를 새로 만들어서,
앞 턴에 열어 쓴 셸이 다시 숨어 있었다. 가장 자연스러운 확인 흐름에서 실패했다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.host.tool_exposure import is_turn_one
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.gates import restore_from_history
from xgen_agent_runtime.tools.registry import ToolRegistry


class _T(Tool):
    def __init__(self, name: str) -> None:
        self._n = name

    @property
    def name(self) -> str:
        return self._n

    @property
    def description(self) -> str:
        return self._n

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content="ok")


LOCAL = ["mcp_local_LocalControl", "mcp_local_Shell", "mcp_local_ListDir", "mcp_local_WriteFile"]


def _fresh_registry() -> ToolRegistry:
    """호스트가 턴마다 하듯 — 새 레지스트리, 턴-1 계획대로(문만 보이고 방은 숨음)."""
    reg = ToolRegistry()
    for n in LOCAL + ["Bash", "Read"]:
        reg.register(_T(n), core=is_turn_one(n))
    return reg


def _history_where_the_gate_was_used() -> List[Dict[str, Any]]:
    """앞 턴들: 문을 열고 ListDir 를 썼다 → 말로 확인을 물었다(도구 0회)."""
    return [
        {"role": "user", "content": "내 PC 작업 폴더 파일 보여줘"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a1", "name": "mcp_local_LocalControl", "input": {}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a1", "content": "map"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a2", "name": "mcp_local_ListDir", "input": {"path": "."}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a2", "content": "d 임시"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "파일은 …입니다."}]},
        {"role": "user", "content": "임시 폴더 지워줘"},
        {"role": "assistant", "content": [{"type": "text", "text": "정말 삭제할까요?"}]},
    ]


class _Recorder(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.tool_names: List[List[str]] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.tool_names.append(sorted(t.get("name") for t in (request.tools or [])))
        return APIResponse(content=[ContentBlock(type="text", text="삭제했습니다.")],
                           stop_reason="end_turn", usage=TokenUsage(input_tokens=10, output_tokens=2), model="fake")


def test_the_confirmation_turn_can_call_the_shell_directly():
    reg = _fresh_registry()
    assert not reg.is_exposed("mcp_local_Shell")  # 새 레지스트리 — 셸은 숨어 있다
    client = _Recorder(api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, registry=reg, turn_input_budget_tokens=None,
    )
    state = PipelineState(session_id="s", model="m")
    state.messages = _history_where_the_gate_was_used()  # 호스트가 싣는 기록
    runner.run_turn(pipe, "정말 삭제해도 좋다", state)
    first = client.tool_names[0]
    assert "mcp_local_Shell" in first, first  # 문을 열어 둔 대화는 방도 열려 있다
    assert "mcp_local_ListDir" in first


def test_restore_opens_a_gates_family_and_used_tools():
    reg = _fresh_registry()
    out = restore_from_history(reg.list_names(), reg.is_exposed, _history_where_the_gate_was_used())
    assert set(out) == {"mcp_local_Shell", "mcp_local_ListDir", "mcp_local_WriteFile"}


def test_a_conversation_without_tool_use_restores_nothing():
    reg = _fresh_registry()
    msgs = [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": [{"type": "text", "text": "네"}]}]
    assert restore_from_history(reg.list_names(), reg.is_exposed, msgs) == []


def test_unregistered_tools_in_history_are_ignored():
    """기록에 있어도 이번 턴에 등록되지 않은 도구(커넥터가 끊김 등)는 열 수 없다."""
    reg = ToolRegistry()
    reg.register(_T("Bash"), core=True)
    msgs = [{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "mcp_local_Shell", "input": {}}]}]
    assert restore_from_history(reg.list_names(), reg.is_exposed, msgs) == []
