"""AgentTurnExecutor 프롬프트/도구 게이트 — 호스트가 주지 않는 것은 약속하지 않는다.

* [DELEGATION_GATE] host.build_turn_delegation 이 실제 백엔드(subagent_manager /
  task_runner / task_registry)를 돌려줄 때만 SDK SubAgent*/Task*/DelegateTask 가
  등록된다. {} (데스크톱 사이드카) → 아무것도 등록 안 됨 + '위임 미배선 — host 미제공'.
* [CLI_BRIDGE] host.cli_bridge_available(provider) 가 False 면 CLI 전용 노트
  (memory_* 이름 규약 / SELF_EVOLUTION / 위임 노트·스태시)를 붙이지 않고 메모리는
  '자동'이라고만 안내한다. 메서드 부재 → True(레거시 서버).
* 한 스위치가 옆 능력을 끌고 내려가지 않는다: 자기진화를 꺼도 위임·메모리는
  그대로다. (내장 도구를 통째로 끄던 enable_builtin_tools 는 폐기됐다 —
  표면을 정하는 축은 tool_exposure 의 계층 하나뿐이다.)
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from xgen_agent_runtime.host import runner as runner_mod
from xgen_agent_runtime.host._constants import (
    MEMORY_AUTO_PROMPT_BLOCK,
    MEMORY_PROMPT_BLOCK,
    SELF_EVOLUTION_PROMPT_BLOCK,
    _delegation_wired,
    default_prompt,
)
from xgen_agent_runtime.host.host import HostServices
from xgen_agent_runtime.host.turn_executor import AgentTurnExecutor
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult


# ── 대역 ──────────────────────────────────────────────────────────────


class _FakeMemoryProvider:
    async def close(self) -> None:  # 파이프라인 조립 실패 경로가 부른다
        return None


class _DelegateTaskStub(Tool):
    @property
    def name(self) -> str:
        return "DelegateTask"

    @property
    def description(self) -> str:
        return "stub"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:  # noqa: A002
        return ToolResult(content="ok")


_WIRED_EXTRAS = {
    "subagent_manager": object(),
    "task_runner": object(),
    "task_registry": object(),
}


class _FakeHost:
    """HostServices 최소 대역 — 실행기가 파이프라인 조립 직전까지 닿는 표면만."""

    def __init__(
        self,
        *,
        delegation_extras: Optional[Dict[str, Any]] = None,
        memory: bool = True,
        cli_bridge: Optional[bool] = None,
    ) -> None:
        self._delegation_extras = delegation_extras
        self._memory = memory
        self.calls: List[str] = []
        self.cli_params: Optional[Dict[str, Any]] = None
        if cli_bridge is not None:
            # 인스턴스 속성으로 주입 — None 이면 메서드 자체가 없는 호스트(레거시).
            self.cli_bridge_available = lambda provider, _v=cli_bridge: _v  # type: ignore[assignment]

    # A
    def setting(self, name: str, default: str = "") -> str:
        return default

    def setting_truthy(self, name: str) -> bool:
        return False

    def resolve_model(self, provider, params):
        return "m"

    def resolve_api_key(self, provider, params):
        return "k"

    def resolve_base_url(self, provider, params):
        return None

    def resolve_credentials(self, provider, params):
        return None

    # B
    def probe_connector_workspace(self, *a, **k):
        return None

    def make_sandbox(self, *a, **k):
        return None

    def agent_workspace_dir(self, workflow_id, *, create=True):
        return "/tmp/ws"

    def workspace_storage_root(self, workflow_id):
        return "/tmp/ws-storage"

    def hydrate_workspace(self, workflow_id, run_dir):
        return None

    def publish_workspace(self, *a, **k):
        return None

    def environment_prompt(self, *a, **k):
        return ""

    # C
    def build_memory_provider(self, workflow_id, interaction_id):
        return _FakeMemoryProvider() if self._memory else None

    # D






    def jobs_prompt_block(self):
        return ""



    # E
    def build_connector_mcp_tools(self, *a, **k):
        return []

    def build_job_tools(self, *a, **k):
        return []

    def register_workflow_self_tools(self, registry, **k):
        return None  # 데스크톱: WorkflowSelf 미제공

    def build_turn_delegation(self, **k):
        self.calls.append("build_turn_delegation")
        return dict(self._delegation_extras) if self._delegation_extras is not None else {}

    def is_report_turn(self, text):
        return False

    def delegation_extra_tool_classes(self):
        return {"DelegateTask": _DelegateTaskStub} if self._delegation_extras else {}

    def delegation_workspace(self, workflow_id):
        return "/tmp/ws"

    def drain_pending_reports(self, *a, **k):
        return ""

    def make_sub_cli_client_factory(self, *a, **k):
        return None

    def register_forged_tools(self, *a, **k):
        return None

    def register_builtin_tools(self, registry, **k):
        return {"tools": [], "extras": {}, "families": []}

    def build_run_tool_context(self, **k):
        return SimpleNamespace(extras=dict(k.get("extras") or {}))

    def load_ssh_servers(self):
        return []

    # H
    def rag_context_builder(self, text, item):
        return None

    def fetch_vllm_max_model_len(self, base_url, model):
        return None

    def agent_vault_root(self, workflow_id):
        return "/tmp/vault"

    def build_turn_memory_llm(self, *a, **k):
        return None

    # G / F
    def finalize_turn(self, **k):
        return None

    def build_cli_runtime(self, provider, params):
        # kwargs 사전 그 자체가 넘어온다 — 위임 스태시(_delegation_extras) 관찰 지점.
        self.cli_params = params
        return object(), None


@pytest.fixture
def capture(monkeypatch):
    """build_pipeline 인자(system_prompt/registry)를 잡고 스트림은 빈 iterator 로."""
    seen: Dict[str, Any] = {}

    def _fake_build_pipeline(**kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(runner_mod, "build_pipeline", _fake_build_pipeline)
    def _fake_stream_turn(*args, **kwargs):
        seen["stream_input"] = args[1]
        return iter([])

    monkeypatch.setattr(runner_mod, "stream_turn", _fake_stream_turn)
    monkeypatch.setattr(runner_mod, "run_turn", lambda *a, **k: "")
    return seen


def _run(host: Any, capture: Dict[str, Any], **over: Any) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        text="hi",
        provider="openai",
        workflow_id="wf-1",
        workflow_name="wf",
        user_id="u1",
        interaction_id="inter-1",
        streaming=True,
        memory_distill=False,
        enable_compaction=False,
    )
    kw.update(over)
    out = AgentTurnExecutor().run(host, **kw)
    assert list(out) == [], "파이프라인 조립 실패 경로로 빠지면 안 된다 (fake 가 불완전)"
    assert "system_prompt" in capture
    return capture


def _registry_names(capture: Dict[str, Any]) -> List[str]:
    reg = capture.get("registry")
    if reg is None:
        return []
    return list(reg.list_names())


def test_structured_image_input_survives_host_executor(capture) -> None:
    host = _FakeHost(delegation_extras={}, memory=False)
    image = {
        "kind": "image",
        "mime_type": "image/png",
        "data": "iVBORw0KGgo=",
        "workspace_path": "uploads/turn-1/example.png",
    }
    seen = _run(
        host,
        capture,
        text={"text": "describe it", "attachments": [image]},
    )
    assert seen["stream_input"] == {
        "text": "describe it",
        "attachments": [image],
        "metadata": {},
    }


def test_openai_content_array_becomes_canonical_image_input(capture) -> None:
    host = _FakeHost(delegation_extras={}, memory=False)
    seen = _run(
        host,
        capture,
        text=[
            {"type": "text", "text": "what is shown?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ],
    )
    assert seen["stream_input"] == {
        "text": "what is shown?",
        "attachments": [
            {"kind": "image", "mime_type": "image/png", "data": "iVBORw0KGgo="}
        ],
        "metadata": {},
    }


# ── [DELEGATION_GATE] ─────────────────────────────────────────────────


def test_delegation_wired_helper_contract() -> None:
    assert not _delegation_wired({})
    assert not _delegation_wired(None)
    assert not _delegation_wired({"agent_depth": 0})
    assert _delegation_wired({"subagent_manager": object()})
    assert _delegation_wired({"task_runner": object(), "task_registry": object()})


def test_sdk_delegation_registered_when_host_provides_backend(capture, caplog) -> None:
    host = _FakeHost(delegation_extras=_WIRED_EXTRAS)
    with caplog.at_level(logging.INFO):
        seen = _run(host, capture)
    names = _registry_names(seen)
    assert "DelegateTask" in names
    assert "SubAgentSpawn" in names and "SubAgentAssign" in names
    assert "TaskCreate" in names and "TaskStop" in names
    assert "위임 미배선" not in caplog.text
    assert "build_turn_delegation" in host.calls


def test_sdk_delegation_not_registered_when_host_returns_empty(capture, caplog) -> None:
    host = _FakeHost(delegation_extras={})
    with caplog.at_level(logging.INFO):
        seen = _run(host, capture)
    names = _registry_names(seen)
    assert not [n for n in names if n.startswith("SubAgent") or n.startswith("Task")]
    assert "DelegateTask" not in names
    assert "위임 미배선 — host 미제공" in caplog.text
    # SDK 경로엔 애초 위임 노트가 없다 — CLI 노트 문구도 섞이지 않는다.
    assert "mcp__connector__DelegateTask" not in seen["system_prompt"]


# ── [CLI_BRIDGE] ──────────────────────────────────────────────────────


def test_host_services_declares_optional_cli_bridge_available() -> None:
    fn = getattr(HostServices, "cli_bridge_available", None)
    assert fn is not None
    assert [p for p in inspect.signature(fn).parameters if p != "self"] == ["provider"]


def test_cli_legacy_host_without_probe_keeps_notes(capture) -> None:
    """메서드 부재 → True: 서버 레거시 동작(노트 전부 유지)."""
    host = _FakeHost(delegation_extras=_WIRED_EXTRAS, cli_bridge=None)
    assert not hasattr(host, "cli_bridge_available")
    seen = _run(host, capture, provider="claude_code")
    sp = seen["system_prompt"]
    assert MEMORY_PROMPT_BLOCK in sp and "mcp__connector__memory_write" in sp
    assert MEMORY_AUTO_PROMPT_BLOCK not in sp
    assert SELF_EVOLUTION_PROMPT_BLOCK in sp and "mcp__connector__WorkflowSelf" in sp
    assert "mcp__connector__DelegateTask" in sp
    assert host.cli_params is not None and "_delegation_extras" in host.cli_params


def test_cli_bridge_available_true_keeps_notes(capture) -> None:
    host = _FakeHost(delegation_extras=_WIRED_EXTRAS, cli_bridge=True)
    seen = _run(host, capture, provider="claude_code")
    sp = seen["system_prompt"]
    assert "mcp__connector__memory_write" in sp
    assert SELF_EVOLUTION_PROMPT_BLOCK in sp
    assert "mcp__connector__DelegateTask" in sp
    assert "_delegation_extras" in host.cli_params


def test_cli_bridge_unavailable_drops_tool_notes_and_memory_is_automatic(capture, caplog) -> None:
    host = _FakeHost(delegation_extras=_WIRED_EXTRAS, cli_bridge=False)
    with caplog.at_level(logging.INFO):
        seen = _run(host, capture, provider="claude_code")
    sp = seen["system_prompt"]
    # 메모리: 도구 광고 없음, 자동 계층 안내만
    assert MEMORY_PROMPT_BLOCK not in sp
    assert "memory_categories(" not in sp and "mcp__connector__memory" not in sp
    assert MEMORY_AUTO_PROMPT_BLOCK in sp
    # self-evolution / 위임: 블록·노트·스태시 전부 없음
    assert SELF_EVOLUTION_PROMPT_BLOCK not in sp and "WorkflowSelf" not in sp
    assert "mcp__connector__DelegateTask" not in sp
    assert "_delegation_extras" not in host.cli_params
    assert "build_turn_delegation" not in host.calls  # 쓸 데 없는 백엔드 생성도 없다
    assert "self-evolution 미배선 — CLI 브릿지 없음" in caplog.text
    assert "위임 미배선 — CLI 브릿지 없음" in caplog.text


def test_codex_bridge_unavailable_memory_automatic_no_self_evolution(capture) -> None:
    host = _FakeHost(delegation_extras={}, cli_bridge=False)
    seen = _run(host, capture, provider="codex")
    sp = seen["system_prompt"]
    assert MEMORY_AUTO_PROMPT_BLOCK in sp
    assert "'connector' MCP server" not in sp
    assert SELF_EVOLUTION_PROMPT_BLOCK not in sp


def test_codex_bridge_available_keeps_connector_note(capture) -> None:
    host = _FakeHost(delegation_extras={}, cli_bridge=True)
    seen = _run(host, capture, provider="codex")
    sp = seen["system_prompt"]
    assert MEMORY_PROMPT_BLOCK in sp and "'connector' MCP server" in sp
    assert SELF_EVOLUTION_PROMPT_BLOCK in sp


def test_sdk_provider_memory_block_unchanged_regardless_of_probe(capture) -> None:
    """SDK 경로는 registry 에 memory 도구를 직접 등록 — 프로브와 무관하게 블록 유지."""
    host = _FakeHost(delegation_extras={}, cli_bridge=False)
    seen = _run(host, capture, provider="openai")
    sp = seen["system_prompt"]
    assert MEMORY_PROMPT_BLOCK in sp
    assert MEMORY_AUTO_PROMPT_BLOCK not in sp
    assert "memory_write" in _registry_names(seen)


# ── System Prompt: 명시적 "" 과 미지정(None/키 없음)을 구분 ────────────────
# kwargs.get("system_prompt") or default_prompt 였을 때는 사용자가 System
# Prompt 를 의도적으로 비워도 조용히 기본 문구로 되돌아가 "정말 비우기" 가
# 불가능했다. 키가 아예 없을 때만 기본값을 쓰도록 고친 회귀 방지 테스트.


def test_empty_system_prompt_is_preserved_not_replaced_by_default(capture) -> None:
    """System Prompt 를 사용자가 명시적으로 비우면("") 기본 문구로 대체되면 안 된다."""
    host = _FakeHost(delegation_extras={}, cli_bridge=False, memory=False)
    seen = _run(host, capture, provider="openai", system_prompt="")
    sp = seen["system_prompt"]
    assert default_prompt not in sp
    assert sp == ""


def test_missing_system_prompt_still_falls_back_to_default(capture) -> None:
    """system_prompt 키 자체를 안 주면(레거시 호출부 등) 여전히 기본 문구를 쓴다."""
    host = _FakeHost(delegation_extras={}, cli_bridge=False, memory=False)
    seen = _run(host, capture, provider="openai")  # system_prompt 미지정
    sp = seen["system_prompt"]
    assert default_prompt in sp


# ── 내장 도구 표면은 설정이 아니라 계층이다 ───────────────────────────
#
# 예전엔 ``enable_builtin_tools`` 스위치가 있었다. 그 스위치는 우리가 **주고 싶은**
# 도구(셸·파일·웹·문서)를 통째로 껐는데, 정작 끄고 싶었던 것은 CLI 하네스가
# 자기 것으로 들고 오는 네이티브 도구였다. 끄면 셸도 파일도 사라져 에이전트가
# 아무것도 못 하고, 켜면 계층이 없어 전부 쏟아졌다 — 어느 쪽도 원하는 상태가
# 아니어서 스위치를 걷어냈다. 남은 축은 하나, ``tool_exposure`` 의 계층이다.


def test_run_ctx_는_자기진화만_꺼도_살아_있다(capture, caplog) -> None:
    """자기진화를 꺼도 위임·메모리는 그대로다.

    WorkflowSelf 는 registry + workflow_id 만 필요하다(편집은 DB, workspace 불필요).
    한 스위치가 옆 능력을 조용히 끌고 내려가던 회귀(감사 HIGH)를 여기서 막는다.
    """
    host = _FakeHost(delegation_extras=_WIRED_EXTRAS, cli_bridge=True)
    with caplog.at_level(logging.INFO):
        seen = _run(host, capture, provider="claude_code", enable_self_evolution=False)
    sp = seen["system_prompt"]
    assert SELF_EVOLUTION_PROMPT_BLOCK not in sp
    assert "mcp__connector__DelegateTask" in sp
    assert "_delegation_extras" in host.cli_params
    assert "mcp__connector__memory_write" in sp


def test_cli_legacy_host_also_keeps_self_evolution(capture) -> None:
    """프로브가 없는 레거시 호스트도 같은 판정(브릿지 있음으로 간주)."""
    host = _FakeHost(delegation_extras=_WIRED_EXTRAS, cli_bridge=None)
    seen = _run(host, capture, provider="claude_code")
    sp = seen["system_prompt"]
    assert SELF_EVOLUTION_PROMPT_BLOCK in sp
    assert "mcp__connector__DelegateTask" in sp
    assert "mcp__connector__memory_write" in sp


# ── 그래프 연결 도구는 CLI 백엔드에도 도달한다 ─────────────────────────
#
# CLI 는 파이프라인 registry 를 보지 못하므로, 실행기가 그걸 **버리지 않고**
# host 의 CLI 런타임 빌더에 넘겨야 브릿지가 광고할 수 있다. 예전엔 정말로 버렸고
# (registry = None), 그래서 캔버스에서 붙인 도구 노드가 claude_code/codex 에
# 영원히 도달하지 않았다.


class _GraphTool(Tool):
    name = "jira_search"
    description = "search jira"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, input, context):  # noqa: A002
        return ToolResult(content="ok")


def test_cli_hands_graph_tools_to_the_bridge_instead_of_dropping_them(capture) -> None:
    host = _FakeHost(delegation_extras={})
    _run(host, capture, provider="claude_code", tools=[_GraphTool()])
    # 파이프라인에는 안 넘긴다(Stage 10 이 안 돈다).
    assert capture.get("registry") is None
    # 대신 CLI 런타임 빌더가 받는 kwargs 에 실려 있다.
    graph = host.cli_params.get("_graph_tools")
    assert graph is not None, "그래프 도구가 버려졌다 — 브릿지가 광고할 수 없다"
    assert "jira_search" in graph.list_names()


def test_sdk_backend_keeps_graph_tools_in_the_pipeline_registry(capture) -> None:
    """SDK 는 원래대로 registry 로 직접 쓴다 — 두 경로가 같은 도구를 갖는다."""
    host = _FakeHost(delegation_extras={})
    _run(host, capture, provider="openai", tools=[_GraphTool()])
    assert "jira_search" in _registry_names(capture)


def test_cli_says_so_when_graph_tools_cannot_be_delivered(capture, caplog) -> None:
    """브릿지가 없으면 정말 못 준다 — 그때 침묵하면 에이전트가 원인을 지어낸다.

    프롬프트에 사실을 실어, 사용자가 물으면 설정 문제라고 정확히 답하게 한다.
    """
    host = _FakeHost(delegation_extras={}, cli_bridge=False)
    out = _run(host, capture, provider="claude_code", tools=[_GraphTool()])
    assert "cannot be delivered on this backend" in out["system_prompt"]
    assert "1 tool(s)" in out["system_prompt"]


def test_no_such_note_when_the_bridge_is_available(capture) -> None:
    host = _FakeHost(delegation_extras={})
    out = _run(host, capture, provider="claude_code", tools=[_GraphTool()])
    assert "cannot be delivered" not in out["system_prompt"]
