"""AgentTurnExecutor 프롬프트/도구 게이트 — 호스트가 주지 않는 것은 약속하지 않는다.

* [ONE_SURFACE] 표면은 provider 와 무관하게 하나다. SDK 는 레지스트리를 파이프라인으로, CLI
  (claude_code·codex)는 **같은 레지스트리**를 ``TurnToolSurface`` 로 묶어 호스트 브릿지에 넘긴다.
  CLI 프롬프트는 SDK 프롬프트 + (Stage 3 가 SDK 에 붙이는 것과 같은) 숨김 목록 + 이름 규약 한 줄이다.
* [CLI_BRIDGE] host.cli_bridge_available(provider) 가 False 면 도구가 CLI 에 닿지 않는다 — 도구를
  약속하지 않고 메모리는 '자동'이라고만 안내한다. 메서드 부재 → True(레거시 서버).
* 하위 에이전트 위임은 4.70.0 에서 제거됐다 — 호스트 프로토콜에도, 표면에도 없다.
* 한 스위치가 옆 능력을 끌고 내려가지 않는다: 자기진화를 꺼도 메모리는 그대로다.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.host import runner as runner_mod
from xgen_agent_runtime.host._constants import (
    MEMORY_AUTO_PROMPT_BLOCK,
    MEMORY_PROMPT_BLOCK,
    SELF_EVOLUTION_PROMPT_BLOCK,
    cli_tool_naming_note,
    default_prompt,
)
from xgen_agent_runtime.host.host import HostServices
from xgen_agent_runtime.host.rollouts import (
    ROLLOUT_ENABLED_SETTING,
    rollout_directory,
)
from xgen_agent_runtime.host.turn_executor import AgentTurnExecutor
from xgen_agent_runtime.host.workspace_fast_path import WORKSPACE_FAST_PATH_SETTING
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.built_in.bash_tool import BashTool


# ── 대역 ──────────────────────────────────────────────────────────────


class _FakeMemoryProvider:
    async def close(self) -> None:  # 파이프라인 조립 실패 경로가 부른다
        return None


class _FakeHost:
    """HostServices 최소 대역 — 실행기가 파이프라인 조립 직전까지 닿는 표면만."""

    def __init__(
        self,
        *,
        memory: bool = True,
        cli_bridge: Optional[bool] = None,
        rollout_enabled: bool = False,
        storage_root: str = "/tmp/ws-storage",
    ) -> None:
        self._memory = memory
        self._rollout_enabled = rollout_enabled
        self._storage_root = storage_root
        self.calls: List[str] = []
        self.cli_params: Optional[Dict[str, Any]] = None
        if cli_bridge is not None:
            # 인스턴스 속성으로 주입 — None 이면 메서드 자체가 없는 호스트(레거시).
            self.cli_bridge_available = lambda provider, _v=cli_bridge: _v  # type: ignore[assignment]

    # A
    def setting(self, name: str, default: str = "") -> str:
        return default

    def setting_truthy(self, name: str) -> bool:
        return self._rollout_enabled if name == ROLLOUT_ENABLED_SETTING else False

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
        return self._storage_root

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
        # kwargs 사전 그 자체가 넘어온다 — 도구 표면(_tool_surface) 관찰 지점.
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
        seen["stream_state"] = args[2]
        seen["stream_kwargs"] = dict(kwargs)
        return iter([])

    monkeypatch.setattr(runner_mod, "stream_turn", _fake_stream_turn)

    def _fake_run_turn(*args, **kwargs):
        seen["run_input"] = args[1]
        seen["run_kwargs"] = dict(kwargs)
        return "done"

    monkeypatch.setattr(runner_mod, "run_turn", _fake_run_turn)
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
    host = _FakeHost(memory=False)
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
    host = _FakeHost(memory=False)
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


def test_opt_in_workspace_fast_path_is_wired_before_the_model_call(
    capture, tmp_path: Path, monkeypatch
) -> None:
    class _FastPathHost(_FakeHost):
        def setting_truthy(self, name: str) -> bool:
            if name == WORKSPACE_FAST_PATH_SETTING:
                return True
            return super().setting_truthy(name)

        def agent_workspace_dir(self, workflow_id, *, create=True):
            return str(tmp_path)

        def workspace_storage_root(self, workflow_id):
            return str(tmp_path.parent)

        def register_builtin_tools(self, registry, **kwargs):
            registry.register(BashTool(), core=True)
            return {"tools": ["Bash"], "extras": {}, "families": ["shell"]}

        def build_run_tool_context(self, **kwargs):
            run_dir = str(kwargs["run_dir"])
            return ToolContext(
                session_id="inter-1",
                working_dir=run_dir,
                allowed_paths=[run_dir],
                extras=dict(kwargs.get("extras") or {}),
            )

    source = tmp_path / "source"
    source.write_text("alpha", encoding="utf-8")
    seen = _run(
        _FastPathHost(memory=False),
        capture,
        text=f"Transform {source} and save the requested output",
    )

    assert "# Complete workspace snapshot" in seen["stream_input"]
    assert '"content":"alpha"' in seen["stream_input"]
    witnessed = seen["stream_state"].shared[SharedKeys.FILE_WITNESSED]
    assert witnessed == ["source", str(source)]
    assert seen["stream_state"].shared[SharedKeys.WORKSPACE_FAST_PATH] == {
        "active": True,
        "reason": "eligible",
        "file_count": 1,
        "total_bytes": 5,
    }

    # If context budgeting would trim the snapshot, retry from the original
    # request and do not claim the model witnessed any file content.
    from xgen_agent_runtime.host import context_budget
    from xgen_agent_runtime.host.context_budget import BudgetFit

    fits = 0

    def _fit(**kwargs):
        nonlocal fits
        fits += 1
        if fits == 1:
            return BudgetFit(
                text="partial snapshot",
                rag_block=kwargs["rag_block"],
                clamped=True,
                window=kwargs["window"],
                budget=1,
                total_before=999,
            )
        return BudgetFit(
            text=kwargs["text"],
            rag_block=kwargs["rag_block"],
            clamped=False,
            window=kwargs["window"],
            budget=999,
            total_before=1,
        )

    monkeypatch.setattr(context_budget, "fit_input_to_budget", _fit)
    capture.clear()
    original = f"Transform {source} and save the requested output"
    fallback = _run(
        _FastPathHost(memory=False),
        capture,
        text=original,
        enable_compaction=True,
        context_window=4096,
    )
    assert fits == 2
    assert fallback["stream_input"] == original
    assert SharedKeys.FILE_WITNESSED not in fallback["stream_state"].shared
    assert fallback["stream_state"].shared[SharedKeys.WORKSPACE_FAST_PATH]["reason"] == (
        "context_budget"
    )


# ── 위임은 없다 ──────────────────────────────────────────────────────


_REMOVED_TOOLS = {
    "DelegationGuide", "DelegateTask", "SubAgentSpawn", "SubAgentSend", "SubAgentList",
    "SubAgentKill", "Task", "TaskCreate", "TaskGet", "TaskList", "TaskOutput", "TaskStop",
    "TaskUpdate", "Agent",
}


def test_host_protocol_has_no_delegation_hooks() -> None:
    for hook in (
        "build_turn_delegation",
        "delegation_extra_tool_classes",
        "delegation_workspace",
        "make_sub_cli_client_factory",
        "is_report_turn",
        "drain_pending_reports",
    ):
        assert not hasattr(HostServices, hook), hook


def test_no_turn_registers_a_delegation_tool(capture) -> None:
    for provider in ("openai", "claude_code", "codex"):
        capture.clear()
        host = _FakeHost()
        _run(host, capture, provider=provider)
        names = set(_surface_names(capture, host))
        assert not names & _REMOVED_TOOLS, provider
        assert "Delegat" not in capture["system_prompt"]


# ── [CLI_BRIDGE] ──────────────────────────────────────────────────────


def test_host_services_declares_optional_cli_bridge_available() -> None:
    fn = getattr(HostServices, "cli_bridge_available", None)
    assert fn is not None
    assert [p for p in inspect.signature(fn).parameters if p != "self"] == ["provider"]


def _surface_names(capture: Dict[str, Any], host: Any) -> List[str]:
    """이 턴에 등록된 도구 이름 — SDK 는 파이프라인 레지스트리, CLI 는 브릿지로 간 표면."""
    reg = capture.get("registry")
    if reg is None and host.cli_params is not None:
        surface = host.cli_params.get("_tool_surface")
        reg = surface.registry if surface is not None else None
    return list(reg.list_names()) if reg is not None else []


class _SelfEditHost(_FakeHost):
    """서버처럼 WorkflowSelf 를 등록하는 호스트."""

    def register_workflow_self_tools(self, registry, **k):
        registry.register(_NamedTool("WorkflowSelf"), core=False)


class _NamedTool(Tool):
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    async def execute(self, input, context):  # noqa: A002
        return ToolResult(content="ok")


@pytest.mark.parametrize("provider", ["claude_code", "codex"])
def test_cli_gets_the_same_registry_and_the_same_prompt_as_sdk(capture, provider) -> None:
    """CLI 표면 = SDK 표면. 프롬프트는 SDK 것 + 숨김 목록 + 이름 규약 한 줄."""
    from xgen_agent_runtime.tools.catalog import deferred_catalog_text

    sdk_host = _SelfEditHost()
    sdk = dict(_run(sdk_host, capture, provider="openai"))
    sdk_names = _registry_names(sdk)
    capture.clear()

    cli_host = _SelfEditHost()
    cli = _run(cli_host, capture, provider=provider)
    assert cli.get("registry") is None, "CLI 는 파이프라인 Stage 10 이 돌지 않는다"
    surface = cli_host.cli_params["_tool_surface"]
    assert list(surface.registry.list_names()) == sdk_names
    assert sorted(t.name for t in surface.registry.list_exposed()) == sorted(
        t.name for t in sdk["registry"].list_exposed()
    )
    assert "memory_write" in sdk_names and "WorkflowSelf" in sdk_names

    catalog = deferred_catalog_text(surface.registry)
    expected = sdk["system_prompt"]
    if catalog:
        expected += "\n\n" + catalog
    expected += cli_tool_naming_note("connector", provider)
    assert cli["system_prompt"] == expected
    assert SELF_EVOLUTION_PROMPT_BLOCK in cli["system_prompt"]
    assert MEMORY_PROMPT_BLOCK in cli["system_prompt"]


def test_cli_legacy_host_without_probe_gets_the_surface(capture) -> None:
    """메서드 부재 → True: 서버 레거시 동작(브릿지 있음)."""
    host = _FakeHost(cli_bridge=None)
    assert not hasattr(host, "cli_bridge_available")
    _run(host, capture, provider="claude_code")
    assert host.cli_params.get("_tool_surface") is not None


def test_cli_bridge_unavailable_drops_tools_and_memory_is_automatic(capture, caplog) -> None:
    host = _SelfEditHost(cli_bridge=False)
    with caplog.at_level(logging.INFO):
        seen = _run(host, capture, provider="claude_code")
    sp = seen["system_prompt"]
    # 메모리: 도구 광고 없음, 자동 계층 안내만
    assert MEMORY_PROMPT_BLOCK not in sp
    assert "memory_categories(" not in sp and "mcp__connector__" not in sp
    assert MEMORY_AUTO_PROMPT_BLOCK in sp
    # 자기진화: 블록·도구 없음
    assert SELF_EVOLUTION_PROMPT_BLOCK not in sp and "WorkflowSelf" not in sp
    assert host.cli_params.get("_tool_surface") is None
    assert "self-evolution 미배선 — CLI 브릿지 없음" in caplog.text


def test_codex_bridge_unavailable_memory_automatic_no_self_evolution(capture) -> None:
    host = _SelfEditHost(cli_bridge=False)
    seen = _run(host, capture, provider="codex")
    sp = seen["system_prompt"]
    assert MEMORY_AUTO_PROMPT_BLOCK in sp
    assert "'connector'" not in sp
    assert SELF_EVOLUTION_PROMPT_BLOCK not in sp


def test_sdk_provider_memory_block_unchanged_regardless_of_probe(capture) -> None:
    """SDK 경로는 registry 에 memory 도구를 직접 등록 — 프로브와 무관하게 블록 유지."""
    host = _FakeHost(cli_bridge=False)
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
    host = _FakeHost(cli_bridge=False, memory=False)
    seen = _run(host, capture, provider="openai", system_prompt="")
    sp = seen["system_prompt"]
    assert default_prompt not in sp
    assert sp == ""


def test_missing_system_prompt_still_falls_back_to_default(capture) -> None:
    """system_prompt 키 자체를 안 주면(레거시 호출부 등) 여전히 기본 문구를 쓴다."""
    host = _FakeHost(cli_bridge=False, memory=False)
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


@pytest.mark.parametrize("provider", ["openai", "claude_code"])
def test_self_evolution_off_leaves_memory_alone(capture, caplog, provider) -> None:
    """자기진화를 꺼도 메모리는 그대로다 — 한 스위치가 옆 능력을 끌고 내려가지 않는다."""
    host = _SelfEditHost(cli_bridge=True)
    with caplog.at_level(logging.INFO):
        seen = _run(host, capture, provider=provider, enable_self_evolution=False)
    sp = seen["system_prompt"]
    assert SELF_EVOLUTION_PROMPT_BLOCK not in sp
    names = _surface_names(seen, host)
    assert "WorkflowSelf" not in names
    assert "memory_write" in names and MEMORY_PROMPT_BLOCK in sp


# ── 그래프 연결 도구도 같은 표면이다 ─────────────────────────────────


class _GraphTool(Tool):
    name = "jira_search"
    description = "search jira"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, input, context):  # noqa: A002
        return ToolResult(content="ok")


@pytest.mark.parametrize("provider", ["openai", "claude_code", "codex"])
def test_graph_tools_are_on_every_backend(capture, provider) -> None:
    host = _FakeHost()
    _run(host, capture, provider=provider, tools=[_GraphTool()])
    assert "jira_search" in _surface_names(capture, host)


def test_cli_says_so_when_graph_tools_cannot_be_delivered(capture, caplog) -> None:
    """브릿지가 없으면 정말 못 준다 — 그때 침묵하면 에이전트가 원인을 지어낸다.

    프롬프트에 사실을 실어, 사용자가 물으면 설정 문제라고 정확히 답하게 한다.
    """
    host = _FakeHost(cli_bridge=False)
    out = _run(host, capture, provider="claude_code", tools=[_GraphTool()])
    assert "cannot be delivered on this backend" in out["system_prompt"]
    assert "1 tool(s)" in out["system_prompt"]


def test_no_such_note_when_the_bridge_is_available(capture) -> None:
    host = _FakeHost()
    out = _run(host, capture, provider="claude_code", tools=[_GraphTool()])
    assert "cannot be delivered" not in out["system_prompt"]


# ── durable rollout host gate ────────────────────────────────────────


def test_rollout_recording_is_opt_in_and_passes_no_path_by_default(capture) -> None:
    host = _FakeHost(memory=False)

    seen = _run(host, capture)

    assert seen["stream_kwargs"]["rollout_path"] is None


def test_rollout_recording_allocates_safe_workflow_scoped_path(
    capture, tmp_path: Path
) -> None:
    host = _FakeHost(
        memory=False,
        rollout_enabled=True,
        storage_root=str(tmp_path),
    )

    seen = _run(host, capture, interaction_id="../../customer/대화")

    path = Path(seen["stream_kwargs"]["rollout_path"])
    assert path.parent == rollout_directory(tmp_path)
    assert path.suffix == ".jsonl"
    assert "customer" not in path.name and "대화" not in path.name
    assert not path.exists(), "the runner, not turn assembly, creates the file"


def test_rollout_recording_reaches_non_stream_runner(capture, tmp_path: Path) -> None:
    host = _FakeHost(
        memory=False,
        rollout_enabled=True,
        storage_root=str(tmp_path),
    )

    output = AgentTurnExecutor().run(
        host,
        text="hi",
        provider="openai",
        workflow_id="wf-1",
        workflow_name="wf",
        user_id="u1",
        interaction_id="inter-1",
        streaming=False,
        memory_distill=False,
        enable_compaction=False,
    )

    assert output == "done"
    assert Path(capture["run_kwargs"]["rollout_path"]).parent == rollout_directory(tmp_path)


def test_rollout_recording_skips_unscoped_turn_with_warning(capture, caplog) -> None:
    host = _FakeHost(memory=False, rollout_enabled=True)

    with caplog.at_level(logging.WARNING):
        seen = _run(host, capture, workflow_id="")

    assert seen["stream_kwargs"]["rollout_path"] is None
    assert "workflow_id is unavailable" in caplog.text


def test_rollout_recording_runs_end_to_end_without_public_result_change(
    monkeypatch, tmp_path: Path
) -> None:
    from xgen_agent_runtime.core.state import TokenUsage
    from xgen_agent_runtime.llm_client import BaseClient, ClientCapabilities
    from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock

    class _Client(BaseClient):
        provider = "fake"
        capabilities = ClientCapabilities()

        async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
            return APIResponse(
                content=[ContentBlock(type="text", text="done")],
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=2, output_tokens=1),
                model="fake-model",
            )

    monkeypatch.setattr(
        runner_mod,
        "build_client",
        lambda *args, **kwargs: _Client(api_key="k"),
    )
    host = _FakeHost(
        memory=False,
        rollout_enabled=True,
        storage_root=str(tmp_path),
    )

    output = AgentTurnExecutor().run(
        host,
        text="hi",
        provider="openai",
        workflow_id="wf-1",
        workflow_name="wf",
        user_id="u1",
        interaction_id="inter-1",
        streaming=False,
        memory_distill=False,
        enable_compaction=False,
    )

    assert output == "done"
    files = list(rollout_directory(tmp_path).glob("rollout-*.jsonl"))
    assert len(files) == 1
    records = [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["type"] == "pipeline.start"
    assert records[-1]["type"] == "pipeline.complete"


# ── CLI 이름 규약은 한 줄이다 ─────────────────────────────────────────
#
# 예전엔 메모리·자기진화·위임마다 CLI 각주가 따로 있었다. 각주가 도구마다 있으면 빠진 도구가
# 생기고(SSH·작업·앱·기기 도구는 각주가 없었다), 빠진 도구는 모델에게 이름을 모르는 것이 된다.
# 이제 이름 규약 하나가 모든 도구에 적용된다.


def test_the_cli_naming_note_is_one_sentence_for_every_tool():
    from xgen_agent_runtime.host import _constants

    for gone in ("cli_memory_note", "cli_self_evolution_note", "cli_delegation_note"):
        assert not hasattr(_constants, gone), gone
    claude = cli_tool_naming_note("connector", "claude_code")
    assert "mcp__connector__X" in claude and "only tools" in claude
    codex = cli_tool_naming_note("connector", "codex")
    assert "'connector'" in codex and "mcp__" not in codex


def test_the_note_follows_the_server_name():
    """브릿지 서버 이름이 바뀌면 각주도 따라간다 — 하드코딩된 'connector' 금지."""
    assert "mcp__local__X" in cli_tool_naming_note("local", "claude_code")
    assert "'local'" in cli_tool_naming_note("local", "codex")


# ── 메모리 지침은 남은 도구를 보고 고른다 (2026-09-21) ──────────────────────────
# 호스트 정책(게스트·동결본)이 memory_write/memory_pin 을 뺀 턴에 "기억하라면 저장하라" 는
# 블록이 그대로 붙어 있었다 — 모델은 그 약속을 파일 쓰기로 메웠다. 문구와 표면은 같은 판정.
from xgen_agent_runtime.host._constants import MEMORY_READONLY_PROMPT_BLOCK  # noqa: E402


class _ReadOnlyMemoryHost(_FakeHost):
    """서버 호스트처럼 내장 도구 등록 뒤 기억 쓰기 도구를 뺀다."""

    def register_builtin_tools(self, registry, **k):
        for name in ("memory_write", "memory_pin"):
            if registry.get(name) is not None:
                registry.unregister(name)
        return {"tools": [], "extras": {}, "families": []}

    def memory_write_available(self, workflow_id):
        return False


def test_sdk_uses_the_readonly_block_when_the_host_removed_write_tools(capture) -> None:
    host = _ReadOnlyMemoryHost(cli_bridge=False)
    seen = _run(host, capture, provider="openai")
    sp = seen["system_prompt"]
    assert MEMORY_READONLY_PROMPT_BLOCK in sp and MEMORY_PROMPT_BLOCK not in sp
    names = _registry_names(seen)
    assert "memory_read" in names and "memory_write" not in names and "memory_pin" not in names


def test_sdk_keeps_the_write_block_when_write_tools_remain(capture) -> None:
    host = _FakeHost(cli_bridge=False)
    seen = _run(host, capture, provider="openai")
    sp = seen["system_prompt"]
    assert MEMORY_PROMPT_BLOCK in sp and MEMORY_READONLY_PROMPT_BLOCK not in sp
    assert "memory_write" in _registry_names(seen)


@pytest.mark.parametrize("provider", ["claude_code", "codex"])
def test_cli_memory_wording_follows_the_same_registry(capture, provider) -> None:
    """CLI 도 같은 레지스트리를 보고 고른다 — 쓰기 도구가 빠지면 읽기 전용 문구, 표면에도 없다."""
    host = _ReadOnlyMemoryHost(cli_bridge=True)
    seen = _run(host, capture, provider=provider)
    sp = seen["system_prompt"]
    assert MEMORY_READONLY_PROMPT_BLOCK in sp and MEMORY_PROMPT_BLOCK not in sp
    names = _surface_names(seen, host)
    assert "memory_read" in names and "memory_write" not in names


def test_cli_keeps_the_write_block_when_write_tools_remain(capture) -> None:
    host = _FakeHost(cli_bridge=True)
    seen = _run(host, capture, provider="claude_code")
    assert MEMORY_PROMPT_BLOCK in seen["system_prompt"]


# ── 목록을 다시 읽지 않는 CLI(codex)는 처음부터 전부 본다 ─────────────────────
#
# 2026-09-30 실측(실제 CLI + 실제 브릿지): Claude Code 는 list_changed 뒤 tools/list 를 다시 읽지만 codex 는
# 한 번도 다시 읽지 않았다 — 문·ToolSearch 로 연 도구를 부르면 "unsupported call". 계층은 토큰 절약이지 능력의
# 경계가 아니므로 codex 는 평면 노출로 돈다.


def test_codex_gets_every_tool_up_front_and_claude_keeps_the_hierarchy(capture) -> None:
    codex_host = _FakeHost()
    _run(codex_host, capture, provider="codex", tools=[_GraphTool()])
    codex_reg = codex_host.cli_params["_tool_surface"].registry
    assert codex_reg.list_deferred() == []
    assert codex_reg.is_exposed("jira_search")

    capture.clear()
    claude_host = _FakeHost()
    _run(claude_host, capture, provider="claude_code", tools=[_GraphTool()])
    claude_reg = claude_host.cli_params["_tool_surface"].registry
    assert not claude_reg.is_exposed("jira_search"), "Claude Code 는 목록을 다시 읽으므로 계층을 지킨다"


# ── [RESULT_FILTER] 호스트의 도구 결과 필터 (4.71.0) ─────────────────────────
#
# 호스트(xgen-workflow)는 관리자 정책에 따라 외부 데이터 도구 결과의 개인정보·금칙어를 가린다.
# 필터는 턴마다 한 번 받아 SDK 파이프라인과 CLI 도구 표면의 **같은** 도구 컨텍스트에 싣는다 —
# 적용은 Stage 10 라우터 한 곳이라 두 경로가 같은 결과를 본다. 훅이 없는 옛 호스트는 그대로 돈다.


async def _mask_filter(tool, result):  # noqa: ANN001
    return result


class _BuiltinToolsHost(_FakeHost):
    """내장 도구가 있어 run_tool_context 가 만들어지는 턴."""

    def register_builtin_tools(self, registry, **kwargs):
        registry.register(BashTool(), core=True)
        return {"tools": ["Bash"], "extras": {}, "families": ["shell"]}

    def build_run_tool_context(self, **kwargs):
        run_dir = str(kwargs["run_dir"])
        return ToolContext(session_id="inter-1", working_dir=run_dir, allowed_paths=[run_dir])


def test_host_protocol_declares_optional_tool_result_filter() -> None:
    fn = getattr(HostServices, "tool_result_filter", None)
    assert fn is not None
    assert [p for p in inspect.signature(fn).parameters if p != "self"] == []
    # 기본 구현은 필터 없음 — 프로토콜을 상속한 호스트가 구현을 빠뜨려도 결과는 그대로.
    assert HostServices.tool_result_filter(object()) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("host_cls", [_FakeHost, _BuiltinToolsHost])
def test_result_filter_reaches_the_sdk_tool_stage(capture, host_cls) -> None:
    host = host_cls()
    host.tool_result_filter = lambda: _mask_filter  # type: ignore[attr-defined]
    seen = _run(host, capture, provider="openai", tools=[_GraphTool()])
    assert seen["tool_result_filter"] is _mask_filter
    if seen.get("tool_context") is not None:
        assert seen["tool_context"].result_filter is _mask_filter


@pytest.mark.parametrize("host_cls", [_FakeHost, _BuiltinToolsHost])
@pytest.mark.parametrize("provider", ["claude_code", "codex"])
def test_result_filter_reaches_the_cli_tool_surface(capture, host_cls, provider) -> None:
    host = host_cls()
    host.tool_result_filter = lambda: _mask_filter  # type: ignore[attr-defined]
    seen = _run(host, capture, provider=provider, tools=[_GraphTool()])
    surface = host.cli_params["_tool_surface"]
    assert surface.tool_context.result_filter is _mask_filter
    # 표면의 Stage 10 은 그 컨텍스트를 그대로 쓴다 — 실행 컨텍스트에도 실린다.
    ctx = surface._stage.build_dispatch_context(surface.state)
    assert ctx.result_filter is _mask_filter
    # CLI 턴은 파이프라인 Stage 10 이 돌지 않는다.
    assert seen["tool_result_filter"] is None


@pytest.mark.parametrize("provider", ["openai", "claude_code"])
def test_host_without_the_hook_runs_unfiltered(capture, provider) -> None:
    host = _FakeHost()
    seen = _run(host, capture, provider=provider, tools=[_GraphTool()])
    assert seen["tool_result_filter"] is None
    if provider == "claude_code":
        assert host.cli_params["_tool_surface"].tool_context.result_filter is None


def test_a_failing_hook_runs_unfiltered(capture, caplog) -> None:
    host = _FakeHost()

    def _boom():
        raise RuntimeError("policy store down")

    host.tool_result_filter = _boom  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING):
        seen = _run(host, capture, provider="openai", tools=[_GraphTool()])
    assert seen["tool_result_filter"] is None
    assert "tool_result_filter" in caplog.text
