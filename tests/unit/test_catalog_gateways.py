"""숨은 도구 카탈로그가 문(게이트웨이) 뒤의 식구를 한 줄로 접는다 (4.33.0).

근거 (2026-09-20, dev 4.32.0 표면): 숨은 도구 46개 중 38개가 이미 턴1에 서 있는 문
(Browser/Delegation/Artifact/Job/SelfExtend)이나 카탈로그 안의 문(DocGuide) 뒤에 있었다.
그런데 카탈로그는 식구마다 한 줄 설명(~22토큰)을 실었다 — 문의 설명이 이미 말하는 능력을
40번 되풀이한 것. 접으니 900 → 286 토큰(Qwen 토크나이저, Job/Artifact 선언 포함).
이름은 남긴다 — ToolSearch("<exact name>") 가 그대로 통해야 한다.
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host import runner, tools as host_tools
from xgen_agent_runtime.stages.s03_system.artifact.default.stage import SystemStage
from xgen_agent_runtime.tools.built_in import SKILL_GATEWAYS, DocGuideTool
from xgen_agent_runtime.tools.built_in.browser_tools import BrowserGuideTool
from xgen_agent_runtime.tools.registry import ToolRegistry
from tests.unit.test_core_deferred_tools import _NamedTool


def _catalog(reg: ToolRegistry) -> str:
    return SystemStage(prompt="sys", tool_registry=reg)._deferred_catalog_text()


# ── 레지스트리 ─────────────────────────────────────────────────────────


def test_declare_gateway_maps_members_and_bumps_version() -> None:
    reg = ToolRegistry().register(_NamedTool("Gate"))
    v = reg.version
    reg.declare_gateway("Gate", ("a", "b", "Gate"))  # 문 자신은 식구가 아니다
    assert reg.version == v + 1
    assert reg.gateway_members("Gate") == ("a", "b")
    assert reg.gateway_of("a") == "Gate" and reg.gateway_of("Gate") is None
    reg.declare_gateway("Gate", ("a", "b"))  # 같은 선언은 조용히
    assert reg.version == v + 1
    reg.declare_gateway("Gate", ("b",))  # 다시 선언하면 빠진 식구는 풀린다
    assert reg.gateway_of("a") is None and reg.gateway_of("b") == "Gate"


# ── 카탈로그 ───────────────────────────────────────────────────────────


def test_members_behind_a_visible_gate_fold_into_one_line() -> None:
    reg = (
        ToolRegistry()
        .register(_NamedTool("BrowserGuide", "Opens the browser room"))
        .register(_NamedTool("BrowserNavigate", "Open a URL"), core=False)
        .register(_NamedTool("BrowserAct", "Click / type"), core=False)
        .register(_NamedTool("NotebookEdit", "Edit notebook cells"), core=False)
    )
    reg.declare_gateway("BrowserGuide", ("BrowserNavigate", "BrowserAct", "BrowserNotRegistered"))
    text = _catalog(reg)
    assert "3 more tools exist" in text
    assert "- via BrowserGuide: BrowserAct, BrowserNavigate" in text  # 이름만, 설명 없이
    assert "Open a URL" not in text and "BrowserNotRegistered" not in text
    assert "- NotebookEdit — Edit notebook cells" in text  # 문이 없는 도구는 그대로
    assert "open when you call that guide" in text


def test_hidden_gate_keeps_its_line_and_carries_its_room() -> None:
    reg = (
        ToolRegistry()
        .register(_NamedTool("Read"))
        .register(_NamedTool("DocGuide", "START HERE for documents"), core=False)
        .register(_NamedTool("DocRender", "Render a document"), core=False)
        .register(_NamedTool("DocBuild", "Build a document"), core=False)
    )
    reg.declare_gateway("DocGuide", ("DocRender", "DocBuild"))
    text = _catalog(reg)
    assert "- DocGuide — START HERE for documents → opens DocBuild, DocRender" in text
    assert "Render a document" not in text
    assert "3 more tools exist" in text


def test_unregistered_gate_does_not_fold() -> None:
    """문이 없으면 식구는 각자 한 줄 — 접을 곳이 없다(문 없이 숨기면 2026-08-18 회귀)."""
    reg = ToolRegistry().register(_NamedTool("Read")).register(_NamedTool("ForgeTool", "Make a tool"), core=False)
    reg.declare_gateway("SelfExtendGuide", ("ForgeTool",))
    assert "- ForgeTool — Make a tool" in _catalog(reg)


@pytest.mark.asyncio
async def test_folding_is_cache_stable_across_activation() -> None:
    reg = (
        ToolRegistry()
        .register(_NamedTool("Gate"))
        .register(_NamedTool("member_a"), core=False)
        .register(_NamedTool("member_b"), core=False)
    )
    reg.declare_gateway("Gate", ("member_a", "member_b"))
    stage = SystemStage(prompt="sys", tool_registry=reg)
    state = PipelineState()
    await stage.execute(None, state)
    before = state.system
    reg.activate("member_a")
    await stage.execute(None, state)
    assert state.system == before


# ── 호스트 배선 ────────────────────────────────────────────────────────


def test_build_pipeline_declares_built_in_gateways() -> None:
    reg = ToolRegistry().register(BrowserGuideTool()).register(DocGuideTool(), core=False)
    for n in ("BrowserNavigate", "DocRender"):
        reg.register(_NamedTool(n), core=False)
    runner.build_pipeline(name="t", provider="openai", model="m", api_key="k", registry=reg, stream=False)
    assert reg.gateway_of("BrowserNavigate") == "BrowserGuide"
    assert reg.gateway_of("DocRender") == "DocGuide"
    assert reg.gateway_members("BrowserGuide") == SKILL_GATEWAYS["BrowserGuide"]
    assert reg.gateway_of("Read") is None


def test_register_declares_a_gateway_from_opens_family_attribute() -> None:
    """호스트 문(JobGuide/ArtifactGuide)은 클래스 속성 한 줄이면 된다."""

    class _Gate(_NamedTool):
        opens_family = ("JobSchedule", "JobList")

    reg = ToolRegistry().register(_Gate("JobGuide")).register(_NamedTool("JobList"), core=False)
    assert reg.gateway_of("JobList") == "JobGuide"
    assert "- via JobGuide: JobList" in _catalog(reg)


def test_adapter_opens_family_metadata_declares_a_gateway() -> None:
    class _Lc:
        name = "ConnectorGuide"
        description = "opens the connector room"
        metadata = {"opens_family": ["ConnectorRead", "ConnectorWrite"]}

        def invoke(self, x):
            return "ok"

    reg = host_tools.adapt_tools([_Lc()], core=True)
    assert reg is not None
    assert reg.gateway_members("ConnectorGuide") == ("ConnectorRead", "ConnectorWrite")
