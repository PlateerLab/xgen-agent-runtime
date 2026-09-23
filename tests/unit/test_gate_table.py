"""문과 방을 한 표에서 — 숨김·열림·검사가 같은 표를 읽는다 (4.56.0).

"문은 보이는데 방이 안 열린다" 가 2주 새 네 번(SSH 09-09 · Delegation · Browser 09-19 ·
LocalControl 09-23). 매번 그 문 하나에 여는 코드를 덧댔다. 여기 테스트는 그 **부류**를 막는다:

* 표에 있는 문은 스스로 열 줄 몰라도 라우터가 연다 — 오늘 사례(여는 선언 없는 커넥터
  LocalControl)를 workflow 패치 **없이** 재현하고 풀리는지 본다.
* 숨긴 가족에 보이는 문이 없으면 숨기지 않는다(fail-open) — SSH 사례.
* 의도된 긴 꼬리(사용자 DB·API·MCP 도구, 문서)는 건드리지 않는다.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from xgen_agent_runtime.host.tool_exposure import TURN_ONE_TOOLS, is_turn_one
from xgen_agent_runtime.stages.s10_tool.artifact.default.routers import RegistryRouter
from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult
from xgen_agent_runtime.tools.gates import (
    GATES,
    family_of,
    gate_of,
    reachability_fixes,
    split_prefix,
)
from xgen_agent_runtime.tools.registry import ToolRegistry


class _Plain(Tool):
    """아무것도 스스로 열지 않는 도구 — 커넥터가 광고한 LocalControl 이 딱 이랬다."""

    def __init__(self, name: str, answer: str = "ok") -> None:
        self._name = name
        self._answer = answer

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=self._answer)


def _surface(names):
    """턴-1 계획대로 등록한 레지스트리 (계획이 아니라 등록 결과를 본다)."""
    reg = ToolRegistry()
    for n in names:
        reg.register(_Plain(n), core=is_turn_one(n))
    return reg


def _call(reg: ToolRegistry, name: str) -> ToolResult:
    ctx = ToolContext(session_id="s")
    ctx.tool_registry = reg
    return asyncio.run(RegistryRouter(reg).route(name, {}, ctx))


# ── 표 자체 ──────────────────────────────────────────────────────────


class TestTable:
    def test_every_visible_gate_stands_on_turn_one(self):
        """표와 턴-1 목록이 어긋나면 SSH 사고가 다시 난다(문이 숨으면 방도 못 찾는다)."""
        missing = [g.name for g in GATES.values() if g.visible and g.name not in TURN_ONE_TOOLS]
        assert missing == []

    def test_every_guide_on_turn_one_is_in_the_table(self):
        guides = [n for n in TURN_ONE_TOOLS if n.endswith("Guide") or n in ("LocalControl", "SshListServers")]
        assert [n for n in guides if n not in GATES] == []

    def test_prefix_split(self):
        assert split_prefix("mcp_local_Shell") == ("mcp_local_", "Shell")
        assert split_prefix("mcp__connector__Foo") == ("mcp__connector__", "Foo")
        assert split_prefix("Bash") == ("", "Bash")


class TestFamilies:
    REG = [
        "mcp_local_LocalControl", "mcp_local_Shell", "mcp_local_WriteFile",
        "mcp_local_BrowserGuide", "mcp_local_BrowserNavigate",
        "BrowserGuide", "BrowserNavigate", "Bash", "Read",
        "JobGuide", "JobSchedule", "JobList", "db_query_orders",
        "mcp_github_BrowserNavigate",
    ]

    def test_local_control_opens_the_rest_of_its_prefix(self):
        fam = family_of("mcp_local_LocalControl", self.REG)
        assert sorted(fam) == ["mcp_local_BrowserNavigate", "mcp_local_Shell", "mcp_local_WriteFile"]

    def test_a_gate_opens_only_its_own_prefix(self):
        assert family_of("BrowserGuide", self.REG) == ["BrowserNavigate"]
        assert family_of("mcp_local_BrowserGuide", self.REG) == ["mcp_local_BrowserNavigate"]

    def test_an_unprefixed_local_control_is_not_a_gate(self):
        """CLI 경로의 LocalControl(접두 없음)이 모든 내장 도구를 가족으로 삼으면 안 된다."""
        assert gate_of("LocalControl") is None
        assert family_of("LocalControl", self.REG) == []

    def test_workflow_owned_families_follow_the_naming_rule(self):
        assert sorted(family_of("JobGuide", self.REG)) == ["JobList", "JobSchedule"]


# ── 라우터: 표에 있는 문은 스스로 몰라도 열린다 ─────────────────────


class TestRouterOpens:
    def test_todays_bug_is_fixed_without_the_workflow_patch(self):
        """2026-09-23 dev: 커넥터 LocalControl 이 여는 선언 없이 감싸져, 지도의
        Shell·WriteFile 을 끝내 못 불렀다. 문이 스스로 몰라도 열려야 한다."""
        reg = _surface(["mcp_local_LocalControl", "mcp_local_Shell", "mcp_local_WriteFile", "Bash"])
        assert reg.is_exposed("mcp_local_LocalControl")
        assert not reg.is_exposed("mcp_local_Shell")

        out = _call(reg, "mcp_local_LocalControl")
        assert reg.is_exposed("mcp_local_Shell")
        assert reg.is_exposed("mcp_local_WriteFile")
        assert "Now callable" in out.content

    def test_a_failed_gate_opens_nothing(self):
        reg = ToolRegistry()

        class _Broken(_Plain):
            async def execute(self, input, context):
                return ToolResult(content="Error: connector offline", is_error=True)

        reg.register(_Broken("mcp_local_LocalControl"), core=True)
        reg.register(_Plain("mcp_local_Shell"), core=False)
        _call(reg, "mcp_local_LocalControl")
        assert not reg.is_exposed("mcp_local_Shell")

    def test_a_gate_that_opens_itself_is_not_announced_twice(self):
        from xgen_agent_runtime.tools.built_in._skill_gateway import open_family, with_opened

        class _SelfOpening(_Plain):
            async def execute(self, input, context):
                opened = open_family(context, ["BrowserNavigate"])
                return ToolResult(content=with_opened("map", opened))

        reg = ToolRegistry()
        reg.register(_SelfOpening("BrowserGuide"), core=True)
        reg.register(_Plain("BrowserNavigate"), core=False)
        out = _call(reg, "BrowserGuide")
        assert out.content.count("Now callable") == 1

    def test_non_gates_do_not_open_anything(self):
        reg = _surface(["Bash", "mcp_local_Shell", "mcp_local_LocalControl"])
        _call(reg, "Bash")
        assert not reg.is_exposed("mcp_local_Shell")


# ── 불변식: 숨긴 가족에는 보이는 문이 있다 ──────────────────────────


class TestReachability:
    def test_a_hidden_member_with_no_gate_is_exposed(self):
        """09-09 SSH: 문(SshListServers)도 방(SshRun)도 숨어 있었다."""
        reg = _surface(["Bash", "SshRun", "SshUpload"])  # 문이 등록되지 않음
        fixes = reachability_fixes(reg.list_names(), reg.is_exposed)
        assert sorted(fixes) == ["SshRun", "SshUpload"]

    def test_a_hidden_gate_is_opened_instead_of_its_room(self):
        """문이 등록돼 있는데 숨어 있으면 문 하나만 연다 — 표면을 최소로 키운다."""
        reg = ToolRegistry()
        reg.register(_Plain("JobGuide"), core=False)
        reg.register(_Plain("JobSchedule"), core=False)
        reg.register(_Plain("JobList"), core=False)
        assert reachability_fixes(reg.list_names(), reg.is_exposed) == ["JobGuide"]

    def test_a_reachable_family_needs_nothing(self):
        reg = _surface(["mcp_local_LocalControl", "mcp_local_Shell", "BrowserGuide", "BrowserNavigate"])
        assert reachability_fixes(reg.list_names(), reg.is_exposed) == []

    def test_the_long_tail_is_left_alone(self):
        """연결된 DB·API·사용자 MCP 도구가 ToolSearch 뒤에 있는 것은 설계다."""
        reg = _surface(["Bash", "db_query_orders", "api_create_invoice", "mcp_github_create_issue"])
        assert reachability_fixes(reg.list_names(), reg.is_exposed) == []

    def test_a_hidden_connector_gate_is_opened(self):
        reg = ToolRegistry()
        reg.register(_Plain("mcp_local_LocalControl"), core=False)
        reg.register(_Plain("mcp_local_Shell"), core=False)
        assert reachability_fixes(reg.list_names(), reg.is_exposed) == ["mcp_local_LocalControl"]

    def test_someone_elses_mcp_server_is_not_ours_to_open(self):
        """이름이 우리 규칙에 맞아도(BrowserNavigate) 그 서버엔 우리 문이 없다 — 긴 꼬리."""
        reg = _surface(["Bash", "mcp_github_BrowserNavigate", "mcp_github_JobList"])
        assert reachability_fixes(reg.list_names(), reg.is_exposed) == []

    def test_documents_stay_behind_tool_search_as_declared(self):
        """DocGuide 는 visible=False — 지금 동작(ToolSearch 로 찾기)을 선언한 것이다."""
        reg = _surface(["Bash", "DocGuide", "DocRender", "DocBuild"])
        assert reachability_fixes(reg.list_names(), reg.is_exposed) == []


class TestStageThree:
    def test_the_surface_is_repaired_before_the_model_sees_it(self):
        from xgen_agent_runtime.core.state import PipelineState
        from xgen_agent_runtime.stages.s03_system.artifact.default.stage import (
            _enforce_gate_reachability,
        )

        reg = _surface(["Bash", "SshRun"])
        opened = _enforce_gate_reachability(reg)
        assert opened == ["SshRun"]
        assert reg.is_exposed("SshRun")
        names = [t["name"] for t in reg.to_api_format(exposed_only=True)]
        assert "SshRun" in names
        assert PipelineState  # import 가 살아 있다
