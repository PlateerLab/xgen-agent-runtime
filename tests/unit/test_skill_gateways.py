"""스킬 게이트웨이 규약 — Guide + 컴팩트 멤버 (DocGuide 동형 점진공개).

브라우저/위임 게이트웨이는 정적 텍스트라 an-web 등 엔진 extra 없이도 돈다.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_FEATURES, get_builtin_tools
from xgen_agent_runtime.tools.built_in.browser_tools import BrowserGuideTool
from xgen_agent_runtime.tools.built_in.delegation_guide_tool import DelegationGuideTool


def test_browser_guide_is_the_gateway_and_members_are_compact():
    """browser 패밀리: BrowserGuide 가 첫 항목(게이트웨이)이고, 무주제=지도 /
    주제=심층 가이드. 멤버 description 은 컴팩트(<250자)라 턴1 컨텍스트를
    태우지 않는다 — 심층 지식은 가이드가 요청 시 공개한다."""
    assert BUILT_IN_TOOL_FEATURES["browser"][0] == "BrowserGuide"
    guide = BrowserGuideTool()
    top = asyncio.run(guide.execute({}, None))
    assert not top.is_error and "[ref=" in top.content and "BrowserAct" in top.content
    act = asyncio.run(guide.execute({"topic": "act"}, None))
    assert "wait_for" in act.content and "network_idle" in act.content
    tools = get_builtin_tools(features=["browser"])
    for name, cls in tools.items():
        desc = cls().description
        assert len(desc) < 250, f"{name} description 이 비대하다 ({len(desc)}자)"
    # 대표 멤버는 게이트웨이 포인터를 든다.
    assert "BrowserGuide('act')" in tools["BrowserAct"]().description
    assert "BrowserGuide('extract')" in tools["BrowserExtract"]().description


def test_delegation_guide_is_the_gateway_of_the_subagent_family():
    """subagent 패밀리: DelegationGuide 가 첫 항목. 무주제=3표면 결정 지도 /
    주제=심층 가이드. 전부 영문."""
    assert BUILT_IN_TOOL_FEATURES["subagent"][0] == "DelegationGuide"
    guide = DelegationGuideTool()
    top = asyncio.run(guide.execute({}, None))
    assert "DelegateTask" in top.content and "SubAgent" in top.content and "Task" in top.content
    sub = asyncio.run(guide.execute({"topic": "subagents"}, None))
    assert "SubAgentSpawn" in sub.content and "inbox" in sub.content.lower()
    has_korean = any("가" <= ch <= "힣" for ch in top.content + sub.content)
    assert not has_korean


# ── 문은 방을 연다 ───────────────────────────────────────────────────
#
# 계층 표면의 약속은 "숨긴다" 가 아니라 "한 겹씩 연다" 다(tool_exposure 참조).
# 그 약속이 성립하려면 게이트웨이 호출이 **실제로 멤버를 활성화**해야 한다.
# 예전에는 지도만 돌려주고 방은 잠긴 채였다: 가이드가 "DelegateTask 를 써라" 고
# 말해 놓고 그 이름은 부를 수 없었다(CLI 백엔드에서는 클라이언트가 로컬에서
# "No such tool available" 로 거절 — 서버가 손쓸 수 없는 실패다).

class _FakeRegistry:
    """activate 만 있는 최소 레지스트리 — 게이트웨이가 보는 표면 그대로."""

    def __init__(self, deferred):
        self.deferred = set(deferred)
        self.activated = []

    def activate(self, name):
        if name in self.deferred:
            self.deferred.discard(name)
            self.activated.append(name)
            return True
        return False


class _Ctx:
    def __init__(self, registry):
        self.tool_registry = registry


def test_delegation_guide_opens_its_family():
    from xgen_agent_runtime.tools.built_in.delegation_guide_tool import DELEGATION_FAMILY

    registry = _FakeRegistry(DELEGATION_FAMILY)
    res = asyncio.run(DelegationGuideTool().execute({}, _Ctx(registry)))

    # 지도가 가리키는 동사가 실제로 열렸다.
    assert "DelegateTask" in registry.activated
    assert "SubAgentSpawn" in registry.activated
    assert set(res.metadata["opened"]) == set(DELEGATION_FAMILY)
    # 무엇이 열렸는지 같은 답 안에서 말한다 — "쓰라" 와 "쓸 수 있다" 가 한 턴에.
    assert "Now callable:" in res.content and "DelegateTask" in res.content


def test_topic_guide_opens_the_family_too():
    """심층 가이드로 바로 들어와도 문은 열린다 — 두 입구가 같은 방이다."""
    from xgen_agent_runtime.tools.built_in.delegation_guide_tool import DELEGATION_FAMILY

    registry = _FakeRegistry(DELEGATION_FAMILY)
    res = asyncio.run(DelegationGuideTool().execute({"topic": "subagents"}, _Ctx(registry)))
    assert "SubAgentAssign" in registry.activated
    assert res.metadata["opened"]


def test_browser_guide_opens_its_family():
    from xgen_agent_runtime.tools.built_in.browser_tools import BROWSER_TOOL_CLASSES

    members = [n for n in BROWSER_TOOL_CLASSES if n != "BrowserGuide"]
    registry = _FakeRegistry(members)
    res = asyncio.run(BrowserGuideTool().execute({}, _Ctx(registry)))
    assert set(registry.activated) == set(members)
    assert "BrowserGuide" not in registry.activated, "문이 자기를 열 필요는 없다"
    assert res.metadata["opened"]


def test_already_open_members_are_not_reported_as_newly_opened():
    """등록되지 않은 이름은 열리지 않는다 — 지도만 나온다."""
    registry = _FakeRegistry([])  # 활성화할 것이 없다
    res = asyncio.run(DelegationGuideTool().execute({}, _Ctx(registry)))
    assert res.metadata["opened"] == []
    assert "Now callable:" not in res.content


def _registry_with_delegation_family(*, core):
    """실 ToolRegistry 에 위임 패밀리를 얹는다 (core=True → flat 표면)."""
    from xgen_agent_runtime.tools.built_in.delegation_guide_tool import DELEGATION_FAMILY
    from xgen_agent_runtime.tools.registry import ToolRegistry

    class _Stub:
        def __init__(self, name):
            self.name = name
            self.description = name
            self.input_schema = {"type": "object", "properties": {}}

        async def execute(self, input, context):  # noqa: A002
            raise AssertionError("실행되지 않아야 한다")

    registry = ToolRegistry()
    for name in DELEGATION_FAMILY:
        registry.register(_Stub(name), core=core)
    return registry, DELEGATION_FAMILY


def test_the_real_registry_reports_exactly_what_this_call_opened():
    """실 ToolRegistry 는 이미 보이는 도구에도 activate() 가 True 를 준다
    (성공한 no-op). 그래서 '열렸다' 의 근거는 반환값이 아니라 **호출 전 상태**
    여야 한다 — 아니면 두 번째 호출이 첫 번째와 똑같이 전 패밀리를 읊는다."""
    registry, family = _registry_with_delegation_family(core=False)
    ctx = _Ctx(registry)

    first = asyncio.run(DelegationGuideTool().execute({}, ctx))
    assert set(first.metadata["opened"]) == set(family)
    assert "Now callable:" in first.content

    second = asyncio.run(DelegationGuideTool().execute({}, ctx))
    assert second.metadata["opened"] == [], "이미 열린 방을 다시 열었다고 말한다"
    assert "Now callable:" not in second.content
    assert "DelegateTask" in second.content, "지도는 여전히 나온다"


def test_flat_surface_gateway_says_nothing_about_opening():
    """평면 표면에서는 처음부터 전부 보인다 — 열 것이 없다."""
    registry, _ = _registry_with_delegation_family(core=True)
    res = asyncio.run(DelegationGuideTool().execute({}, _Ctx(registry)))
    assert res.metadata["opened"] == []
    assert "Now callable:" not in res.content


def test_guide_still_answers_without_a_registry():
    """레지스트리가 없는 호출(테스트·베어 하네스)에서도 지도는 나온다."""
    res = asyncio.run(DelegationGuideTool().execute({}, None))
    assert not res.is_error and "DelegateTask" in res.content
    assert res.metadata["opened"] == []


def test_activation_failure_never_costs_the_map():
    class _Broken:
        def activate(self, name):
            raise RuntimeError("registry is on fire")

    res = asyncio.run(DelegationGuideTool().execute({}, _Ctx(_Broken())))
    assert not res.is_error and "Delegation skill" in res.content
