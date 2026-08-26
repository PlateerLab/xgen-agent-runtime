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
