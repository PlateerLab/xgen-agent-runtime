"""'도구를 만들어 달라' 는 요청은 ForgeTool 로 간다 — 앱(아티팩트)으로 새지 않는다.

실측(2026-09-24, dev 턴-1 표면을 그대로 재현한 claude CLI + MCP 스텁, sonnet):
"요약 도구", "환율 계산 도구", "주식 시세 툴", "PDF 표 → 엑셀 도구" 처럼 **사람이 쓸 도구**를
시키면 도구 요청 10개 중 5개가 ArtifactGuide → ArtifactCreate 로 갔다. 입구에 선 두 문 중
ArtifactGuide 는 "사람이 OPEN 해서 쓰는 것" 을, SelfExtendGuide 는 "extend yourself" 를
말했다. 사람이 쓸 도구는 앞쪽으로 읽혔다.

XGEN 화면에서 "도구" 는 ForgeTool 이 만든 것이다([Agent 생성 도구], 에이전트 [도구] 탭,
"'이 작업을 도구로 만들어 둬' 라고 요청해 보세요"). 그래서 문의 설명이 그 말의 주인을 밝힌다.

이 테스트는 문장을 외우지 않는다 — **모델이 길을 고르는 데 쓰는 사실**만 본다:
문이 "도구" 요청을 자기 것이라고 말하는가, 그리고 ToolSearch 가 도구 제작의 흔한 말로
ForgeTool 을 찾는가(예전에는 "create tool" 이 ArtifactGuide 를, "make tool" 이 "없다" 를 돌려줬다).
"""

from __future__ import annotations

import pytest

from xgen_agent_runtime.host.forged_tools import ForgeTool
from xgen_agent_runtime.tools.built_in.self_extend_guide_tool import _MAP, SelfExtendGuideTool
from xgen_agent_runtime.tools.built_in.tool_search_tool import _rank


class _NoStore:
    def list(self):
        return []


def _forge() -> ForgeTool:
    return ForgeTool(workflow_id="wf", workspace_dir="/tmp", registry=None, store=_NoStore())


def test_the_self_extension_door_claims_tool_requests_first():
    first = SelfExtendGuideTool().description.split("—")[0]
    assert "TOOL" in first and "make" in first, first
    assert "ForgeTool" in SelfExtendGuideTool().description


def test_forge_tool_says_what_the_user_calls_it():
    desc = _forge().description
    assert desc.startswith("Make"), desc[:60]
    assert "Not an app" in desc


@pytest.mark.parametrize("query", ["create tool", "make tool", "build tool", "tool"])
def test_tool_search_finds_forge_tool_with_everyday_words(query):
    t = _forge()
    assert _rank({"name": t.name, "description": t.description, "input_schema": t.input_schema}, query) > 0


def test_the_map_tells_a_tool_from_an_app():
    assert "make a TOOL" in _MAP
    assert "not an app" in _MAP


def test_an_opened_room_is_reachable_even_before_the_client_refreshes():
    """CLI 클라이언트는 문이 연 도구를 같은 턴에 못 볼 수 있다(2026-09-24 실측: Claude Code
    2.1.236 은 158회 중 70회 "No such tool available", Codex 0.156.1 은 턴 안에서 목록을
    갱신하지 않았다). 첫 턴 표면의 ToolBatch 는 이름으로 어떤 도구든 부르므로 그 길을 알려 준다."""
    from xgen_agent_runtime.tools.built_in._skill_gateway import with_opened

    text = with_opened("map", ["ForgeTool"])
    assert "Now callable: ForgeTool." in text
    assert 'ToolBatch(tool="<name>"' in text


def test_tool_search_says_the_same_when_it_activates():
    import inspect

    from xgen_agent_runtime.tools.built_in import tool_search_tool

    assert 'ToolBatch(tool="<name>"' in inspect.getsource(tool_search_tool.ToolSearchTool.execute)
