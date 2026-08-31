"""계층 표면의 첫 턴에 무엇이 서는가.

이 규칙은 **눈으로 확인되지 않는다**. 표면이 무너져도 에이전트는 조용히 다르게
행동할 뿐이고("무슨 도구가 있냐"에 재고 목록을 읊거나, 셸이 있는데 없다고 하거나),
로그에는 아무 오류도 남지 않는다. 실제로 그렇게 무너져 있었다 — Bash·웹·브라우저는
숨고 위임 6종·작업 4종이 첫 턴에 쏟아졌다. 그래서 여기서 못박는다.
"""
from xgen_agent_runtime.host.tool_exposure import (
    TURN_ONE_TOOLS,
    is_turn_one,
    registers_core,
)


def test_기본_명령은_첫_턴에_있다():
    # 셸을 여는 문은 셸이다 — 게이트웨이 뒤에 두면 열 방법이 없다.
    for name in ("Bash", "Read", "Write", "Edit", "Glob", "Grep"):
        assert is_turn_one(name), name


def test_패밀리는_문_하나만_내놓는다():
    # 문은 보이고, 그 뒤의 식구들은 보이지 않는다.
    for gate, member in (
        ("JobGuide", "JobSchedule"),
        ("DelegationGuide", "SubAgentSpawn"),
        ("DelegationGuide", "TaskCreate"),
        ("BrowserGuide", "BrowserNavigate"),
    ):
        assert is_turn_one(gate), gate
        assert not is_turn_one(member), member


def test_문서편집은_검색해야_나온다():
    # 사용자 지시: "Documents 편집은 검색해야 나오는거고".
    for name in ("DocGuide", "DocBuild", "DocEdit", "DocGenerate", "DocRender"):
        assert not is_turn_one(name), name


def test_도구_제작은_문을_두지_않는다():
    """숨겼더니 에이전트가 패키지를 깔 수 있다는 걸 모른 채 후퇴했다(2026-08-18).

    이 넷은 표면을 무너뜨린 쪽이 아니다 — 무너뜨린 것은 위임 12종·작업 3종·
    문서 8종처럼 **패밀리를 통째로** 올린 쪽이었다.
    """
    for name in ("ForgeTool", "ListForgedTools", "DeleteForgedTool", "PythonEnv"):
        assert is_turn_one(name), name


def test_기억과_자기확장은_첫_턴에_있다():
    for name in (
        "memory_write", "memory_read", "memory_list",
        "memory_search", "memory_pin", "memory_categories",
        "WorkflowSelf", "ForgeTool", "FileCloud", "ToolSearch",
    ):
        assert is_turn_one(name), name


def test_웹_통로는_브라우저가_없는_표면의_유일한_바깥이다():
    # 웹 대화에는 커넥터 브라우저가 없다 — WebFetch/WebSearch 까지 숨기면
    # 에이전트가 바깥을 볼 방법이 사라진다.
    assert is_turn_one("WebFetch") and is_turn_one("WebSearch")


def test_모르는_이름은_첫_턴이_아니다():
    # 화이트리스트다. 새 도구가 조용히 끼어들면 표면은 한 줄씩 무너진다.
    assert not is_turn_one("SomeNewTool")
    assert not is_turn_one("")
    assert not is_turn_one(None)


def test_flat_은_전부_선노출한다():
    assert registers_core("DocBuild", flat=True)
    assert not registers_core("DocBuild", flat=False)
    assert registers_core("Bash", flat=False)


def test_첫_턴_표면은_스물몇_개를_넘지_않는다():
    # 정확한 수를 박지 않는 이유: 추가는 있을 수 있다. 다만 "패밀리를 통째로
    # 올렸다"는 실수는 이 선을 반드시 넘는다.
    assert len(TURN_ONE_TOOLS) <= 25, sorted(TURN_ONE_TOOLS)


def test_한_묶음_안에서도_계층이_갈린다():
    """커넥터가 도구를 한 뭉치로 건네도 표면은 뭉치째 올라가지 않는다.

    커넥터를 연결하는 순간 브라우저 조작 6종이 통째로 첫 턴에 올라오던 자리다.
    """
    from xgen_agent_runtime.host.tools import adapt_tools
    from xgen_agent_runtime.tools import build_tool

    def _fake(name):
        return build_tool(
            name=name,
            description=name,
            input_schema={"type": "object", "properties": {}},
            execute=lambda _input, _ctx: name,
        )

    registry = adapt_tools(
        [_fake("BrowserGuide"), _fake("BrowserNavigate"), _fake("Bash")],
        core=lambda name: registers_core(name, flat=False),
    )
    assert sorted(registry.core_names()) == ["Bash", "BrowserGuide"]
    assert [t.name for t in registry.list_deferred()] == ["BrowserNavigate"]

    everything = adapt_tools([_fake("BrowserNavigate"), _fake("Bash")], core=True)
    assert sorted(everything.core_names()) == ["Bash", "BrowserNavigate"]


def test_MCP_를_지나온_이름도_같은_도구다():
    """커넥터 도구는 ``mcp_local_*`` 로, CLI 표면은 ``mcp__connector__*`` 로 온다.

    표면이 이름을 바꾼다고 계층이 달라지면 커넥터를 연결한 사용자만 다른 규칙을
    받는다 — 브라우저 조작 6종이 통째로 첫 턴에 서던 자리다.
    """
    assert is_turn_one("mcp_local_BrowserGuide")
    assert not is_turn_one("mcp_local_BrowserNavigate")
    assert is_turn_one("mcp__connector__WorkflowSelf")
    assert is_turn_one("mcp_local_Shell")
    assert not is_turn_one("mcp_local_DocBuild")
