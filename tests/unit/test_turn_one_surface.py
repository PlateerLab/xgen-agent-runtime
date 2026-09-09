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
        ("ArtifactGuide", "ArtifactSave"),
    ):
        assert is_turn_one(gate), gate
        assert not is_turn_one(member), member


def test_문서편집은_검색해야_나온다():
    # 사용자 지시: "Documents 편집은 검색해야 나오는거고".
    for name in ("DocGuide", "DocBuild", "DocEdit", "DocGenerate", "DocRender"):
        assert not is_turn_one(name), name


#: 입구에 서지 **않는** 문과 그 이유. 여기 없는 문은 전부 입구에 서야 한다.
#:
#: 이 표가 있는 이유: 문을 만들면서 입구에 세우는 것을 잊으면 아무 신호도 나지
#: 않는다. 도구는 등록돼 있고 부르면 돌기 때문에 테스트도 로그도 조용하다.
#: 에이전트만 그 능력이 없는 것처럼 행동한다 — 그게 SSH 에게 실제로 일어난 일이다.
_DOORS_THAT_STAY_HIDDEN = {
    # 사용자 지시. 위 테스트가 같은 사실을 못박는다.
    "DocGuide": "Documents 편집은 검색해야 나온다 (사용자 지시)",
}


def test_선언된_문은_전부_입구에_선다():
    """계층의 약속은 "숨긴다" 가 아니라 "문 하나만 보이고 방은 그 뒤" 다.

    문까지 숨기면 그 능력은 **없는 것과 같다.** ToolSearch 로 정확한 단어를
    맞혀야만 닿는데, 필요한 순간의 에이전트는 그 단어를 모른다.

    2026-09-09 실증: SSH 서버를 두 대 등록해 둔 사용자의 턴이 배포 단계에서
    "서버 SSH 인증 정보가 이 세션에 없습니다" 로 멈췄다. SshListServers 는
    등록돼 있었고 부르면 돌았다 — 입구에 없었을 뿐이다.
    """
    from xgen_agent_runtime.tools.built_in import SKILL_GATEWAYS

    for gateway in SKILL_GATEWAYS:
        if gateway in _DOORS_THAT_STAY_HIDDEN:
            assert not is_turn_one(gateway), (
                f"{gateway} 는 숨기기로 한 문인데 입구에 섰다 — "
                f"이유: {_DOORS_THAT_STAY_HIDDEN[gateway]}"
            )
            continue
        assert is_turn_one(gateway), (
            f"{gateway} 가 입구에 없다. 문을 숨기면 그 패밀리는 없는 것과 같다. "
            f"의도한 것이라면 _DOORS_THAT_STAY_HIDDEN 에 이유와 함께 적어라."
        )


def test_문_뒤의_방은_계속_숨어_있다():
    """문을 입구에 올렸다고 방까지 딸려 올라오면 계층이 사라진다."""
    from xgen_agent_runtime.tools.built_in import SKILL_GATEWAYS

    for gateway, family in SKILL_GATEWAYS.items():
        for member in family:
            assert not is_turn_one(member), f"{gateway} 의 방 {member} 이 입구에 섰다"


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
        "WorkflowSelf", "ForgeTool", "ToolSearch",
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
    #
    # 26 (2026-09-08, 25 →): SystemPackages 하나. `command not found` 를 만난
    # 에이전트가 고칠 길을 못 찾고 후퇴하거나, 셸 apt 로 깔았다가 다음 세션에
    # 다시 잃는 것을 막는다 — PythonEnv 가 첫 턴에 있는 것과 같은 이유다.
    # 같은 날 Shell 이 빠지고 LocalControl 이 들어와 순증은 +1 이다.
    #
    # 27 (2026-09-09, 26 →): SshListServers 하나 — **빠져 있던 문을 채운 것**이지
    # 새 능력을 올린 것이 아니다. 다른 문은 전부 입구에 서 있었고 SSH 만
    # 방까지 숨어 있었다. 이 문은 다른 문보다 싸다: 호스트 게이트
    # (feature:ssh_enabled)가 먼저 걸러서 **서버를 실제로 등록한 세션에만**
    # 등록되므로, 나머지 세션의 입구는 26개 그대로다.
    #
    # 이 선을 다시 올리려면 **한 도구씩** 이유를 여기 적어야 한다. 이 주석이
    # 길어지는 것이 곧 표면이 넓어졌다는 신호다.
    assert len(TURN_ONE_TOOLS) <= 27, sorted(TURN_ONE_TOOLS)


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
    assert not is_turn_one("mcp_local_DocBuild")


def test_자기_환경을_고치는_길은_문_뒤에_두지_않는다():
    """``command not found`` / ``ModuleNotFoundError`` 앞에서 후퇴하지 않으려면,
    고칠 길이 **그 자리에** 보여야 한다.

    PythonEnv 를 숨겼더니 에이전트가 자기 환경에 패키지를 깔 수 있다는 걸 모른 채
    마크다운으로 후퇴했다(2026-08-18). SystemPackages 도 같은 이유로 첫 턴에 둔다 —
    숨기면 셸 apt 로 깔고 다음 세션에 다시 잃는다(2026-09-08: git 이 두 번 사라졌다).
    """
    assert is_turn_one("PythonEnv")
    assert is_turn_one("SystemPackages")
    assert is_turn_one("mcp__connector__SystemPackages")


def test_사용자_PC_는_문_뒤에_있다():
    """장소가 다르면 문 뒤다.

    예전에는 사용자 PC 셸(``Shell``)이 문 없이 첫 턴에 섰다 — 우리 ``Bash`` 옆에,
    같은 층의 동사로. 모델은 둘을 바꿔 써도 되는 것으로 읽었고, 샌드박스 작업을
    하던 턴이 사용자 PC 셸을 불러 42분을 타임아웃으로 태웠다(2026-09-08 실측).

    이제 입구에는 문(``LocalControl``) 하나만 서고, 그 PC 의 셸·파일·브라우저·
    앱·오피스는 전부 그 뒤에 있다.
    """
    assert is_turn_one("LocalControl")
    assert is_turn_one("mcp__connector__LocalControl")
    for member in ("Shell", "ShellJob", "Open", "ReadFile", "WriteFile",
                   "ListDir", "Search", "Clipboard", "Notify"):
        assert not is_turn_one(member), member
        assert not is_turn_one(f"mcp_local_{member}"), member


def test_아티팩트는_문이_첫_턴에_있다():
    """만든 것을 대화에 찍는 대신 화면으로 내놓는 길 — 문이 안 보이면 그런 길이
    있다는 것 자체를 모른다. 멤버는 문 뒤다."""
    assert is_turn_one("ArtifactGuide")
    for member in ("ArtifactSave", "ArtifactList", "ArtifactDelete"):
        assert not is_turn_one(member), member
