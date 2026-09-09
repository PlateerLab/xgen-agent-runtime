"""How many tools the agent sees at once.

Our tool surface is **hierarchical**, not flat. The agent always sees its basic
tools — web, files, shell, delegation, memory, the search tool of every attached
knowledge source. Everything beyond that (connected API / DB / MCP nodes, which
can be hundreds of schemas) is announced by name and one line, and the agent
pulls in the full schema with ``ToolSearch`` when it actually needs it.

That is the point: a tool list is a map, not an inventory. Sending every schema
up front spends the context window on things the turn will never call, and a
model that reads a hundred near-identical schemas picks worse than one that
reads five and drills into the right one.

Two settings:

``hierarchy``
    The default. Basic tools visible, the rest discovered on demand.

``flat``
    Every connected schema up front. An escape hatch for models that cannot
    drive a discovery step; it costs tokens on every request.

Older workflows stored ``all`` (everything up front) or ``search`` (defer).
Both now resolve to ``hierarchy`` — the hierarchy is the platform's behaviour,
and an agent that wants the flat surface says so explicitly.
"""

from __future__ import annotations

import re

#: Basic tools stay visible; the rest is discovered with ToolSearch.
HIERARCHY = "hierarchy"
#: Every connected tool schema is sent up front.
FLAT = "flat"

#: Values that mean "send everything up front".
_FLAT_ALIASES = frozenset({FLAT, "all_upfront", "upfront"})


def normalize_exposure(value: object) -> str:
    """Resolve a stored ``tool_exposure`` to ``hierarchy`` or ``flat``.

    Unknown values resolve to ``hierarchy`` rather than raising: an exposure
    setting is a preference, and a typo in it should not stop a turn from
    running.
    """
    text = str(value or "").strip().lower()
    return FLAT if text in _FLAT_ALIASES else HIERARCHY


def sends_every_schema(value: object) -> bool:
    """Should connected tool nodes be registered as immediately visible?

    Only the flat surface says yes. Under the hierarchy they are registered
    deferred, and ``ToolSearch`` activates the ones a turn actually needs.
    """
    return normalize_exposure(value) == FLAT


# ── 턴 1 표면 ────────────────────────────────────────────────────────
#
# "계층적"은 **적게 보여 준다**가 아니라 **한 겹씩 보여 준다**는 뜻이다. 첫 턴에
# 보이는 것은 능력의 목록이 아니라 능력으로 가는 **입구**의 목록이어야 한다:
# 바로 쓰는 기본 도구 몇 개와, 나머지를 여는 문 하나씩.
#
# 이 상수가 없던 동안 등록부는 정반대로 굴었다 — Bash·웹·브라우저는 숨고,
# 위임 6종·작업 4종·메모리 6종이 전부 첫 턴에 쏟아졌다. 그래서 "무슨 도구가
# 있냐"는 물음에 에이전트가 재고 목록을 읊고, 정작 셸은 못 찾았다.
#
# 이름을 여기 한 곳에 모은 이유도 그것이다. 등록 지점은 다섯 군데(내장 패밀리·
# 메모리·작업·위임·커넥터)에 흩어져 있고, 각자가 자기 판단으로 ``core=True`` 를
# 쓰면 표면은 아무도 의도하지 않은 모양이 된다.

#: 첫 턴에 스키마까지 보이는 도구.
#:
#: 각 줄은 **입구 하나**다. 패밀리 전체를 올리는 줄은 없다 — 기본 명령만 예외인데,
#: 그건 게이트웨이를 둘 수 없는 종류의 도구이기 때문이다(셸을 여는 문은 셸이다).
TURN_ONE_TOOLS = frozenset(
    {
        # 1. 기본 명령 — 셸과 파일. 여기가 막히면 나머지가 다 무의미하다.
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        # 2. 도구 발견 — 아래 계층 전부로 가는 문.
        "ToolSearch",
        # 3. 기억 — 도구가 곧 능력이라 게이트웨이를 둘 것이 없다.
        "memory_write",
        "memory_read",
        "memory_list",
        "memory_search",
        "memory_pin",
        "memory_categories",
        # 4. 영구 작업 — JobSchedule/JobList/JobCancel 은 이 문 뒤에.
        "JobGuide",
        # 4-b. 아티팩트(사용자가 여는 화면) — ArtifactSave/List/Delete 는 이 문 뒤에.
        #      만든 것을 대화에 찍는 대신 화면으로 내놓는 길이라, 문이 안 보이면
        #      그런 길이 있다는 것 자체를 모른다.
        "ArtifactGuide",
        # 5. 위임 — DelegateTask/SubAgent*/Task* 는 이 문 뒤에.
        "DelegationGuide",
        # 6. 도구 제작 — 이 넷은 문을 두지 않는다. 숨겼더니 에이전트가 자기
        #    환경에 패키지를 깔 수 있다는 걸 모른 채 ModuleNotFoundError 앞에서
        #    후퇴했다(2026-08-18 실증). 스키마 넷은 표면을 무너뜨린 쪽이 아니다.
        "ForgeTool",
        "ListForgedTools",
        "DeleteForgedTool",
        "PythonEnv",
        #    같은 이유로 **시스템 패키지**도 여기 있다. 에이전트가
        #    `command not found` 를 만났을 때 고칠 길이 문 뒤에 있으면, 그
        #    사실을 모른 채 후퇴하거나 셸 apt 로 깔고 다음 세션에 다시 잃는다
        #    (2026-09-08 실증: git 이 그렇게 두 번 사라졌다).
        "SystemPackages",
        # 7. 자기 진화 — 한 도구가 action 으로 자기 안을 연다.
        "WorkflowSelf",
        # 웹 — 브라우저가 없는 표면(웹 대화)의 유일한 바깥 통로라 항상 둔다.
        "WebFetch",
        "WebSearch",
        # 브라우저 — 실제 조작 도구는 이 문 뒤에. an-web·커넥터 양쪽이 같은 이름을
        #   쓰므로, 어느 쪽이 이번 턴의 주인이든 표면의 모양은 같다.
        "BrowserGuide",
        # 로컬 컨트롤 — 사용자 PC 를 조작하는 **문**. 방(셸·파일·브라우저·앱·
        #   오피스·사용자가 붙인 MCP 서버)은 이 문 뒤에 있다.
        #
        #   예전에는 그 방의 셸(``Shell``)이 문 없이 여기 서 있었고, 주석은
        #   "우리 쪽 Bash 와 같은 층의 동사" 라고 적혀 있었다. 그건 **틀린
        #   말이었다** — 같은 동사지만 **다른 기계**다. 입구에 나란히 세워 두면
        #   모델은 둘을 바꿔 써도 되는 것으로 읽고, 실제로 그랬다: 샌드박스
        #   작업을 하던 턴이 사용자 PC 셸을 불러 42분을 타임아웃으로 태웠다.
        #
        #   장소가 다른 능력은 전부 **문을 입구에 세우고 방을 그 뒤에** 둔다 —
        #   브라우저(BrowserGuide)·위임(DelegationGuide)이 그 규약이다.
        "LocalControl",
        # SSH — 사용자가 등록한 **원격 서버**. 여기도 장소가 다른 능력이라 같은
        #   규약을 따른다: 문(SshListServers)만 서고 SshRun/Upload/Download 는
        #   그 뒤다.
        #
        #   4.15.0 의 주석은 "SSH(SshListServers)가 이미 그 규약이다" 라고 적었지만
        #   **사실이 아니었다** — 방도 문도 둘 다 숨어 있었다. 그래서 서버를 두 대
        #   등록해 둔 사용자의 턴이 배포 단계에서 "서버 SSH 인증 정보가 이 세션에
        #   없습니다" 로 멈췄다(2026-09-09 실증). 도구는 등록돼 있었고 부르면 돌았다.
        #   에이전트가 그 존재를 알 길이 없었을 뿐이다.
        #
        #   이 문이 다른 문보다 싼 이유: 호스트 게이트(``feature:ssh_enabled``)가
        #   먼저 걸러서, **서버를 실제로 등록한 세션에만** 등록된다. 나머지 세션의
        #   입구는 지금과 똑같다.
        "SshListServers",
    }
)

#: 우리 도구가 MCP 를 지나며 얻는 접두 — ``mcp_local_BrowserNavigate``,
#: ``mcp__connector__WorkflowSelf``. 같은 도구인데 표면에 따라 이름이 달라진다.
_MCP_PREFIX = re.compile(r"^mcp_{1,2}[A-Za-z0-9-]+_{1,2}")


def is_turn_one(name: object) -> bool:
    """이 도구가 계층 표면의 첫 턴에 보이는가.

    MCP 를 지나온 이름(``mcp_local_BrowserNavigate``)도 같은 도구로 본다. 표면이
    이름을 바꾼다고 계층이 달라지면, 커넥터를 연결한 사용자만 다른 규칙을 받는다.

    모르는 이름은 **아니다** — 계층은 화이트리스트다. 새 도구가 조용히 첫 턴에
    끼어들면 표면은 한 번에 무너지지 않고 한 줄씩 무너진다. (그 대가로 남의 MCP
    서버가 우리 게이트웨이와 똑같은 이름을 쓰면 첫 턴에 선다 — 스키마 하나가 더
    보이는 것뿐이라, 우리 도구가 표면에 따라 사라지는 쪽보다 낫다.)
    """
    text = str(name or "")
    return text in TURN_ONE_TOOLS or _MCP_PREFIX.sub("", text) in TURN_ONE_TOOLS


def registers_core(name: object, *, flat: bool) -> bool:
    """등록 지점이 물어야 할 단 하나의 질문.

    ``flat`` 이면 전부 선노출(탈출구), 아니면 턴 1 표면만.
    """
    return True if flat else is_turn_one(name)
