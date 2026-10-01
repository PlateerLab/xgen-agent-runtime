"""How many tools the agent sees at once.

Our tool surface is **hierarchical**, not flat. The agent always sees its basic
tools — web, files, shell, memory, the search tool of every attached
knowledge source. Everything beyond that (connected API / DB / MCP nodes, which
can be hundreds of schemas) is announced by name and one line, and the agent
pulls in the full schema with ``ToolSearch`` when it actually needs it.

That is the point: a tool list is a map, not an inventory. Sending every schema
up front spends the context window on things the turn will never call, and a
model that reads a hundred near-identical schemas picks worse than one that
reads five and drills into the right one.

Three settings:

``hierarchy``
    The default. Basic tools visible, the rest discovered on demand.

``flat``
    Every connected schema up front. An escape hatch for models that cannot
    drive a discovery step; it costs tokens on every request.

``map``
    The hierarchy, narrowed by the agent's memory. When the host already knows
    where this turn's answer lives (``HostServices.turn_tool_map``), only that
    turn's tools and a small base stay visible; everything else stays registered
    and discoverable. Turns the host has no map for run as ``hierarchy``.

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
#: The hierarchy, narrowed per turn by the host's memory map.
MAP = "map"

#: Values that mean "send everything up front".
_FLAT_ALIASES = frozenset({FLAT, "all_upfront", "upfront"})


def normalize_exposure(value: object) -> str:
    """Resolve a stored ``tool_exposure`` to ``hierarchy`` or ``flat``.

    Unknown values resolve to ``hierarchy`` rather than raising: an exposure
    setting is a preference, and a typo in it should not stop a turn from
    running.
    """
    text = str(value or "").strip().lower()
    if text == MAP:
        return MAP
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
#: 1. 기본 명령 — 셸과 파일. 여기가 막히면 나머지가 다 무의미하다.
TURN_ONE_BASICS = frozenset({"Bash", "Read", "Write", "Edit", "Glob", "Grep"})
#: 2. 도구 발견 — 아래 계층 전부로 가는 문. 2-b. 목록 작업(ToolBatch) — 같은 도구를 여러 입력으로 한 왕복에.
#:    첫 턴에 보여야 항목마다 따로 부르는 습관(왕복 N회)이 처음부터 생기지 않는다.
#:    2-c. 표 내보내기(TableExport) 는 여기 있었다 — 28일 실측 5회(턴의 0.1%)에 318토큰이 매 호출에 실렸다.
#:    카탈로그로 내린다(ToolSearch 로 열림).
TURN_ONE_DISCOVERY = frozenset({"ToolSearch", "ToolBatch"})
#: 3. 기억 — 도구가 곧 능력이라 게이트웨이를 둘 것이 없다.
TURN_ONE_MEMORY = frozenset(
    {"memory_write", "memory_read", "memory_list", "memory_search", "memory_pin", "memory_categories"}
)

TURN_ONE_TOOLS = TURN_ONE_BASICS | TURN_ONE_DISCOVERY | TURN_ONE_MEMORY | frozenset(
    {
        # 4. 영구 작업 — JobSchedule/JobList/JobCancel 은 이 문 뒤에.
        "JobGuide",
        # 4-b. 앱(사용자가 여는 웹 앱) — AppCreate/Publish/Status/List/Delete
        #      는 이 문 뒤에. 만든 것을 대화에 찍는 대신 화면으로 내놓는 길이라, 문이
        #      안 보이면 그런 길이 있다는 것 자체를 모른다. "도구를 만들어 달라" 는 이
        #      문이 아니라 6(SelfExtendGuide → ForgeTool)의 몫이다 — 두 문의 설명이 그
        #      경계를 말한다(tests/unit/test_tool_request_routing.py).
        "AppGuide",
        # 6. 자기확장 — 제작(ForgeTool·ListForgedTools·DeleteForgedTool)·환경
        #    (PythonEnv·SystemPackages)·자기진화(WorkflowSelf)는 **문 하나** 뒤에.
        #
        #    예전 주석: "이 넷은 문을 두지 않는다. 숨겼더니 에이전트가 자기 환경에
        #    패키지를 깔 수 있다는 걸 모른 채 후퇴했다(2026-08-18)". 그 회귀의 원인은
        #    숨긴 것이 아니라 **문이 없던 것**이었다 — 위임·브라우저·작업이 문으로
        #    해결한 것과 같다. 2026-09-20 실측(Qwen 토크나이저): 여섯 스키마가
        #    2,854토큰 = 고정 프리픽스 8.6k 의 33% 를 **모든 호출**에 실었고, 쓰인
        #    턴은 각각 0~2.4% 였다. 문의 설명이 능력을 말해 인식은 남고, 쓰는 턴에만
        #    왕복 하나가 는다.
        "SelfExtendGuide",
        # 웹 — 바깥으로 가는 일반 통로(검색·페이지 읽기). 모든 표면에 항상 둔다.
        "WebFetch",
        "WebSearch",
        # 브라우저 — 사용자 PC 의 XGEN 브라우저 탭(커넥터 ``mcp_local_BrowserGuide``)만 있다.
        #   조작 도구 6종은 이 문 뒤에. 서버 브라우저 엔진(an-web)은 제거됐다.
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
        #   브라우저(BrowserGuide)·작업(JobGuide)이 그 규약이다.
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
#: CLI 브릿지가 커넥터 도구에 한 겹 더 씌우는 접두.
_CLI_PREFIX = "mcp__connector__"


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


# ── 기억 지도 표면 (map) ─────────────────────────────────────────────
#
# 같은 질문을 전에 답한 적이 있으면, 그때 통한 도구만 있으면 된다. 호스트(기억 게이트웨이)가 턴마다
# 열 도구를 정하고(``HostServices.turn_tool_map``), 등록이 다 끝난 뒤 첫 화면을 그 도구와 아래 기본으로
# 좁힌다. 나머지는 **등록된 채 숨긴다**: ToolSearch 로 이름을 부르면 열리고, 숨긴 가족의 문은 Stage 3 의
# 도달성 검사가 다시 세운다(tools.gates). 지도의 도구가 빈 결과나 오류를 두 번 내면 Stage 3 이 이 턴의
# 표면을 지도 이전으로 되돌린다(:func:`map_fallback_due`, :func:`restore_tool_map`).
#
# 등록 지점은 지도를 모른다 — 지도는 등록 계획(``registers_core``)이 아니라 다 지은 표면 위에 한 번 얹는다.
# 그래야 등록 지점 다섯 곳이 각자 지도를 해석하지 않는다.

#: 지도 표면에서도 늘 보이는 도구 — 턴 1 표의 발견 · 기억 그룹 그대로(표는 하나다).
MAP_BASE_TOOLS = TURN_ONE_DISCOVERY | TURN_ONE_MEMORY

#: 지도의 도구가 이만큼 빈 결과·오류를 내면 지도 이전 표면으로 되돌린다.
MAP_FALLBACK_MISSES = 2


def uses_tool_map(value: object) -> bool:
    """이 에이전트가 기억 지도 표면을 골랐는가."""
    return normalize_exposure(value) == MAP


def _bare(name: object) -> str:
    """접두를 끝까지 벗긴 이름. CLI 브릿지를 지난 커넥터 도구는 접두가 두 겹이다
    (``mcp__connector__mcp_local_ReadFile``)."""
    text = str(name or "")
    while True:
        stripped = _MCP_PREFIX.sub("", text)
        if stripped == text:
            return text
        text = stripped


def _read_only(tool) -> bool:
    try:
        return bool(tool.capabilities({}).read_only)
    except Exception:  # noqa: BLE001
        return False


def resolve_tool_map(registry, open_names) -> list:
    """호스트가 말한 것을 이번 턴 레지스트리의 도구 이름으로 푼다.

    호스트(기억 게이트웨이)는 이름표를 들고 있지 않다. 말할 수 있는 것은 네 가지다:
      * ``family:<이름>[:ro]`` — 내장 도구 가족(:data:`BUILT_IN_TOOL_FEATURES`). ``:ro`` 면 읽기 전용 멤버만.
      * ``folder[:ro]`` — 이 대화에 연결된 폴더의 기기 도구(:func:`host.local_folders.is_folder_tool`).
      * ``gate:<이름>`` — 문(:mod:`tools.gates`). 문만 세우고 방은 문이 열리게 둔다(라우터 규약).
      * 그 밖은 도구 이름 그대로(MCP 접두는 벗겨 비교). 등록되지 않은 이름은 무시한다.
    """
    from xgen_agent_runtime.host.local_folders import is_folder_tool
    from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_FEATURES
    from xgen_agent_runtime.tools.gates import gate_of

    names = list(registry.list_names())
    registry_names = set(names)
    by_bare = {}
    for n in names:
        by_bare.setdefault(_bare(n), []).append(n)
    out: list = []

    def add(n):
        if n not in out:
            out.append(n)

    def add_group(members, qual):
        # ``:ro`` 는 읽기 전용 멤버만. 아무도 능력을 말하지 않는 가족(주석 없는 옛 기기 · 외부 MCP)이면 다 연다 -
        # 빈 표면보다 넓은 표면이 낫다(되돌림 규칙과 같은 방향).
        picked = [n for n in members if qual != "ro" or _read_only(registry.get(n))]
        for n in picked or list(members):
            add(n)

    for raw in open_names or ():
        token = str(raw or "").strip()
        if not token:
            continue
        kind, _, rest = token.partition(":")
        if kind == "family" and rest:
            fam, _, qual = rest.partition(":")
            add_group([n for member in BUILT_IN_TOOL_FEATURES.get(fam, ()) for n in by_bare.get(member, ())], qual)
        elif kind == "folder":
            add_group([n for n in names if is_folder_tool(n)], rest)
        elif kind == "gate" and rest:
            for n in names:
                if _bare(n) == rest and gate_of(n) is not None:
                    add(n)
        else:
            # 등록된 이름 그대로가 먼저(CLI 브릿지 접두 한 겹은 벗긴다) - 다른 서버의 같은 이름(mcp_other_ReadFile)과
            # 섞지 않는다. 접두 없는 이름(ReadFile)만 접두를 벗겨 맞춘다.
            exact = token[len(_CLI_PREFIX):] if token.startswith(_CLI_PREFIX) else token
            if exact in registry_names:
                add(exact)
            else:
                for n in by_bare.get(_bare(token), ()):
                    add(n)
    return out


def apply_tool_map(registry, open_names) -> dict:
    """첫 화면을 기본(:data:`MAP_BASE_TOOLS`)과 지도가 연 도구로 좁힌다.

    ``open_names`` 는 호스트가 말한 것(:func:`resolve_tool_map` 이 푼다). 반환값은 되돌릴 때 쓰는 기록
    ``{"open", "hidden", "core", "activated"}`` 이다.
    """
    names = list(registry.list_names())
    wanted = set(resolve_tool_map(registry, open_names))   # 등록된 이름 그대로 비교한다(다른 서버의 같은 이름은 다른 도구)
    record = {
        "open": [],
        "hidden": [],
        "core": {n: bool(registry.is_core(n)) for n in names},
        "activated": list(registry.activated_names()),
    }
    for name in names:
        if name in wanted:
            record["open"].append(name)
        if name in wanted or name in MAP_BASE_TOOLS:
            registry.activate(name)
        elif registry.is_exposed(name):
            registry.set_core(name, False)
            registry.deactivate(name)
            record["hidden"].append(name)
    return record


def restore_tool_map(registry, record: dict) -> None:
    """:func:`apply_tool_map` 이전 표면으로 되돌린다(기록에 없는 이름은 그대로)."""
    activated = set(record.get("activated") or ())
    for name, core in (record.get("core") or {}).items():
        if registry.get(name) is None:
            continue
        registry.set_core(name, bool(core))
        if name in activated:
            registry.activate(name)
        else:
            registry.deactivate(name)


def _field(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(_field(b, "text") or "") for b in content)
    return str(content or "")


def turn_messages(messages) -> list:
    """이번 턴의 메시지 - 마지막 사용자 발화(도구 결과가 아닌 것) 뒤."""
    msgs = list(messages or ())
    start = 0
    for i, msg in enumerate(msgs):
        if _field(msg, "role") != "user":
            continue
        content = _field(msg, "content")
        if isinstance(content, str) or not any(_field(b, "type") == "tool_result" for b in (content or [])):
            start = i + 1
    return msgs[start:]


def map_fallback_due(messages, open_names) -> bool:
    """이번 턴(마지막 사용자 발화 뒤)에서 지도의 도구가 빈 결과나 오류를 :data:`MAP_FALLBACK_MISSES` 번 냈는가.

    ``open_names`` 는 등록된 이름 그대로(:func:`apply_tool_map` 의 기록). 지도가 연 도구가 없으면(답을 기억에서
    그대로 내는 턴) 어떤 도구든 센다 — 그 턴에 도구가 두 번 빗나갔다면 기억이 틀렸을 수 있으니 계층형으로 돌아가
    찾게 한다.
    """
    wanted = {str(n) for n in (open_names or ()) if str(n or "").strip()}
    calls = {}
    misses = 0
    for msg in turn_messages(messages):
        content = _field(msg, "content")
        if not isinstance(content, list):
            continue
        for block in content:
            kind = _field(block, "type")
            if kind == "tool_use":
                calls[str(_field(block, "id") or "")] = str(_field(block, "name") or "")
            elif kind == "tool_result":
                name = calls.get(str(_field(block, "tool_use_id") or ""), "")
                if wanted and name not in wanted:
                    continue
                if _field(block, "is_error") or not _result_text(_field(block, "content")).strip():
                    misses += 1
    return misses >= MAP_FALLBACK_MISSES
