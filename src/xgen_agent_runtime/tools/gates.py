"""문(gate)과 그 뒤의 방(family)을 **한 표**에서 정한다.

왜 필요한가 — "문은 보이는데 방이 안 열린다" 가 2주 새 네 번
-------------------------------------------------------------
턴-1 표면(``host.tool_exposure``)은 기본 동사만 세우고, 나머지 능력은 가족마다 **문 하나**
뒤에 숨긴다. 숨기는 일은 한 곳(``TURN_ONE_TOOLS``)이 한꺼번에 하는데, **여는 일은 문마다
따로** 선언해야 했다 — 내장 안내 도구는 모듈마다 ``open_family(...)`` 를 부르고, 어댑트된
LangChain 도구는 ``metadata["opens_family"]`` 를 달아야 했다. 빠뜨려도 아무 오류가 없었다.
그래서 같은 모양의 사고가 되풀이됐다:

* 09-09 ``SshListServers`` — 방도 문도 둘 다 숨어 "인증 정보가 없다" 로 멈춤.
* ``DelegationGuide`` — 지도만 주고 방을 잠가 둠.
* 09-19 ``BrowserGuide``(커넥터) — 안내만 100회 되풀이, 입력 약 750만 토큰.
* 09-23 ``LocalControl``(커넥터, SDK 경로) — 지도의 Shell·WriteFile 을 한 번도 못 부르고
  브라우저 도구에 ``rm -rf`` 를 우겨넣음. 부르지도 않은 쓰기로 "PC 에 저장했다" 고 답함.

이 모듈이 그 관계의 **유일한 출처**다. 여기 적힌 문이 불리면 그 가족은 문의 구현과 무관하게
열린다(:func:`family_of` — Stage 10 라우터가 부른다). 그리고 매 표면마다 "숨긴 가족에는 보이는
문이 있다" 를 검사한다(:func:`reachability_fixes` — Stage 3 가 부른다). 규칙을 **기억**하지
않고 **검사**한다.

가족은 이름 규칙이다
--------------------
목록을 두 저장소에 나눠 적지 않으려고, 가족은 대부분 **이름 규칙**으로 적는다
(``Browser*``, ``Doc*``, ``Job*`` …). workflow 가 가진 가족(작업·아티팩트)도 이름 규칙이라
여기서 따로 목록을 들고 있을 필요가 없다. 그리고 문과 방은 **같은 MCP 접두**를 가져야 같은
가족이다 — 커넥터의 ``mcp_local_BrowserGuide`` 는 ``mcp_local_Browser*`` 를 열고, 내장
``BrowserGuide`` 는 접두 없는 ``Browser*`` 를 연다. 남의 MCP 서버 도구가 우리 규칙에
우연히 맞아도 접두가 달라 섞이지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Set, Tuple

__all__ = ["GATES", "Gate", "family_of", "gate_of", "reachability_fixes", "split_prefix"]

#: MCP 를 지나며 붙는 접두 — ``mcp_local_Shell``, ``mcp__connector__Foo``.
#: (host.tool_exposure 와 같은 정규식 — 표면 판정과 가족 판정이 같은 이름 해석을 쓴다.)
_MCP_PREFIX = re.compile(r"^mcp_{1,2}[A-Za-z0-9-]+_{1,2}")


def split_prefix(name: str) -> Tuple[str, str]:
    """``mcp_local_Shell`` → ``("mcp_local_", "Shell")``. 접두가 없으면 ``("", name)``."""
    text = str(name or "")
    m = _MCP_PREFIX.match(text)
    return (m.group(0), text[m.end() :]) if m else ("", text)


def _named(*names: str) -> Callable[[str], bool]:
    wanted = frozenset(names)
    return lambda base: base in wanted


def _starts(stem: str, gate: str) -> Callable[[str], bool]:
    return lambda base: base.startswith(stem) and base != gate


def _lazy_family(module: str, attr: str) -> Callable[[str], bool]:
    """내장 모듈이 들고 있는 목록을 그대로 쓴다(두 벌로 적지 않는다). 처음 쓸 때 읽는다."""
    cache: Dict[str, frozenset] = {}

    def member(base: str) -> bool:
        if "v" not in cache:
            import importlib

            try:
                cache["v"] = frozenset(getattr(importlib.import_module(module), attr))
            except Exception:  # noqa: BLE001 — 모듈이 없는 배포(선택 기능)면 빈 가족
                cache["v"] = frozenset()
        return base in cache["v"]

    return member


@dataclass(frozen=True)
class Gate:
    #: 접두를 뗀 문 이름.
    name: str
    #: 접두를 뗀 이름이 이 문의 가족인가.
    member: Callable[[str], bool]
    #: 이 문이 턴-1 에 **보여야 하는가**. False 는 "그 가족은 의도적으로 ToolSearch 로 찾는
    #: 긴 꼬리" 라는 선언이다 — 불변식이 그 문을 억지로 세우지 않는다.
    visible: bool = True
    #: 접두 **없는** 문(내장)만 가족을 여는가. 로컬 컨트롤처럼 "같은 접두의 전부" 를 여는 문은
    #: 접두가 있을 때만 의미가 있다 — 접두 없는 LocalControl(CLI 경로)이 모든 내장 도구를
    #: 가족으로 삼는 사고를 막는다.
    prefixed_only: bool = False


#: 문 → 가족. **새 가족을 문 뒤에 숨기려면 여기 한 줄을 더한다** — 숨김(턴-1 계획)과 열림
#: (라우터)과 검사(Stage 3)가 전부 이 표를 읽는다.
GATES: Dict[str, Gate] = {
    g.name: g
    for g in (
        Gate("BrowserGuide", _starts("Browser", "BrowserGuide")),
        Gate("JobGuide", _starts("Job", "JobGuide")),
        Gate("ArtifactGuide", _starts("Artifact", "ArtifactGuide")),
        Gate("SshListServers", _named("SshRun", "SshUpload", "SshDownload")),
        Gate(
            "DelegationGuide",
            _lazy_family(
                "xgen_agent_runtime.tools.built_in.delegation_guide_tool", "DELEGATION_FAMILY"
            ),
        ),
        Gate(
            "SelfExtendGuide",
            _lazy_family(
                "xgen_agent_runtime.tools.built_in.self_extend_guide_tool", "SELF_EXTEND_FAMILY"
            ),
        ),
        # 사용자 PC(커넥터) — 같은 접두의 **나머지 전부**. 셸·파일·앱·오피스, 이 PC 의 브라우저,
        # 사용자가 커넥터에 붙인 MCP 서버까지. 다른 문(BrowserGuide 등)은 가족에서 뺀다.
        Gate("LocalControl", lambda base: True, prefixed_only=True),
        # 문서 — 현재 턴-1 에 없다(문도 ToolSearch 로 찾는다). 호출당 약 211 토큰이라 세울지는
        # 따로 판단한다. 여기서는 **지금의 동작을 선언**만 한다: 불변식이 억지로 세우지 않는다.
        Gate("DocGuide", _starts("Doc", "DocGuide"), visible=False),
    )
}


def gate_of(name: str) -> Gate | None:
    """이 이름이 문이면 그 :class:`Gate`."""
    prefix, base = split_prefix(name)
    gate = GATES.get(base)
    if gate is None or (gate.prefixed_only and not prefix):
        return None
    return gate


def _owner(name: str, present: Set[str]) -> Tuple[str, Gate] | None:
    """이 도구가 속한 가족의 (문의 전체 이름, 문). 문 자신이나 어느 가족도 아니면 None.

    **접두가 있는 도구는 같은 접두의 문이 실제로 있을 때만** 그 가족이다. 사용자가 붙인
    MCP 서버(``mcp_github_…``)는 우리 문을 갖고 있지 않다 — 이름이 우연히 규칙에 맞아도
    (예: 그 서버의 ``BrowserNavigate``) 그건 그 서버의 카탈로그이고 ToolSearch 뒤의 긴
    꼬리다. 접두 없는 도구(우리 내장)만 "문이 없으면 방을 연다" 의 대상이다(09-09 SSH).

    한 도구가 두 문에 걸리면(커넥터 ``mcp_local_BrowserNavigate`` 는 BrowserGuide 와
    LocalControl 둘 다) **좁은 쪽**(prefixed_only 가 아닌 문)을 주인으로 본다.
    """
    prefix, base = split_prefix(name)
    if base in GATES:
        return None
    best: Tuple[str, Gate] | None = None
    for gate in GATES.values():
        if gate.prefixed_only and not prefix:
            continue
        if prefix and (prefix + gate.name) not in present:
            continue
        if gate.member(base):
            cand = (prefix + gate.name, gate)
            if best is None or (best[1].prefixed_only and not gate.prefixed_only):
                best = cand
    return best


def family_of(gate_name: str, registered: Iterable[str]) -> List[str]:
    """``gate_name`` 을 부르면 열려야 할 도구들 — 등록된 것 중 **같은 접두**의 가족."""
    gate = gate_of(gate_name)
    if gate is None:
        return []
    prefix, _ = split_prefix(gate_name)
    out: List[str] = []
    for name in registered:
        p, base = split_prefix(name)
        if p != prefix or base in GATES:
            continue
        if gate.member(base):
            out.append(name)
    return out


def reachability_fixes(registered: Iterable[str], is_exposed: Callable[[str], bool]) -> List[str]:
    """ "숨긴 가족에는 보이는 문이 있다" 를 지키려면 **무엇을 열어야 하는가**.

    숨겨진 도구마다 주인 문을 찾는다:

    * 문이 등록돼 있고 ``visible`` 인데 **숨어 있으면** → 문을 연다(스키마 하나로 복구).
    * 문이 **아예 등록되지 않았으면** → 그 도구 자신을 연다. 문이 없는데 숨기면 모델이 그
      도구에 닿을 길은 ToolSearch 뿐이고, 약한 모델은 거기까지 가지 않는다(09-09 SSH).
    * 문이 ``visible=False`` 면 손대지 않는다 — 의도된 긴 꼬리다.

    어느 가족도 아닌 숨긴 도구(연결된 DB·API 노드·사용자 MCP 의 수백 개)는 건드리지 않는다.
    그것들이 ToolSearch 뒤에 있는 것은 설계다.

    반환값은 열어야 할 이름 목록(중복 없음). 부르는 쪽이 열고 이벤트를 남긴다.
    """
    names = list(registered)
    present: Set[str] = set(names)
    fixes: List[str] = []
    seen: Set[str] = set()
    for name in names:
        if is_exposed(name):
            continue
        owner = _owner(name, present)
        if owner is None:
            continue
        gate_full, gate = owner
        if not gate.visible:
            continue
        target = gate_full if gate_full in present else name
        if target in present and is_exposed(target):
            continue
        if target not in seen:
            seen.add(target)
            fixes.append(target)
    return fixes
