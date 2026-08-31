"""도구 표면은 계층적이다.

기본 도구는 언제나 보이고, 연결된 API/DB/MCP 노드는 이름과 한 줄로만 알려 둔 뒤
필요할 때 ToolSearch 로 스키마를 끌어온다. 도구 목록은 재고 목록이 아니라 지도다 —
이번 턴에 부르지도 않을 수백 개의 스키마에 컨텍스트를 쓰면 모델은 더 많이 읽고 더
못 고른다.

예전에는 반대가 기본이었다(`all` = 전부 선노출). 그래서 저장된 값들이 무엇으로
읽히는지가 이 파일의 핵심이다.
"""

from xgen_agent_runtime.host.tool_exposure import (
    FLAT,
    HIERARCHY,
    normalize_exposure,
    sends_every_schema,
)


def test_기본은_계층형이다():
    # 아무것도 고르지 않은 에이전트는 계층형으로 동작한다.
    assert normalize_exposure(None) == HIERARCHY
    assert normalize_exposure("") == HIERARCHY
    assert normalize_exposure("   ") == HIERARCHY


def test_예전_값들은_계층형으로_읽힌다():
    # `all`(전부 선노출)과 `search`(전부 유예)는 계층 이전의 두 극단이다.
    # 둘 다 계층형으로 수렴한다 — 계층이 플랫폼의 동작이고, 플랫한 표면은
    # 명시적으로 고른 에이전트만 갖는다.
    assert normalize_exposure("all") == HIERARCHY
    assert normalize_exposure("search") == HIERARCHY
    assert normalize_exposure("hierarchy") == HIERARCHY


def test_플랫은_명시적으로_고른다():
    assert normalize_exposure("flat") == FLAT
    assert normalize_exposure("FLAT") == FLAT, "대소문자로 갈리면 안 된다"
    assert normalize_exposure(" flat ") == FLAT


def test_모르는_값은_계층형으로_떨어진다():
    # 노출 설정은 취향이다. 오타 하나로 턴이 죽으면 안 된다.
    assert normalize_exposure("nonsense") == HIERARCHY
    assert normalize_exposure(123) == HIERARCHY
    assert normalize_exposure(["flat"]) == HIERARCHY


def test_연결된_도구를_선노출할지_한_줄로_답한다():
    # 이 판정이 곧 registry 등록의 core 여부다.
    assert sends_every_schema("flat") is True
    assert sends_every_schema("hierarchy") is False
    assert sends_every_schema("all") is False, "예전 기본값도 이제 계층형이다"
    assert sends_every_schema(None) is False
