"""
Tool Indicator Registry — Self-Adaptive UI Phase 0a (PR 1)

LangChain tool 실행 시점에 UI 인디케이터(라벨·아이콘·예상 소요)를 SSE 로
흘려보낼 수 있도록, tool 이름별 메타데이터를 한 곳에 모아둔다.

설계:
- 인디케이터는 UI 표시를 위한 *부가* 메타데이터일 뿐, 기존 tool_call/tool_result
  이벤트 키와 절대 충돌하지 않는다 (기존 컨슈머 100% 호환).
- 등록되지 않은 tool 도 DEFAULT_INDICATOR 로 fallback 되므로 항상 안전.
- 향후 도구 기반 (BaseTool) 에 `_xgen_indicator` 속성을 두는 방식도 가능
  하지만 1차에는 중앙 레지스트리만 둔다 (모든 도구를 한눈에 보기 좋음).

본 모듈은 외부 의존성 0. 순수 Python dict + helper 함수.

참고: SELF_ADAPTIVE_UI_PLAN.md §5.2.1
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# 기본 인디케이터 (등록되지 않은 도구의 fallback)
# ---------------------------------------------------------------------------

DEFAULT_INDICATOR: Dict[str, Any] = {
    "display_label": None,        # None → 호출자가 tool_name 자체를 라벨로 사용
    "verb_running": "실행 중...",
    "verb_done": "완료",
    "icon": "tool",
    "category": "generic",
    "expected_duration_ms": 1500,
    "render_hint": "inline",       # 'inline' | 'surface' | 'hide'
}


# ---------------------------------------------------------------------------
# 도구별 인디케이터 — Phase 0a 등록 항목
#
# 키: tool name (LangChain Tool.name 그대로)
# 값: indicator dict (DEFAULT_INDICATOR 키와 동일)
#
# Phase 1 이후 차트·표·A2UI 도구가 추가되면 본 dict 에 항목 추가.
# ---------------------------------------------------------------------------

TOOL_INDICATORS: Dict[str, Dict[str, Any]] = {
    # ── Retrieval / RAG ────────────────────────────────────────────────────
    "WorkflowSelf": {
        "display_label": "자기 진화",
        "verb_running": "자기 워크플로를 수정하는 중...",
        "verb_done": "자기 진화 적용",
        "icon": "tool",
        "category": "self",
        "render_hint": "inline",
    },
    "vectordb_retrieval_tool": {
        "display_label": "지식 검색",
        "verb_running": "지식을 찾는 중...",
        "verb_done": "검색 완료",
        "icon": "search",
        "category": "retrieval",
        "expected_duration_ms": 2500,
        "render_hint": "inline",
    },

    # ── Database / Structured Data ─────────────────────────────────────────
    "list_tables": {
        "display_label": "DB 테이블 목록",
        "verb_running": "테이블 조회 중...",
        "verb_done": "테이블 목록 가져옴",
        "icon": "database",
        "category": "data",
        "expected_duration_ms": 600,
        "render_hint": "inline",
    },
    "get_schema": {
        "display_label": "테이블 스키마",
        "verb_running": "스키마 조회 중...",
        "verb_done": "스키마 가져옴",
        "icon": "database",
        "category": "data",
        "expected_duration_ms": 500,
        "render_hint": "inline",
    },
    "query": {
        "display_label": "DB 쿼리",
        "verb_running": "데이터 조회 중...",
        "verb_done": "데이터 가져옴",
        "icon": "table",
        "category": "data",
        "expected_duration_ms": 1200,
        "render_hint": "inline",
    },

    # ── A2UI / Self-Adaptive UI 도구 (Phase 1 이후 활성화) ─────────────────
    "a2ui_list_components": {
        "display_label": "UI 부품 탐색",
        "verb_running": "부품 목록 조회 중...",
        "verb_done": "부품 목록",
        "icon": "layout",
        "category": "genui",
        "expected_duration_ms": 200,
        "render_hint": "hide",   # 빠르고 노이즈 — UI 표시 생략
    },
    "a2ui_inspect_component": {
        "display_label": "UI 부품 상세",
        "verb_running": "부품 명세 확인 중...",
        "verb_done": "확인",
        "icon": "layout",
        "category": "genui",
        "expected_duration_ms": 200,
        "render_hint": "hide",
    },
    "a2ui_validate_messages": {
        "display_label": "UI 메시지 검증",
        "verb_running": "검증 중...",
        "verb_done": "검증 완료",
        "icon": "check",
        "category": "genui",
        "expected_duration_ms": 300,
        "render_hint": "hide",
    },
    "a2ui_finalize_surface": {
        "display_label": "화면 합성",
        "verb_running": "화면을 만드는 중...",
        "verb_done": "화면 준비됨",
        "icon": "layout-grid",
        "category": "genui",
        "expected_duration_ms": 800,
        "render_hint": "hide",   # surface 자체가 결과 → chip 으로 또 표시하면 중복
    },
}


# ---------------------------------------------------------------------------
# 조회 헬퍼
# ---------------------------------------------------------------------------

def get_indicator(tool_name: str) -> Dict[str, Any]:
    """
    tool_name 에 해당하는 인디케이터 메타데이터를 반환한다.

    등록되지 않은 도구도 DEFAULT_INDICATOR 를 복사해 반환하여
    호출자는 None 체크 없이 안전하게 사용 가능.

    Args:
        tool_name: LangChain Tool.name (예: "vectordb_retrieval_tool")

    Returns:
        indicator dict. 반드시 다음 키 포함:
            display_label, verb_running, verb_done, icon,
            category, expected_duration_ms, render_hint, tool_name

    Example:
        >>> ind = get_indicator("vectordb_retrieval_tool")
        >>> ind["display_label"]
        '지식 검색'
        >>> ind = get_indicator("some_unknown_tool")
        >>> ind["category"]
        'generic'
    """
    base = TOOL_INDICATORS.get(tool_name, DEFAULT_INDICATOR)
    # DEFAULT_INDICATOR 를 복사할 때 mutation 방지 + tool_name 첨부
    return {**base, "tool_name": tool_name}


def is_hidden(tool_name: str) -> bool:
    """
    인디케이터를 UI 에 노출하지 않을 도구인지 (render_hint == 'hide').

    A2UI 내부 탐색 도구처럼 사용자가 알 필요 없는 노이즈 도구를
    프론트에서 chip 으로 표시하지 않기 위한 편의 함수.
    """
    return get_indicator(tool_name).get("render_hint") == "hide"
