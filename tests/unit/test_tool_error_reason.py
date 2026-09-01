"""실패한 도구는 **왜** 실패했는지 함께 알린다.

`tool.call_complete` 는 이름·성패·소요시간만 실었다. 호스트는 그 사건에 내용이
없어 ``result_sink`` 를 봤는데, 그건 그래프 포트 도구만 채운다(adapt_tools 의
래퍼) — 내장·작업·제작 도구의 실패는 늘 빈 문자열이었고 "tool execution failed"
라는 고정 문구로 뭉개졌다.

프로드에서 JobSchedule 이 여덟 번 실패했고, [전체로그]에는 같은 줄만 여덟 번
쌓였다. 무엇을 고쳐야 하는지 아무 데도 없었다.
"""
from xgen_agent_runtime.stages.s10_tool.artifact.default.executors import (
    _emit_call_complete,
    _error_text,
)


def _emit(result):
    seen = []
    _emit_call_complete(lambda t, d: seen.append((t, d)), {"tool_name": "JobSchedule"}, result, 12)
    return seen[0][1]


def test_실패는_사유를_달고_나온다():
    data = _emit({"is_error": True, "content": "Exactly one of tool / script is required."})
    assert data["is_error"] is True
    assert data["error"] == "Exactly one of tool / script is required."


def test_성공은_결과를_싣지_않는다():
    """성공 결과는 크고, 모델은 이미 받았다 — 사건에 또 실을 이유가 없다."""
    data = _emit({"is_error": False, "content": "x" * 10_000})
    assert "error" not in data


def test_사유가_없으면_빈_문자열이지_거짓말이_아니다():
    assert _emit({"is_error": True})["error"] == ""


def test_아주_긴_사유는_자른다():
    data = _emit({"is_error": True, "content": "y" * 50_000})
    assert 0 < len(data["error"]) <= 2000


def test_내용이_문자열이_아니어도_읽는다():
    assert _error_text({"content": {"reason": "nope"}}) != ""
    assert _error_text({"display_text": "보조 문구"}) == "보조 문구"


def test_호스트가_사건의_사유를_쓴다():
    """result_sink 가 비어도 UI/트레이스에 이유가 남아야 한다."""
    from xgen_agent_runtime.host.runner import _tool_end_event

    evt = _tool_end_event("JobSchedule", "cron_expr must have 5 fields", is_error=True)
    assert evt["type"] == "tool_error"
    assert evt["error"] == "cron_expr must have 5 fields"
    # 사유를 정말 못 얻은 경우에만 고정 문구.
    assert _tool_end_event("X", "", is_error=True)["error"] == "tool execution failed"
