"""에이전트 턴은 오늘 날짜(요일 포함)를 안다 — SDK 경로(host/runner.build_pipeline).

2026-09-26 dev 실사용 점검: "다음주 화요일 회의를 목요일로 옮긴다는 메일" 에 모델이 날짜를 비워 두고
"(예: 6월 12일 목요일)" 을 예로 들었다. DateTimeBlock 은 있었지만 SDK 경로의 시스템 프롬프트 조립에
빠져 있었다(메모리 없는 경로 = 정적 프롬프트, 메모리 경로 = 기억 블록만).
"""
from __future__ import annotations

import re

from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock

_DATE = re.compile(r"Current date: \d{4}-\d{2}-\d{2} \((Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) \d{2}:\d{2}")


class _Client(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, **kw):
        super().__init__(**kw)
        self.requests = []

    async def _send(self, request, *, purpose=""):
        self.requests.append(request)
        return APIResponse(content=[ContentBlock(type="text", text="ok")], stop_reason="end_turn",
                           usage=TokenUsage(input_tokens=10, output_tokens=2), model="fake")


def _first_request_text(pipe, client) -> str:
    runner.run_turn(pipe, "다음주 화요일 회의 안내 메일 써줘", PipelineState(session_id="t", model="m"))
    req = client.requests[0]
    return f"{getattr(req, 'system', '')}\n{req.messages}"


def test_memoryless_turn_request_carries_todays_date_with_weekday():
    client = _Client(api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, max_iterations=2, turn_input_budget_tokens=None, system_prompt="You are helpful.",
    )
    text = _first_request_text(pipe, client)
    assert _DATE.search(text), text[:600]
    assert "You are helpful." in text


def test_date_rides_outside_the_cached_system_prefix():
    """날짜는 분 단위로 바뀐다 — 시스템 프롬프트(캐시 접두)가 아니라 턴 맥락에 실린다."""
    client = _Client(api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, max_iterations=2, turn_input_budget_tokens=None, system_prompt="You are helpful.",
    )
    runner.run_turn(pipe, "hi", PipelineState(session_id="t", model="m"))
    system = str(getattr(client.requests[0], "system", ""))
    assert "You are helpful." in system
    assert not _DATE.search(system), system
