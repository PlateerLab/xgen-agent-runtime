"""호스트 경로(build_pipeline → run_turn)에서 단기 기억 창이 실제로 돈다.

재현: 2026-10-20 dev(claude-sonnet-5-5, DB Memory 노드 없는 Agent Geny). 앞 턴 답변이 "1. … 2. … 3. …
1~3번은 바로 진행해도 되나요?" 로 끝났고, 사용자가 "1~3번 전부 하자" 고 했다. 다음 턴의 첫 생각은
"my previous truncated answer" 였다. 저장(STM)은 온전했지만, build_pipeline 이 Stage 2 에 창의 출처를
주지 않아 창(4.36.0)은 한 번도 돌지 않았고, 지난 턴은 발화마다 1,200자 머리만 남은 시스템 프롬프트
불릿(L0 recent_turns)으로만 실렸다. 창의 단위 시험은 ContextStage(provider=…) 를 직접 만들어 이
빈틈을 보지 못했다. 이 시험은 호스트처럼 턴마다 파이프라인을 새로 만든다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from xgen_agent_runtime.core.state import PipelineState, TokenUsage
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock
from xgen_agent_runtime.memory.providers.ephemeral import EphemeralMemoryProvider

#: 1,200자 L0 상한보다 길고, 목록이 꼬리에 있는 답 — 잘리면 바로 이 목록이 사라진다.
LONG_ANSWER = (
    "점검 결과입니다. " + "게시본의 정책 파일과 업로드 경로를 하나씩 확인했습니다. " * 60
    + "\n\n## 제안\n\n1. 정책 파일의 도구 지시를 정리합니다.\n2. 업로드 잔여물을 삭제합니다.\n"
    "3. 짧은 영상 1건으로 앱을 끝까지 돌립니다.\n\n1~3번은 바로 진행해도 되나요?"
)


class _Recorder(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, answer: str, **kw: Any) -> None:
        super().__init__(**kw)
        self._answer = answer
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        return APIResponse(
            content=[ContentBlock(type="text", text=self._answer)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            model="fake",
        )


def _turn(provider: EphemeralMemoryProvider, text: str, answer: str) -> _Recorder:
    """호스트가 하듯 턴마다 새 파이프라인과 새 상태로 한 턴을 돈다."""
    client = _Recorder(answer, api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="anthropic", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, memory_provider=provider, turn_input_budget_tokens=None,
    )
    state = PipelineState(session_id="sess-1", model="m")
    runner.run_turn(pipe, text, state)
    return client


def _system_text(request: Any) -> str:
    system = request.system
    if isinstance(system, list):
        return "\n".join(str(b.get("text", "")) for b in system if isinstance(b, dict))
    return str(system or "")


def _texts(messages: List[Dict[str, Any]], role: str) -> List[str]:
    out: List[str] = []
    for m in messages:
        if m.get("role") != role:
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out += [str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return out


def test_the_previous_answer_comes_back_whole_as_a_message():
    provider = EphemeralMemoryProvider()
    _turn(provider, "게시된 앱을 점검해 줘", LONG_ANSWER)
    second = _turn(provider, "1~3번 전부 진행하자", "진행합니다.")

    request = second.requests[0]
    # 지난 턴은 대화 자리(messages)에 있다. 목록까지 온전하다.
    assert LONG_ANSWER in _texts(request.messages, "assistant")
    assert "게시된 앱을 점검해 줘" in _texts(request.messages, "user")
    # 이번 지시는 맨 끝 사용자 메시지다(턴 맥락 블록이 같은 메시지에 붙는다).
    assert "1~3번 전부 진행하자" in _texts([request.messages[-1]], "user")
    # 같은 턴을 시스템 프롬프트의 잘린 지식 불릿으로 한 번 더 싣지 않는다.
    assert "recent_turns" not in _system_text(request)


def test_the_window_is_not_recorded_again():
    """창은 STM 에서 왔다. 다음 턴 끝에 STM 에 다시 적히면 턴마다 기록이 불어난다."""
    provider = EphemeralMemoryProvider()
    _turn(provider, "첫 질문", "첫 답")
    _turn(provider, "둘째 질문", "둘째 답")

    import asyncio

    rows = asyncio.run(provider.stm().recent(n=50))
    texts = [r.content if isinstance(r.content, str) else json.dumps(r.content, ensure_ascii=False) for r in rows]
    assert sum("첫 질문" in t for t in texts) == 1
    assert sum("첫 답" in t for t in texts) == 1
    assert sum("둘째 질문" in t for t in texts) == 1


def test_the_window_source_does_not_inject_provider_layers():
    """창의 출처는 provider 자리가 아니다. provider 자리면 provider.retrieve 가 STM 원문을 예산 밖에서 싣는다."""
    provider = EphemeralMemoryProvider()
    client = _Recorder("답", api_key="k")
    pipe = runner.build_pipeline(
        name="t", provider="anthropic", model="m", api_key="k", llm_client=client, stream=False,
        enable_compaction=False, memory_provider=provider, turn_input_budget_tokens=None,
    )
    stage = pipe.get_stage(2)
    assert stage.short_term_source is provider
    assert stage.provider is None
