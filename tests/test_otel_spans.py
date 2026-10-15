"""런타임이 LLM 호출·도구 실행마다 OTel span 을 내고, 그 span 이 호스트 턴 아래에 달린다.

실측(2026-10-15, dev Tempo): Geny 턴 trace 에 ``chat`` span 이 하나도 없었고, LLM 의
``POST /v1/chat/completions`` 와 도구가 부른 HTTP·MCP 호출이 에이전트 턴 밖(워크플로우 바로
아래)에 흩어져 어느 호출이 병목인지 읽을 수 없었다. 호스트는 사건만 받아서는 호출 구간도
부모 관계도 되살릴 수 없다 — 그래서 일이 일어나는 런타임이 span 을 연다.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace as otel_trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from xgen_agent_runtime import observability  # noqa: E402
from xgen_agent_runtime.observability import spans as obs_spans  # noqa: E402
from xgen_agent_runtime.core.state import PipelineState  # noqa: E402
from xgen_agent_runtime.host import runner  # noqa: E402
from xgen_agent_runtime.llm_client.base import BaseClient, ClientCapabilities  # noqa: E402
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock, TokenUsage  # noqa: E402
from xgen_agent_runtime.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult  # noqa: E402
from xgen_agent_runtime.tools.registry import ToolRegistry  # noqa: E402

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
_tracer = _provider.get_tracer("test")


@pytest.fixture(autouse=True)
def _spans(monkeypatch):
    """전역 provider 는 건드리지 않는다 — OTel 은 첫 설정만 받는다."""
    monkeypatch.setattr(obs_spans, "get_tracer", lambda: _tracer)
    monkeypatch.delenv(observability.ENV_SWITCH, raising=False)
    _exporter.clear()
    yield
    _exporter.clear()


class _Fetch(Tool):
    """도구 안에서 바깥 호출을 하는 도구 — 그 호출이 도구 span 아래에 달려야 한다."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    @property
    def name(self) -> str:
        return "fetch"

    @property
    def description(self) -> str:
        return "fetch"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object"}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        with _tracer.start_as_current_span("POST"):  # HTTP 자동 계측 span 자리
            pass
        if self.fail:
            return ToolResult(content="boom", is_error=True)
        return ToolResult(content="12345")


class _Client(BaseClient):
    """1번째 응답: 도구 호출. 2번째: 최종 답. 호출마다 HTTP span 을 하나 낸다."""

    provider = "fake"
    capabilities = ClientCapabilities()

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.requests: List[Any] = []

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        self.requests.append(request)
        with _tracer.start_as_current_span("POST /v1/chat/completions"):
            pass
        if len(self.requests) == 1:
            return APIResponse(
                content=[ContentBlock(type="tool_use", tool_use_id="t1", tool_name="fetch", tool_input={})],
                stop_reason="tool_use",
                usage=TokenUsage(input_tokens=100, output_tokens=7, cache_read_input_tokens=40),
                model="m-snapshot",
            )
        return APIResponse(
            content=[ContentBlock(type="text", text="done")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=130, output_tokens=3),
            model="m-snapshot",
        )


def _run_under_host_turn(tool: _Fetch) -> str:
    reg = ToolRegistry()
    reg.register(tool, core=True)
    pipe = runner.build_pipeline(
        name="t", provider="openai", model="m", api_key="k", llm_client=_Client(api_key="k"),
        stream=False, enable_compaction=False, registry=reg,
    )
    # 호스트가 연 턴 span 을 현재 span 으로 두고 턴을 돈다 — 런타임은 전용 이벤트 루프에서 돌지만
    # 루프가 호출 스레드의 context 를 이어받아야 한다.
    with _tracer.start_as_current_span("invoke_agent host"):
        return runner.run_turn(pipe, "가져와", PipelineState(session_id="s", model="m"))


def _by_name(name: str) -> List[Any]:
    return [s for s in _exporter.get_finished_spans() if s.name == name]


def _parent_name(span: Any) -> str:
    parent_id = span.parent.span_id if span.parent is not None else None
    for s in _exporter.get_finished_spans():
        if s.context.span_id == parent_id:
            return s.name
    return ""


def test_each_llm_call_is_a_chat_span_under_the_host_turn_with_its_own_usage() -> None:
    assert _run_under_host_turn(_Fetch()) == "done"

    chats = _by_name("chat m")
    assert len(chats) == 2
    assert all(_parent_name(s) == "invoke_agent host" for s in chats)
    first, second = sorted(chats, key=lambda s: s.start_time)
    assert first.attributes["gen_ai.operation.name"] == "chat"
    assert first.attributes["gen_ai.provider.name"] == "fake"
    assert first.attributes["gen_ai.response.model"] == "m-snapshot"
    assert first.attributes["gen_ai.usage.input_tokens"] == 100
    assert first.attributes["gen_ai.usage.output_tokens"] == 7
    assert first.attributes["gen_ai.usage.cache_read.input_tokens"] == 40
    assert tuple(first.attributes["gen_ai.response.finish_reasons"]) == ("tool_use",)
    assert first.attributes["xgen.llm.tool_calls"] == 1
    assert second.attributes["gen_ai.usage.input_tokens"] == 130
    assert "gen_ai.usage.cache_read.input_tokens" not in second.attributes

    # 벤더 SDK 의 HTTP span 은 그 호출 span 아래에 달린다.
    http = _by_name("POST /v1/chat/completions")
    assert len(http) == 2 and all(_parent_name(s) == "chat m" for s in http)


def test_tool_run_is_an_execute_tool_span_and_its_calls_nest_under_it() -> None:
    _run_under_host_turn(_Fetch())

    (tool,) = _by_name("execute_tool fetch")
    assert _parent_name(tool) == "invoke_agent host"
    assert tool.attributes["gen_ai.tool.name"] == "fetch"
    assert tool.attributes["xgen.tool.is_error"] is False
    assert tool.attributes["xgen.tool.result_size"] == 5
    assert tool.status.status_code != StatusCode.ERROR
    (inner,) = _by_name("POST")
    assert _parent_name(inner) == "execute_tool fetch"


def test_failed_tool_is_an_error_span_without_its_text() -> None:
    _run_under_host_turn(_Fetch(fail=True))

    (tool,) = _by_name("execute_tool fetch")
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.attributes["xgen.tool.is_error"] is True
    assert tool.attributes["error.type"] == "tool_error"
    assert "boom" not in str(dict(tool.attributes))


def test_switch_off_emits_nothing(monkeypatch) -> None:
    monkeypatch.setenv(observability.ENV_SWITCH, "0")
    assert observability.spans_enabled() is False
    _run_under_host_turn(_Fetch())
    assert not _by_name("chat m") and not _by_name("execute_tool fetch")
    # 런타임 span 이 빠져도 HTTP span 은 그대로 호스트 턴 아래에 남는다.
    assert all(_parent_name(s) == "invoke_agent host" for s in _by_name("POST /v1/chat/completions"))


def test_broken_tracer_never_breaks_the_turn(monkeypatch) -> None:
    def _boom() -> Any:
        raise RuntimeError("tracer down")

    monkeypatch.setattr(obs_spans, "get_tracer", _boom)
    assert _run_under_host_turn(_Fetch()) == "done"


def test_spans_enabled_reflects_the_api_install() -> None:
    assert observability.spans_enabled() is (otel_trace is not None)
