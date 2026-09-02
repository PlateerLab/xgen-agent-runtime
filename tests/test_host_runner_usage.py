"""host.runner 계약 — usage 청크 / record_failed_starts 게이트 / build_cli_client prewarm.

* ``stream_turn`` 은 파이프라인 종료 후 ``{"type": "usage", "data": {...}}`` 를 정확히
  한 번(마지막) yield 한다 — 성공·오류 무관, 취소 시 없음, 사용량 0 이면 없음.
* ``run_turn`` 은 ``usage_sink`` 로 같은 shape 를 노출한다.
* ``host.record_failed_starts=False`` 면 "출력 0 + 실패/취소" 턴의 실행 기록을 건너뛴다.
* ``build_cli_client(prewarm_spawn=...)`` 은 None 이면 클라이언트 기본값(미전달), 아니면 전달.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from xgen_agent_runtime import PipelineState
from xgen_agent_runtime.core.state import TokenUsage
from xgen_agent_runtime.events.types import PipelineEvent
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.llm_client import BaseClient, ClientCapabilities
from xgen_agent_runtime.llm_client.types import APIResponse, ContentBlock

_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "total_cost_usd",
    "model",
    "provider",
}


# ── 가짜 파이프라인(덕 타입) — stream_turn 이 쓰는 표면만 구현 ──────────


class _FakePipeline:
    """``run_stream`` 이 스크립트된 이벤트를 내고, 스크립트의 callable 은 state 조작 훅."""

    def __init__(self, script: List[Any], *, provider: str = "fake_provider") -> None:
        self._script = script
        self._provider = provider
        self.closed = False
        self._memory_provider: Any = None
        self._memory_distill_spec: Any = None

    async def run_stream(self, text: str, state: PipelineState) -> AsyncIterator[PipelineEvent]:
        state.begin_turn()
        for item in self._script:
            if callable(item):
                item(state)
                continue
            kind, data = item
            yield PipelineEvent(type=kind, data=dict(data))

    async def run(self, text: str, state: PipelineState) -> Any:
        state.begin_turn()
        final = ""
        error: Optional[str] = None
        for item in self._script:
            if callable(item):
                item(state)
                continue
            kind, data = item
            if kind == "text.delta":
                final += data.get("text", "")
            elif kind == "pipeline.error":
                error = str(data.get("error"))

        class _R:
            success = error is None
            text = final

        _R.error = error  # type: ignore[attr-defined]
        return _R()

    async def aclose(self) -> None:
        self.closed = True

    def _resolved_provider_name(self, state: PipelineState) -> str:
        return self._provider


def _track(input_tokens: int, output_tokens: int, *, cost: Optional[float] = None,
           cache_read: int = 0, cache_create: int = 0):
    """Stage 7 DefaultTracker 가 하는 일을 흉내 — per-call TokenUsage 를 state 에 쌓는다."""

    def _apply(state: PipelineState) -> None:
        u = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_create,
            cost_usd=cost,
        )
        state.token_usage += u
        state.turn_token_usage.append(u)

    return _apply


def _calc_cost(cost: float):
    def _apply(state: PipelineState) -> None:
        state.accumulate_cost(cost)

    return _apply


def _state(model: str = "m-1") -> PipelineState:
    return PipelineState(session_id="s1", model=model)


# ── stream_turn: usage 청크 ──────────────────────────────────────────


def test_stream_turn_yields_usage_exactly_once_at_end() -> None:
    pipe = _FakePipeline([
        ("text.delta", {"text": "안"}),
        _track(10, 2, cache_read=5, cache_create=1),
        ("tool.call_start", {"name": "Bash", "input": {"cmd": "ls"}}),
        ("tool.call_complete", {"name": "Bash", "is_error": False}),
        ("text.delta", {"text": "녕"}),
        _track(20, 3),
        ("pipeline.complete", {"result": "안녕"}),
    ])
    out = list(runner.stream_turn(pipe, "hi", _state("gpt-x")))
    usages = [c for c in out if isinstance(c, dict) and c.get("type") == "usage"]
    assert len(usages) == 1
    assert out[-1] is usages[0]  # 마지막 항목
    data = usages[0]["data"]
    assert set(data) == _USAGE_KEYS
    assert data["input_tokens"] == 30 and data["output_tokens"] == 5
    assert data["cache_read_tokens"] == 5 and data["cache_creation_tokens"] == 1
    assert data["total_cost_usd"] is None  # 비용 정보 없음 → null
    assert data["model"] == "gpt-x" and data["provider"] == "fake_provider"
    # 텍스트는 그대로 흐른다
    assert [c for c in out if isinstance(c, str)] == ["안", "녕"]
    assert pipe.closed


def test_stream_turn_usage_prefers_provider_reported_cost() -> None:
    # CLI(claude_code) 경로: result envelope 의 total_cost_usd → TokenUsage.cost_usd
    pipe = _FakePipeline([
        ("text.delta", {"text": "x"}),
        _track(100, 10, cost=0.0123),
        _calc_cost(0.5),  # Stage 7 계산기 추정치 — provider 보고값이 있으면 무시
        ("pipeline.complete", {"result": "x"}),
    ])
    out = list(runner.stream_turn(pipe, "hi", _state()))
    assert out[-1]["data"]["total_cost_usd"] == pytest.approx(0.0123)


def test_stream_turn_usage_falls_back_to_calculator_cost() -> None:
    pipe = _FakePipeline([
        ("text.delta", {"text": "x"}),
        _track(100, 10),
        _calc_cost(0.25),
        ("pipeline.complete", {"result": "x"}),
    ])
    out = list(runner.stream_turn(pipe, "hi", _state()))
    assert out[-1]["data"]["total_cost_usd"] == pytest.approx(0.25)


def test_stream_turn_usage_also_on_pipeline_error() -> None:
    pipe = _FakePipeline([
        ("text.delta", {"text": "partial"}),
        _track(7, 1),
        ("pipeline.error", {"error": "boom"}),
    ])
    out = list(runner.stream_turn(pipe, "hi", _state()))
    assert "partial" in out and any(isinstance(c, str) and "[ERROR] boom" in c for c in out)
    assert isinstance(out[-1], dict) and out[-1]["type"] == "usage"
    assert out[-1]["data"]["input_tokens"] == 7


def test_stream_turn_no_usage_when_nothing_tracked() -> None:
    pipe = _FakePipeline([("pipeline.error", {"error": "guard rejected"})])
    out = list(runner.stream_turn(pipe, "hi", _state()))
    assert not any(isinstance(c, dict) and c.get("type") == "usage" for c in out)


def test_stream_turn_no_usage_when_cancelled() -> None:
    flag = {"v": False}

    def _flip(state: PipelineState) -> None:
        flag["v"] = True

    pipe = _FakePipeline([
        ("text.delta", {"text": "a"}),
        _track(5, 5),
        _flip,
        ("text.delta", {"text": "b"}),
        ("pipeline.complete", {"result": "ab"}),
    ])
    out = list(runner.stream_turn(pipe, "hi", _state(), cancel_check=lambda: flag["v"]))
    # cancel_check 는 이벤트 경계마다 — flip 직후 나온 "b" 까지는 흐르고 거기서 멈춘다.
    assert out == ["a", "b"]  # 취소 → 이후 텍스트도 usage 도 없다


def test_turn_usage_provider_from_client_when_pipeline_lacks_resolver() -> None:
    class _Client:
        provider = "openai"

    state = _state("m")
    state.llm_client = _Client()
    state.turn_token_usage.append(TokenUsage(input_tokens=1, output_tokens=1))
    data = runner.turn_usage(object(), state)
    assert data is not None and data["provider"] == "openai"
    assert runner.turn_usage(object(), _state()) is None


# ── 실제 build_pipeline + BaseClient 로 끝-끝 (Stage 6→7 추적 경로) ─────


class _UsageClient(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities()

    async def _send(self, request: Any, *, purpose: str = "") -> APIResponse:
        return APIResponse(
            content=[ContentBlock(type="text", text="done")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=11, output_tokens=4, cache_read_input_tokens=2),
            model="fake-model-9",
        )


def test_stream_turn_real_pipeline_emits_usage_from_stage7() -> None:
    pipe = runner.build_pipeline(
        name="t",
        provider="openai",
        model="cfg-model",
        api_key="k",
        llm_client=_UsageClient(api_key="k"),
        stream=False,
        enable_compaction=False,
    )
    out = list(runner.stream_turn(pipe, "hi", _state("cfg-model")))
    assert out[-1]["type"] == "usage"
    data = out[-1]["data"]
    assert data["input_tokens"] == 11 and data["output_tokens"] == 4
    assert data["cache_read_tokens"] == 2 and data["cache_creation_tokens"] == 0
    assert data["model"] == "fake-model-9" and data["provider"] == "fake"
    assert "done" in "".join(c for c in out if isinstance(c, str))


def test_run_turn_fills_usage_sink() -> None:
    pipe = runner.build_pipeline(
        name="t",
        provider="openai",
        model="cfg-model",
        api_key="k",
        llm_client=_UsageClient(api_key="k"),
        stream=False,
        enable_compaction=False,
    )
    sink: Dict[str, Any] = {}
    text = runner.run_turn(pipe, "hi", _state("cfg-model"), usage_sink=sink)
    assert text == "done"
    assert set(sink) == _USAGE_KEYS and sink["input_tokens"] == 11 and sink["output_tokens"] == 4


# ── host-owned rollout recorder lifecycle ───────────────────────────


def _rollout_types(path: Path) -> List[str]:
    return [
        json.loads(line)["type"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_stream_turn_writes_and_shuts_down_durable_rollout(tmp_path: Path) -> None:
    path = tmp_path / "rollouts" / "stream.jsonl"
    pipe = runner.build_pipeline(
        name="t",
        provider="openai",
        model="cfg-model",
        api_key="k",
        llm_client=_UsageClient(api_key="k"),
        stream=False,
        enable_compaction=False,
    )

    out = list(runner.stream_turn(pipe, "hi", _state("cfg-model"), rollout_path=path))

    assert "done" in "".join(chunk for chunk in out if isinstance(chunk, str))
    types = _rollout_types(path)
    assert types[0] == "pipeline.start"
    assert types[-1] == "pipeline.complete"


def test_run_turn_preserves_existing_session_runtime_around_rollout(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    pipe = runner.build_pipeline(
        name="t",
        provider="openai",
        model="cfg-model",
        api_key="k",
        llm_client=_UsageClient(api_key="k"),
        stream=False,
        enable_compaction=False,
    )
    state = _state("cfg-model")
    existing_runtime = SimpleNamespace(marker="preserved")
    state.session_runtime = existing_runtime

    assert runner.run_turn(pipe, "hi", state, rollout_path=path) == "done"
    assert state.session_runtime is existing_runtime
    assert _rollout_types(path)[-1] == "pipeline.complete"


def test_rollout_storage_failure_surfaces_through_existing_error_result(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    pipe = runner.build_pipeline(
        name="t",
        provider="openai",
        model="cfg-model",
        api_key="k",
        llm_client=_UsageClient(api_key="k"),
        stream=False,
        enable_compaction=False,
    )

    state = _state()
    result = runner.run_turn(pipe, "hi", state, rollout_path=blocker / "run.jsonl")

    assert result.startswith("[ERROR]")
    assert state.session_runtime is None


def test_closing_stream_early_drains_rollout_prefix_and_restores_runtime(
    tmp_path: Path,
) -> None:
    class _BlockingPipeline:
        def __init__(self) -> None:
            self.runtime: Any = None
            self.closed = False

        def attach_runtime(self, *, session_runtime: Any) -> None:
            self.runtime = session_runtime

        async def run_stream(
            self, text: str, state: PipelineState
        ) -> AsyncIterator[PipelineEvent]:
            self.runtime.rollout_recorder.record_nowait(
                PipelineEvent(type="pipeline.start", session_id=state.session_id)
            )
            yield PipelineEvent(type="text.delta", data={"text": "partial"})
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            self.closed = True

    path = tmp_path / "cancelled.jsonl"
    pipeline = _BlockingPipeline()
    state = _state()
    original_runtime = SimpleNamespace(marker="original")
    state.session_runtime = original_runtime
    stream = runner.stream_turn(pipeline, "hi", state, rollout_path=path)

    assert next(stream) == "partial"
    stream.close()

    assert pipeline.closed
    assert state.session_runtime is original_runtime
    assert _rollout_types(path) == ["pipeline.start"]


# ── record_failed_starts 게이트 ──────────────────────────────────────


class _FakeMemoryProvider:
    async def close(self) -> None:
        return None


@pytest.fixture
def record_calls(monkeypatch) -> List[Dict[str, Any]]:
    import xgen_agent_runtime.host.execution_record as er

    calls: List[Dict[str, Any]] = []

    async def _fake_record(provider: Any, **kw: Any) -> None:
        calls.append(dict(kw))

    monkeypatch.setattr(er, "record_turn_execution", _fake_record)
    return calls


class _Host:
    def __init__(self, record_failed_starts: bool) -> None:
        self.record_failed_starts = record_failed_starts


def _with_memory(script: List[Any]) -> _FakePipeline:
    pipe = _FakePipeline(script)
    pipe._memory_provider = _FakeMemoryProvider()
    return pipe


def test_failed_start_not_recorded_when_host_opts_out(record_calls) -> None:
    pipe = _with_memory([("pipeline.error", {"error": "spawn failed"})])
    list(runner.stream_turn(pipe, "hi", _state(), host=_Host(False)))
    assert record_calls == []


def test_failed_start_recorded_by_default_and_for_opt_in_host(record_calls) -> None:
    pipe = _with_memory([("pipeline.error", {"error": "spawn failed"})])
    list(runner.stream_turn(pipe, "hi", _state()))  # host 미전달 → 기록(기존 동작)
    assert len(record_calls) == 1 and record_calls[0]["success"] is False
    pipe2 = _with_memory([("pipeline.error", {"error": "spawn failed"})])
    list(runner.stream_turn(pipe2, "hi", _state(), host=_Host(True)))
    assert len(record_calls) == 2


def test_failure_after_output_is_recorded_even_when_host_opts_out(record_calls) -> None:
    pipe = _with_memory([("text.delta", {"text": "some"}), ("pipeline.error", {"error": "late"})])
    list(runner.stream_turn(pipe, "hi", _state(), host=_Host(False)))
    assert len(record_calls) == 1
    assert record_calls[0]["output_text"] == "some" and record_calls[0]["error"] == "late"


def test_cancel_before_output_not_recorded_when_host_opts_out(record_calls) -> None:
    pipe = _with_memory([("tool.call_start", {"name": "Bash", "input": {}}), ("text.delta", {"text": "z"})])
    list(runner.stream_turn(pipe, "hi", _state(), host=_Host(False), cancel_check=lambda: True))
    assert record_calls == []


def test_success_without_text_is_still_recorded(record_calls) -> None:
    pipe = _with_memory([("pipeline.complete", {"result": ""})])
    list(runner.stream_turn(pipe, "hi", _state(), host=_Host(False)))
    assert len(record_calls) == 1 and record_calls[0]["success"] is True


def test_run_turn_failed_start_gate(record_calls) -> None:
    pipe = _with_memory([("pipeline.error", {"error": "x"})])
    assert runner.run_turn(pipe, "hi", _state(), host=_Host(False)).startswith("[ERROR]")
    assert record_calls == []
    pipe2 = _with_memory([("pipeline.error", {"error": "x"})])
    runner.run_turn(pipe2, "hi", _state())
    assert len(record_calls) == 1


# ── build_cli_client(prewarm_spawn) ─────────────────────────────────


@pytest.fixture
def captured_cli(monkeypatch) -> Dict[str, Any]:
    import xgen_agent_runtime.llm_client.claude_code as cc

    captured: Dict[str, Any] = {}

    class _Capture:
        def __init__(self, **kw: Any) -> None:
            captured.clear()
            captured.update(kw)

    monkeypatch.setattr(cc, "ClaudeCodeCLIClient", _Capture)
    return captured


def test_build_cli_client_prewarm_default_untouched(captured_cli) -> None:
    runner.build_cli_client(auth_mode="api_key", api_key="sk")
    assert "prewarm_spawn" not in captured_cli  # 클라이언트 기본(env) 유지
    assert captured_cli["bare_mode"] is True and captured_cli["api_key"] == "sk"


def test_build_cli_client_prewarm_passthrough(captured_cli) -> None:
    runner.build_cli_client(auth_mode="api_key", api_key="sk", prewarm_spawn=False)
    assert captured_cli["prewarm_spawn"] is False
    runner.build_cli_client(auth_mode="api_key", api_key="sk", prewarm_spawn=True)
    assert captured_cli["prewarm_spawn"] is True


def test_claude_code_client_accepts_prewarm_spawn_kwarg() -> None:
    import inspect

    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient

    assert "prewarm_spawn" in inspect.signature(ClaudeCodeCLIClient.__init__).parameters


# ── 네이티브 도구 전면 차단 계약 ────────────────────────────────────


def test_natives_fully_disallowed_when_local_tools_off(captured_cli) -> None:
    """``allow_local_tools=False`` → 카탈로그 **전체**가 차단된다.

    예전엔 fs/셸 9종만 막아 WebSearch·WebFetch·TodoWrite 세 네이티브가 살아남았다
    (실측). 규약은 "네이티브는 전부 끄고 우리 런타임 도구만 쓴다" 이므로 Bash 를
    포함한 카탈로그 전체가 나가야 한다 — 같은 능력은 MCP 표면이 준다.
    """
    runner.build_cli_client(auth_mode="api_key", api_key="sk", allow_local_tools=False)
    disallowed = set(captured_cli.get("disallow_tools", ()))
    assert set(runner.CLI_NATIVE_TOOL_CATALOG) <= disallowed
    assert "Bash" in disallowed and "WebSearch" in disallowed and "TodoWrite" in disallowed
    assert "CronCreate" in disallowed  # 세션 한정 스케줄 도구는 언제나 차단


def test_build_cli_client_adds_caller_disallows(captured_cli) -> None:
    """호출자 추가분(위임 시 CLI 자체 Task/Agent)은 차단 목록에 합쳐진다.

    ``allow_local_tools=True`` 는 sub-worker 팩토리에만 남은 예외다 — 거기엔 아직
    MCP 브릿지가 없어 네이티브가 유일한 도구 경로다.
    """
    runner.build_cli_client(
        auth_mode="api_key", api_key="sk", allow_local_tools=True,
        disallow_tools_extra=("Task", "Agent"),
    )
    disallowed = set(captured_cli.get("disallow_tools", ()))
    assert {"Task", "Agent"} <= disallowed
    assert "Bash" not in disallowed  # 카탈로그는 열려 있다(예외 경로)
    assert "CronCreate" in disallowed  # 스케줄 도구는 항상 차단
