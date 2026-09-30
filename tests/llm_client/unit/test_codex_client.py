"""CodexCLIClient — end-to-end against the fake ``codex`` binary."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.core.errors import APIError, ErrorCategory
from xgen_agent_runtime.llm_client.codex import CodexCLIClient

FAKE_CODEX = str(Path(__file__).resolve().parents[2] / "_fixtures" / "fake_codex.py")


def _client(scenario: str = "ok_stream", **kwargs) -> CodexCLIClient:
    env_extras = dict(kwargs.pop("env_extras", None) or {})
    env_extras.setdefault("FAKE_CODEX_SCENARIO", scenario)
    defaults = dict(
        binary_path=FAKE_CODEX,
        workspace_dir=os.getcwd(),
        api_key="sk-fake",
        timeout_s=10.0,
        env_extras=env_extras,
    )
    defaults.update(kwargs)
    return CodexCLIClient(**defaults)


def _mc(model: str = "gpt-5.3-codex") -> ModelConfig:
    return ModelConfig(model=model)


@pytest.mark.asyncio
async def test_oneshot_returns_text_and_usage():
    client = _client()
    response = await client.create_message(
        model_config=_mc(), messages=[{"role": "user", "content": "hi"}]
    )
    assert response.text == "fake codex answer"
    assert response.usage.input_tokens == 12
    assert response.usage.cache_read_input_tokens == 2
    # The thread id was captured for next-turn resume.
    assert client._session_hint == {"session_id": "thr_1", "resume": True}
    assert response.raw["cli_version"].startswith("codex-cli")


@pytest.mark.asyncio
async def test_streaming_yields_events_then_message_complete():
    client = _client()
    events = []
    async for event in client.create_message_stream(
        model_config=_mc(), messages=[{"role": "user", "content": "hi"}]
    ):
        events.append(event)
    types = [e["type"] for e in events]
    assert types[-1] == "message_complete"
    assert "text_delta" in types and "thinking_delta" in types
    final = events[-1]["response"]
    assert final.text == "fake codex answer"
    assert final.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_auth_failure_is_fatal_category():
    client = _client("auth_fail")
    with pytest.raises(APIError) as ei:
        await client.create_message(
            model_config=_mc(), messages=[{"role": "user", "content": "hi"}]
        )
    assert ei.value.category == ErrorCategory.CLI_AUTH_FAILED


@pytest.mark.asyncio
async def test_argv_and_stdin_reach_the_binary():
    """The system prompt and flattened history travel over stdin; the
    sandbox / json flags are on argv."""
    client = _client("echo_argv")
    response = await client.create_message(
        model_config=_mc(),
        messages=[{"role": "user", "content": "질문입니다"}],
        system="너는 XGEN 에이전트다.",
    )
    import json

    payload = json.loads(response.text)
    assert payload["argv"][0] == "exec"
    assert "--json" in payload["argv"]
    assert payload["argv"][-1] == "-"
    assert "너는 XGEN 에이전트다." in payload["stdin"]
    assert "질문입니다" in payload["stdin"]


@pytest.mark.asyncio
async def test_resume_sends_only_new_delta_not_system_or_full_history():
    client = _client("echo_argv", session_hint={"session_id": "thr_existing", "resume": True})
    response = await client.create_message(
        model_config=_mc(),
        messages=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ],
        system="old system prompt",
    )
    import json

    payload = json.loads(response.text)
    assert payload["argv"][:3] == ["exec", "resume", "thr_existing"]
    assert "new question" in payload["stdin"]
    assert "old question" not in payload["stdin"]
    assert "old answer" not in payload["stdin"]
    assert "old system prompt" not in payload["stdin"]


@pytest.mark.asyncio
async def test_missing_binary_is_cli_not_found():
    client = CodexCLIClient(binary_path="/totally/missing/codex", api_key="k")
    with pytest.raises(APIError) as ei:
        await client.create_message(
            model_config=_mc(), messages=[{"role": "user", "content": "hi"}]
        )
    assert ei.value.category == ErrorCategory.CLI_NOT_FOUND


def test_subscription_mode_never_leaks_the_api_key():
    """oauth(구독) 모드에 OPENAI_API_KEY 를 흘리면 청구 채널이 조용히
    뒤집힌다 — Claude 백엔드와 같은 계약."""
    client = _client(auth_mode="oauth")
    assert "OPENAI_API_KEY" not in client._env_extras()
    client2 = _client(auth_mode="api_key")
    assert client2._env_extras()["OPENAI_API_KEY"] == "sk-fake"


# ---------------------------------------------------------------------------
# audit #26 — CLI-internal tool items reach the stream + Stage 6 (source=cli)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_surfaces_tool_use_and_tool_result_for_cli_items():
    client = _client("ok_stream_tools")
    events = []
    async for event in client.create_message_stream(
        model_config=_mc(), messages=[{"role": "user", "content": "ls"}]
    ):
        events.append(event)
    types = [e["type"] for e in events]
    assert types == [
        "tool_use", "tool_result", "tool_use", "tool_result", "text_delta", "message_complete"
    ], types
    assert events[0]["name"] == "Bash" and events[0]["id"] == "item_1"
    assert events[1]["tool_use_id"] == "item_1" and events[1]["content"] == "a.txt\nb.txt\n"
    assert events[2]["name"] == "mcp__connector__memory_search"
    assert events[3]["tool_use_id"] == "item_2" and events[3]["is_error"] is False
    final = events[-1]["response"]
    assert final.text == "two files"
    assert [b.type for b in final.content] == ["text"]
    assert "unknown_line_count" in final.raw and final.raw["unknown_line_count"] == 0


@pytest.mark.asyncio
async def test_codex_tool_items_reach_stage6_as_cli_tool_calls():
    """Same outermost surface the Claude CLI test pins: api.tool_use +
    api.cli_tool_call companion + api.tool_result, all source="cli", so
    ``host.runner.stream_turn`` renders agent_event tool_call/tool_result
    for Codex without any backend-specific branch."""
    from xgen_agent_runtime import Pipeline
    from xgen_agent_runtime.stages.s01_input import InputStage
    from xgen_agent_runtime.stages.s06_api import APIStage
    from xgen_agent_runtime.stages.s21_yield import YieldStage

    pipeline = Pipeline()
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage())
    pipeline.register_stage(YieldStage())
    pipeline.attach_runtime(llm_client=_client("ok_stream_tools"))

    events = []
    async for event in pipeline.run_stream("list /tmp/x"):
        events.append(event)
    assert events[-1].type == "pipeline.complete", [e.type for e in events]

    tool_uses = [e for e in events if e.type == "api.tool_use"]
    assert [e.data["name"] for e in tool_uses] == ["Bash", "mcp__connector__memory_search"]
    assert all(e.data["source"] == "cli" for e in tool_uses)
    cli_calls = [e for e in events if e.type == "api.cli_tool_call"]
    assert [c.data for c in cli_calls] == [t.data for t in tool_uses]
    results = [e for e in events if e.type == "api.tool_result"]
    assert [r.data["tool_use_id"] for r in results] == ["item_1", "item_2"]
    assert all(r.data["source"] == "cli" for r in results)
    # The tool items never reach the history as tool_use blocks (no Stage 10 ghost dispatch).
    text = [e for e in events if e.type == "text.delta"]
    assert [t.data["text"] for t in text] == ["two files"]


# ---------------------------------------------------------------------------
# 4.71.0 — the stream path decides like one-shot: warnings are not fatal
# ---------------------------------------------------------------------------
#
# 실측(codex-cli 0.159.2): codex 는 치명적이지 않은 경고도 {"type": "error"} 줄로 낸다
# ("Model metadata for `gpt-5.3-codex` not found. Defaulting to fallback metadata …").
# 스트림 경로가 줄마다 바로 올리면 경고 하나가 codex 턴 전체를 죽였다 — 같은 출력을
# one-shot(_send)은 성공으로 읽는다. 스트림도 끝에서 판정한다.

import json as _json
from types import SimpleNamespace

_WARNING = (
    "Model metadata for `gpt-5.3-codex` not found. Defaulting to fallback metadata; "
    "this can degrade performance and cause issues."
)


class _ScriptedRunner:
    """CLIProcessRunner 대역 — ``stream`` 이 정해 둔 JSONL 줄을 내보낸다."""

    def __init__(self, lines, *, exit_error: Exception | None = None) -> None:
        self._lines = [_json.dumps(obj).encode() + b"\n" for obj in lines]
        self._exit_error = exit_error

    async def run_oneshot(self, argv, **_kw):
        return SimpleNamespace(returncode=0, stdout=b"codex-cli 0.159.2\n", stderr=b"")

    async def stream(self, argv, *, stdin_iter=None, prespawned=None):
        if stdin_iter is not None:
            async for _chunk in stdin_iter:
                pass
        for line in self._lines:
            yield line
        if self._exit_error is not None:
            raise self._exit_error


def _scripted(lines, **kw) -> CodexCLIClient:
    runner = _ScriptedRunner(lines, **kw)
    return CodexCLIClient(api_key="sk-fake", runner_factory=lambda **_k: runner)


async def _collect(client: CodexCLIClient):
    return [
        e
        async for e in client.create_message_stream(
            model_config=_mc(), messages=[{"role": "user", "content": "hi"}]
        )
    ]


_THREAD = {"type": "thread.started", "thread_id": "thr_1"}
_ANSWER = {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "done"}}
_TURN_DONE = {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 1}}


@pytest.mark.asyncio
async def test_stream_warning_error_line_then_answer_completes(caplog):
    client = _scripted([_THREAD, {"type": "error", "message": _WARNING}, _ANSWER, _TURN_DONE])
    import logging

    with caplog.at_level(logging.WARNING, logger="xgen_agent_runtime.llm_client.codex"):
        events = await _collect(client)
    assert events[-1]["type"] == "message_complete"
    assert events[-1]["response"].text == "done"
    assert "Model metadata" in caplog.text, "the warning is logged, not swallowed"


@pytest.mark.asyncio
async def test_stream_tool_call_counts_as_output_after_a_warning():
    tool_done = {
        "type": "item.completed",
        "item": {
            "id": "i2",
            "type": "command_execution",
            "command": "ls",
            "aggregated_output": "a\n",
            "exit_code": 0,
            "status": "completed",
        },
    }
    client = _scripted([{"type": "error", "message": _WARNING}, tool_done, _TURN_DONE])
    events = await _collect(client)
    assert events[-1]["type"] == "message_complete"
    assert "tool_use" in [e["type"] for e in events]


@pytest.mark.asyncio
async def test_stream_error_then_turn_failed_raises_with_that_message():
    client = _scripted(
        [_THREAD, {"type": "error", "message": "stream disconnected: 502"}, {"type": "turn.failed"}]
    )
    with pytest.raises(APIError) as ei:
        await _collect(client)
    assert ei.value.category == ErrorCategory.CLI_PROTOCOL_ERROR
    assert "stream disconnected: 502" in str(ei.value)


@pytest.mark.asyncio
async def test_stream_turn_failed_uses_its_own_message_when_present():
    client = _scripted(
        [
            {"type": "error", "message": _WARNING},
            _ANSWER,
            {"type": "turn.failed", "error": {"message": "context window exceeded"}},
        ]
    )
    with pytest.raises(APIError) as ei:
        await _collect(client)
    assert "context window exceeded" in str(ei.value)


@pytest.mark.asyncio
async def test_stream_error_only_then_eof_raises_with_that_message():
    client = _scripted([_THREAD, {"type": "error", "message": "model not supported"}])
    with pytest.raises(APIError) as ei:
        await _collect(client)
    assert ei.value.category == ErrorCategory.CLI_PROTOCOL_ERROR
    assert "model not supported" in str(ei.value)


@pytest.mark.asyncio
async def test_stream_auth_phrase_fails_immediately():
    client = _scripted(
        [{"type": "error", "message": "401 Unauthorized: invalid api key"}, _ANSWER, _TURN_DONE]
    )
    events = []
    with pytest.raises(APIError) as ei:
        async for e in client.create_message_stream(
            model_config=_mc(), messages=[{"role": "user", "content": "hi"}]
        ):
            events.append(e)
    assert ei.value.category == ErrorCategory.CLI_AUTH_FAILED
    assert events == [], "auth failure must not wait for the rest of the stream"


@pytest.mark.asyncio
async def test_stream_non_zero_exit_still_raises():
    from xgen_agent_runtime.llm_client._cli_runtime import CLIProtocolError

    client = _scripted(
        [{"type": "error", "message": _WARNING}, _ANSWER],
        exit_error=CLIProtocolError("CLI 'codex' exited with code 1: boom"),
    )
    with pytest.raises(APIError) as ei:
        await _collect(client)
    assert ei.value.category == ErrorCategory.CLI_PROTOCOL_ERROR
    assert "exited with code 1" in str(ei.value)
