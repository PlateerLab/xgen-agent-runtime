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
