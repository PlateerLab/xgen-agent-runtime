"""LLM 호출 하나가 턴을 무한정 붙잡지 못한다 (2026-09-23 안정성 감사 F2).

무엇이 있었나: 어느 클라이언트도 SDK 에 timeout 을 넘기지 않아 SDK 기본값(읽기 600초,
재시도 2회)이 적용됐고, API 스테이지가 그 위에서 타임아웃까지 "복구 가능" 으로 보고
최대 3회 더 시도했다 — 4 × 3 × 600초 ≈ **호출 하나에 최악 2시간**. 폐쇄망의 게이트웨이가
연결만 받고 응답을 흘리지 않으면 턴이 그동안 실행 스레드를 쥐고 있었다.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from xgen_agent_runtime.core.errors import APIError, ErrorCategory
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.llm_client import timeouts
from xgen_agent_runtime.stages.s06_api import APIStage, MockProvider
from xgen_agent_runtime.stages.s06_api.artifact.default import stage as stage_mod


# ── 한 곳에서 정한 값 ──────────────────────────────────────────────


class TestTheLimitsLiveInOnePlace:
    def test_defaults(self, monkeypatch):
        for name in ("XGEN_LLM_CONNECT_TIMEOUT_S", "XGEN_LLM_FIRST_CHUNK_TIMEOUT_S",
                     "XGEN_LLM_IDLE_TIMEOUT_S", "XGEN_LLM_REQUEST_TIMEOUT_S",
                     "XGEN_LLM_TIMEOUT_RETRIES"):
            monkeypatch.delenv(name, raising=False)
        assert timeouts.connect_timeout_s() == 10.0
        assert timeouts.first_chunk_timeout_s() == 180.0
        assert timeouts.idle_timeout_s() == 120.0
        assert timeouts.request_timeout_s() == 600.0
        assert timeouts.timeout_retries() == 1

    def test_env_overrides_are_read_at_call_time(self, monkeypatch):
        monkeypatch.setenv("XGEN_LLM_IDLE_TIMEOUT_S", "7")
        assert timeouts.idle_timeout_s() == 7.0
        monkeypatch.setenv("XGEN_LLM_IDLE_TIMEOUT_S", "nonsense")
        assert timeouts.idle_timeout_s() == 120.0
        monkeypatch.setenv("XGEN_LLM_IDLE_TIMEOUT_S", "-3")
        assert timeouts.idle_timeout_s() == 120.0

    def test_sdk_retries_are_off_so_they_do_not_multiply(self):
        kw = timeouts.sdk_client_kwargs()
        assert kw["max_retries"] == 0
        assert kw["timeout"].connect == timeouts.connect_timeout_s()

    def test_the_socket_waits_longer_than_the_watchdog(self):
        """짧으면 SDK 가 먼저 끊어 '무엇을 기다리다 끊겼는지' 를 스테이지가 말하지 못한다."""
        read = timeouts.sdk_read_timeout_s()
        assert read > timeouts.first_chunk_timeout_s()
        assert read > timeouts.idle_timeout_s()
        assert read > timeouts.request_timeout_s()


    def test_the_timeout_is_built_from_the_sdks_own_class(self):
        """anthropic 1.x·openai 3.x 는 httpx → httpx2 로 옮겨 다른 패키지의 Timeout 을 생성자에서
        거절한다(TypeError). 그래서 SDK 가 내보내는 Timeout 클래스로 만든다 — 판과 무관하게 맞다.
        (2026-09-23 4.52.0 main CI 가 anthropic 1.8.0 에서 이것으로 깨졌다.)"""

        class _Timeout:
            def __init__(self, read, *, connect, pool):
                self.read, self.connect, self.pool = read, connect, pool

        class _FakeSdk:
            Timeout = _Timeout

        t = timeouts.sdk_timeout(_FakeSdk)
        assert isinstance(t, _Timeout)
        assert t.connect == timeouts.connect_timeout_s()
        assert t.read == timeouts.sdk_read_timeout_s()
        assert isinstance(timeouts.sdk_client_kwargs(_FakeSdk)["timeout"], _Timeout)

    def test_real_sdks_get_their_own_timeout_type(self):
        import anthropic
        import openai

        assert isinstance(timeouts.sdk_timeout(anthropic), anthropic.Timeout)
        assert isinstance(timeouts.sdk_timeout(openai), openai.Timeout)

    def test_without_an_sdk_it_is_plain_httpx(self):
        import httpx

        assert isinstance(timeouts.sdk_timeout(None), httpx.Timeout)


# ── 모든 SDK 클라이언트가 그 값을 쓴다 ─────────────────────────────


def _sdk(client):
    return client._get_client()


class TestEveryClientUsesThem:
    def test_anthropic(self):
        from xgen_agent_runtime.llm_client.anthropic import AnthropicClient

        sdk = _sdk(AnthropicClient(api_key="k"))
        assert sdk.max_retries == 0
        assert sdk.timeout.connect == timeouts.connect_timeout_s()

    def test_openai_and_its_family(self):
        from xgen_agent_runtime.llm_client.openai import OpenAIClient
        from xgen_agent_runtime.llm_client.vllm import VLLMClient

        for cls in (OpenAIClient, VLLMClient):
            sdk = _sdk(cls(api_key="k", base_url="http://x/v1"))
            assert sdk.max_retries == 0, cls
            assert sdk.timeout.connect == timeouts.connect_timeout_s(), cls

    def test_google_always_carries_a_timeout(self):
        from xgen_agent_runtime.llm_client.google import GoogleClient

        opts = GoogleClient(api_key="k")._http_options()
        assert opts["timeout"] == timeouts.genai_timeout_ms()

    def test_no_client_builds_an_sdk_without_the_shared_limits(self):
        """새 클라이언트를 추가하며 잊으면 여기서 걸린다."""
        from pathlib import Path

        import xgen_agent_runtime.llm_client as pkg

        root = Path(pkg.__file__).parent
        for name in ("anthropic.py", "openai.py", "azure_foundry.py", "bedrock.py"):
            src = (root / name).read_text("utf-8")
            assert "sdk_client_kwargs" in src, name
            # SDK 모듈을 넘겨야 그 SDK 의 Timeout 클래스로 만든다(httpx2 판에서 깨지지 않게).
            assert "sdk_client_kwargs()" not in src, f"{name}: SDK 모듈 없이 불렀다"


# ── 스트림 감시 ───────────────────────────────────────────────────


async def _drain(stream):
    return [c async for c in stream]


async def _chunks(*items, gap=0.0, first_delay=0.0):
    if first_delay:
        await asyncio.sleep(first_delay)
    for i, item in enumerate(items):
        if i and gap:
            await asyncio.sleep(gap)
        yield item


class TestTheStreamWatchdog:
    @pytest.mark.asyncio
    async def test_a_stream_that_never_starts_is_cut(self):
        stream = stage_mod._watched_stream(
            _chunks({"type": "text_delta", "text": "x"}, first_delay=5),
            first_chunk_s=0.05, idle_s=10)
        with pytest.raises(APIError) as err:
            await _drain(stream)
        assert err.value.category == ErrorCategory.TIMEOUT
        assert "응답을 시작하지" in str(err.value)

    @pytest.mark.asyncio
    async def test_a_stream_that_stalls_midway_is_cut(self):
        stream = stage_mod._watched_stream(
            _chunks({"type": "text_delta", "text": "a"}, {"type": "text_delta", "text": "b"},
                    gap=5),
            first_chunk_s=10, idle_s=0.05)
        seen = []
        with pytest.raises(APIError) as err:
            async for c in stream:
                seen.append(c)
        assert [c["text"] for c in seen] == ["a"]
        assert "멈췄습니다" in str(err.value)

    @pytest.mark.asyncio
    async def test_header_chunks_do_not_end_the_first_wait(self):
        """머리 청크(사용량 등)가 곧바로 와도 '첫 내용' 대기는 계속 요청 시각부터 잰다."""
        async def gen():
            yield {"type": "usage", "input_tokens": 1}
            await asyncio.sleep(0.2)
            yield {"type": "text_delta", "text": "late"}

        with pytest.raises(APIError):
            await _drain(stage_mod._watched_stream(gen(), first_chunk_s=0.1, idle_s=10))

    @pytest.mark.asyncio
    async def test_a_healthy_stream_passes_untouched(self):
        items = [{"type": "text_delta", "text": w} for w in "abc"]
        out = await _drain(stage_mod._watched_stream(
            _chunks(*items, gap=0.01), first_chunk_s=1, idle_s=1))
        assert out == items

    @pytest.mark.asyncio
    async def test_the_stream_is_closed_when_cut(self):
        closed = {}

        async def gen():
            try:
                await asyncio.sleep(5)
                yield {"type": "text_delta", "text": "x"}
            finally:
                closed["yes"] = True

        with pytest.raises(APIError):
            await _drain(stage_mod._watched_stream(gen(), first_chunk_s=0.05, idle_s=1))
        assert closed.get("yes") is True


# ── 스테이지 배선 ─────────────────────────────────────────────────


class _Stalling(MockProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.attempts = 0

    async def create_message_stream(self, request):  # noqa: ANN001
        self.attempts += 1
        await asyncio.sleep(5)
        yield {"type": "text_delta", "text": "late"}


def _state(sid="s"):
    st = PipelineState(session_id=sid)
    st.add_message("user", "hi")
    return st


class TestTheStageEnforcesIt:
    @pytest.mark.asyncio
    async def test_a_timeout_is_retried_once_not_three_times(self, monkeypatch):
        monkeypatch.setenv("XGEN_LLM_FIRST_CHUNK_TIMEOUT_S", "0.05")
        monkeypatch.setenv("XGEN_LLM_TIMEOUT_RETRIES", "1")
        provider = _Stalling(default_text="x")
        stage = APIStage(provider=provider)
        t0 = time.monotonic()
        with pytest.raises(APIError) as err:
            await stage.execute("in", _state())
        assert err.value.category == ErrorCategory.TIMEOUT
        assert provider.attempts == 2, "타임아웃은 한 번만 더 시도해야 한다"
        assert time.monotonic() - t0 < 10

    @pytest.mark.asyncio
    async def test_retries_for_timeouts_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("XGEN_LLM_FIRST_CHUNK_TIMEOUT_S", "0.05")
        monkeypatch.setenv("XGEN_LLM_TIMEOUT_RETRIES", "0")
        provider = _Stalling(default_text="x")
        with pytest.raises(APIError):
            await APIStage(provider=provider).execute("in", _state())
        assert provider.attempts == 1

    @pytest.mark.asyncio
    async def test_non_streaming_calls_are_bounded_too(self, monkeypatch):
        monkeypatch.setenv("XGEN_LLM_REQUEST_TIMEOUT_S", "0.05")
        monkeypatch.setenv("XGEN_LLM_TIMEOUT_RETRIES", "0")

        class _Slow(MockProvider):
            async def create_message(self, request):  # noqa: ANN001
                await asyncio.sleep(5)

        stage = APIStage(provider=_Slow(default_text="x"))
        stage.update_config({"stream": False})
        with pytest.raises(APIError) as err:
            await stage.execute("in", _state())
        assert err.value.category == ErrorCategory.TIMEOUT

    def test_cli_backends_are_not_watched(self):
        """CLI 는 스트림 안에서 도구를 직접 돌린다 — 몇 분짜리 Bash 가 정상이다."""
        import inspect

        src = inspect.getsource(stage_mod.APIStage._call_streaming)
        assert 'if source != "cli":' in src
        assert src.index('if source != "cli":') < src.index("_watched_stream(")
