"""API 클라이언트가 벤더 SDK 연결 풀을 닫는다 (`BaseClient.aclose`, 4.73.0).

닫지 않고 버린 SDK 클라이언트는 가비지 컬렉터가 나중에 닫는데, 그 사이 새 연결이 같은 소켓 번호를 받으면 새
연결이 닫혀 연결 상한(10초)까지 멈췄다(xgen-workflow 앱 LLM 실측). 턴은 끝에 ``Pipeline.aclose`` 가 이것을 부른다.
"""

from __future__ import annotations

import asyncio

import pytest

from xgen_agent_runtime.llm_client.openai import OpenAIClient


class _Async:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


class _Genai:
    class _Aio:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    def __init__(self):
        self.aio = self._Aio()
        self.sync_closed = 0

    def close(self):
        self.sync_closed += 1


def test_the_sdk_client_is_closed_once_and_rebuilt_on_next_use():
    client = OpenAIClient(api_key="k")
    sdk = _Async()
    client._client = sdk
    asyncio.run(client.aclose())
    asyncio.run(client.aclose())
    assert sdk.closed == 1 and client._client is None, "한 번만 닫고, 다시 불러도 괜찮다"


def test_google_genai_closes_its_async_side():
    pytest.importorskip("google.genai")
    from xgen_agent_runtime.llm_client.google import GoogleClient

    client = GoogleClient(api_key="k")
    sdk = _Genai()
    client._client = sdk
    asyncio.run(client.aclose())
    assert sdk.aio.closed == 1 and sdk.sync_closed == 0


def test_a_real_openai_sdk_client_is_really_closed():
    pytest.importorskip("openai")
    client = OpenAIClient(api_key="k", base_url="http://127.0.0.1:9/v1")
    sdk = client._get_client()
    assert not sdk.is_closed()
    asyncio.run(client.aclose())
    assert sdk.is_closed()
    assert client._get_client() is not sdk, "닫은 뒤 다시 쓰면 새로 만든다"


def test_a_client_without_an_sdk_has_nothing_to_close():
    client = OpenAIClient(api_key="k")
    asyncio.run(client.aclose())  # 만들기 전 — 할 일이 없다


def test_a_failing_close_does_not_raise():
    class _Bad:
        async def close(self):
            raise RuntimeError("boom")

    client = OpenAIClient(api_key="k")
    client._client = _Bad()
    asyncio.run(client.aclose())
    assert client._client is None
