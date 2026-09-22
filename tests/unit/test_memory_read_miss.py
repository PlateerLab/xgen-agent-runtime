"""없는 노트를 물었을 때 금고가 가진 것을 보여 준다 (4.48.0).

근거 (dev `agent_trace_spans` 2026-09-02~20): `memory_read` 의
``Note not found`` 45건 / 14 대화 = **대화당 3.2회**. 파일명은 금고가 붙이는
것이라 모델이 맞힐 수 없는데, "없다" 만 돌려주니 모델이 또 지어냈다.
실제 기록의 추측 예: ``conversations/conn-wf_…__user__깃허브-레포-주소를-….md``,
``executions/exec-0002-422d6b6c.md``, ``critical/사용자-이름.md``.
"""

from __future__ import annotations

import json

import pytest

from xgen_agent_runtime.host.memory_tools import build_memory_tools
from xgen_agent_runtime.memory.provider import Scope
from xgen_agent_runtime.memory.providers.file.provider import FileMemoryProvider


def _tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"{name} not built: {[t.name for t in tools]}")


async def _vault(tmp_path):
    provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
    await provider.initialize()
    return provider


@pytest.mark.asyncio
async def test_miss_on_an_empty_vault_says_so_plainly(tmp_path):
    provider = await _vault(tmp_path)
    try:
        tools = build_memory_tools(provider)
        result = await _tool(tools, "memory_read").execute(
            {"filename": "critical/사용자-이름.md"}, None
        )
        assert result.is_error is True
        body = json.loads(result.content)
        assert "Note not found" in body["error"]
        assert body["available"] == []
        assert "memory_write" in body["hint"]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_miss_offers_the_real_filenames(tmp_path):
    provider = await _vault(tmp_path)
    try:
        tools = build_memory_tools(provider)
        write = _tool(tools, "memory_write")
        await write.execute(
            {"title": "사용자 이름", "content": "손성준", "category": "critical"}, None
        )
        await write.execute(
            {"title": "배포 절차", "content": "ArgoCD 수동 sync", "category": "critical"}, None
        )

        result = await _tool(tools, "memory_read").execute(
            {"filename": "critical/사용자-이름.md"}, None
        )
        assert result.is_error is True
        body = json.loads(result.content)
        assert body["did_you_mean"], body
        assert body["available_count"] >= 2
        # 추측을 반복하지 말고 목록/검색을 쓰라고 말한다.
        assert "memory_search" in body["hint"] and "memory_list" in body["hint"]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_unknown_category_falls_back_to_the_whole_vault(tmp_path):
    provider = await _vault(tmp_path)
    try:
        tools = build_memory_tools(provider)
        await _tool(tools, "memory_write").execute(
            {"title": "세션 기록", "content": "x", "category": "notes"}, None
        )
        result = await _tool(tools, "memory_read").execute(
            {"filename": "conversations/conn-wf_1789721412638_naq7zec.md"}, None
        )
        body = json.loads(result.content)
        # 'conversations' 카테고리는 없다 → 금고 전체에서 후보를 낸다.
        assert body["did_you_mean"], body
        assert "scanned_category" not in body
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_a_real_note_still_reads_normally(tmp_path):
    provider = await _vault(tmp_path)
    try:
        tools = build_memory_tools(provider)
        written = json.loads(
            (
                await _tool(tools, "memory_write").execute(
                    {"title": "배포 절차", "content": "본문", "category": "notes"}, None
                )
            ).content
        )
        result = await _tool(tools, "memory_read").execute(
            {"filename": written["filename"]}, None
        )
        assert result.is_error is False
        assert json.loads(result.content)["body"].strip() == "본문"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_a_broken_index_does_not_swallow_the_error(tmp_path):
    provider = await _vault(tmp_path)

    class _Boom:
        async def list_notes(self, **_kw):
            raise RuntimeError("index down")

    try:
        tools = build_memory_tools(provider)
        provider.index = lambda: _Boom()  # type: ignore[method-assign]
        result = await _tool(tools, "memory_read").execute({"filename": "a/b.md"}, None)
        assert result.is_error is True
        assert "Note not found" in json.loads(result.content)["error"]
    finally:
        await provider.close()
