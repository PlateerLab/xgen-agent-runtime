"""memory_wire 왕복 — RPC 로 오가는 타입이 **객체로** 복원되어야 한다.

회귀(2026-08-23 실기): MemoryChunk 는 stages.s02_context.types 에 있는데
_registry() 가 xgen_agent_runtime.memory 만 스캔해 미등록 → RetrievalResult.chunks
가 dict 로 격하 → s02 의 `chunk.content` 에서 'dict' object has no attribute
'content' 로 커넥터 로컬 턴이 통째로 죽었다.
"""
from __future__ import annotations

from xgen_agent_runtime.host import memory_wire as w
from xgen_agent_runtime.memory import RetrievalResult
from xgen_agent_runtime.stages.s02_context.types import MemoryChunk


def _roundtrip(obj):
    return w.load(w.dump(obj))


def test_memory_chunk_roundtrips_to_object():
    c = MemoryChunk(key="name", content="사용자 이름은 홍길동", source="pinned")
    out = _roundtrip(c)
    assert isinstance(out, MemoryChunk)
    assert out.content == "사용자 이름은 홍길동"
    assert out.source == "pinned"


def test_retrieval_result_chunks_are_objects_not_dicts():
    rr = RetrievalResult(
        chunks=[
            MemoryChunk(key="a", content="fact A", source="pinned"),
            MemoryChunk(key="b", content="fact B", source="stm"),
        ],
        layer_breakdown={},
        total_chars=12,
        cost=None,
        metadata={},
    )
    out = _roundtrip(rr)
    assert isinstance(out, RetrievalResult)
    assert out.chunks and all(isinstance(c, MemoryChunk) for c in out.chunks)
    # s02 가 하는 접근 — dict 면 AttributeError 로 죽는다.
    assert "\n\n".join(c.content for c in out.chunks) == "fact A\n\nfact B"


def test_registry_contains_wire_crossing_types():
    reg = w._registry()
    assert "MemoryChunk" in reg
    assert "RetrievalResult" in reg
    assert "Turn" in reg
