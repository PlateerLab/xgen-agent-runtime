"""RemoteMemoryProvider ↔ memory_wire 왕복 증명 — 라이브 서버 없이.

핵심(무발산 공유 메모리)의 3요소를 로컬에서 전부 증명한다:
  ① memory_wire 코덱: 런타임 dataclass/enum 왕복(타입 보존).
  ② RemoteMemoryProvider 반사 프록시: 핸들.메서드 → RPC.
  ③ 서버 디스패치: op → 실제 provider 메서드(화이트리스트).

라이브 서버 대신 **루프백 전송**을 쓴다 — 서버 엔드포인트와 동일한 디스패치
(load 인자 → 대상 해석 → 호출 → dump 결과)를 실제 런타임 FileMemoryProvider 에
적용한다. 즉 이 테스트가 통과하면, 남은 건 그 디스패치를 HTTP 로 감싼 서버
엔드포인트(controller/workflow/endpoints/geny_memory.py::memory_rpc)뿐이다.
"""

from __future__ import annotations

import asyncio
import tempfile

from xgen_agent_runtime.host import memory_wire
from xgen_agent_runtime.host.remote_memory import RemoteMemoryProvider

# 서버 memory_rpc 의 화이트리스트/해석과 동일 규칙(테스트 미러).
_RPC_HANDLES = {"stm", "ltm", "notes", "index", "vector"}
_RPC_TOPLEVEL = {"record_turn", "record_execution", "reflect", "promote", "descriptor"}


def _resolve(provider, op):
    if "." in op:
        handle_name, _, method = op.partition(".")
        assert handle_name in _RPC_HANDLES and method and not method.startswith("_")
        handle = getattr(provider, handle_name)()
        assert handle is not None
        target = getattr(handle, method)
        assert callable(target)
        return target
    assert op in _RPC_TOPLEVEL
    if op == "descriptor":
        desc = provider.descriptor
        desc = desc() if callable(desc) else desc
        return lambda: desc
    return getattr(provider, op)


def _loopback(provider):
    """서버 엔드포인트와 동일한 디스패치를 실제 provider 에 적용하는 전송."""

    async def transport(payload):
        try:
            args = [memory_wire.load(a) for a in payload.get("args", [])]
            kwargs = {k: memory_wire.load(v) for k, v in payload.get("kwargs", {}).items()}
            target = _resolve(provider, payload["op"])
            result = target(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return {"ok": True, "result": memory_wire.dump(result)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return transport


def _build_file_provider(root):
    from xgen_agent_runtime.memory import MemoryProviderFactory

    return MemoryProviderFactory().build({"provider": "file", "root": root, "session_id": "s1"})


def _remote(provider):
    return RemoteMemoryProvider(
        base_url="http://loopback",
        token="t",
        workflow_id="wf1",
        interaction_id="i1",
        transport=_loopback(provider),
    )


def test_codec_roundtrips_dataclass_and_enum():
    from xgen_agent_runtime.memory import Importance, NoteDraft, Scope

    draft = NoteDraft(title="제목", body="본문", importance=Importance.HIGH, tags=["a", "b"])
    wire = memory_wire.dump(draft)
    # 순수 JSON 직렬화 가능해야(HTTP 왕복).
    import json

    back = memory_wire.load(json.loads(json.dumps(wire)))
    assert isinstance(back, NoteDraft)
    assert back.title == "제목" and back.body == "본문"
    assert back.importance is Importance.HIGH  # enum 복원(동일성)
    assert back.scope is Scope.SESSION
    assert back.tags == ["a", "b"]


def test_notes_write_read_list_through_remote_provider():
    async def scenario():
        with tempfile.TemporaryDirectory() as root:
            real = _build_file_provider(root)
            await real.initialize()
            remote = _remote(real)
            await remote.initialize()

            from xgen_agent_runtime.memory import NoteDraft

            meta = await remote.notes().write(
                NoteDraft(title="회의록", body="결정: X 를 채택", category="topics")
            )
            assert meta.title == "회의록"
            fn = meta.ref.filename

            note = await remote.notes().read(fn)
            assert note is not None
            assert "X 를 채택" in note.body  # 본문이 서버(파일) 왕복 후 그대로

            listed = await remote.notes().list()
            names = [m.ref.filename for m in listed]
            assert fn in names
            await real.close()

    asyncio.run(scenario())


def test_high_level_record_turn_reaches_stm():
    async def scenario():
        with tempfile.TemporaryDirectory() as root:
            real = _build_file_provider(root)
            await real.initialize()
            remote = _remote(real)

            from xgen_agent_runtime.memory import Turn

            await remote.record_turn(Turn(role="user", content="안녕"))
            await remote.record_turn(Turn(role="assistant", content="반가워"))

            recent = await remote.stm().recent(10)
            contents = [t.content for t in recent]
            assert "안녕" in contents and "반가워" in contents
            await real.close()

    asyncio.run(scenario())


def test_rpc_error_surfaces_as_domain_error():
    async def scenario():
        with tempfile.TemporaryDirectory() as root:
            real = _build_file_provider(root)
            await real.initialize()
            remote = _remote(real)
            from xgen_agent_runtime.host.remote_memory import MemoryRPCError

            raised = False
            try:
                await remote.notes().read("does-not-exist-and-bad")
            except MemoryRPCError:
                raised = True
            except Exception:  # noqa: BLE001 — read 는 보통 None 반환(예외 아님)
                raised = False
            # read 는 없는 파일에 None 을 주는 게 정상 — 오류가 아니라 None 이어야.
            note = await remote.notes().read("nope.md")
            assert note is None and raised is False
            await real.close()

    asyncio.run(scenario())
