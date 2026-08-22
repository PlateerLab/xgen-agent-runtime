"""RemoteMemoryProvider — MemoryProvider 프로토콜을 **서버 RPC** 위에 구현.

로컬↔웹 무발산: 실행은 로컬(사이드카)이지만 **메모리 vault 는 서버(로그인 계정)
소유**다. 이 provider 는 파이프라인/메모리툴이 부르는 모든 메모리 연산을 서버의
실제 provider 로 그대로 넘긴다 — 로컬 턴이 남긴 기억이 곧 웹에서도 보이고, 웹이
남긴 기억을 로컬이 읽는다(같은 workflow 축).

구현 전략(반사 프록시): 프로토콜은 핸들 5개(stm/ltm/notes/index/vector) × 여러
async 메서드 + 고수준 4개(record_turn/record_execution/reflect/promote). 이를
일일이 나열하지 않고, 핸들 접근을 프록시로 잡아 ``<handle>.<method>(*a, **k)`` 를
RPC ``{op:"notes.read", args, kwargs}`` 로 보낸다. 인자/결과는 memory_wire 의
타입-태그 코덱으로 왕복(양쪽에 런타임이 있어 원래 타입으로 복원).

전송: httpx(런타임 의존). 서버 엔드포인트:
  POST {base}/api/agentflow/geny-memory/{workflow_id}/rpc
  body {op, args:[enc], kwargs:{enc}, interaction_id}  → {ok, result:enc} | {ok:false, error}
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from xgen_agent_runtime.host import memory_wire


class MemoryRPCError(RuntimeError):
    """서버 메모리 RPC 실패(엔드포인트 오류/거부)."""


class _HandleProxy:
    """핸들(notes/stm/…) 프록시 — 메서드 접근을 RPC 로 변환.

    프로토콜 핸들 메서드는 전부 async 이므로, 여기서 반환하는 콜러블도 async."""

    __slots__ = ("_provider", "_handle")

    def __init__(self, provider: "RemoteMemoryProvider", handle: str) -> None:
        self._provider = provider
        self._handle = handle

    def __getattr__(self, method: str):
        if method.startswith("_"):
            raise AttributeError(method)

        async def call(*args: Any, **kwargs: Any) -> Any:
            return await self._provider._rpc(f"{self._handle}.{method}", args, kwargs)

        call.__name__ = method
        return call


class RemoteMemoryProvider:
    """서버 메모리 vault 의 원격 핸들. MemoryProvider 프로토콜 준수(반사)."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        workflow_id: str,
        interaction_id: str = "",
        timeout_s: float = 30.0,
        transport: Any = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._workflow_id = workflow_id
        self._interaction_id = interaction_id
        self._timeout = float(timeout_s)
        self._transport = transport  # 테스트 주입: async (payload)->dict
        self._descriptor: Any = None
        self._has_vector: Optional[bool] = None

    @property
    def _url(self) -> str:
        return f"{self._base}/api/agentflow/geny-memory/{self._workflow_id}/rpc"

    # ── 전송 ─────────────────────────────────────────────────────────────
    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._transport is not None:
            return await self._transport(payload)
        import httpx  # 런타임 의존(지연 import — 테스트는 transport 주입).

        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _rpc(self, op: str, args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
        payload = {
            "op": op,
            "args": [memory_wire.dump(a) for a in args],
            "kwargs": {k: memory_wire.dump(v) for k, v in kwargs.items()},
            "interaction_id": self._interaction_id,
        }
        try:
            resp = await self._post(payload)
        except Exception as exc:  # noqa: BLE001 — 전송 실패를 도메인 오류로.
            raise MemoryRPCError(f"memory rpc transport failed ({op}): {exc}") from exc
        if not isinstance(resp, dict) or not resp.get("ok", False):
            detail = (resp or {}).get("error") if isinstance(resp, dict) else resp
            raise MemoryRPCError(f"memory rpc {op} failed: {detail}")
        return memory_wire.load(resp.get("result"))

    # ── 라이프사이클 ──────────────────────────────────────────────────────
    async def initialize(self) -> None:
        """서버가 vault 를 이미 소유하므로 여기선 descriptor 만 당겨 캐시한다
        (capability 게이팅: vector() 유무). 실패해도 조용히 진행(무기억 아님 —
        메모리 op 는 여전히 RPC 로 시도된다)."""
        try:
            self._descriptor = await self._rpc("descriptor", (), {})
            caps = getattr(self._descriptor, "capabilities", None) or set()
            layers = getattr(self._descriptor, "layers", None) or set()
            names = {getattr(x, "value", str(x)) for x in list(caps) + list(layers)}
            self._has_vector = any("vector" in n or "semantic" in n for n in names)
        except Exception:  # noqa: BLE001
            self._descriptor = None
            self._has_vector = None

    async def close(self) -> None:
        return None

    def descriptor(self) -> Any:
        """캐시된 서버 descriptor. 미초기화면 최소 placeholder(크래시 방지)."""
        if self._descriptor is not None:
            return self._descriptor
        return _minimal_descriptor()

    # ── 핸들(sync 반환, 메서드는 async RPC) ──────────────────────────────
    def stm(self) -> Any:
        return _HandleProxy(self, "stm")

    def ltm(self) -> Any:
        return _HandleProxy(self, "ltm")

    def notes(self) -> Any:
        return _HandleProxy(self, "notes")

    def index(self) -> Any:
        return _HandleProxy(self, "index")

    def vector(self) -> Any:
        """서버에 벡터 레이어가 있으면 프록시, 없으면 None(capability 게이팅).

        초기화 전(미상)엔 프록시를 준다(best-effort). 서버가 벡터 없음을
        확정했으면 None → 시맨틱 경로가 깔끔히 비활성."""
        if self._has_vector is False:
            return None
        return _HandleProxy(self, "vector")

    # ── 고수준(파이프라인) ────────────────────────────────────────────────
    async def record_turn(self, turn: Any) -> None:
        await self._rpc("record_turn", (turn,), {})

    async def record_execution(self, summary: Any) -> Any:
        return await self._rpc("record_execution", (summary,), {})

    async def reflect(self, ctx: Any) -> Sequence[Any]:
        return await self._rpc("reflect", (ctx,), {})

    async def promote(self, ref: Any, to: Any) -> Any:
        return await self._rpc("promote", (ref, to), {})


def _minimal_descriptor() -> Any:
    """런타임이 있으면 진짜 MemoryDescriptor, 없으면 단순 네임스페이스."""
    try:
        from xgen_agent_runtime.memory import (
            Layer,
            MemoryDescriptor,
            Scope,
        )

        return MemoryDescriptor(
            name="remote",
            version="0",
            layers={Layer.SHORT_TERM} if hasattr(Layer, "SHORT_TERM") else set(),
            capabilities=set(),
            backends=[],
            scope=Scope.SESSION if hasattr(Scope, "SESSION") else None,  # type: ignore[arg-type]
            description="server-backed remote memory (pre-init)",
        )
    except Exception:  # noqa: BLE001
        import types

        return types.SimpleNamespace(
            name="remote", version="0", layers=set(), capabilities=set(), backends=[]
        )
