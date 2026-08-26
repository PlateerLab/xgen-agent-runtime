"""RemoteForgedToolStore — forged-tool 스펙 저장소의 서버 RPC 구현.

:class:`~xgen_agent_runtime.host.forged_tools.ForgedToolSpecStore` 를 서버
엔드포인트 위에 구현한다. 도구 **스펙**은 계정 자산이라 서버 DB 가 원본이고,
로컬 턴도 웹 턴과 같은 목록을 본다 — 여기서 만든 도구가 웹에서도 보이고, 그
반대도 같다.

**스크립트 실행은 여기로 오지 않는다.** 스크립트는 동기화되는 ``workspace/``
안에 있으므로 로컬 턴은 자기 PC 에서 직접 돌린다(서버 턴이 러너 세션에서 돌리는
것과 같은 규칙 — 실행지는 ``ToolContext.sandbox`` 가 정한다). 이 클래스가 나르는
것은 스펙 CRUD 와 통계뿐이다.

실패 방침은 연산마다 다르다. 이건 취향이 아니라 **오답의 비용**이 다르기 때문이다:

* ``list``/``get`` 실패 → 빈 목록. 도구가 잠깐 안 보이는 것뿐이다.
* ``save`` 실패 → **예외를 올린다**. 여기서 성공했다고 답하면 에이전트는 도구를
  만들었다고 믿는데 다음 턴에 없다 (서버 store 의 "DB 가 원본" 방침과 동일).
* ``record_call``/``mark_tested`` 실패 → 무시. 통계가 도구 실행을 깨면 안 된다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("xgen_agent_runtime.host.remote_forged_store")


class RemoteForgedToolStoreError(RuntimeError):
    """저장소 쓰기가 서버에 도달하지 못했다."""


class RemoteForgedToolStore:
    """서버 ``/forged-tools`` RPC 위의 스펙 저장소."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        path: str,
        timeout_s: float = 30.0,
        verify: Any = True,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._path = path if path.startswith("/") else f"/{path}"
        self._timeout = float(timeout_s)
        self._verify = verify

    # ── 전송 ─────────────────────────────────────────────────────────
    def _rpc(self, op: str, **payload: Any) -> Optional[Dict[str, Any]]:
        """한 번의 RPC. 네트워크/HTTP 실패는 None, 서버가 낸 실패는 ``ok:false``."""
        try:
            import httpx

            with httpx.Client(timeout=self._timeout, verify=self._verify) as client:
                resp = client.post(
                    f"{self._base}{self._path}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"op": op, **payload},
                )
                resp.raise_for_status()
                data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("forged-tool store RPC 실패 (op=%s): %s", op, exc)
            return None

    def _spec_of(self, raw: Any) -> Optional[Any]:
        from xgen_agent_runtime.host.forged_tools import ForgedToolSpec

        if not isinstance(raw, dict):
            return None
        try:
            return ForgedToolSpec.from_dict(raw)
        except Exception:  # noqa: BLE001 — 스펙 하나가 목록 전체를 죽이지 않는다
            logger.warning("forged-tool 스펙 해석 실패 (스킵)", exc_info=True)
            return None

    # ── ForgedToolSpecStore 계약 ─────────────────────────────────────
    def list(self) -> List[Any]:
        data = self._rpc("list")
        if not data or not data.get("ok"):
            return []
        out = [self._spec_of(r) for r in (data.get("specs") or [])]
        return [s for s in out if s is not None]

    def get(self, name: str) -> Optional[Any]:
        data = self._rpc("get", name=str(name))
        if not data or not data.get("ok"):
            return None
        return self._spec_of(data.get("spec"))

    def save(self, spec: Any) -> Any:
        """저장 — 실패하면 **여기서 실패한다**.

        조용히 성공으로 답하면 에이전트는 도구를 만들었다고 믿고 다음 턴에
        "그런 도구 없음"을 만난다. 서버 store 가 DB 실패를 삼키지 않는 것과 같은
        이유다.
        """
        data = self._rpc("save", spec=spec.to_dict())
        if data is None:
            raise RemoteForgedToolStoreError(
                "서버에 도구 스펙을 저장하지 못했습니다 (서버에 닿지 못함). "
                "네트워크가 끊겼거나 로그인 세션이 만료됐을 수 있습니다."
            )
        if not data.get("ok"):
            raise RemoteForgedToolStoreError(
                f"서버가 도구 스펙 저장을 거부했습니다: {data.get('error') or '알 수 없는 오류'}"
            )
        saved = self._spec_of(data.get("spec"))
        return saved if saved is not None else spec

    def delete(self, name: str) -> bool:
        data = self._rpc("delete", name=str(name))
        if not data or not data.get("ok"):
            return False
        return bool(data.get("removed"))

    def record_call(self, name: str, *, error: Optional[str] = None) -> None:
        # 통계는 유실돼도 된다 — 도구 실행을 깨는 것보다 훨씬 싸다.
        self._rpc("record_call", name=str(name), error=error)

    def mark_tested(self, name: str, *, ok: bool, error: Optional[str] = None) -> None:
        self._rpc("mark_tested", name=str(name), ok=bool(ok), error=error)
