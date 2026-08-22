"""ServerBridge — 커넥터 사이드카가 **서버(로그인 계정)** 의 라이브 공유 상태에
닿는 인증 RPC 클라이언트.

로컬↔웹 무발산의 핵심: 실행은 로컬이지만 **메모리 같은 계정 상태는 서버가
진실**이다. 이 브릿지가 서버의 메모리 저장소를 원격 provider 로 열어, 로컬 턴이
읽고 쓴 기억이 곧 웹에서도 보인다(같은 workflow 축).

전송: httpx(런타임 의존) 로 서버 내부 RPC 엔드포인트를 호출한다. 토큰은 커넥터가
로그인 세션에서 발급받아 사이드카 요청에 실어 준다.
"""

from __future__ import annotations

from typing import Any, Optional


class ServerBridge:
    """서버 RPC 클라이언트. 지금은 메모리 provider 를 연다(공유 기억).

    미연결/실패는 조용히 None 으로 degrade — 로컬 전용 상태를 만들지 않는다
    (발산 방지: 기억이 갈라지느니 이번 턴은 무기억)."""

    def __init__(self, base_url: str, token: str, *, timeout_s: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout_s)

    # ── memory (서버 저장소 공유) ─────────────────────────────────────────
    def build_memory_provider(self, workflow_id: str, interaction_id: str) -> Optional[Any]:
        """서버 메모리를 원격 provider 로 연다 — 웹과 같은 workflow 저장소.

        RemoteMemoryProvider 는 MemoryProvider 프로토콜을 서버 RPC 위에 구현한다.
        구성 실패(엔드포인트 부재 등)는 None → 무기억 degrade."""
        try:
            from xgen_agent_runtime.host.remote_memory import RemoteMemoryProvider

            return RemoteMemoryProvider(
                base_url=self._base,
                token=self._token,
                workflow_id=str(workflow_id or ""),
                interaction_id=str(interaction_id or ""),
                timeout_s=self._timeout,
            )
        except Exception:  # noqa: BLE001
            return None
