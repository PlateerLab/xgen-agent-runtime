"""ServerBridge — 커넥터 사이드카가 **서버(로그인 계정)** 의 라이브 공유 상태에
닿는 인증 RPC 클라이언트.

로컬↔웹 무발산의 핵심: 실행은 로컬이지만 **메모리 같은 계정 상태는 서버가
진실**이다. 이 브릿지가 서버의 메모리 저장소를 원격 provider 로 열어, 로컬 턴이
읽고 쓴 기억이 곧 웹에서도 보인다(같은 workflow 축).

전송: httpx(런타임 의존) 로 서버 내부 RPC 엔드포인트를 호출한다. 토큰은 커넥터가
로그인 세션에서 발급받아 사이드카 요청에 실어 준다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ServerBridge:
    """서버 RPC 클라이언트. 지금은 메모리 provider 를 연다(공유 기억).

    미연결/실패는 조용히 None 으로 degrade — 로컬 전용 상태를 만들지 않는다
    (발산 방지: 기억이 갈라지느니 이번 턴은 무기억)."""

    def __init__(
        self, base_url: str, token: str, *, timeout_s: float = 30.0, verify: Any = True
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout_s)
        #: TLS 검증 — httpx verify 인자(True/False/CA 파일 경로). 커넥터의 사설 인증서 정책.
        self._verify = verify

    @property
    def base_url(self) -> str:
        """서버(로그인 계정 저장소)의 **도달 가능한** base URL.

        LocalHostServices 의 LLM 프록시 재작성이 재사용한다 — 내부 서빙(vLLM 등)
        base_url 은 PC 에서 못 닿으므로, 런타임이 이 서버 URL 로 LLM 호출을
        프록시한다 (connector 런타임 → xgen-server → 내부 provider)."""
        return self._base

    @property
    def token(self) -> str:
        """서버 인증 토큰(사용자 세션) — 프록시 엔드포인트 ``_authorize`` 가 그대로 검증."""
        return self._token

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
                verify=self._verify,
            )
        except Exception:  # noqa: BLE001
            return None

    # ── RAG (서버 자산 호출) ──────────────────────────────────────────────
    def rag_search(self, workflow_id: str, text: str, search_params: Any) -> Optional[str]:
        """서버 RAG 검색을 호출해 ``[DOC_n]`` 컨텍스트 블록을 받는다(동기).

        RAG 서비스/컬렉션은 **서버 자산** — 로컬은 search_params(직렬화 가능)만 갖고 서버가
        rag_service 클라이언트로 실제 검색을 돌린다(메모리와 같은 '서버 호출' 원칙).
        빌드 파이프라인(루프 이전)에서 동기로 불리므로 blocking httpx 로 안전하다.
        실패는 None → 그 RAG 아이템은 컨텍스트 없이 진행(턴 불변)."""
        wf = str(workflow_id or "").strip()
        if not wf or not isinstance(search_params, dict):
            return None
        try:
            import httpx

            url = f"{self._base}/api/agentflow/geny-memory/{wf}/rag-search"
            with httpx.Client(timeout=max(self._timeout, 300.0), verify=self._verify) as client:
                resp = client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"text": str(text or ""), "search_params": search_params},
                )
                resp.raise_for_status()
                data = resp.json()
            if not isinstance(data, dict) or not data.get("ok"):
                return None
            block = data.get("block")
            return str(block) if block else None
        except Exception:  # noqa: BLE001
            return None

    # ── connector 내장 도구 중계 (브라우저 — 사용자가 보는 XGEN 탭) ─────────
    def connector_mcp_call(
        self, path: str, server: str, tool: str, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """커넥터 내장 도구 실행을 **서버를 경유해** 커넥터로 중계한다(동기).

        로컬 턴은 사용자 PC 에서 돌지만 커넥터의 브라우저는 Electron 메인 프로세스
        안에 있고 사이드카에서 직접 가는 채널이 없다 — 그래서 메모리·RAG·자기진화와
        같은 '서버 호출' 경로를 쓴다: 런타임 → 서버 RPC → 역방향 WS → 커넥터.

        ``path`` 는 컨텍스트 메타가 실어 준 값(런타임이 경로를 지어내지 않는다).
        반환은 서버 응답 dict(``{"ok","content"}`` 또는 ``{"ok":false,"error"}``),
        네트워크/인가 실패는 None — 호출부가 에러 텍스트로 degrade 한다.
        브라우저 조작은 페이지 로드를 포함하므로 타임아웃을 넉넉히 잡는다.
        """
        p = str(path or "").strip()
        if not p or not tool:
            return None
        try:
            import httpx

            url = f"{self._base}{p if p.startswith('/') else '/' + p}"
            with httpx.Client(timeout=max(self._timeout, 300.0), verify=self._verify) as client:
                resp = client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"server": str(server or ""), "tool": str(tool), "args": dict(args or {})},
                )
                resp.raise_for_status()
                data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None

    # ── workflow-self (자기진화 — 그래프는 서버 자산, 편집은 서버 RPC) ──────
    def forged_tool_store(self, path: str) -> Optional[Any]:
        """서버 DB 의 forged-tool **스펙 저장소**를 원격 store 로 연다.

        스펙만 서버로 간다 — 스크립트 실행은 로컬이다(스크립트가 동기화되는
        workspace 안에 있으므로). 메모리와 같은 '상태는 서버, 실행은 여기' 규약.
        """
        p = str(path or "").strip()
        if not p:
            return None
        try:
            from xgen_agent_runtime.host.remote_forged_store import RemoteForgedToolStore

            return RemoteForgedToolStore(
                base_url=self._base,
                token=self._token,
                path=p,
                timeout_s=self._timeout,
                verify=self._verify,
            )
        except Exception:  # noqa: BLE001
            return None

    def workflow_self(self, path: str, input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """서버의 실물 WorkflowSelfTool 실행을 위임한다(동기, blocking httpx).

        ``path`` 는 컨텍스트 메타가 실어 준 RPC 경로(버전 정합 — 런타임이 경로를
        지어내지 않는다). 반환 ``{"ok","content","is_error","metadata"}`` 또는
        네트워크/인가 실패 시 None — 호출부(프록시 도구)가 에러 텍스트로 degrade.
        """
        p = str(path or "").strip()
        if not p:
            return None
        try:
            import httpx

            url = f"{self._base}{p if p.startswith('/') else '/' + p}"
            with httpx.Client(timeout=max(self._timeout, 300.0), verify=self._verify) as client:
                resp = client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"input": dict(input or {})},
                )
                resp.raise_for_status()
                data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception:  # noqa: BLE001
            return None
