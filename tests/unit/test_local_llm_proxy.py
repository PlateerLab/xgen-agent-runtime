"""로컬 실행에서 내부(vLLM 등) LLM = 서버 프록시 — LocalHostServices 가 context.llm_proxy
마커를 보고 base_url/api_key 를 **서버 브릿지 경유**로 재작성하는지 검증.

내부 서빙 base_url 은 커넥터 PC 에서 도달 불가라, 서버는 그 URL·모델키를 싣지 않고
마커만 싣는다. 런타임은 서버 브릿지(base_url·토큰 보유)로 LLM 호출을 프록시한다:
    connector 런타임 → xgen-server /llm-proxy → 내부 provider
프록시 base_url 은 OpenAI 호환이라 런타임 OpenAI 클라이언트가 그대로 쓴다(변경 0).
"""
from __future__ import annotations

from xgen_agent_runtime.host.local_host import LocalHostServices


class _FakeBridge:
    def __init__(self, base="https://xgen.example.com", token="user-sess-tok"):
        self._base = base.rstrip("/")
        self._tok = token

    @property
    def base_url(self) -> str:
        return self._base

    @property
    def token(self) -> str:
        return self._tok


_MARKER = {"provider": "vllm", "path": "/api/agentflow/geny-memory/wf7/llm-proxy/v1"}


def test_proxy_rewrites_base_url_and_api_key(tmp_path):
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={"llm_proxy": _MARKER},
        server_bridge=_FakeBridge(),
    )
    # base_url = 브릿지 서버 URL + 프록시 경로 (OpenAI 클라이언트가 /chat/completions 를 append)
    assert (
        host.resolve_base_url("vllm", {})
        == "https://xgen.example.com/api/agentflow/geny-memory/wf7/llm-proxy/v1"
    )
    # api_key = 브릿지 토큰(사용자 세션) — 프록시 _authorize 가 검증, 실키는 서버에서 주입
    assert host.resolve_api_key("vllm", {}) == "user-sess-tok"


def test_proxy_only_for_marked_provider(tmp_path):
    # 마커 provider 와 다른 provider 는 프록시하지 않는다(종전 base_urls/api_keys 경로).
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={
            "llm_proxy": _MARKER,
            "base_urls": {"openai": "https://api.openai.com/v1"},
            "api_keys": {"openai": "sk-pub"},
        },
        server_bridge=_FakeBridge(),
    )
    assert host.resolve_base_url("openai", {}) == "https://api.openai.com/v1"
    assert host.resolve_api_key("openai", {}) == "sk-pub"


def test_no_proxy_without_bridge(tmp_path):
    # 브릿지 없음(오프라인) → 마커 있어도 프록시 불가 → 종전 경로(base_urls 없으면 None).
    host = LocalHostServices(str(tmp_path / "ws"), context={"llm_proxy": _MARKER})
    assert host.resolve_base_url("vllm", {}) is None
    assert host.resolve_api_key("vllm", {}) == ""


def test_node_explicit_still_wins(tmp_path):
    # 노드 명시 base_url/api_key(params)는 프록시보다 우선(사용자가 직접 지정한 값).
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={"llm_proxy": _MARKER},
        server_bridge=_FakeBridge(),
    )
    params = {"base_url": "http://my-own:8000/v1", "api_key": "my-key"}
    assert host.resolve_base_url("vllm", params) == "http://my-own:8000/v1"
    assert host.resolve_api_key("vllm", params) == "my-key"


def test_proxy_needs_token(tmp_path):
    # 토큰 없는 브릿지 → 프록시 인증 불가 → 프록시 안 함(방어).
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={"llm_proxy": _MARKER, "base_urls": {"vllm": "http://internal:8000/v1"}},
        server_bridge=_FakeBridge(token=""),
    )
    # base_url 재작성도 토큰 없으면 안 함 → context.base_urls 폴백
    assert host.resolve_base_url("vllm", {}) == "http://internal:8000/v1"
