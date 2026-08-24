"""로컬 실행에서 RAG = 서버 호출 — LocalHostServices.rag_context_builder 가 search_params
만 갖고 서버 RAG RPC(ServerBridge.rag_search)로 위임하는지 검증.

RAG 서비스/컬렉션은 서버 자산이라 로컬로 오지 않는다(rag_service=None). 로컬은 search_params
+ workflow_id 로 서버를 호출해 [DOC_n] 블록을 받는다.
"""
from __future__ import annotations

from xgen_agent_runtime.host.local_host import LocalHostServices


class _FakeBridge:
    def __init__(self):
        self.calls = []

    def rag_search(self, workflow_id, text, search_params):
        self.calls.append((workflow_id, text, search_params))
        return "[DOC_1] result"


def test_rag_context_builder_delegates_to_server(tmp_path):
    bridge = _FakeBridge()
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={"workflow_id": "wf9"},
        server_bridge=bridge,
    )
    item = {"rag_service": None, "search_params": {"collection_name": "c1", "top_k": 5}}
    block = host.rag_context_builder("질문", item)
    assert block == "[DOC_1] result"
    assert bridge.calls == [("wf9", "질문", {"collection_name": "c1", "top_k": 5})]


def test_rag_context_builder_none_without_bridge_or_params(tmp_path):
    # 브릿지 없음 → None
    h1 = LocalHostServices(str(tmp_path / "w1"), context={"workflow_id": "wf"})
    assert h1.rag_context_builder("q", {"search_params": {"x": 1}, "rag_service": None}) is None
    # search_params 없음 → None
    h2 = LocalHostServices(str(tmp_path / "w2"), context={"workflow_id": "wf"}, server_bridge=_FakeBridge())
    assert h2.rag_context_builder("q", {"rag_service": None}) is None
    # workflow_id 없음 → None
    h3 = LocalHostServices(str(tmp_path / "w3"), context={}, server_bridge=_FakeBridge())
    assert h3.rag_context_builder("q", {"search_params": {"x": 1}, "rag_service": None}) is None


def test_server_bridge_rag_search_posts(monkeypatch):
    from xgen_agent_runtime.host.server_bridge import ServerBridge

    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True, "block": "[DOC_1] X"}

    class _Client:
        def __init__(self, **kw):
            seen["client_kw"] = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            seen["url"] = url
            seen["headers"] = headers
            seen["json"] = json
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "Client", _Client)
    b = ServerBridge("https://s", "tok", verify=False)
    out = b.rag_search("wf1", "질문", {"collection_name": "c"})
    assert out == "[DOC_1] X"
    assert seen["url"] == "https://s/api/agentflow/geny-memory/wf1/rag-search"
    assert seen["headers"]["Authorization"] == "Bearer tok"
    assert seen["json"] == {"text": "질문", "search_params": {"collection_name": "c"}}
    assert seen["client_kw"]["verify"] is False
    # ok:false → None
    class _RespBad(_Resp):
        def json(self):
            return {"ok": False, "error": "x"}

    monkeypatch.setattr(_Client, "post", lambda self, url, headers=None, json=None: _RespBad())
    assert b.rag_search("wf1", "q", {"c": 1}) is None
