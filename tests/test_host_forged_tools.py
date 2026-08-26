"""자가제작 도구(forged tools) — 엔진 계약 + 로컬 배선.

핵심 주장 하나: **엔진은 하나이고 저장소만 갈린다.** 스펙은 계정 자산이라 서버 DB
가 원본이고(로컬은 같은 인터페이스의 RPC 프록시), 스크립트 실행지는 다른 도구와
똑같이 ``ToolContext.sandbox`` 가 정한다. 그래서 웹에서 만든 도구가 로컬에서 그대로
살아나고, 그 반대도 같다.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import pytest

from xgen_agent_runtime.host.forged_tools import (
    ForgedToolSpec,
    forged_tool_instances,
    register_forged_tools,
)
from xgen_agent_runtime.stages.s10_tool import RegistryRouter
from xgen_agent_runtime.tools import ToolRegistry
from xgen_agent_runtime.tools.base import ToolContext


class _MemStore:
    """ForgedToolSpecStore 의 최소 구현 — 서버 DB 대역."""

    def __init__(self) -> None:
        self.specs: Dict[str, ForgedToolSpec] = {}
        self.calls: List[tuple] = []

    def list(self) -> List[ForgedToolSpec]:
        return list(self.specs.values())

    def get(self, name: str) -> Optional[ForgedToolSpec]:
        return self.specs.get(name)

    def save(self, spec: ForgedToolSpec) -> ForgedToolSpec:
        self.specs[spec.name] = spec
        return spec

    def delete(self, name: str) -> bool:
        return self.specs.pop(name, None) is not None

    def record_call(self, name: str, *, error: Optional[str] = None) -> None:
        self.calls.append((name, error))

    def mark_tested(self, name: str, *, ok: bool, error: Optional[str] = None) -> None:
        if name in self.specs:
            self.specs[name].verified = ok


def _ws(tmp_path) -> str:
    return str(tmp_path)


def _ctx(ws: str) -> ToolContext:
    return ToolContext(session_id="i1", working_dir=ws, allowed_paths=[ws], sandbox=None)


def _write_script(ws: str, name: str = "greet.py") -> str:
    path = os.path.join(ws, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "import json, sys\n"
            "i = json.load(sys.stdin)\n"
            "print(json.dumps({'msg': 'hello ' + i.get('who', '?')}))\n"
        )
    return name


def test_forge_then_restore_then_run(tmp_path) -> None:
    """제작 → (저장소) → 새 세션 복원 → 실행. 이게 이 기능의 전부다."""
    ws, store = _ws(tmp_path), _MemStore()
    entry = _write_script(ws)

    reg = ToolRegistry()
    summary = register_forged_tools(
        reg, workflow_id="wf1", workspace_dir=ws, store=store, core=True
    )
    assert set(summary["authoring"]) == {
        "ForgeTool", "ListForgedTools", "DeleteForgedTool", "PythonEnv"
    }
    assert summary["restored"] == []

    res = asyncio.run(RegistryRouter(reg).route("ForgeTool", {
        "name": "Greet", "description": "인사한다", "entrypoint": entry,
        "input_schema": {
            "type": "object", "properties": {"who": {"type": "string"}}, "required": ["who"],
        },
        "test_input": {"who": "xgen"},
    }, _ctx(ws)))
    assert not res.is_error, res.content
    # 등록 게이트: 실행 테스트를 통과해야 verified 로 저장된다.
    assert store.specs["Greet"].verified is True

    # 새 세션 — 저장소만으로 도구가 되살아난다.
    reg2 = ToolRegistry()
    restored = register_forged_tools(
        reg2, workflow_id="wf1", workspace_dir=ws, store=store, core=True
    )
    assert restored["restored"] == ["Greet"]
    out = asyncio.run(RegistryRouter(reg2).route("Greet", {"who": "hrjang"}, _ctx(ws)))
    assert out.is_error is False and out.content == {"msg": "hello hrjang"}


def test_unverified_tools_are_never_advertised(tmp_path) -> None:
    """미검증 도구는 노출하지 않는다 — 깨진 도구가 호출돼 실패하는 걸 원천 차단."""
    ws, store = _ws(tmp_path), _MemStore()
    _write_script(ws)
    store.save(ForgedToolSpec(
        name="Broken", description="d", entrypoint="greet.py", verified=False,
    ))
    reg = ToolRegistry()
    register_forged_tools(reg, workflow_id="wf1", workspace_dir=ws, store=store, core=True)
    assert "Broken" not in reg.list_names()
    assert "Broken" not in {t.name for t in forged_tool_instances(
        workflow_id="wf1", workspace_dir=ws, store=store,
    )}


def test_missing_script_is_not_advertised_locally(tmp_path) -> None:
    """로컬은 실행지가 이 PC 라 스크립트 존재를 **여기서** 확인할 수 있다.

    (러너에 있을 땐 확인할 수 없어 ``sandboxed=True`` 로 통과시킨다 — 그걸
    '없음'으로 읽으면 모든 도구가 조용히 사라진다.)
    """
    ws, store = _ws(tmp_path), _MemStore()
    store.save(ForgedToolSpec(
        name="Gone", description="d", entrypoint="deleted.py", verified=True,
    ))
    reg = ToolRegistry()
    register_forged_tools(reg, workflow_id="wf1", workspace_dir=ws, store=store, core=True)
    assert "Gone" not in reg.list_names()

    reg2 = ToolRegistry()
    register_forged_tools(
        reg2, workflow_id="wf1", workspace_dir=ws, store=store, core=True, sandboxed=True
    )
    assert "Gone" in reg2.list_names()


def test_script_runs_in_the_workspace_not_the_cwd(tmp_path) -> None:
    """실행 위치는 workspace 다 — 여기가 어긋나면 파일이 엉뚱한 곳에 생긴다."""
    ws, store = _ws(tmp_path), _MemStore()
    with open(os.path.join(ws, "where.py"), "w", encoding="utf-8") as fh:
        fh.write("import json, os, sys\nsys.stdin.read()\nprint(json.dumps({'cwd': os.getcwd()}))\n")
    store.save(ForgedToolSpec(
        name="Where", description="d", entrypoint="where.py", verified=True,
        input_schema={"type": "object", "properties": {}},
    ))
    reg = ToolRegistry()
    register_forged_tools(reg, workflow_id="wf1", workspace_dir=ws, store=store, core=True)
    out = asyncio.run(RegistryRouter(reg).route("Where", {}, _ctx(ws)))
    assert os.path.realpath(out.content["cwd"]) == os.path.realpath(ws)


def test_failing_script_reports_stderr_and_records(tmp_path) -> None:
    """실패는 stderr 를 달고 돌아오고 통계에 남는다 — 조용히 성공으로 위장하지 않는다."""
    ws, store = _ws(tmp_path), _MemStore()
    with open(os.path.join(ws, "boom.py"), "w", encoding="utf-8") as fh:
        fh.write("import sys\nsys.stderr.write('kaboom\\n')\nsys.exit(3)\n")
    store.save(ForgedToolSpec(
        name="Boom", description="d", entrypoint="boom.py", verified=True,
        input_schema={"type": "object", "properties": {}},
    ))
    reg = ToolRegistry()
    register_forged_tools(reg, workflow_id="wf1", workspace_dir=ws, store=store, core=True)
    out = asyncio.run(RegistryRouter(reg).route("Boom", {}, _ctx(ws)))
    assert out.is_error and "kaboom" in out.content
    assert store.calls and store.calls[-1][0] == "Boom" and store.calls[-1][1]


# ── 로컬 호스트 배선 ────────────────────────────────────────────────────


class _StoreBridge:
    def __init__(self, store: Any) -> None:
        self._store = store
        self.asked: List[str] = []

    base_url = "https://xgen.example"
    token = "t"

    def forged_tool_store(self, path: str) -> Any:
        self.asked.append(path)
        return self._store


def _local_host(ws: str, *, ctx: Dict[str, Any], bridge: Any = None):
    from xgen_agent_runtime.host.local_host import LocalHostServices

    return LocalHostServices(ws, context=ctx, server_bridge=bridge)


def test_local_host_wires_forged_tools_from_server_store(tmp_path) -> None:
    """로컬 턴도 같은 엔진 — 스펙은 서버 store, 실행은 이 PC."""
    ws, store = _ws(tmp_path), _MemStore()
    _write_script(ws)
    store.save(ForgedToolSpec(
        name="Greet", description="인사", entrypoint="greet.py", verified=True,
        input_schema={"type": "object", "properties": {"who": {"type": "string"}}},
    ))
    bridge = _StoreBridge(store)
    host = _local_host(ws, ctx={"forged_tools": {"enabled": True, "path": "/rpc"}}, bridge=bridge)

    reg = ToolRegistry()
    host.register_forged_tools(reg, workflow_id="wf1", workspace_dir=ws, core=True, sandboxed=False)
    assert bridge.asked == ["/rpc"]
    assert {"Greet", "ForgeTool", "PythonEnv"} <= set(reg.list_names())
    out = asyncio.run(RegistryRouter(reg).route("Greet", {"who": "local"}, _ctx(ws)))
    assert out.content == {"msg": "hello local"}


@pytest.mark.parametrize(
    "ctx, bridge_present",
    [
        ({}, True),                                                   # 메타 없음(구서버)
        ({"forged_tools": {"enabled": False}}, True),                 # 관리자 kill-switch
        ({"forged_tools": {"enabled": True, "path": ""}}, True),      # 경로 없음
        ({"forged_tools": {"enabled": True, "path": "/rpc"}}, False),  # 브릿지 없음(미로그인)
    ],
)
def test_local_host_stays_silent_without_a_store(tmp_path, ctx, bridge_present) -> None:
    """저장소가 없으면 **제작 도구도** 띄우지 않는다.

    띄워 놓고 저장이 안 되면 에이전트는 도구를 만들었다고 믿고 다음 턴에 잃는다 —
    "도구가 없다"보다 "만들었다는데 없다"가 훨씬 나쁘다.
    """
    ws = _ws(tmp_path)
    host = _local_host(ws, ctx=ctx, bridge=_StoreBridge(_MemStore()) if bridge_present else None)
    reg = ToolRegistry()
    host.register_forged_tools(reg, workflow_id="wf1", workspace_dir=ws, core=True, sandboxed=False)
    assert reg.list_names() == []


def test_remote_store_save_failure_is_loud(tmp_path) -> None:
    """저장 실패를 삼키면 안 된다 — 성공했다고 답하면 다음 턴에 도구가 없다."""
    from xgen_agent_runtime.host.remote_forged_store import (
        RemoteForgedToolStore,
        RemoteForgedToolStoreError,
    )

    store = RemoteForgedToolStore(
        base_url="http://127.0.0.1:1", token="t", path="/rpc", timeout_s=0.2
    )
    # 읽기는 조용히 degrade (도구가 잠깐 안 보이는 것뿐)
    assert store.list() == []
    assert store.get("x") is None
    assert store.delete("x") is False
    store.record_call("x")  # 통계 실패는 무시 — raise 하지 않는다
    with pytest.raises(RemoteForgedToolStoreError):
        store.save(ForgedToolSpec(name="X", description="d", entrypoint="a.py"))


def test_remote_store_speaks_the_server_contract() -> None:
    """RPC 봉투가 서버 엔드포인트(geny_memory.forged_tool_store_rpc)와 같은 모양인가."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from xgen_agent_runtime.host.remote_forged_store import RemoteForgedToolStore

    seen: List[Dict[str, Any]] = []

    class _H(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            spec = ForgedToolSpec(name="A", description="d", entrypoint="a.py").to_dict()
            raw = json.dumps({"ok": True, "specs": [spec], "spec": spec, "removed": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a: Any) -> None:
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        s = RemoteForgedToolStore(
            base_url=f"http://127.0.0.1:{httpd.server_address[1]}", token="tok", path="/rpc"
        )
        assert [t.name for t in s.list()] == ["A"]
        assert s.get("A").name == "A"
        assert s.save(ForgedToolSpec(name="A", description="d", entrypoint="a.py")).name == "A"
        assert s.delete("A") is True
        s.mark_tested("A", ok=True)
    finally:
        httpd.shutdown()
    assert [b["op"] for b in seen] == ["list", "get", "save", "delete", "mark_tested"]
    assert seen[2]["spec"]["name"] == "A"
    assert seen[4]["ok"] is True
