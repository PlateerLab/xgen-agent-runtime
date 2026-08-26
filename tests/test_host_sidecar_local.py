"""데스크톱 호스트 계약 — sidecar v2 프로토콜 + LocalHostServices 정합성.

커넥터(Node)가 의존하는 표면을 고정한다:
* sidecar: dict 스트림 항목(agent_event/canvas_command)은 텍스트로 강등되지 않고
  전용 ``tool`` / ``canvas_command`` 이벤트로 올라간다; 데몬 모드는 id 상관·취소·ping.
* LocalHostServices: HostServices 프로토콜 전 메서드 존재 + 서명 일치, CLI 네이티브
  도구 허용(allow_local_tools), CLI 홈 격리(CODEX_HOME/CLAUDE_CONFIG_DIR), 중앙
  codex 자격증명 물질화, 서버와 같은 built-in 패밀리 kill-switch.
* CLI env 화이트리스트: Windows 부트스트랩 변수·프록시·CODEX_HOME 통과.
"""

from __future__ import annotations

import inspect
import io
import json
import sys
import threading
import time
from typing import Any, Dict, Iterator, List

import pytest

from xgen_agent_runtime.host import sidecar
from xgen_agent_runtime.host.host import HostServices
from xgen_agent_runtime.host.local_host import LocalHostServices


# ── sidecar: 이벤트 정규화 ────────────────────────────────────────────


def test_normalize_event_keeps_tool_events_structured() -> None:
    tool = {"type": "agent_event", "data": {"type": "tool_call", "tool_name": "Bash"}}
    assert sidecar._normalize_event("hi") == {"type": "chunk", "text": "hi"}
    assert sidecar._normalize_event(tool) == {"type": "tool", "data": tool["data"]}
    canvas = {"type": "canvas_command", "data": {"op": "add"}}
    assert sidecar._normalize_event(canvas) == {"type": "canvas_command", "data": {"op": "add"}}


class _FakeExecutor:
    """AgentTurnExecutor 대역 — str 과 dict 가 섞인 스트림."""

    items: List[Any] = []
    seen_kwargs: Dict[str, Any] = {}

    def run(self, host: Any, **kwargs: Any) -> Iterator[Any]:
        _FakeExecutor.seen_kwargs = dict(kwargs)
        for it in _FakeExecutor.items:
            if callable(it):
                it()
                continue
            yield it


@pytest.fixture
def fake_executor(monkeypatch, tmp_path):
    import xgen_agent_runtime.host.turn_executor as te

    monkeypatch.setattr(te, "AgentTurnExecutor", _FakeExecutor)
    _FakeExecutor.items = []
    return {"ws": str(tmp_path / "ws")}


def _req(ws: str, **opts: Any) -> Dict[str, Any]:
    return {
        "workspace_dir": ws,
        "provider": "openai",
        "text": "hello",
        "context": {"api_keys": {"openai": "sk"}, "settings": {}},
        "options": {"interaction_id": "i1", "workflow_id": "wf", **opts},
    }


def test_run_turn_request_emits_started_tool_and_done(fake_executor) -> None:
    _FakeExecutor.items = [
        "안",
        {"type": "agent_event", "data": {"type": "tool_call", "tool_name": "Bash", "tool_input": "{}"}},
        {"type": "agent_event", "data": {"type": "tool_result", "tool_name": "Bash", "result": "ok"}},
        "녕",
    ]
    events = list(sidecar.run_turn_request(_req(fake_executor["ws"])))
    kinds = [e["type"] for e in events]
    assert kinds == ["started", "chunk", "tool", "tool", "chunk", "done"]
    assert events[0]["surface"] == "connector_local"
    assert events[2]["data"]["tool_name"] == "Bash"
    # done 텍스트에는 dict repr 이 섞이지 않는다 (v1 회귀 금지)
    assert events[-1]["text"] == "안녕"
    # 옵션이 run() kwargs 로 평탄화되고 streaming 기본 True
    assert _FakeExecutor.seen_kwargs["streaming"] is True
    assert _FakeExecutor.seen_kwargs["interaction_id"] == "i1"


def test_run_turn_request_reports_missing_workspace() -> None:
    events = list(sidecar.run_turn_request({"provider": "openai", "text": "x"}))
    assert events == [{"type": "error", "message": "workspace_dir 가 없습니다."}]


def test_run_turn_request_cancel_check_stops_stream(fake_executor) -> None:
    flag = {"v": False}

    def _flip() -> None:
        flag["v"] = True

    _FakeExecutor.items = ["a", _flip, "b", "c"]
    events = list(sidecar.run_turn_request(_req(fake_executor["ws"]), cancel_check=lambda: flag["v"]))
    kinds = [e["type"] for e in events]
    assert kinds == ["started", "chunk", "cancelled"]


# ── sidecar: 데몬 모드 ───────────────────────────────────────────────


class _PipeIn:
    """stdin 대역 — 라인을 밀어 넣을 수 있는 이터러블."""

    def __init__(self) -> None:
        self._q: List[str] = []
        self._cv = threading.Condition()
        self._closed = False

    def push(self, obj: Any) -> None:
        with self._cv:
            self._q.append(json.dumps(obj) + "\n")
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def __iter__(self):
        while True:
            with self._cv:
                while not self._q and not self._closed:
                    self._cv.wait(timeout=0.05)
                if self._q:
                    yield self._q.pop(0)
                    continue
                if self._closed:
                    return


def _events_of(out: io.StringIO) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def test_daemon_ping_turn_and_shutdown(fake_executor) -> None:
    _FakeExecutor.items = ["x", "y"]
    inp = _PipeIn()
    out = io.StringIO()
    daemon = sidecar.SidecarDaemon(inp, out)
    th = threading.Thread(target=daemon.serve, daemon=True)
    th.start()
    inp.push({"op": "ping", "id": "p1"})
    inp.push({"op": "turn", "id": "t1", **_req(fake_executor["ws"])})
    deadline = time.time() + 5
    while time.time() < deadline:
        evs = _events_of(out)
        if any(e.get("id") == "t1" and e["type"] == "done" for e in evs):
            break
        time.sleep(0.02)
    inp.push({"op": "shutdown"})
    inp.close()
    th.join(timeout=5)
    evs = _events_of(out)
    assert evs[0]["type"] == "ready" and evs[0]["protocol"] == sidecar.SIDECAR_PROTOCOL_VERSION
    pong = next(e for e in evs if e["type"] == "pong")
    assert pong["id"] == "p1" and pong["runtime_version"]
    t1 = [e for e in evs if e.get("id") == "t1"]
    assert [e["type"] for e in t1] == ["started", "chunk", "chunk", "done"]
    assert t1[-1]["text"] == "xy"


def test_daemon_cancel_emits_cancelled(fake_executor) -> None:
    gate = threading.Event()

    def _wait() -> None:
        gate.wait(timeout=5)

    _FakeExecutor.items = ["a", _wait, "b"]
    inp = _PipeIn()
    out = io.StringIO()
    daemon = sidecar.SidecarDaemon(inp, out)
    th = threading.Thread(target=daemon.serve, daemon=True)
    th.start()
    inp.push({"op": "turn", "id": "t2", **_req(fake_executor["ws"])})
    deadline = time.time() + 5
    while time.time() < deadline and not any(
        e.get("id") == "t2" and e["type"] == "chunk" for e in _events_of(out)
    ):
        time.sleep(0.02)
    inp.push({"op": "cancel", "id": "t2"})
    time.sleep(0.1)
    gate.set()
    deadline = time.time() + 5
    while time.time() < deadline and not any(
        e.get("id") == "t2" and e["type"] in ("cancelled", "done") for e in _events_of(out)
    ):
        time.sleep(0.02)
    inp.push({"op": "shutdown"})
    inp.close()
    th.join(timeout=5)
    t2 = [e["type"] for e in _events_of(out) if e.get("id") == "t2"]
    assert t2 == ["started", "chunk", "cancelled"]


def test_daemon_unknown_op_and_bad_json(fake_executor) -> None:
    inp = _PipeIn()
    out = io.StringIO()
    daemon = sidecar.SidecarDaemon(inp, out)
    th = threading.Thread(target=daemon.serve, daemon=True)
    th.start()
    inp._q.append("not json\n")
    inp.push({"op": "bogus", "id": "z"})
    time.sleep(0.2)
    inp.push({"op": "shutdown"})
    inp.close()
    th.join(timeout=5)
    errs = [e for e in _events_of(out) if e["type"] == "error"]
    assert any("bad command" in e["message"] for e in errs)
    assert any("unknown op" in e["message"] for e in errs)


# ── LocalHostServices: 프로토콜 정합성 ───────────────────────────────


def test_local_host_satisfies_host_services_protocol(tmp_path) -> None:
    host = LocalHostServices(str(tmp_path / "ws"))
    assert isinstance(host, HostServices)
    # 서명 동일성 — 인자 이름 순서까지(무발산 계약).
    for name, proto_fn in inspect.getmembers(HostServices, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        impl = getattr(LocalHostServices, name, None)
        assert impl is not None, name
        p_params = [p for p in inspect.signature(proto_fn).parameters if p != "self"]
        i_params = [p for p in inspect.signature(impl).parameters if p != "self"]
        if i_params and i_params[-1] == "kwargs":
            continue  # **kwargs 수용 구현(finalize_turn/build_run_tool_context)
        assert p_params == i_params, f"{name}: {p_params} != {i_params}"


def test_local_host_settings_prefer_server_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_AUTH_MODE", "api_key")
    host = LocalHostServices(str(tmp_path / "ws"), context={"settings": {"CODEX_AUTH_MODE": "oauth"}})
    assert host.setting("CODEX_AUTH_MODE") == "oauth"
    assert host.setting("NOPE", "dflt") == "dflt"
    assert host.resolve_model("claude_code", {}) == ""
    host2 = LocalHostServices(str(tmp_path / "ws2"), context={"settings": {"CODEX_MODEL_DEFAULT": "gpt-5.3-codex"}})
    assert host2.resolve_model("codex", {}) == "gpt-5.3-codex"
    assert host2.resolve_model("codex", {"model": "m"}) == "m"


def test_local_host_builtin_families_mirror_server_killswitch(tmp_path) -> None:
    host = LocalHostServices(str(tmp_path / "ws"))
    assert host.builtin_families() == ["web", "documents", "browser", "workflow", "filesystem", "shell"]
    off = LocalHostServices(
        str(tmp_path / "ws"),
        context={"settings": {"GENY_TOOLS_BROWSER_ENABLED": "false", "GENY_TOOLS_SHELL_ENABLED": "0"}},
    )
    assert "browser" not in off.builtin_families() and "shell" not in off.builtin_families()
    assert "filesystem" in off.builtin_families()


def test_local_host_agent_vault_root_outside_workspace(tmp_path) -> None:
    ws = tmp_path / "agent-a"
    host = LocalHostServices(str(ws))
    root = host.agent_vault_root("wf")
    assert not root.startswith(str(ws) + "/")
    assert root.startswith(str(tmp_path))


def test_local_host_codex_cli_runtime_isolates_home_and_materializes(tmp_path, monkeypatch) -> None:
    import xgen_agent_runtime.host.runner as runner

    captured: Dict[str, Any] = {}

    def _fake_codex(**kw: Any) -> str:
        captured.update(kw)
        return "client"

    monkeypatch.setattr(runner, "build_codex_cli_client", _fake_codex)
    codex_home = tmp_path / "codex-home"
    cred = json.dumps({"tokens": {"access_token": "a"}})
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={
            "api_keys": {"openai": "sk"},
            "settings": {
                "CODEX_ENABLED": "true",
                "CODEX_AUTH_MODE": "oauth",
                "CODEX_CREDENTIALS_JSON": cred,
                "CODEX_BINARY_PATH": "/opt/codex",
                "CODEX_TIMEOUT_S": "120",
                "XGEN_LOCAL_CODEX_HOME": str(codex_home),
            },
        },
    )
    client, cleanup = host.build_cli_runtime("codex", {})
    assert client == "client" and cleanup is None
    assert captured["auth_mode"] == "oauth" and captured["api_key"] == ""  # 구독 모드는 키 미전달
    assert captured["env_extras"] == {"CODEX_HOME": str(codex_home)}
    assert captured["timeout_s"] == 120.0 and captured["binary_path"] == "/opt/codex"
    assert (codex_home / "auth.json").read_text() == cred
    # api_key 모드: 키 전달, 물질화 없음
    host2 = LocalHostServices(
        str(tmp_path / "ws"),
        context={"api_keys": {"openai": "sk"}, "settings": {"CODEX_ENABLED": "1", "CODEX_AUTH_MODE": "api_key"}},
    )
    host2.build_cli_runtime("codex", {})
    assert captured["auth_mode"] == "api_key" and captured["api_key"] == "sk"
    assert captured["env_extras"] is None


def test_local_host_claude_cli_runtime_blocks_natives_and_isolates(tmp_path, monkeypatch) -> None:
    import xgen_agent_runtime.host.runner as runner

    captured: Dict[str, Any] = {}

    def _fake_claude(**kw: Any) -> str:
        captured.update(kw)
        return "client"

    monkeypatch.setattr(runner, "build_cli_client", _fake_claude)
    claude_home = tmp_path / "claude-home"
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={
            "api_keys": {"anthropic": "ak"},
            "settings": {
                "CLAUDE_CODE_ENABLED": "true",
                "CLAUDE_CODE_AUTH_MODE": "setup_token",
                "CLAUDE_CODE_OAUTH_TOKEN": "tok",
                "CLAUDE_CODE_BINARY_PATH": "/opt/claude",
                "CLAUDE_CODE_MAX_BUDGET_USD": "2.5",
                "XGEN_LOCAL_CLAUDE_CONFIG_DIR": str(claude_home),
            },
        },
    )
    host.build_cli_runtime("claude_code", {})
    # 네이티브는 전면 차단 — 같은 능력은 루프백 MCP 가 우리 런타임 도구로 준다
    # (registry 스태시가 없는 이 호출은 mcp_config 도 None: 도구 0개가 정직한 결과).
    assert captured["allow_local_tools"] is False
    assert captured["auth_mode"] == "setup_token" and captured["oauth_token"] == "tok"
    assert captured["api_key"] == ""
    assert captured["extra_env"] == {"CLAUDE_CONFIG_DIR": str(claude_home)}
    assert captured["max_budget_usd"] == 2.5
    assert captured["mcp_config"] is None
    assert claude_home.is_dir()


# ── CLI env 화이트리스트 ────────────────────────────────────────────


def test_cli_env_whitelist_passes_cli_homes_and_proxies(monkeypatch) -> None:
    from xgen_agent_runtime.llm_client import _cli_runtime as rt

    parent = {
        "HOME": "/h", "PATH": "/bin", "CODEX_HOME": "/iso/codex", "CLAUDE_CONFIG_DIR": "/iso/claude",
        "HTTPS_PROXY": "http://p:3128", "NO_PROXY": "localhost", "SECRET": "x",
        "OPENAI_API_KEY": "leak",
    }
    env = rt.scrub_env(parent, whitelist=rt._ENV_WHITELIST_POSIX, extras={"X": "1"})
    assert env["CODEX_HOME"] == "/iso/codex" and env["CLAUDE_CONFIG_DIR"] == "/iso/claude"
    assert env["HTTPS_PROXY"] == "http://p:3128" and env["X"] == "1"
    assert "SECRET" not in env and "OPENAI_API_KEY" not in env


def test_cli_env_whitelist_windows_bootstrap_vars(monkeypatch) -> None:
    from xgen_agent_runtime.llm_client import _cli_runtime as rt

    for v in ("USERPROFILE", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT", "COMSPEC", "PATHEXT", "TEMP", "TMP"):
        assert v in rt._ENV_WHITELIST_WINDOWS
    assert rt._ENV_WHITELIST_POSIX <= rt._ENV_WHITELIST_WINDOWS
    # Windows 는 변수명 대소문자 무시 — 부모 철자를 보존하면서 매칭한다.
    monkeypatch.setattr(rt.sys, "platform", "win32")
    env = rt.scrub_env({"SystemRoot": "C:\\Windows", "Path": "C:\\bin", "Secret": "x"},
                       whitelist=rt._ENV_WHITELIST_WINDOWS)
    assert env == {"SystemRoot": "C:\\Windows", "Path": "C:\\bin"}
    if sys.platform != "win32":
        assert rt.DEFAULT_ENV_WHITELIST == rt._ENV_WHITELIST_POSIX


def test_local_host_cli_gates_and_hydrate_tristate(tmp_path, monkeypatch) -> None:
    host = LocalHostServices(str(tmp_path / "ws"), context={"settings": {"CODEX_ENABLED": "false"}})
    with pytest.raises(ValueError, match="CODEX_ENABLED"):
        host.build_cli_runtime("codex", {})
    host2 = LocalHostServices(str(tmp_path / "ws"), context={"settings": {"CLAUDE_CODE_ENABLED": "0"}})
    with pytest.raises(ValueError, match="CLAUDE_CODE_ENABLED"):
        host2.build_cli_runtime("claude_code", {})
    assert host.hydrate_workspace("wf", str(tmp_path)) is None  # 해당 없음(경고 X)


def test_sidecar_turn_cancel_is_per_turn_hook(fake_executor) -> None:
    """cancel_check 는 kwargs 로 실행기에 전달되고(턴 단위), interaction 스코프 레지스트리는 건드리지 않는다."""
    from xgen_agent_runtime.host.cancel_context import is_cancelled

    _FakeExecutor.items = ["a"]
    list(sidecar.run_turn_request(_req(fake_executor["ws"]), cancel_check=lambda: False))
    assert callable(_FakeExecutor.seen_kwargs.get("cancel_check"))
    assert is_cancelled("i1") is False


def test_server_bridge_passes_tls_verify() -> None:
    from xgen_agent_runtime.host.server_bridge import ServerBridge

    p = ServerBridge("https://s", "t", verify=False).build_memory_provider("wf", "i")
    assert p is not None and p._verify is False
    p2 = ServerBridge("https://s", "t", verify="/ca.pem").build_memory_provider("wf", "i")
    assert p2._verify == "/ca.pem"


# ── sidecar: 메모리 오프라인 진단 notice ──────────────────────────────


def _req_with_server(ws: str, **opts: Any) -> Dict[str, Any]:
    r = _req(ws, **opts)
    r["server"] = {"url": "https://xgen.example", "token": "tok"}
    return r


def test_run_turn_request_emits_memory_offline_notice_on_bridge_failure(
    fake_executor, monkeypatch
) -> None:
    """브릿지 구성이 실패하면(메모리 기대됨) 첫 청크 이전에 notice 를 1회 낸다."""
    import xgen_agent_runtime.host.server_bridge as sb

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("tls handshake failed")

    monkeypatch.setattr(sb, "ServerBridge", _boom)
    _FakeExecutor.items = ["안", "녕"]
    events = list(sidecar.run_turn_request(_req_with_server(fake_executor["ws"])))
    kinds = [e["type"] for e in events]
    # started 다음, 첫 chunk 이전에 notice 가 정확히 한 번.
    assert kinds == ["started", "notice", "chunk", "chunk", "done"]
    notice = events[1]
    assert notice["data"]["code"] == "memory_offline"
    assert "무기억" in notice["data"]["message"]
    # 폴백이 아니다 — 로컬 실행은 계속돼 done 으로 닫힌다.
    assert events[-1]["type"] == "done" and events[-1]["text"] == "안녕"


def test_run_turn_request_no_notice_when_no_server(fake_executor) -> None:
    """server 미지정(의도적 오프라인)은 실패가 아니므로 notice 를 내지 않는다."""
    _FakeExecutor.items = ["x"]
    events = list(sidecar.run_turn_request(_req(fake_executor["ws"])))
    assert "notice" not in [e["type"] for e in events]


def test_run_turn_request_memory_offline_notice_nonstreaming(
    fake_executor, monkeypatch
) -> None:
    """비스트리밍 턴에서도 notice 는 done 이전에 방출된다."""
    import xgen_agent_runtime.host.server_bridge as sb

    monkeypatch.setattr(sb, "ServerBridge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))

    class _StrExec:
        def run(self, host: Any, **kwargs: Any) -> str:
            return "done-text"

    import xgen_agent_runtime.host.turn_executor as te

    monkeypatch.setattr(te, "AgentTurnExecutor", _StrExec)
    events = list(
        sidecar.run_turn_request(_req_with_server(fake_executor["ws"], streaming=False))
    )
    kinds = [e["type"] for e in events]
    assert kinds == ["started", "notice", "done"]
    assert events[1]["data"]["code"] == "memory_offline"


# ── LocalHostServices: 메모리 오프라인 신호 헬퍼 ──────────────────────


class _FakeBridge:
    """ServerBridge 대역 — build_memory_provider 결과를 주입."""

    def __init__(self, result: Any = object(), raises: bool = False) -> None:
        self._result = result
        self._raises = raises

    def build_memory_provider(self, workflow_id: str, interaction_id: str) -> Any:
        if self._raises:
            raise RuntimeError("rpc transport failed")
        return self._result


def test_local_host_marks_offline_when_provider_build_returns_none(tmp_path) -> None:
    """브릿지는 있으나 첫 provider 구성이 None(엔드포인트 부재 등) → 오프라인 신호."""
    host = LocalHostServices(str(tmp_path / "ws"), server_bridge=_FakeBridge(result=None))
    assert host.memory_offline() is False
    assert host.build_memory_provider("wf", "i") is None
    assert host.memory_offline() is True


def test_local_host_marks_offline_when_first_rpc_raises(tmp_path) -> None:
    """브릿지 첫 RPC 가 던지면 무기억 degrade + 오프라인 신호."""
    host = LocalHostServices(str(tmp_path / "ws"), server_bridge=_FakeBridge(raises=True))
    assert host.build_memory_provider("wf", "i") is None
    assert host.memory_offline() is True


def test_local_host_no_offline_signal_when_bridge_missing(tmp_path) -> None:
    """브릿지 미연결(서버 url 없음=의도적 오프라인)은 실패가 아니다 — 신호 안 섬."""
    host = LocalHostServices(str(tmp_path / "ws"), server_bridge=None)
    assert host.build_memory_provider("wf", "i") is None
    assert host.memory_offline() is False


def test_local_host_no_offline_when_memory_ok(tmp_path) -> None:
    """정상 provider 를 열면 신호가 서지 않는다."""
    prov = object()
    host = LocalHostServices(str(tmp_path / "ws"), server_bridge=_FakeBridge(result=prov))
    assert host.build_memory_provider("wf", "i") is prov
    assert host.memory_offline() is False


# ── LocalHostServices: 3.8.0 — 경로 격리·CLI 네이티브 사전 허용·이력 dict·플래그 ──


def test_local_host_run_tool_context_confines_allowed_paths(tmp_path) -> None:
    """allowed_paths 는 항상 [workspace, *extra_allowed] — None(무제한)이 되면 안 된다."""
    ws = tmp_path / "ws"
    host = LocalHostServices(str(ws))
    ctx = host.build_run_tool_context(interaction_id="i1", run_dir=str(ws), extras={"a": 1})
    assert ctx.allowed_paths == [str(ws)]
    assert ctx.working_dir == str(ws) and ctx.sandbox is None
    # ⚠ extras 다 — metadata 가 아니다. 도구는 ctx.extras 만 읽는다(docs api_key,
    # ssh servers, subagent_manager…). metadata 로 실으면 전부 조용히 미배선이 된다.
    assert ctx.extras["a"] == 1 and ctx.metadata == {}
    # 로컬은 호스트가 곧 실행 대상 — sandbox 없음이 설계라는 표식이 실린다.
    from xgen_agent_runtime.tools.base import HOST_IS_EXECUTION_TARGET

    assert ctx.extras[HOST_IS_EXECUTION_TARGET] is True
    extra = tmp_path / "cloud"
    ctx2 = host.build_run_tool_context(run_dir=str(ws), extra_allowed=[str(extra), str(extra), ""])
    assert ctx2.allowed_paths == [str(ws), str(extra)]
    # run_dir 미지정 → 워크스페이스
    assert host.build_run_tool_context().allowed_paths == [str(ws)]


def test_local_host_file_tools_reject_paths_outside_workspace(tmp_path) -> None:
    """호스트(sandbox=None) 경로에서 Read/Write 가 path guard 로 워크스페이스 밖을 거부한다."""
    import asyncio

    from xgen_agent_runtime.tools.built_in.read_tool import ReadTool
    from xgen_agent_runtime.tools.built_in.write_tool import WriteTool

    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret", encoding="utf-8")
    host = LocalHostServices(str(ws))
    ctx = host.build_run_tool_context(run_dir=str(ws))

    async def _go():
        w_in = await WriteTool().execute({"file_path": "note.md", "content": "hi"}, ctx)
        w_out = await WriteTool().execute({"file_path": str(outside / "x.txt"), "content": "no"}, ctx)
        w_trav = await WriteTool().execute({"file_path": "../outside/y.txt", "content": "no"}, ctx)
        r_in = await ReadTool().execute({"file_path": str(ws / "note.md")}, ctx)
        r_out = await ReadTool().execute({"file_path": str(outside / "secret.txt")}, ctx)
        return w_in, w_out, w_trav, r_in, r_out

    w_in, w_out, w_trav, r_in, r_out = asyncio.run(_go())
    assert not w_in.is_error and (ws / "note.md").read_text(encoding="utf-8") == "hi"
    assert w_out.is_error and "outside allowed" in w_out.content
    assert w_trav.is_error and not (outside / "y.txt").exists()
    assert not (outside / "x.txt").exists()
    assert not r_in.is_error and "hi" in r_in.content
    assert r_out.is_error and "top secret" not in r_out.content


def _registry_stash(ws: str) -> Dict[str, Any]:
    """turn_executor 가 CLI 백엔드에 넘기는 스태시(레지스트리 + run ctx)."""
    from xgen_agent_runtime.tools import ToolRegistry
    from xgen_agent_runtime.tools.base import ToolContext
    from xgen_agent_runtime.tools.built_in import get_builtin_tools

    reg = ToolRegistry()
    for name, cls in get_builtin_tools(features=["shell"]).items():
        reg.register(cls(), core=True)
    ctx = ToolContext(session_id="i1", working_dir=ws, allowed_paths=[ws], sandbox=None)
    return {"_tool_registry": reg, "_run_tool_context": ctx}


def test_local_host_claude_preallows_loopback_surface_only(tmp_path, monkeypatch) -> None:
    """--print 자동 거부 방지: 사전 허용되는 것은 **루프백 MCP 서버 하나**뿐이고
    CLI 네이티브는 전면 차단된다(같은 능력은 우리 런타임 도구가 준다)."""
    import xgen_agent_runtime.host.runner as runner
    from xgen_agent_runtime.host import local_host as lh

    captured: Dict[str, Any] = {}

    def _fake_claude(**kw: Any) -> str:
        captured.update(kw)
        return "client"

    monkeypatch.setattr(runner, "build_cli_client", _fake_claude)
    claude_home = tmp_path / "claude-home"
    ws = tmp_path / "ws"
    ws.mkdir()
    host = LocalHostServices(
        str(ws),
        context={
            "api_keys": {"anthropic": "ak"},
            "settings": {
                "CLAUDE_CODE_ENABLED": "true",
                "CLAUDE_CODE_AUTH_MODE": "api_key",
                "XGEN_LOCAL_CLAUDE_CONFIG_DIR": str(claude_home),
            },
        },
    )
    _client, cleanup = host.build_cli_runtime("claude_code", _registry_stash(str(ws)))
    try:
        settings_file = claude_home / lh.CLAUDE_LOCAL_SETTINGS_FILENAME
        assert captured["settings_path"] == str(settings_file)
        allow = json.loads(settings_file.read_text(encoding="utf-8"))["permissions"]["allow"]
        expect = f"mcp__{lh.CLI_MCP_SERVER_NAME}"
        assert allow == [expect] and tuple(captured["allow_tools"]) == (expect,)
        # 네이티브 전면 차단 — runner 가 카탈로그 전체를 disallow 로 만든다.
        assert captured["allow_local_tools"] is False
        assert not captured.get("disallow_tools_extra")
        assert captured["permission_mode"] == "default"
        # 루프백이 실제로 물렸는지: mcp_config 가 shim 을 모듈 실행으로 가리킨다.
        server = captured["mcp_config"]["mcpServers"][lh.CLI_MCP_SERVER_NAME]
        assert server["args"] == ["-m", "xgen_agent_runtime.host.cli_mcp_shim"]
        assert server["env"]["XGEN_MCP_URL"].startswith("http://127.0.0.1:")
        assert server["env"]["XGEN_MCP_TOKEN"]
        # 표면이 곧 도구 전부 → CLI 자체 tool-search 재유예는 끈다.
        assert captured["extra_env"]["ENABLE_TOOL_SEARCH"] == "false"
    finally:
        if cleanup:
            cleanup()


def test_local_host_claude_without_registry_has_no_tools(tmp_path, monkeypatch) -> None:
    """레지스트리가 없으면 mcp_config 도 없다 — 네이티브도 차단이므로 도구 0개.
    조용히 네이티브로 되돌아가지 않는 것이 요점(그건 가드 없는 실행이 된다)."""
    import xgen_agent_runtime.host.runner as runner

    captured: Dict[str, Any] = {}
    monkeypatch.setattr(runner, "build_cli_client", lambda **kw: captured.update(kw) or "c")
    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={"api_keys": {"anthropic": "ak"}, "settings": {"CLAUDE_CODE_ENABLED": "true"}},
    )
    _c, cleanup = host.build_cli_runtime("claude_code", {})
    assert cleanup is None
    assert captured["mcp_config"] is None
    assert captured["allow_local_tools"] is False
    assert tuple(captured["allow_tools"]) == ()


def test_local_host_claude_argv_carries_allow_surface(tmp_path) -> None:
    """실제 build_cli_client → argv 에 --allowedTools/--settings 가 실리고,
    네이티브 카탈로그 전체가 --disallowedTools 로 차단된다."""
    from xgen_agent_runtime.host import local_host as lh
    from xgen_agent_runtime.host.runner import CLI_NATIVE_TOOL_CATALOG
    from xgen_agent_runtime.llm_client.types import APIRequest

    claude_home = tmp_path / "claude-home"
    ws = tmp_path / "ws"
    ws.mkdir()
    host = LocalHostServices(
        str(ws),
        context={
            "api_keys": {"anthropic": "ak"},
            "settings": {
                "CLAUDE_CODE_ENABLED": "true",
                "CLAUDE_CODE_BINARY_PATH": "/bin/true",
                "XGEN_LOCAL_CLAUDE_CONFIG_DIR": str(claude_home),
            },
        },
    )
    client, cleanup = host.build_cli_runtime("claude_code", _registry_stash(str(ws)))
    try:
        argv = client._build_argv(
            APIRequest(model="claude-sonnet-4-5", messages=[{"role": "user", "content": "x"}])
        )
        allowed = argv[argv.index("--allowedTools") + 1].split()
        assert allowed == [f"mcp__{lh.CLI_MCP_SERVER_NAME}"]
        assert argv[argv.index("--settings") + 1] == str(
            claude_home / lh.CLAUDE_LOCAL_SETTINGS_FILENAME
        )
        assert "--permission-mode" not in argv  # default 유지(허용 목록으로 충분)
        disallowed = set(argv[argv.index("--disallowedTools") + 1].split())
        # 네이티브 카탈로그 **전체** — Bash 도 포함된다(런타임 레지스트리가 대신 준다).
        assert set(CLI_NATIVE_TOOL_CATALOG) <= disallowed
        assert "Bash" in disallowed
        assert "CronCreate" in disallowed  # 세션 한정 스케줄 도구도 여전히 차단
    finally:
        if cleanup:
            cleanup()


def test_local_host_codex_runs_workspace_write_without_bypass(tmp_path, monkeypatch) -> None:
    """codex: exec(헤드리스=승인 never) + --sandbox workspace-write, bypass 미사용."""
    from xgen_agent_runtime.llm_client.types import APIRequest

    host = LocalHostServices(
        str(tmp_path / "ws"),
        context={
            "api_keys": {"openai": "sk"},
            "settings": {"CODEX_ENABLED": "1", "CODEX_AUTH_MODE": "api_key", "CODEX_BINARY_PATH": "/bin/true"},
        },
    )
    client, _ = host.build_cli_runtime("codex", {})
    assert client._sandbox_mode == "workspace-write" and client._bypass_sandbox is False
    argv = client._build_argv(APIRequest(model="gpt-5.3-codex", messages=[{"role": "user", "content": "x"}]))
    assert argv[0] == "exec"
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv


def test_local_host_flags_for_runner_and_executor(tmp_path) -> None:
    host = LocalHostServices(str(tmp_path / "ws"))
    assert LocalHostServices.record_failed_starts is False
    assert getattr(host, "record_failed_starts", True) is False
    # 로컬 CLI 표면은 프로세스 안 루프백이라 선행 조건이 없다 — 서버 연결도,
    # 커넥터 카탈로그도 필요 없다. 실행기는 이 값으로 CLI 도구 표면 존재를 읽는다.
    for provider in ("claude_code", "codex"):
        assert host.cli_bridge_available(provider) is True
        assert host.cli_local_tool_surface(provider) is True


def test_history_messages_accepts_plain_dicts() -> None:
    from xgen_agent_runtime.host.memory import history_messages

    class _Msg:
        def __init__(self, t: str, c: Any) -> None:
            self.type, self.content = t, c

    raw = [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": [{"type": "text", "text": "반가워"}]},
        {"role": "system", "content": "drop"},
        {"role": "user", "content": ""},
        {"type": "human", "content": "직렬화 type 키"},
        _Msg("ai", "객체도 섞여도 됨"),
        "garbage",
        {"content": "no role"},
    ]
    assert history_messages(raw) == [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "반가워"},
        {"role": "user", "content": "직렬화 type 키"},
        {"role": "assistant", "content": "객체도 섞여도 됨"},
    ]
    # 기존 동작 유지: 튜플 모양·None·비리스트
    assert history_messages((raw[:1], "ctx")) == [{"role": "user", "content": "안녕"}]
    assert history_messages(None) == [] and history_messages("x") == []


def test_workflow_self_proxy_registers_from_context_meta() -> None:
    """자기진화(로컬) — 그래프는 서버 자산, 편집은 서버 RPC.

    컨텍스트 메타(enabled+path+실물 description/schema)가 있으면 WorkflowSelf
    프록시가 core 등록되고, 실행은 bridge.workflow_self 로 위임된다. 메타 없음
    (구서버)/비활성이면 조용히 미등록 — 프롬프트 블록 게이트(registry.get)가
    유령 안내를 막는다.
    """
    import asyncio

    from xgen_agent_runtime.tools import ToolRegistry

    class _FakeBridge:
        base_url = "http://srv"
        token = "t"

        def __init__(self) -> None:
            self.calls: list = []

        def workflow_self(self, path: str, input: Dict[str, Any]) -> Dict[str, Any]:
            self.calls.append((path, dict(input)))
            return {"ok": True, "content": "srv-result", "is_error": False, "metadata": {"a": 1}}

    meta = {
        "enabled": True,
        "path": "/api/agentflow/geny-memory/wf1/workflow-self",
        "description": "Edit your OWN workflow graph…",
        "input_schema": {"type": "object", "properties": {"action": {"type": "string"}}},
    }
    bridge = _FakeBridge()
    host = LocalHostServices("/tmp/ws-wsp", context={"workflow_self": meta}, server_bridge=bridge)
    reg = ToolRegistry()
    host.register_workflow_self_tools(reg, workflow_id="wf1", user_id=7, workflow_name="n")
    tool = reg.get("WorkflowSelf")
    assert tool is not None and reg.is_core("WorkflowSelf")
    assert tool.to_api_format()["description"].startswith("Edit your OWN")
    res = asyncio.run(tool.execute({"action": "guidance"}, None))
    assert res.content == "srv-result" and not res.is_error
    assert bridge.calls == [(meta["path"], {"action": "guidance"})]
    # 서버 실패 → 에러 ToolResult 로 degrade (그래프 불변 안내)
    bridge.workflow_self = lambda path, input: None  # type: ignore[assignment]
    res2 = asyncio.run(tool.execute({"action": "graph"}, None))
    assert res2.is_error and "not changed" in res2.content

    # 메타 없음/비활성 → 미등록
    for ctx in ({}, {"workflow_self": {"enabled": False}}):
        r = ToolRegistry()
        LocalHostServices("/tmp/ws-wsp", context=ctx, server_bridge=bridge).register_workflow_self_tools(
            r, workflow_id="wf1", user_id=7, workflow_name="n"
        )
        assert r.get("WorkflowSelf") is None


def test_connector_browser_tools_replace_anweb_locally() -> None:
    """브라우저 표면은 언제나 하나 — 커넥터 브라우저가 켜진 로컬 턴.

    커넥터 브라우저(사용자가 보는 XGEN 탭)는 Electron 이 소유하고 사이드카에서
    직접 갈 채널이 없다 → 서버 중계 프록시로 노출하고, 같은 턴에 an-web
    (headless 별도 세션) 패밀리는 등록하지 않는다. 브라우저가 꺼져 있으면
    (= 서버가 메타를 싣지 않으면) 정반대로 an-web 이 표면을 담당한다.
    """
    import asyncio

    meta = {
        "enabled": True,
        "server": "local",
        "path": "/api/agentflow/geny-memory/wf1/connector-mcp-call",
        "tools": [
            {
                "name": "mcp_local_BrowserTabs",
                "tool": "BrowserTabs",
                "description": "Manage XGEN browser pages on the user desktop.",
                "input_schema": {"type": "object", "properties": {"action": {"type": "string"}}},
            },
            {
                "name": "mcp_local_BrowserNavigate",
                "tool": "BrowserNavigate",
                "description": "Navigate one XGEN browser page.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }

    class _FakeBridge:
        base_url = "http://srv"
        token = "t"

        def __init__(self) -> None:
            self.calls: list = []

        def connector_mcp_call(self, path: str, server: str, tool: str, args: Dict[str, Any]):
            self.calls.append((path, server, tool, dict(args)))
            return {"ok": True, "content": f"ran {tool}"}

    bridge = _FakeBridge()
    host = LocalHostServices(
        "/tmp/ws-connector-browser", context={"connector_browser": meta}, server_bridge=bridge
    )

    tools = host.build_connector_mcp_tools(7, "connector")
    assert [t.name for t in tools] == ["mcp_local_BrowserTabs", "mcp_local_BrowserNavigate"]
    # 서버 실행과 **같은 이름/스키마**로 광고된다(로컬↔서버 표면 동일).
    api = tools[0].to_api_format()
    assert api["name"] == "mcp_local_BrowserTabs"
    assert api["input_schema"]["properties"]["action"]["type"] == "string"

    # an-web 패밀리는 이 턴에 빠진다 — 두 브라우저가 동시에 열리지 않는다.
    assert "browser" not in host.builtin_families()
    assert "web" in host.builtin_families() and "shell" in host.builtin_families()

    # 각 프록시가 자기 원격 도구를 부른다(루프 캡처 회귀 가드) + workflow_id 는 서버가 보완.
    asyncio.run(tools[0].execute({"action": "list"}, None))
    asyncio.run(tools[1].execute({}, None))
    assert [c[2] for c in bridge.calls] == ["BrowserTabs", "BrowserNavigate"]
    assert bridge.calls[0][0] == meta["path"] and bridge.calls[0][1] == "local"

    # 서버/커넥터 불통 → '브라우저는 그대로' 에러로 degrade (턴은 죽지 않는다).
    bridge.connector_mcp_call = lambda *_a, **_k: None  # type: ignore[assignment]
    res = asyncio.run(tools[0].execute({"action": "list"}, None))
    assert res.is_error and "unreachable" in res.content

    # 브라우저 OFF(메타 없음) → 프록시 없음 + an-web 복귀.
    off = LocalHostServices("/tmp/ws-connector-browser", context={}, server_bridge=bridge)
    assert off.build_connector_mcp_tools(7, "connector") == []
    assert "browser" in off.builtin_families()

    # 메타가 있어도 enabled=False / 도구 없음이면 an-web 유지(fail-open).
    for bad in ({"enabled": False, "tools": meta["tools"]}, {"enabled": True, "tools": []}):
        h = LocalHostServices("/tmp/ws-cb", context={"connector_browser": bad}, server_bridge=bridge)
        assert h.build_connector_mcp_tools(7, "connector") == []
        assert "browser" in h.builtin_families()


def test_local_cli_loopback_serves_live_registry(tmp_path, monkeypatch) -> None:
    """로컬 CLI 턴의 도구 표면 = **이 턴의 살아 있는 레지스트리**.

    turn_executor 가 스태시한 registry/ToolContext 를 그대로 루프백 JSON-RPC 로
    열고 shim 이 그 앞단 프록시가 된다. 여기서 확인하는 것은 "설정이 그럴듯한가"
    가 아니라 **실제로 그 도구가 그 컨텍스트로 실행되는가** 다 — 레지스트리와
    ToolContext 가 하나여야 서버/로컬의 표면·경로 가드가 어긋나지 않는다.
    """
    import urllib.request

    import xgen_agent_runtime.host.runner as runner
    from xgen_agent_runtime.host import local_host as lh

    captured: Dict[str, Any] = {}
    monkeypatch.setattr(runner, "build_cli_client", lambda **kw: captured.update(kw) or "c")

    ws = tmp_path / "ws"
    ws.mkdir()
    host = LocalHostServices(
        str(ws),
        context={
            "api_keys": {"anthropic": "ak"},
            "settings": {"CLAUDE_CODE_ENABLED": "true", "CLAUDE_CODE_AUTH_MODE": "api_key"},
        },
    )
    _c, cleanup = host.build_cli_runtime("claude_code", _registry_stash(str(ws)))
    server = captured["mcp_config"]["mcpServers"][lh.CLI_MCP_SERVER_NAME]
    base, token = server["env"]["XGEN_MCP_URL"], server["env"]["XGEN_MCP_TOKEN"]

    def rpc(method: str, params: Any = None, *, tok: str = token) -> Dict[str, Any]:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                           "params": params or {}}).encode()
        req = urllib.request.Request(
            base + server["env"]["XGEN_MCP_PATH"], data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    try:
        assert rpc("initialize")["result"]["protocolVersion"] == "2024-11-05"
        names = {t["name"] for t in rpc("tools/list")["result"]["tools"]}
        assert "Bash" in names  # 네이티브가 아니라 **우리** Bash 가 표면이다
        # 스태시한 ToolContext 그대로 실행된다 — cwd 가 동기화 워크스페이스.
        out = rpc("tools/call", {"name": "Bash", "arguments": {"command": "pwd"}})["result"]
        assert out["isError"] is False
        assert str(ws) in out["content"][0]["text"]
        # 토큰이 없으면 아무것도 안 열린다(루프백이라도 같은 PC 의 다른 프로세스가 있다).
        with pytest.raises(urllib.error.HTTPError) as exc:
            rpc("tools/list", tok="wrong")
        assert exc.value.code == 401
    finally:
        cleanup()

    # 회수 후에는 포트가 닫힌다 — 턴이 끝나면 도구 표면도 사라져야 한다.
    with pytest.raises(Exception):
        rpc("tools/list")


def test_local_cli_loopback_reports_tool_surface_changes(tmp_path) -> None:
    """턴 중간에 레지스트리가 바뀌면 _meta 로 알린다 — shim 이 그걸 보고
    list_changed 를 밀어 CLI 가 **같은 턴 안에서** 새 도구를 쓴다."""
    from xgen_agent_runtime.host.local_tool_mcp import LocalToolMcpServer
    from xgen_agent_runtime.tools import ToolRegistry
    from xgen_agent_runtime.tools.base import Tool, ToolContext, ToolResult

    class _Spawner(Tool):
        def __init__(self, reg: Any) -> None:
            self._reg = reg

        @property
        def name(self) -> str:
            return "Spawn"

        @property
        def description(self) -> str:
            return "registers another tool"

        @property
        def input_schema(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
            self._reg.register(_Quiet(), core=True)
            return ToolResult(content="ok")

    class _Quiet(Tool):
        @property
        def name(self) -> str:
            return "Quiet"

        @property
        def description(self) -> str:
            return "no-op"

        @property
        def input_schema(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(content="quiet")

    reg = ToolRegistry()
    reg.register(_Spawner(reg), core=True)
    srv = LocalToolMcpServer(reg, ToolContext(session_id="s", working_dir=str(tmp_path)))
    srv.start()
    try:
        changed = srv._dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                 "params": {"name": "Spawn", "arguments": {}}})
        assert changed["result"]["_meta"]["genyToolsChanged"] is True
        # 표면이 바뀌지 않는 호출에는 붙지 않는다 (매번 붙으면 신호가 죽는다).
        quiet = srv._dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                               "params": {"name": "Quiet", "arguments": {}}})
        assert "_meta" not in quiet["result"]
    finally:
        srv.stop()
