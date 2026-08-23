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
    assert host.builtin_families() == ["web", "documents", "browser", "workflow", "filesystem", "shell", "meta"]
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
        context={"api_keys": {"openai": "sk"}, "settings": {"CODEX_AUTH_MODE": "api_key"}},
    )
    host2.build_cli_runtime("codex", {})
    assert captured["auth_mode"] == "api_key" and captured["api_key"] == "sk"
    assert captured["env_extras"] is None


def test_local_host_claude_cli_runtime_allows_native_tools_and_isolates(tmp_path, monkeypatch) -> None:
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
                "CLAUDE_CODE_AUTH_MODE": "setup_token",
                "CLAUDE_CODE_OAUTH_TOKEN": "tok",
                "CLAUDE_CODE_BINARY_PATH": "/opt/claude",
                "CLAUDE_CODE_MAX_BUDGET_USD": "2.5",
                "XGEN_LOCAL_CLAUDE_CONFIG_DIR": str(claude_home),
            },
        },
    )
    host.build_cli_runtime("claude_code", {})
    assert captured["allow_local_tools"] is True  # 로컬엔 브릿지가 없다 — 네이티브 도구 필수
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
