"""커넥터 로컬 실행 사이드카 — Node(커넥터)↔Python 계약 (v2: 상주 데몬 + 구조화 이벤트).

데스크톱 커넥터는 커넥터-세션 턴을 서버에 보내는 대신 **이 사이드카를 로컬
프로세스로 스폰**해, 서버 웹과 **같은 AgentTurnExecutor** 를 사용자 PC 에서
돌린다(무발산). 핵심: 실행 host 는 ``LocalHostServices`` — make_sandbox=None 이라
런타임 Bash/Read/Write 가 이 PC 에서 직접 돌고, codex/claude_code 는 로컬
프로세스로 스폰된다. agent 코드는 서버와 한 줄도 다르지 않다.

두 가지 실행 모드 (Node 가 고른다):

1. **원샷** (인자 없음, v1 호환): stdin 전체를 JSON 요청 하나로 읽고, 이벤트를
   stdout JSON-lines 로 흘린 뒤 종료한다.
2. **데몬** (``--serve``): 프로세스가 상주하며 stdin 의 JSON-lines **명령**을 받고
   stdout 으로 JSON-lines **이벤트**를 흘린다. 커넥터가 턴마다 Python 을 새로
   띄우지 않아 첫 토큰 지연이 사라진다(Windows 의 Python 기동 수 초 절감).
   턴은 각각 워커 스레드에서 돌며, 이벤트에는 ``id`` 가 붙어 다중 세션을 구분한다.

stdin 요청(턴) — **상태는 서버가 해석해 넘긴다**(로컬↔웹 무발산)::

    {
      "id": "t-123",                                  # 데몬 모드 필수(이벤트 상관키)
      "op": "turn",                                   # 데몬 모드 (원샷은 생략)
      "workspace_dir": "/path/to/synced/agent-folder",# 필수(서버와 sync 되는 폴더)
      "provider": "openai" | "anthropic" | "codex" | "claude_code" | ...,
      "text": "사용자 메시지",
      "context": {                       # 서버가 **로그인 계정**으로 해석해 넘긴 상태
        "api_keys": {"openai": "sk-...", "anthropic": "..."},
        "base_urls": {"vllm": "..."},
        "credentials": {"bedrock": {...}},
        "settings": {"CODEX_BINARY_PATH": "...", "CODEX_AUTH_MODE": "oauth", ...}
      },
      "server": {"url": "https://xgen...", "token": "..."},  # 라이브 브릿지(메모리 등)
      "options": {"model": "...", "workflow_id": "...", "interaction_id": "...",
                  "streaming": true, ...}                     # 저장된 에이전트 설정
    }

데몬 모드의 다른 명령::

    {"id": "t-123", "op": "cancel"}     # 진행 중 턴 협조 취소(cancel_context)
    {"op": "ping"}                       # → {"type": "pong", ...}
    {"op": "shutdown"}                   # 상주 종료

stdout 이벤트(JSON-lines; 데몬 모드는 ``id`` 포함)::

    {"type": "started", "pid": 123, "surface": "connector_local"}
    {"type": "chunk", "text": "..."}                 # 스트리밍 출력 조각(0..n)
    {"type": "tool", "data": {"type": "tool_call"|"tool_result"|"tool_error",
                               "tool_name": "...", ...}}   # 도구 활동(웹과 같은 shape)
    {"type": "canvas_command", "data": {...}}        # self-evolution 사이드채널
    {"type": "usage", "data": {"input_tokens": int, "output_tokens": int,
                                "cache_read_tokens": int|null,
                                "cache_creation_tokens": int|null,
                                "total_cost_usd": float|null,
                                "model": str|null, "provider": str|null}}
                                                     # 턴 토큰/비용 집계(0..1회, done 직전)
    {"type": "done", "text": "..."}                  # 완료(전체 텍스트)
    {"type": "error", "message": "..."}              # 치명 오류
    {"type": "cancelled"}                            # cancel 로 중단됨

``usage`` 는 runner.stream_turn 이 파이프라인 종료 후 한 번 내는 집계를 **1급
이벤트**로 올린 것이다(``meta`` 로 감싸지 않는다) — 커넥터는 이를
TurnReport.usage 로 report-turn 에 실어 서버 토큰 컬럼을 채운다. 사용량이
기록되지 않은 턴(API 호출 0회)에는 나오지 않는다.

종료 이벤트는 ``done`` / ``error`` / ``cancelled`` 중 **정확히 하나**다. cancel 요청이
관측된 턴은 스트림이 자연 종료한 뒤라도 ``done`` 이 아니라 ``cancelled`` 로 닫는다
(커넥터가 "취소했는데 done" 을 받아 완료로 오판하던 레이스 차단).

v1(커넥터 ≤1.64) 은 chunk/done/error 만 해석했고 그 외 dict 청크를 ``str()`` 로
텍스트에 섞어 넣었다 — v2 는 도구 이벤트를 전용 ``tool`` 이벤트로 올린다.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Dict, Iterator, Mapping, Optional, TextIO

SIDECAR_PROTOCOL_VERSION = 2
SURFACE = "connector_local"


def _runtime_version() -> str:
    try:
        from importlib.metadata import version

        return version("xgen-agent-runtime")
    except Exception:  # noqa: BLE001
        return "unknown"


def _normalize_event(chunk: Any) -> Dict[str, Any]:
    """AgentTurnExecutor 스트림 항목 → 사이드카 이벤트.

    str → chunk. dict 는 runner.stream_turn 의 사이드채널(agent_event /
    canvas_command / usage) — 절대 텍스트로 강등하지 않는다."""
    if isinstance(chunk, str):
        return {"type": "chunk", "text": chunk}
    if isinstance(chunk, dict):
        kind = chunk.get("type")
        data = chunk.get("data")
        if kind == "agent_event" and isinstance(data, dict):
            return {"type": "tool", "data": data}
        if kind == "canvas_command":
            return {"type": "canvas_command", "data": data}
        if kind == "usage" and isinstance(data, dict):
            # 턴 토큰/비용 집계 — 1급 이벤트(meta 로 감싸지 않는다).
            return {"type": "usage", "data": data}
        return {"type": "meta", "data": chunk}
    return {"type": "chunk", "text": str(chunk)}


def run_turn_request(
    req: Mapping[str, Any], *, cancel_check: Optional[Any] = None
) -> Iterator[Dict[str, Any]]:
    """turn 요청을 로컬에서 실행하고 이벤트를 yield 한다(계약의 핵심 — 테스트 대상).

    Node 는 이 결과를 stdout JSON-lines 로 받는다. 실패는 error 이벤트로 승격
    (사이드카가 조용히 죽으면 커넥터가 매달린다)."""
    from xgen_agent_runtime.host.local_host import LocalHostServices
    from xgen_agent_runtime.host.turn_executor import AgentTurnExecutor

    workspace_dir = str(req.get("workspace_dir") or "").strip()
    if not workspace_dir:
        yield {"type": "error", "message": "workspace_dir 가 없습니다."}
        return

    options = dict(req.get("options") or {})
    streaming = bool(options.get("streaming", True))

    # 라이브 서버 브릿지(메모리 등 공유 상태) — {url, token} 있으면 구성.
    bridge = None
    server = req.get("server") or {}
    if isinstance(server, dict) and server.get("url"):
        try:
            from xgen_agent_runtime.host.server_bridge import ServerBridge

            tls = server.get("tls") if isinstance(server.get("tls"), dict) else {}
            bridge = ServerBridge(
                str(server["url"]),
                str(server.get("token") or ""),
                verify=(tls.get("ca_file") or (False if tls.get("verify") is False else True)),
            )
        except Exception:  # noqa: BLE001 — 브릿지 실패는 무기억으로 degrade(발산 방지)
            bridge = None

    host = LocalHostServices(
        workspace_dir,
        context=req.get("context"),  # 서버가 계정으로 해석한 키/설정
        server_bridge=bridge,
    )

    # AgentTurnExecutor.run 의 kwargs 로 평탄화 (서버 노드가 kwargs 로 받는 것과 동일 표면).
    kwargs: Dict[str, Any] = {
        "provider": str(req.get("provider") or "openai"),
        "text": req.get("text") or "",
        "streaming": streaming,
        **options,
    }
    kwargs.setdefault("node_name", "agent")
    if cancel_check is not None:
        # 턴 단위 취소 훅 — interaction 스코프 레지스트리를 쓰지 않아 같은 대화의 다음
        # 턴을 오염시키지 않는다(실행기 _cancelled 가 먼저 본다).
        kwargs["cancel_check"] = cancel_check
    yield {
        "type": "started",
        "pid": os.getpid(),
        "surface": SURFACE,
        "provider": kwargs["provider"],
        "workspace_dir": host.agent_workspace_dir(""),
    }

    try:
        result = AgentTurnExecutor().run(host, **kwargs)
    except Exception as exc:  # noqa: BLE001 — 사이드카는 조용히 죽지 않는다
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return

    if streaming and hasattr(result, "__iter__") and not isinstance(result, str):
        acc = []
        try:
            for chunk in result:
                if cancel_check is not None and cancel_check():
                    try:
                        result.close()
                    except Exception:  # noqa: BLE001
                        pass
                    yield {"type": "cancelled"}
                    return
                ev = _normalize_event(chunk)
                if ev["type"] == "chunk":
                    acc.append(ev["text"])
                yield ev
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            return
        if cancel_check is not None and cancel_check():
            # 레이스: 마지막 청크와 스트림 종료 사이에 cancel 이 들어왔다 — 실행기
            # 루프가 협조 중단했든 자연 종료했든, 취소가 관측된 턴은 절대 done 으로
            # 닫지 않는다(커넥터가 완료로 오판 → 다음 턴 상태 오염).
            yield {"type": "cancelled"}
            return
        yield {"type": "done", "text": "".join(acc)}
    else:
        if cancel_check is not None and cancel_check():
            yield {"type": "cancelled"}
            return
        yield {"type": "done", "text": result if isinstance(result, str) else str(result)}


# ── 데몬 모드 ────────────────────────────────────────────────────────────


class _Emitter:
    """stdout 프로토콜 채널 — 스레드 안전 JSON-lines writer."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._lock = threading.Lock()

    def emit(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._out.write(line + "\n")
                self._out.flush()
            except (BrokenPipeError, OSError):
                # 커넥터가 사라졌다 — 더 쓸 곳이 없다. 데몬은 stdin EOF 로 종료된다.
                pass


class SidecarDaemon:
    """stdin 명령 → 워커 스레드 턴 → stdout 이벤트. 테스트는 in/out 스트림 주입."""

    def __init__(self, inp: Any, out: TextIO) -> None:
        self._in = inp
        self._emitter = _Emitter(out)
        self._turns: Dict[str, Dict[str, Any]] = {}  # id → {"cancel": Event, "interaction_id": str}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ── 명령 처리 ──
    def _handle(self, cmd: Dict[str, Any]) -> None:
        op = str(cmd.get("op") or "turn")
        tid = str(cmd.get("id") or "")
        if op == "ping":
            self._emitter.emit(
                {
                    "id": tid,
                    "type": "pong",
                    "pid": os.getpid(),
                    "protocol": SIDECAR_PROTOCOL_VERSION,
                    "runtime_version": _runtime_version(),
                    "python": sys.version.split()[0],
                    "surface": SURFACE,
                }
            )
            return
        if op == "shutdown":
            self._stop.set()
            return
        if op == "cancel":
            self._cancel(tid)
            return
        if op == "turn":
            if not tid:
                self._emitter.emit(
                    {"id": "", "type": "error", "message": "turn 요청에 id 가 없습니다."}
                )
                return
            th = threading.Thread(
                target=self._run_turn, args=(tid, cmd), name=f"sidecar-turn-{tid}", daemon=True
            )
            th.start()
            return
        self._emitter.emit({"id": tid, "type": "error", "message": f"unknown op: {op}"})

    def _cancel(self, tid: str) -> None:
        with self._lock:
            entry = self._turns.get(tid)
        if entry is None:
            self._emitter.emit(
                {"id": tid, "type": "error", "message": "취소할 턴이 없습니다(이미 종료)."}
            )
            return
        # per-turn Event 만 세운다 — 실행기에는 cancel_check 로 전달돼 있어 다음 스트림 경계에서
        # 멈춘다. interaction 스코프 request_cancel 은 쓰지 않는다(같은 대화의 다음 턴 오염).
        entry["cancel"].set()

    def _run_turn(self, tid: str, req: Dict[str, Any]) -> None:
        cancel = threading.Event()
        interaction_id = str((req.get("options") or {}).get("interaction_id") or "")
        with self._lock:
            self._turns[tid] = {"cancel": cancel, "interaction_id": interaction_id}
        try:
            for ev in run_turn_request(req, cancel_check=cancel.is_set):
                ev["id"] = tid
                self._emitter.emit(ev)
        except Exception as exc:  # noqa: BLE001
            self._emitter.emit(
                {"id": tid, "type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            with self._lock:
                self._turns.pop(tid, None)

    # ── 루프 ──
    def serve(self) -> int:
        self._emitter.emit(
            {
                "type": "ready",
                "pid": os.getpid(),
                "protocol": SIDECAR_PROTOCOL_VERSION,
                "runtime_version": _runtime_version(),
                "python": sys.version.split()[0],
                "surface": SURFACE,
            }
        )
        for raw in self._in:
            if self._stop.is_set():
                break
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            line = raw.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except ValueError as exc:
                self._emitter.emit({"type": "error", "message": f"bad command: {exc}"})
                continue
            if not isinstance(cmd, dict):
                continue
            self._handle(cmd)
        # stdin EOF / shutdown → 진행 중 턴 취소 후 종료.
        with self._lock:
            ids = list(self._turns.keys())
        for tid in ids:
            self._cancel(tid)
        return 0


def _protocol_stdout() -> TextIO:
    """프로토콜 전용 stdout — 라이브러리의 잡다한 print 가 JSON-lines 를 깨지 않게
    실제 fd 1 을 복제해 쓰고, sys.stdout 은 stderr 로 돌린다."""
    fd = os.dup(1)
    out = os.fdopen(fd, "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = sys.stderr
    return out


def main(argv: Any = None) -> int:
    """CLI 진입 — ``--serve`` 면 데몬, 아니면 원샷(stdin JSON 하나)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--serve" in args:
        out = _protocol_stdout()
        return SidecarDaemon(sys.stdin.buffer, out).serve()
    try:
        req = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(json.dumps({"type": "error", "message": f"bad request: {exc}"}) + "\n")
        sys.stdout.flush()
        return 2
    out = _protocol_stdout()
    for event in run_turn_request(req):
        out.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
