"""커넥터 로컬 실행 사이드카 — Node(커넥터)↔Python 계약.

데스크톱 커넥터는 커넥터-세션 턴을 서버에 보내는 대신 **이 사이드카를 로컬
프로세스로 스폰**해, 서버 웹과 **같은 AgentTurnExecutor** 를 사용자 PC 에서
돌린다(무발산). Node 는 turn 요청(JSON)을 stdin 으로 주고, 사이드카는 이벤트를
JSON-lines 로 stdout 에 흘린다(스트리밍 청크 → done/error).

핵심: 실행 host 는 ``LocalHostServices`` — make_sandbox=None 이라 런타임
Bash/Read/Write 가 이 PC 에서 직접 돌고, codex/claude_code 는 로컬 프로세스로
스폰된다. agent 코드는 서버와 한 줄도 다르지 않다.

프로토콜(stdin 요청) — **상태는 서버가 해석해 넘긴다**(로컬↔웹 무발산)::

    {
      "workspace_dir": "/path/to/synced/agent-folder",   # 필수(서버와 sync 되는 폴더)
      "provider": "openai" | "anthropic" | "codex" | "claude_code" | ...,
      "text": "사용자 메시지",
      "context": {                       # 서버가 **로그인 계정**으로 해석해 넘긴 상태
        "api_keys": {"openai": "sk-...", "anthropic": "..."},   # 계정 키
        "base_urls": {"vllm": "..."},
        "credentials": {"bedrock": {...}},
        "settings": {"CODEX_BINARY_PATH": "...", ...}           # 관리자 설정
      },
      "server": {"url": "https://xgen...", "token": "..."},  # 라이브 브릿지(메모리 등)
      "options": {"model": "...", "temperature": 0.7, "workflow_id": "...",
                  "interaction_id": "...", "streaming": true, ...}  # 저장된 에이전트 설정
    }

stdout 이벤트(JSON-lines)::

    {"type": "chunk", "text": "..."}     # 스트리밍 출력 조각(0..n)
    {"type": "done", "text": "..."}      # 완료(비스트리밍이면 전체 텍스트)
    {"type": "error", "message": "..."}  # 치명 오류
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterator, Mapping


def run_turn_request(req: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
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

            bridge = ServerBridge(str(server["url"]), str(server.get("token") or ""))
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

    try:
        result = AgentTurnExecutor().run(host, **kwargs)
    except Exception as exc:  # noqa: BLE001 — 사이드카는 조용히 죽지 않는다
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return

    if streaming and hasattr(result, "__iter__") and not isinstance(result, str):
        acc = []
        try:
            for chunk in result:
                s = chunk if isinstance(chunk, str) else str(chunk)
                acc.append(s)
                yield {"type": "chunk", "text": s}
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            return
        yield {"type": "done", "text": "".join(acc)}
    else:
        yield {"type": "done", "text": result if isinstance(result, str) else str(result)}


def main(argv: Any = None) -> int:
    """CLI 진입 — stdin JSON 요청을 읽어 stdout JSON-lines 이벤트를 흘린다."""
    try:
        req = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(json.dumps({"type": "error", "message": f"bad request: {exc}"}) + "\n")
        sys.stdout.flush()
        return 2
    for event in run_turn_request(req):
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
