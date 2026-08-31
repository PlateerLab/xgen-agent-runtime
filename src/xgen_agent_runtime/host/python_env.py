"""PythonEnv — 에이전트의 **세션 파이썬 환경**을 1급으로 만드는 도구.

지금까지 의존성 선언 대상은 forged tool 하나뿐이었다 (도구별 env_id).
"내 파이썬에 python-pptx 깔아줘" 의 대상 — 세션 전체의 기본 환경 — 이
없어서, 에이전트는 ModuleNotFoundError 앞에서 막다른 길을 겪었다
(2026-08-18 실증: pptx 를 못 쓴다고 판단하고 마크다운으로 후퇴).

설계:

    매니페스트  workspace 의 ``.xgeny/python-env.json`` — 동기화로 영속
                (파드 재시작·스케일 이동 무관), 에이전트가 Read 로 직접
                볼 수 있는 **자기 환경의 일부**다.
    환경        러너의 콘텐츠 주소 환경 (ensure_env → env_id). 같은 목록
                이면 같은 id — 어느 파드든 아티팩트에서 재수화된다.
    적용        session.env_id — 이후 모든 Bash/python/제작 도구(자체
                env 없는 것)가 이 환경의 인터프리터로 돈다. install 은
                **이 턴 안에서 즉시** 적용된다.

ad-hoc ``pip install`` (Bash) 과의 관계: 그것도 동작한다 — 세션 HOME 의
user-site 에 앉아 그 세션이 사는 동안 유지된다. 영속이 필요하면 이 도구다.
둘의 역할이 겹치지 않고, 어느 쪽도 막지 않는다. **둘 다 같은 sandbox 세션
안에서 일어난다** — 설치한 곳과 실행하는 곳이 갈릴 여지가 없다.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("xgen_agent_runtime.host.python_env")

#: 매니페스트 파일 — workspace 기준 상대 경로 (동기화 대상 = 영속).
ENV_FILE = ".xgeny/python-env.json"

_NAME_SPLIT_RE = re.compile(r"[<>=!~\[;@ ]")


def _req_name(spec: str) -> str:
    """요구사항 문자열("httpx>=0.27")의 정규화된 패키지 이름 (PEP 503)."""
    head = _NAME_SPLIT_RE.split(str(spec).strip(), 1)[0]
    return re.sub(r"[-_.]+", "-", head).lower()


def merge_requirements(stored: Sequence[str], add: Sequence[str]) -> List[str]:
    """이름 기준 병합 — 새로 준 스펙이 같은 이름의 기존 핀을 대체한다."""
    out: List[str] = []
    adds = {_req_name(s): str(s).strip() for s in add if str(s).strip()}
    for s in stored:
        s = str(s).strip()
        if s and _req_name(s) not in adds:
            out.append(s)
    out.extend(adds.values())
    return out


def remove_requirements(stored: Sequence[str], names: Sequence[str]) -> List[str]:
    drop = {_req_name(n) for n in names if str(n).strip()}
    return [str(s).strip() for s in stored if str(s).strip() and _req_name(s) not in drop]


def parse_env_file(data: bytes) -> Dict[str, Any]:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"packages": [], "env_id": ""}
    if not isinstance(raw, dict):
        return {"packages": [], "env_id": ""}
    pkgs = raw.get("packages")
    return {
        "packages": [str(p) for p in pkgs if str(p).strip()] if isinstance(pkgs, list) else [],
        "env_id": str(raw.get("env_id") or ""),
    }


def load_session_env(session: Any) -> None:
    """prepare() 에서: 저장된 매니페스트의 env_id 를 세션에 적용한다.

    동기 컨텍스트(턴 경계)라 세션의 ``_sync`` HTTP 를 쓴다 — attach 가 이미
    같은 왕복을 하는 자리다. 파일이 없거나 읽기 실패면 기본 환경 그대로
    (fail-open: 환경 로드가 턴을 막으면 안 된다).
    """
    try:
        import base64

        data = session._sync(
            "GET",
            "/api/sandbox/workspace/file",
            params={"workflow_id": session.workflow_id, "path": ENV_FILE},
            owner=session.workflow_id,
        )
        parsed = parse_env_file(base64.b64decode(data.get("content_b64") or ""))
        if parsed["env_id"]:
            session.env_id = parsed["env_id"]
            logger.info(
                "python-env: 세션 환경 적용 %s (%d개 패키지)",
                parsed["env_id"][:12],
                len(parsed["packages"]),
            )
    except Exception:  # noqa: BLE001 — 404 포함: 매니페스트 없음 = 기본 환경
        logger.debug("python-env: 매니페스트 없음/읽기 실패 — 기본 환경", exc_info=True)


class PythonEnvTool:
    """세션 파이썬 환경 관리 — install / remove / list.

    executor ``Tool`` 계약 (forged_tools 와 같은 duck-type; ``_bind_tool_base``
    로 런타임에 Tool 상속).
    """

    def __init__(self, *, workflow_id: str, workspace_dir: Optional[str] = None) -> None:
        self._workflow_id = str(workflow_id)
        self._workspace_dir = str(workspace_dir) if workspace_dir else ""

    def _workspace(self) -> str:
        """로컬 폴백용 workspace 경로 — 주어졌으면 그것, 아니면 workflow_id 로 유도."""
        # 호출부(register_forged_tools / forged_tool_instances)가 항상 실행 중인
        # 턴의 workspace 를 준다. 예전엔 여기서 workflow_id 로 서버 경로를
        # 유도하는 폴백이 있었는데, 그건 **서버 파드의 경로**라 로컬 실행에서
        # 쓰면 엉뚱한 트리를 만진다 — 없으면 없다고 답하는 편이 정직하다.
        return self._workspace_dir or ""

    @property
    def name(self) -> str:
        return "PythonEnv"

    @property
    def description(self) -> str:
        return (
            "Manage THIS agent's persistent Python environment (your Bash/python and "
            "saved tools without their own env run on it). action=install adds pip "
            "packages — resolved, installed into an isolated content-addressed env, "
            "applied to this session immediately, and REMEMBERED across sessions/pods "
            f"(manifest: {ENV_FILE} in your workspace). remove deletes by name; list "
            "shows the current manifest. Ad-hoc `pip install` in Bash also works but "
            "only lives until the pod restarts — use this for anything you want to keep."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["install", "remove", "list"],
                    "description": "install: add/upgrade packages. remove: drop by name. list: show manifest.",
                },
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        'pip requirement strings, e.g. ["python-pptx", "httpx>=0.27"]. '
                        "Required for install/remove."
                    ),
                },
            },
            "required": ["action"],
        }

    def capabilities(self, *_a: Any, **_k: Any) -> Any:
        from xgen_agent_runtime.tools.base import ToolCapabilities

        # 환경 빌드는 수 분까지 갈 수 있다 (컴파일 패키지). 직렬로.
        return ToolCapabilities(concurrency_safe=False, timeout_s=900.0)

    # ── 저장 (workspace 파일 = 단일 소스) ────────────────────────

    async def _load(self, sandbox: Any) -> Dict[str, Any]:
        try:
            return parse_env_file(await sandbox.read_bytes(ENV_FILE))
        except FileNotFoundError:
            return {"packages": [], "env_id": ""}
        except Exception:  # noqa: BLE001 — 깨진 파일은 빈 매니페스트로
            logger.warning("python-env: 매니페스트 읽기 실패 — 빈 것으로 간주", exc_info=True)
            return {"packages": [], "env_id": ""}

    async def _save(self, sandbox: Any, packages: List[str], env_id: str) -> None:
        body = json.dumps(
            {"packages": packages, "env_id": env_id},
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        await sandbox.write_bytes(ENV_FILE, body)

    # ── 실행 ─────────────────────────────────────────────────────

    async def execute(self, input: Dict[str, Any], context: Any) -> Any:  # noqa: A002
        from xgen_agent_runtime.tools.base import ToolResult

        sandbox = getattr(context, "sandbox", None)
        if sandbox is None:
            # 예전엔 workspace 안에 `pip install --target` 으로 로컬 env 를 만드는
            # 폴백이 있었다. 그게 두 번째 세계를 만들었다 — 그 디렉터리를
            # PYTHONPATH 에 얹는 건 제작 도구뿐이라, Bash 로 테스트하는 에이전트는
            # "설치했는데 못 찾는다" 를 반복했다(프로드 실증). 환경은 하나여야 한다.
            return ToolResult(
                content=(
                    "파이썬 환경을 관리할 실행 환경이 없습니다. 이 에이전트의 "
                    "sandbox 세션에 붙은 뒤 다시 시도하세요."
                ),
                is_error=True,
            )

        action = str(input.get("action") or "").strip()
        packages = [str(p).strip() for p in (input.get("packages") or []) if str(p).strip()]
        stored = await self._load(sandbox)

        if action == "list":
            return ToolResult(
                content={
                    "packages": stored["packages"],
                    "env_id": stored["env_id"] or None,
                    "manifest": ENV_FILE,
                    "active": bool(stored["env_id"]) and sandbox.env_id == stored["env_id"],
                }
            )

        if action not in ("install", "remove"):
            return ToolResult(
                content=f"알 수 없는 action: {action!r} (install|remove|list)", is_error=True
            )
        if not packages:
            return ToolResult(content=f"{action} 에는 packages 가 필요합니다.", is_error=True)

        if action == "install":
            merged = merge_requirements(stored["packages"], packages)
        else:
            merged = remove_requirements(stored["packages"], packages)

        if not merged:
            # 전부 제거 → 기본 인터프리터로 복귀.
            await self._save(sandbox, [], "")
            sandbox.env_id = ""
            return ToolResult(
                content={
                    "packages": [],
                    "env_id": None,
                    "message": "세션 환경을 비웠습니다 — 기본 인터프리터로 돌아갑니다 (이번 턴부터).",
                }
            )

        try:
            env_id, pinned = await sandbox.ensure_env(merged)
        except Exception as exc:  # noqa: BLE001 — 원인을 에이전트가 읽어야 한다
            return ToolResult(
                content=(
                    f"환경 구성 실패: {exc}\n패키지 이름/버전을 확인하세요. "
                    "기존 환경은 그대로 유지됩니다."
                ),
                is_error=True,
            )

        final = pinned or merged
        await self._save(sandbox, final, env_id)
        # 이번 턴 안에서 즉시 적용 — 다음 Bash/python 부터 이 환경이다.
        sandbox.env_id = env_id
        return ToolResult(
            content={
                "packages": final,
                "env_id": env_id,
                "message": (
                    f"세션 환경 적용 완료 — 지금부터 Bash/python 이 이 환경으로 돕니다. "
                    f"{len(final)}개 패키지(전이 포함, 정확한 핀). 이후 세션·다른 파드에서도 "
                    "자동 복원됩니다."
                ),
            }
        )
