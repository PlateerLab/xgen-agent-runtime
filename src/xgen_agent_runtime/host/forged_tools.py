"""Forged tools — 에이전트가 스스로 만든 도구가 다음 턴에도 도구로 남는다.

세션이 스크립트를 짜고 ``ForgeTool`` 로 등록하면 다음 턴부터 그 스크립트가 이름과
스키마를 가진 **도구 하나**가 된다. 이건 권한 확대가 아니다 — 에이전트는 지금도
Write 로 스크립트를 쓰고 Bash 로 실행할 수 있다. 여기서 더해지는 것은
*이름·스키마·영속성·발견가능성* 이다.

**계약** (러너의 ``run_tool`` 과 동일 — 실행지가 바뀌어도 스크립트는 그대로다):

    입력  검증된 tool input 이 **stdin 에 JSON** 으로 들어간다
    출력  결과를 **stdout 에 JSON** 으로 찍는다
          (처리된 실패는 ``{"error": "..."}``)
    실패  0 이 아닌 종료 코드 → ``ToolResult(is_error=True)`` + stderr 꼬리

**실행지는 ToolContext.sandbox 하나로 갈린다** — 다른 모든 도구와 같은 규칙이다.
러너가 붙어 있으면 그 세션에서, 없으면(데스크톱 로컬 턴) 이 PC 의 workspace 에서
서브프로세스로 돈다. 스크립트는 동기화되는 ``workspace/`` 안에 있으므로 양쪽이
**같은 파일**을 본다.

**스펙 저장소는 포트다** (:class:`ForgedToolSpecStore`). 스펙은 계정 자산이라
서버가 원본을 갖는다: 서버 실행은 DB 를 직접 쓰고(xgen-workflow 의
``ForgedToolStore``), 로컬 실행은 같은 인터페이스의 RPC 프록시를 쓴다 — 메모리·
RAG 와 같은 '상태는 서버, 실행은 여기' 규약이다. 엔진이 저장소를 직접 만들지
않는 이유가 그것이다.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger("xgen_agent_runtime.host.forged_tools")


class ForgedToolSpecStore(Protocol):
    """도구 스펙 저장소 — 엔진이 요구하는 최소 계약.

    구현은 호스트가 준다: 서버는 DB(원본), 로컬은 같은 DB 를 향하는 서버 RPC
    프록시. 엔진은 어느 쪽인지 알 필요가 없고, 알면 안 된다.
    """

    def list(self) -> List["ForgedToolSpec"]: ...

    def get(self, name: str) -> Optional["ForgedToolSpec"]: ...

    def save(self, spec: "ForgedToolSpec") -> "ForgedToolSpec": ...

    def delete(self, name: str) -> bool: ...

    def record_call(self, name: str, *, error: Optional[str] = None) -> None: ...

    def mark_tested(self, name: str, *, ok: bool, error: Optional[str] = None) -> None: ...


#: 도구 이름 규칙 — LLM API 의 tool name 제약(영문 시작, 영숫자/_/-)과 동일.
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")

#: 실행기 허용 목록. 임의 문자열을 그대로 exec 하면 그 자체가 명령 주입이다.
_RUNTIMES = {
    "python3": ("python3", "python"),
    "python": ("python3", "python"),
    "node": ("node",),
    "bash": ("bash", "sh"),
    "sh": ("sh", "bash"),
}

#: 자식 프로세스에 물려주는 환경변수 — **allowlist**. 파드 환경변수에는 DB
#: 비밀번호·API 키가 들어 있고, 에이전트가 만든 스크립트에 그걸 통째로 넘기면
#: 한 줄짜리 스크립트가 곧 시크릿 유출 경로가 된다 (executor 2.51.1 감사 S3
#: 의 Bash env 스크럽과 같은 규칙).
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")

#: 결과/에러 문자열 상한 — 폭주하는 스크립트가 턴을 삼키지 않게.
_STDOUT_CAP = 200_000
_STDERR_CAP = 4_000

#: 스크립트 1회 실행 시간 상한 (스펙이 더 크게 요구해도 여기서 자른다).
_MAX_TIMEOUT_S = 600.0
_DEFAULT_TIMEOUT_S = 60.0
#: 등록 게이트/사람 테스트가 한 번 실행할 때의 시간 상한 — 검증 실행이
#: 오래 매달려 forge 응답을 막지 않게 한다 (도구 자체 timeout_s 와 별개).
_TEST_TIMEOUT_CAP_S = 120.0


# ── 스펙 ──────────────────────────────────────────────────────────────


@dataclass
class ForgedToolSpec:
    """저장되는 도구 한 개의 정의."""

    name: str
    description: str
    entrypoint: str  # workspace 기준 상대 경로
    runtime: str = "python3"
    input_schema: Dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    argv: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    timeout_s: float = _DEFAULT_TIMEOUT_S
    enabled: bool = True
    #: 이 도구가 필요로 하는 파이썬 패키지 (``["pandas", "httpx==0.28.1"]``).
    #: 러너가 이걸로 격리 환경을 세우고, 그 결과가 :attr:`env_id` 다.
    dependencies: List[str] = field(default_factory=list)
    #: 러너가 발급한 환경 식별자 (매니페스트의 sha256). 같은 의존성이면 같은
    #: 값이라 에이전트가 100개여도 환경은 하나다.
    env_id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    #: 관측용 — 뷰에서 "쓰이고 있는 도구"를 구분하기 위한 최소 통계.
    calls: int = 0
    errors: int = 0
    last_used_at: Optional[float] = None
    last_error: Optional[str] = None
    #: 등록 전 **실제 실행 테스트**를 통과했는가. 새로 만드는 스펙은 아직
    #: 검증되지 않았으므로 False 로 시작하고, ForgeTool 이 테스트를 돌려
    #: 통과하면 True 로 올린다. 미검증(False) 도구는 에이전트에게 노출되지
    #: 않는다. DB 에서 읽어 온 기존 도구는 grandfather(1) 다.
    verified: bool = False
    #: 마지막 테스트 실패 사유 (통과하면 ''/None).
    last_test_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ForgedToolSpec":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - dataclass 공개 계약
        return cls(**{k: v for k, v in (raw or {}).items() if k in known})


class ForgedToolError(ValueError):
    """스펙이 규칙을 어겼다 (이름/경로/실행기)."""


def _safe_file_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "tool"))


def resolve_runtime(runtime: str) -> str:
    """실행기 이름 → 실제 실행 파일 절대 경로. 허용 목록 밖이면 거부."""
    key = str(runtime or "python3").strip().lower()
    candidates = _RUNTIMES.get(key)
    if not candidates:
        raise ForgedToolError(
            f"runtime '{runtime}' 은(는) 허용되지 않습니다 (가능: {', '.join(sorted(_RUNTIMES))})"
        )
    for c in candidates:
        found = shutil.which(c)
        if found:
            return found
    raise ForgedToolError(f"runtime '{runtime}' 을(를) 이 서버에서 찾을 수 없습니다")


def resolve_entrypoint(workspace_dir: str, entrypoint: str) -> str:
    """entrypoint(상대 경로) → workspace 안의 절대 경로. 탈출은 거부.

    심볼릭 링크까지 펼쳐서(``realpath``) 비교한다 — 에이전트가 workspace 안에
    ``ln -s /etc/passwd`` 를 걸어두고 그걸 entrypoint 로 지정하는 경로를 막는다.
    """
    ep = str(entrypoint or "").strip()
    if not ep:
        raise ForgedToolError("entrypoint(workspace 안 스크립트 경로)가 필요합니다")
    if os.path.isabs(ep):
        raise ForgedToolError("entrypoint 는 workspace 기준 상대 경로여야 합니다")
    root = os.path.realpath(workspace_dir)
    target = os.path.realpath(os.path.join(root, ep))
    if target != root and not target.startswith(root + os.sep):
        raise ForgedToolError("entrypoint 가 workspace 밖을 가리킵니다")
    return target


def validate_spec(
    spec: ForgedToolSpec, workspace_dir: str, *, check_file: bool = True
) -> ForgedToolSpec:
    """저장 전 정규화 + 검증. 문제가 있으면 :class:`ForgedToolError`.

    ``check_file=False`` 는 **파일이 이 파드에 없을 때** 쓴다. 스크립트가
    러너 세션에 있으면 여기서 ``os.path.isfile`` 로 확인할 수 없고, 그걸
    "없음" 으로 읽으면 모든 도구가 조용히 사라진다 — 호출자가 러너에 물어보고
    (:meth:`SandboxSession.exists`) 그 결과를 들고 온다.
    """
    name = str(spec.name or "").strip()
    if not _TOOL_NAME_RE.match(name):
        raise ForgedToolError(
            "도구 이름은 영문으로 시작하고 영문/숫자/_/- 만 쓸 수 있습니다 (최대 64자)"
        )
    spec.name = name
    spec.description = str(spec.description or "").strip() or name
    resolve_runtime(spec.runtime)  # 허용 목록 검사 (경로는 실행 시점에 다시 푼다)
    target = resolve_entrypoint(workspace_dir, spec.entrypoint)
    if check_file and not os.path.isfile(target):
        raise ForgedToolError(
            f"스크립트를 찾을 수 없습니다: {spec.entrypoint} "
            "— 먼저 workspace 에 파일을 만든 뒤 등록하세요"
        )
    if not isinstance(spec.input_schema, dict) or spec.input_schema.get("type") != "object":
        raise ForgedToolError('input_schema 는 {"type": "object", ...} 형태여야 합니다')
    spec.argv = [str(a) for a in (spec.argv or [])]
    spec.env = {str(k): str(v) for k, v in dict(spec.env or {}).items()}
    try:
        spec.timeout_s = float(spec.timeout_s or _DEFAULT_TIMEOUT_S)
    except (TypeError, ValueError):
        spec.timeout_s = _DEFAULT_TIMEOUT_S
    spec.timeout_s = max(1.0, min(_MAX_TIMEOUT_S, spec.timeout_s))
    return spec


# ── 저장소 ────────────────────────────────────────────────────────────


# ── 스펙 저장소 (DB) ──────────────────────────────────────────────────
#
# 스펙은 **파드 밖**에 있어야 한다. xgen-workflow 는 replicas 2 이고 PVC 가
# 없으므로, 파드 로컬 JSON 이면 파드 A 에서 만든 도구가 파드 B 에서 보이지
# 않는다 — 라운드로빈이라 **호출할 때마다 있다 없다 한다.**
#
# 테이블은 xgen-core 가 소유한다 (``xgeny_tool_specs``, APPLICATION_MODELS
# 단일 등록소). 여기서는 읽고 쓰기만 한다.

# ── 실행 도구 ─────────────────────────────────────────────────────────


_MISSING_MOD_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")


def _missing_module_hint(stderr: str) -> str:
    """ModuleNotFoundError 를 **막다른 길이 아니게** 만든다.

    2026-08-18 실증: 에이전트가 이 오류를 "이 환경에선 그 패키지를 못 쓴다"
    로 읽고 기능을 포기했다 — 설치 경로(도구별 dependencies / 세션 PythonEnv /
    ad-hoc pip)가 셋이나 있는데 실패 메시지가 아무것도 가리키지 않았기
    때문이다. 실패한 그 순간, 그 실패에만, 다음 행동을 붙인다.
    """
    m = _MISSING_MOD_RE.search(stderr or "")
    if not m:
        return ""
    mod = m.group(1).split(".")[0]
    return (
        f"\n\n[안내] 파이썬 패키지 '{mod}' 가 이 도구의 실행 환경에 없습니다. 해결 경로:\n"
        f"  1) 이 도구 전용: ForgeTool 로 dependencies 에 pip 패키지명을 넣어 재등록 "
        f"(격리 환경에 자동 설치, 이후 호출부터 적용)\n"
        f"  2) 세션 전체: PythonEnv(action=install) — 이후 세션에서도 유지\n"
        f"  (모듈명과 pip 패키지명이 다를 수 있습니다 — 예: pptx → python-pptx)"
    )


def _child_env(
    spec_env: Dict[str, str], extra_pythonpath: Optional[List[str]] = None
) -> Dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # 로컬 폴백 환경(도구별 + 세션 PythonEnv)을 PYTHONPATH 앞에 얹는다 — 러너 없이도
    # 의존성 도구가 자기 패키지를 import 할 수 있게 한다.
    paths = [p for p in (extra_pythonpath or []) if p and os.path.isdir(p)]
    if paths:
        existing = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    env.update({str(k): str(v) for k, v in (spec_env or {}).items()})
    return env


# ── 로컬(무-sandbox) 파이썬 환경 폴백 ──────────────────────────────────
# 러너(sandbox)가 없을 때도 의존성 도구가 동작하도록, workspace 안에
# `pip install --target <dir>` 로 격리 디렉터리를 만들고 실행 시 PYTHONPATH 에
# 얹는다. 러너의 content-addressed env 를 로컬로 근사한 것 — 같은 핀이면 같은 dir,
# 멱등(marker), workspace 동기화로 파드/재시작을 가로질러 유지된다.
_LOCAL_ENV_ROOT = ".xgeny/tool-envs"  # 도구별 (deps 해시)
_SESSION_LOCAL_ENV = ".xgeny/python-local-env"  # 세션 전체 (PythonEnv 무-sandbox 폴백)
_LOCAL_ENV_TIMEOUT = 600


def _deps_hash(dependencies: List[str]) -> str:
    canon = "\n".join(sorted(str(d).strip() for d in (dependencies or []) if str(d).strip()))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def _tool_env_dir(workspace_dir: str, dependencies: List[str]) -> str:
    return os.path.join(str(workspace_dir), _LOCAL_ENV_ROOT, _deps_hash(dependencies))


def _session_local_env_dir(workspace_dir: str) -> str:
    return os.path.join(str(workspace_dir), _SESSION_LOCAL_ENV)


def _pin_from_target(env_dir: str) -> List[str]:
    """`pip install --target` 디렉터리의 *.dist-info 로부터 name==version 핀 목록."""
    pins: List[str] = []
    try:
        for name in os.listdir(env_dir):
            if name.endswith(".dist-info"):
                base = name[: -len(".dist-info")]
                if "-" in base:
                    pkg, ver = base.rsplit("-", 1)
                    pins.append(f"{pkg.replace('_', '-')}=={ver}")
    except Exception:  # noqa: BLE001
        pass
    return sorted(set(pins))


def _pip_install_target(env_dir: str, dependencies: List[str]) -> "subprocess.CompletedProcess":  # type: ignore  # noqa: F821
    import subprocess

    os.makedirs(env_dir, exist_ok=True)
    tail = [
        "install",
        "--no-input",
        "--disable-pip-version-check",
        "--target",
        env_dir,
        *[str(d) for d in dependencies],
    ]
    # 1) 이 인터프리터의 pip(가장 정확). 2) PATH 의 pip3/pip 실행 파일 폴백 —
    # 일부 환경(uv 등)은 `python -m pip` 가 없어도 pip 실행 파일은 있다.
    bases: List[List[str]] = [[sys.executable, "-m", "pip"]]
    for exe in ("pip3", "pip"):
        if shutil.which(exe):
            bases.append([exe])
    last = None
    for base in bases:
        proc = subprocess.run(
            [*base, *tail], capture_output=True, text=True, timeout=_LOCAL_ENV_TIMEOUT
        )
        last = proc
        if proc.returncode == 0:
            return proc
        # 'No module named pip' 는 이 진입점만의 문제 → 다음 후보 시도.
        # 그 외(패키지명 오류 등)는 실제 실패이므로 즉시 반환.
        if "no module named pip" not in ((proc.stderr or "") + (proc.stdout or "")).lower():
            return proc
    return last  # type: ignore[return-value]


async def _ensure_local_env(workspace_dir: str, dependencies: List[str]) -> Tuple[str, List[str]]:
    """sandbox 없이 workspace 로컬 격리 디렉터리에 deps 를 설치한다.

    반환 ``("local:<hash>", pinned)``. 멱등 — marker 가 있으면 재설치하지 않는다.
    """
    deps = [str(d).strip() for d in (dependencies or []) if str(d).strip()]
    h = _deps_hash(deps)
    env_dir = _tool_env_dir(workspace_dir, deps)
    marker = os.path.join(env_dir, ".installed")
    if os.path.isfile(marker):
        try:
            pinned = [ln for ln in open(marker, encoding="utf-8").read().splitlines() if ln.strip()]
        except Exception:  # noqa: BLE001
            pinned = deps
        return f"local:{h}", pinned or deps
    proc = await asyncio.to_thread(_pip_install_target, env_dir, deps)
    if getattr(proc, "returncode", 1) != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "pip install 실패")[-1500:])
    pinned = _pin_from_target(env_dir) or deps
    try:
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("\n".join(pinned))
    except Exception:  # noqa: BLE001
        pass
    return f"local:{h}", pinned


def _local_pythonpath(workspace_dir: str, dependencies: List[str]) -> List[str]:
    """도구별 로컬 env + 세션 로컬 env 를 PYTHONPATH 후보로 돌려준다(존재하는 것만)."""
    out: List[str] = []
    if dependencies:
        out.append(_tool_env_dir(workspace_dir, dependencies))
    out.append(_session_local_env_dir(workspace_dir))
    return out


class ForgedScriptTool:
    """저장된 스크립트 하나를 도구로 노출한다 (executor ``Tool`` 계약).

    ``xgen_agent_runtime.tools.base.Tool`` 을 런타임에 상속한다 — 모듈 임포트 시점에
    executor 를 요구하지 않기 위해 :func:`_tool_base` 로 늦게 묶는다.
    """

    def __init__(
        self,
        spec: ForgedToolSpec,
        *,
        workspace_dir: str,
        store: Optional["ForgedToolSpecStore"] = None,
    ) -> None:
        self._spec = spec
        self._workspace = str(workspace_dir)
        self._store = store

    # ── executor Tool 계약 ──
    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        deps = self._spec.dependencies
        env_note = (
            f" 격리 환경에 {', '.join(deps[:5])}" + (" 외" if len(deps) > 5 else "") + " 설치됨."
            if deps
            else ""
        )
        return (
            f"{self._spec.description}\n"
            f"(이 에이전트가 만들어 저장한 도구 — `{self._spec.runtime} "
            f"{self._spec.entrypoint}` 를 workspace 에서 실행합니다.{env_note})"
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._spec.input_schema

    def capabilities(self, *_a: Any, **_k: Any) -> Any:
        from xgen_agent_runtime.tools.base import ToolCapabilities

        # 무엇을 하는 스크립트인지 알 수 없으니 보수적으로 — 직렬 실행.
        return ToolCapabilities(concurrency_safe=False, timeout_s=self._spec.timeout_s + 5)

    async def execute(self, input: Dict[str, Any], context: Any) -> Any:  # noqa: A002 - executor 계약
        from xgen_agent_runtime.tools.base import ToolResult

        spec = self._spec
        payload_early = json.dumps(dict(input or {}), ensure_ascii=False).encode("utf-8")

        # 러너가 붙어 있으면 **거기서** 돈다. 스크립트도 그 트리에 있고,
        # 이 도구가 요구한 패키지도 그쪽 환경에만 있다 — 여기서 돌리면
        # ModuleNotFoundError 가 나거나, 더 나쁘게는 이 파드의 다른 버전으로
        # 조용히 다른 답을 낸다.
        sandbox = getattr(context, "sandbox", None)
        if sandbox is not None:
            try:
                res = await _run_in_sandbox(sandbox, spec, payload_early)
            except Exception as exc:  # noqa: BLE001
                self._record(str(exc))
                return ToolResult(content=f"도구 '{spec.name}' 실행 실패: {exc}", is_error=True)
            return self._interpret(
                res.stdout.decode("utf-8", "replace")[:_STDOUT_CAP],
                res.stderr.decode("utf-8", "replace")[-_STDERR_CAP:],
                int(res.rc),
            )

        try:
            runtime = resolve_runtime(spec.runtime)
            script = resolve_entrypoint(self._workspace, spec.entrypoint)
        except ForgedToolError as exc:
            self._record(str(exc))
            return ToolResult(
                content=f"도구 '{spec.name}' 을(를) 실행할 수 없습니다: {exc}", is_error=True
            )
        if not os.path.isfile(script):
            msg = (
                f"스크립트가 없어졌습니다: {spec.entrypoint} "
                "— workspace 에서 삭제되었을 수 있습니다."
            )
            self._record(msg)
            return ToolResult(content=msg, is_error=True)

        # 러너 없이 로컬 실행: 의존성이 있으면 로컬 격리 env 를 보장(멱등)하고,
        # 도구별 env + 세션 PythonEnv 로컬 폴백을 PYTHONPATH 에 얹어 import 가 되게 한다.
        if spec.dependencies:
            try:
                await _ensure_local_env(self._workspace, spec.dependencies)
            except Exception as exc:  # noqa: BLE001 — 준비 실패해도 실행은 시도
                logger.warning("로컬 도구 환경 준비 실패 (실행은 시도): %s", exc)
        pythonpath = _local_pythonpath(self._workspace, spec.dependencies)

        payload = json.dumps(dict(input or {}), ensure_ascii=False).encode("utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                runtime,
                script,
                *spec.argv,
                cwd=self._workspace,
                env=_child_env(spec.env, pythonpath),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            self._record(str(exc))
            return ToolResult(content=f"도구 '{spec.name}' 기동 실패: {exc}", is_error=True)

        try:
            out, err = await asyncio.wait_for(proc.communicate(payload), timeout=spec.timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:  # noqa: BLE001
                pass
            msg = f"도구 '{spec.name}' 이(가) {spec.timeout_s:g}초 안에 끝나지 않았습니다"
            self._record(msg)
            return ToolResult(content=msg, is_error=True)

        return self._interpret(
            out.decode("utf-8", "replace")[:_STDOUT_CAP],
            err.decode("utf-8", "replace")[-_STDERR_CAP:],
            proc.returncode or 0,
        )

    def _interpret(self, stdout: str, stderr: str, rc: int) -> Any:
        """실행 결과 → ToolResult. **로컬과 러너가 같은 해석을 쓴다.**

        갈라 두면 "이 백엔드에서만 다르게 실패하는 도구" 가 생기고, 그건
        에이전트가 원인을 짚을 수 없는 종류의 차이다.
        """
        from xgen_agent_runtime.tools.base import ToolResult

        spec = self._spec
        if rc != 0:
            msg = f"도구 '{spec.name}' 실패 (exit {rc})"
            self._record(f"{msg}: {stderr[-500:]}")
            return ToolResult(
                content=f"{msg}\n--- stderr ---\n{stderr or '(없음)'}"
                + _missing_module_hint(stderr),
                is_error=True,
                metadata={"exit_code": rc, "stderr": stderr},
            )

        # 계약: stdout 은 JSON. 아니면 원문 텍스트로 돌려준다 (스크립트가
        # 사람이 읽는 출력을 내는 경우까지 죽이지 않는다).
        content: Any = stdout.strip()
        try:
            parsed = json.loads(stdout) if stdout.strip() else {}
            content = parsed
            if isinstance(parsed, dict) and parsed.get("error"):
                self._record(str(parsed["error"]))
                return ToolResult(content=parsed, is_error=True)
        except json.JSONDecodeError:
            pass
        self._record(None)
        return ToolResult(content=content, metadata={"stderr": stderr} if stderr else {})

    def _record(self, error: Optional[str]) -> None:
        """호출 통계 기록 (동기 DB UPDATE).

        이 도구의 execute 는 CLI 브릿지 경유 시 **서빙 루프**, sub-agent 경유
        시 **공유 위임 루프**에서 돈다 — 루프가 감지되면 워커로 보내고 결과는
        기다리지 않는다 (통계는 유실보다 루프 정지가 훨씬 비싸다).
        """
        if self._store is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._store.record_call(self._spec.name, error=error)
            return
        task = loop.create_task(
            asyncio.to_thread(self._store.record_call, self._spec.name, error=error)
        )
        task.add_done_callback(_swallow_record_failure)


def _swallow_record_failure(task: "asyncio.Task") -> None:
    exc = task.exception() if not task.cancelled() else None
    if exc is not None:
        logger.debug("도구 호출 통계 기록 실패 (무시)", exc_info=exc)


# ── 격리 실행 환경 ────────────────────────────────────────────────────


async def _ensure_env(sandbox: Any, dependencies: List[str]) -> tuple:
    """의존성 목록 → ``(env_id, 확정된 핀 목록)``.

    에이전트는 ``["pandas"]`` 처럼 버전 없이 적어도 된다 — 러너가 정확한
    버전으로 확정한 뒤 그 결과의 sha256 을 env_id 로 낸다. **확정된 핀을
    스펙에 기록하는 것이 중요하다**: 안 그러면 같은 도구가 시점마다 다른
    것을 설치하게 되고, "어제는 됐는데 오늘은 안 된다" 는 원인을 찾을 수 없다.

    같은 핀 목록이면 같은 id 라, 에이전트가 100개여도 ``pandas`` 환경은
    하나이고 빌드도 한 번이다. 멱등이라 "이미 있는지" 를 먼저 물을 필요가 없다.
    """
    ensure = getattr(sandbox, "ensure_env", None)
    if not callable(ensure):
        raise RuntimeError("이 실행 기반은 도구 환경을 만들 수 없습니다")
    env_id, pinned = await ensure(list(dependencies))
    if not env_id:
        raise RuntimeError("러너가 환경 id 를 돌려주지 않았습니다")
    return str(env_id), [str(p) for p in (pinned or [])]


async def _run_in_sandbox(sandbox: Any, spec: "ForgedToolSpec", payload: bytes) -> Any:
    """도구 스크립트를 러너 세션에서 실행한다.

    로컬 실행과 **같은 계약**이다 (stdin JSON → stdout JSON). 달라지는 것은
    어디서 도는가와, 이 도구가 요구한 패키지가 있는 인터프리터로 돈다는 것뿐이다.

    ``cwd`` 를 **명시한다.** entrypoint 는 workspace 기준 상대 경로이고, 다른 모든
    도구(sb_run)는 이미 세션 workdir 을 명시해서 넘긴다. 여기만 비워 두면 실행
    위치가 러너의 기본값에 달리고, 그 기본값이 바뀌는 날 "도구는 등록됐는데
    스크립트를 못 찾는다"가 된다 — 아무 로그도 그 이유를 말해 주지 않는다.
    """
    from xgen_agent_runtime.tools._xgeny_sandbox import _cwd

    kwargs: Dict[str, Any] = {
        "stdin": payload,
        "env": dict(spec.env or {}),
        "timeout_s": float(spec.timeout_s),
        "cwd": _cwd(sandbox, ""),
    }
    # "local:" 은 무-sandbox 로컬 폴백 마커라 러너 env_id 가 아니다 — 넘기지 않는다
    # (러너에 붙어 실행되면 deps 가 없을 수 있으나, 로컬 폴백으로 만든 도구를 러너에서
    #  도로 돌리는 경우는 드물고, 넘기면 러너가 알 수 없는 env 라 실패한다).
    if spec.env_id and not spec.env_id.startswith("local:"):
        kwargs["env_id"] = spec.env_id
    return await sandbox.exec([spec.runtime, spec.entrypoint, *spec.argv], **kwargs)


# ── 제작/관리 도구 (에이전트가 쓰는 것) ───────────────────────────────


class ForgeTool:
    """에이전트가 자기 도구를 만들어 저장하는 도구."""

    def __init__(
        self,
        *,
        workflow_id: str,
        workspace_dir: str,
        registry: Any,
        store: "ForgedToolSpecStore",
    ) -> None:
        self._workflow_id = str(workflow_id)
        self._workspace = str(workspace_dir)
        self._registry = registry
        self._store = store

    @property
    def name(self) -> str:
        return "ForgeTool"

    @property
    def description(self) -> str:
        return (
            "Turn a script you wrote in your workspace into a REUSABLE TOOL that "
            "persists across sessions — this is how you permanently extend yourself. "
            "Write the script first (Write), then register it here. Contract: your "
            "script receives the tool input as JSON on stdin and must print its result "
            'as JSON on stdout (print {"error": "..."} for a handled failure). '
            "VERIFY-BEFORE-REGISTER: when you register, the tool is RUN ONCE with your "
            "`test_input` on the exact same path a real call takes (input is validated "
            "against `input_schema`, then executed). It is registered and made callable "
            "ONLY if that test run succeeds. If the test fails, the tool is NOT exposed "
            "and this call returns the failure output — read it, fix your script (or the "
            "input_schema / test_input), and register again. So provide a representative "
            "`test_input` that actually exercises the script. The tool is callable from "
            "your NEXT turn and restored automatically in every future session. "
            "Re-registering the same name updates it (and re-runs the verification)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Tool name the model will call (letters, digits, _ and -).",
                },
                "description": {
                    "type": "string",
                    "description": "What the tool does and when to use it — written for yourself, later.",
                },
                "entrypoint": {
                    "type": "string",
                    "description": "Path to the script, RELATIVE to your workspace (e.g. 'tools/fx.py').",
                },
                "input_schema": {
                    "type": "object",
                    "description": "JSON Schema (type: object) for the tool's input. Omit for a no-argument tool.",
                },
                "runtime": {
                    "type": "string",
                    "enum": sorted(_RUNTIMES),
                    "description": "Interpreter to run the script with. Default python3.",
                },
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra command-line arguments appended after the script path.",
                },
                "env": {
                    "type": "object",
                    "description": "Literal environment variables for the script. Do NOT put secrets here.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": f"Per-call time limit in seconds (max {_MAX_TIMEOUT_S:g}). Default {_DEFAULT_TIMEOUT_S:g}.",
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Python packages the script imports, as pip requirements "
                        "(e.g. ['pandas', 'httpx>=0.27', 'tabulate==0.9.0']). You do NOT need "
                        "to know exact versions — they are resolved and pinned for you, and the "
                        "pinned set is what gets installed from then on, so the tool cannot "
                        "drift later. They are installed ONCE into an isolated environment that "
                        "is reused on every later call, so declare them here instead of "
                        "pip-installing at run time. Leave empty if the script only uses the "
                        "standard library."
                    ),
                },
                "test_input": {
                    "type": "object",
                    "description": (
                        "A representative sample input the tool is TESTED with before it is "
                        "registered. The tool is run once with this exact value (validated "
                        "against input_schema, then executed) and is registered ONLY if that "
                        "run succeeds. Use realistic values that exercise the script's real "
                        "path — not a placeholder. If input_schema declares required fields, "
                        "test_input MUST provide them. Omit only for a genuinely no-argument "
                        "tool (it is then tested with {})."
                    ),
                },
            },
            "required": ["name", "description", "entrypoint"],
        }

    def capabilities(self, *_a: Any, **_k: Any) -> Any:
        from xgen_agent_runtime.tools.base import ToolCapabilities

        return ToolCapabilities(concurrency_safe=False)

    async def execute(self, input: Dict[str, Any], context: Any) -> Any:  # noqa: A002
        from xgen_agent_runtime.tools.base import ToolResult

        args = dict(input or {})
        name = str(args.get("name") or "").strip()
        # 스토어는 동기 DB — 이 코루틴이 도는 루프(서빙/위임)를 잡으면 안 된다.
        existing = await asyncio.to_thread(self._store.get, name) if name else None
        # 내장/다른 도구 이름을 덮어쓰면 그 도구가 사라진다 — 자기 것만 갱신 허용.
        if existing is None and name and self._registry is not None:
            try:
                if self._registry.get(name) is not None:
                    return ToolResult(
                        content=(
                            f"'{name}' 은(는) 이미 있는 도구 이름입니다 — 다른 이름을 쓰세요."
                        ),
                        is_error=True,
                    )
            except Exception:  # noqa: BLE001
                pass
        spec = ForgedToolSpec(
            name=name,
            description=str(args.get("description") or ""),
            entrypoint=str(args.get("entrypoint") or ""),
            runtime=str(args.get("runtime") or "python3"),
            input_schema=args.get("input_schema") or {"type": "object", "properties": {}},
            argv=list(args.get("argv") or []),
            env=dict(args.get("env") or {}),
            timeout_s=args.get("timeout_s") or _DEFAULT_TIMEOUT_S,
            dependencies=[
                str(d).strip() for d in (args.get("dependencies") or []) if str(d).strip()
            ],
            created_at=existing.created_at if existing else 0.0,
            calls=existing.calls if existing else 0,
            errors=existing.errors if existing else 0,
        )
        # 스크립트가 어디 있는지는 실행지가 정한다. 러너가 붙어 있으면 파일도
        # 거기 있으므로 이 파드의 파일시스템을 봐서는 안 된다 — 그러면 항상
        # "스크립트를 찾을 수 없습니다" 가 된다.
        _sandbox = getattr(context, "sandbox", None)
        try:
            validate_spec(spec, self._workspace, check_file=(_sandbox is None))
        except ForgedToolError as exc:
            return ToolResult(content=f"도구를 만들 수 없습니다: {exc}", is_error=True)
        if _sandbox is not None:
            try:
                found = await _sandbox.exists(spec.entrypoint)
            except Exception as exc:  # noqa: BLE001 — 확인 실패를 '없음' 으로 읽지 않는다
                logger.warning("스크립트 존재 확인 실패 (등록은 진행): %s", exc)
                found = True
            if not found:
                return ToolResult(
                    content=(
                        f"스크립트를 찾을 수 없습니다: {spec.entrypoint} "
                        "— 먼저 workspace 에 파일을 만든 뒤 등록하세요"
                    ),
                    is_error=True,
                )

        # 의존성이 있으면 **등록 시점에** 환경을 세운다. 첫 호출로 미루면
        # 에이전트가 도구를 부른 순간 몇 분을 기다리게 되고, 그 지연이
        # "도구가 고장났다" 로 읽힌다.
        if spec.dependencies:
            sandbox = getattr(context, "sandbox", None)
            if sandbox is None:
                # 러너(sandbox)가 없으면 workspace 로컬 격리 환경으로 폴백한다 —
                # pip install --target 후 실행 시 PYTHONPATH 에 얹어, 러너 없이도
                # 의존성 도구가 그대로 동작한다(모든 Agent-XGeny 가 자기 환경을 갖도록).
                try:
                    spec.env_id, _pinned = await _ensure_local_env(
                        self._workspace, spec.dependencies
                    )
                    if _pinned:
                        spec.dependencies = _pinned
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(
                        content=(
                            f"로컬 의존성 설치에 실패했습니다: {exc}\n"
                            "패키지 이름과 버전을 확인하세요(모듈명≠pip명일 수 있음: pptx→python-pptx)."
                        ),
                        is_error=True,
                    )
            else:
                try:
                    spec.env_id, _pinned = await _ensure_env(sandbox, spec.dependencies)
                    if _pinned:
                        # 확정된 핀으로 갈아 끼운다 — 재현의 근거는 에이전트가 적은
                        # 이름이 아니라 실제로 설치된 버전이다.
                        spec.dependencies = _pinned
                except Exception as exc:  # noqa: BLE001 — 원인을 에이전트가 읽어야 한다
                    return ToolResult(
                        content=(
                            f"의존성 설치에 실패했습니다: {exc}\n패키지 이름과 버전을 확인하세요."
                        ),
                        is_error=True,
                    )

        # ── 등록 전 검증 (verify-before-register) ──
        # 스크립트를 **에이전트가 부를 때와 똑같은 경로**로 한 번 실행한다.
        # 통과해야만 verified=사용가능 으로 등록한다. 실패하면 미검증 초안으로만
        # 저장하고(노출 안 함) 실패 출력을 그대로 돌려준다 — 미검증(깨진) 도구가
        # 그대로 등록/호출되는 것을 원천 차단한다. (프롬프트가 아니라 로직으로.)
        _test_input = args.get("test_input")
        if not isinstance(_test_input, dict):
            _test_input = {}
        _test_result = None
        try:
            _test_result = await run_forged_tool_test(spec, self._workspace, context, _test_input)
            _test_ok = not bool(getattr(_test_result, "is_error", False))
            _fail_text = "" if _test_ok else _test_failure_text(_test_result)
        except Exception as exc:  # noqa: BLE001 — 테스트 자체가 터진 것도 실패다
            _test_ok = False
            _fail_text = f"테스트 실행 중 예외: {exc}"
        spec.verified = bool(_test_ok)
        spec.last_test_error = "" if _test_ok else _fail_text[:2000]

        try:
            await asyncio.to_thread(self._store.save, spec)
        except Exception as exc:  # noqa: BLE001 — 원인을 에이전트가 읽어야 한다
            # 저장이 실패했는데 성공했다고 답하면, 다음 턴에 "그런 도구 없음" 이
            # 되고 에이전트는 자기가 뭘 잘못했는지 알 수 없다. 여기서 끊는다.
            logger.warning("도구 저장 실패: %s/%s", self._workflow_id, spec.name, exc_info=True)
            return ToolResult(
                content=(
                    f"도구 '{spec.name}' 를 저장하지 못했습니다: {exc}\n"
                    "스크립트는 workspace 에 그대로 있으니 잠시 뒤 다시 등록해 보세요."
                ),
                is_error=True,
            )

        if not _test_ok:
            # 미검증 — **등록/노출하지 않는다.** 초안은 저장돼 있으니(사람이
            # [도구] 화면에서 볼 수 있음) 사라지진 않지만, 에이전트는 고쳐서 다시
            # 등록해야 쓸 수 있다. 계약 위반(입력을 stdin JSON 으로 안 읽거나
            # 결과를 stdout JSON 으로 안 내는 것)이 여기서 대부분 걸린다.
            return ToolResult(
                content={
                    "ok": False,
                    "name": spec.name,
                    "verified": False,
                    "registered": False,
                    "test_input": _test_input,
                    "test_output": _fail_text or "(출력 없음)",
                    "message": (
                        f"도구 '{spec.name}' 는 등록 전 테스트에 실패해 **등록되지 않았습니다** "
                        "(미검증 초안으로만 저장). 위 test_output 을 보고 원인을 고치세요. "
                        "계약: 입력을 stdin 의 JSON 으로 받아 결과를 stdout 에 JSON 으로 "
                        '출력해야 하며, 처리된 실패는 {"error": "..."} 로 알립니다. '
                        "스크립트(또는 input_schema/test_input)를 고친 뒤 다시 등록하면 "
                        "재검증됩니다."
                    ),
                },
                is_error=True,
            )

        registered = _register_one(self._registry, spec, self._workspace, self._store, replace=True)
        verb = "갱신" if existing else "등록"
        return ToolResult(
            content={
                "ok": True,
                "name": spec.name,
                "action": "updated" if existing else "created",
                "verified": True,
                "dependencies": list(spec.dependencies),
                "env_id": spec.env_id or None,
                "message": (
                    f"도구 '{spec.name}' {verb} 완료 — 등록 전 테스트를 통과(검증됨)했고, "
                    "다음 턴부터 호출할 수 있으며 이후 세션에서도 자동으로 복원됩니다."
                    + (
                        f" 의존성 {len(spec.dependencies)}개를 격리 환경에 설치했습니다 "
                        "(다음 호출부터는 설치 없이 바로 실행)."
                        if spec.dependencies
                        else ""
                    )
                    + ("" if registered else " (이번 턴 등록은 건너뜀 — 다음 세션에서 복원)")
                ),
            }
        )


class ListForgedTools:
    """저장된 자기 도구 목록."""

    def __init__(self, *, workflow_id: str, store: "ForgedToolSpecStore") -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "ListForgedTools"

    @property
    def description(self) -> str:
        return "List the tools you built and saved (name, what it does, script, usage stats)."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def capabilities(self, *_a: Any, **_k: Any) -> Any:
        from xgen_agent_runtime.tools.base import ToolCapabilities

        return ToolCapabilities(read_only=True, concurrency_safe=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> Any:  # noqa: A002
        from xgen_agent_runtime.tools.base import ToolResult

        tools = [
            {
                "name": s.name,
                "description": s.description,
                "entrypoint": s.entrypoint,
                "runtime": s.runtime,
                "enabled": s.enabled,
                "calls": s.calls,
                "errors": s.errors,
                "last_error": s.last_error,
            }
            for s in sorted(await asyncio.to_thread(self._store.list), key=lambda x: x.name)
        ]
        return ToolResult(content={"tools": tools, "count": len(tools)})


class DeleteForgedTool:
    """저장된 자기 도구 삭제 (스크립트 파일은 남긴다)."""

    def __init__(self, *, workflow_id: str, registry: Any, store: "ForgedToolSpecStore") -> None:
        self._store = store
        self._registry = registry

    @property
    def name(self) -> str:
        return "DeleteForgedTool"

    @property
    def description(self) -> str:
        return (
            "Remove one of the tools you built. The script file stays in your "
            "workspace; only the tool registration is removed."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Tool name to remove."}},
            "required": ["name"],
        }

    def capabilities(self, *_a: Any, **_k: Any) -> Any:
        from xgen_agent_runtime.tools.base import ToolCapabilities

        return ToolCapabilities(destructive=True, concurrency_safe=False)

    async def execute(self, input: Dict[str, Any], context: Any) -> Any:  # noqa: A002
        from xgen_agent_runtime.tools.base import ToolResult

        name = str((input or {}).get("name") or "").strip()
        if not await asyncio.to_thread(self._store.get, name):
            return ToolResult(content=f"'{name}' 은(는) 저장된 도구가 아닙니다", is_error=True)
        await asyncio.to_thread(self._store.delete, name)
        try:
            if self._registry is not None:
                self._registry.unregister(name)
        except Exception:  # noqa: BLE001
            pass
        return ToolResult(content={"ok": True, "name": name, "message": f"도구 '{name}' 삭제됨"})


# ── 등록 (세션 빌드 시 복원 + 제작 도구 배선) ─────────────────────────


def _bind_tool_base(obj: Any) -> Any:
    """executor ``Tool`` 을 런타임 베이스로 묶은 인스턴스를 돌려준다.

    이 모듈은 executor 없이도 임포트 가능해야 한다 (스토어/검증은 API 서버가
    executor 를 로드하지 않고도 쓴다). 그래서 상속을 임포트 시점이 아니라
    등록 시점에 만든다.
    """
    from xgen_agent_runtime.tools.base import Tool

    cls = type(obj)
    if issubclass(cls, Tool):
        return obj
    bound = type(f"{cls.__name__}Bound", (cls, Tool), {})
    obj.__class__ = bound
    return obj


def _do_register(registry: Any, tool: Any, core: Optional[bool]) -> None:
    """executor 레지스트리는 ``register(tool, core=...)`` 를 받는다 (점진공개)."""
    if core is None:
        registry.register(tool)
    else:
        registry.register(tool, core=core)


def _register_one(
    registry: Any,
    spec: ForgedToolSpec,
    workspace_dir: str,
    store: "ForgedToolSpecStore",
    *,
    replace: bool = False,
    core: Optional[bool] = None,
) -> bool:
    if registry is None:
        return False
    try:
        if replace:
            try:
                registry.unregister(spec.name)
            except Exception:  # noqa: BLE001 — 없으면 그만
                pass
        _do_register(
            registry,
            _bind_tool_base(ForgedScriptTool(spec, workspace_dir=workspace_dir, store=store)),
            core,
        )
        return True
    except Exception:  # noqa: BLE001 — 도구 하나가 세션을 깨뜨리지 않는다
        logger.warning("forged tool 등록 실패 (스킵): %s", spec.name, exc_info=True)
        return False


async def run_forged_tool_test(
    spec: ForgedToolSpec,
    workspace_dir: str,
    context: Any,
    test_input: Optional[Dict[str, Any]] = None,
) -> Any:
    """스펙을 **에이전트가 부를 때와 똑같은 경로**로 한 번 실행한다.

    같은 ``ForgedScriptTool`` · 같은 ``RegistryRouter`` (입력 jsonschema 검증
    포함) · 같은 실행지(``context.sandbox``)를 쓴다 — 그래서 여기서 통과하면
    에이전트도 통과한다. **등록 게이트(ForgeTool)** 와 **사람의 [테스트] 버튼**
    (geny_tools ``/test``) 이 이 함수를 공유해, "테스트는 되는데 에이전트가
    쓰면 안 되는" 상태가 생기지 않게 한다.

    반환은 도구의 ``ToolResult`` (``is_error`` 로 성공/실패 판정). 통계(store)는
    건드리지 않는다 (``store=None``).
    """
    from xgen_agent_runtime.stages.s10_tool import RegistryRouter
    from xgen_agent_runtime.tools import ToolRegistry

    ti = test_input if isinstance(test_input, dict) else {}
    capped = min(float(spec.timeout_s or _DEFAULT_TIMEOUT_S), _TEST_TIMEOUT_CAP_S)
    spec_for_test = ForgedToolSpec.from_dict({**spec.to_dict(), "timeout_s": capped})
    tool = _bind_tool_base(ForgedScriptTool(spec_for_test, workspace_dir=workspace_dir, store=None))
    registry = ToolRegistry()
    registry.register(tool, core=True)
    return await RegistryRouter(registry).route(spec.name, ti, context)


def _test_failure_text(result: Any) -> str:
    """테스트 실패 ToolResult 에서 사람이/에이전트가 읽을 사유를 뽑는다."""
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content.strip()
    try:
        return json.dumps(content, ensure_ascii=False)[:2000]
    except Exception:  # noqa: BLE001
        return str(content)[:2000]


def register_forged_tools(
    registry: Any,
    *,
    workflow_id: str,
    workspace_dir: str,
    store: "ForgedToolSpecStore",
    core: Optional[bool] = None,
    sandboxed: bool = False,
) -> Dict[str, Any]:
    """저장된 도구를 복원하고, 제작/관리 도구를 배선한다.

    ``core`` 는 점진공개 정책 (내장 도구와 동일하게 전달). 단 **제작 도구
    3종은 항상 core** 다 — ToolSearch 뒤에 숨기면 모델이 자기확장 능력이
    있다는 걸 모른 채 지나간다 (Geny 위임 도구에서 실증된 회귀와 같은 함정).

    반환 ``{"restored": [...], "authoring": [...]}``.

    ``sandboxed`` — 스크립트가 러너 세션에 있으면 이 파드에서 파일 존재를
    확인할 수 없다. 그걸 "없음" 으로 읽으면 **모든 도구가 조용히 사라진다.**
    실제로 없는 경우는 호출 시점에 러너가 알려 준다 (그때 메시지가 더 정확하다).
    """
    if not workflow_id:
        return {"restored": [], "authoring": []}
    restored: List[str] = []
    for spec in store.list():
        # 미검증(등록 전 실행 테스트 미통과) 도구는 에이전트에게 노출하지
        # 않는다 — 깨진 도구가 호출되어 실패하는 것을 원천 차단한다. 사람이
        # [도구] 화면에서 테스트로 통과시키거나, 에이전트가 고쳐 다시 등록하면
        # verified 가 되어 다음 세션부터 복원된다.
        if not spec.enabled or not spec.verified:
            continue
        try:
            validate_spec(spec, workspace_dir, check_file=not sandboxed)
        except ForgedToolError as exc:
            # 스크립트가 지워졌거나 규칙이 바뀌었다 — 조용히 빠뜨리지 않고 남긴다.
            logger.info("forged tool '%s' 복원 건너뜀: %s", spec.name, exc)
            continue
        if _register_one(registry, spec, workspace_dir, store, core=core):
            restored.append(spec.name)

    # PythonEnv — 세션 파이썬 환경 관리. 제작 도구와 같은 자기확장 축이라
    # 같은 자리에서, 같은 core 정책으로 등록한다 (숨기면 에이전트는 자기
    # 환경에 패키지를 깔 수 있다는 걸 모른 채 ModuleNotFoundError 앞에서
    # 후퇴한다 — 2026-08-18 실증).
    from xgen_agent_runtime.host.python_env import PythonEnvTool

    authoring: List[str] = []
    for tool in (
        ForgeTool(
            workflow_id=workflow_id,
            workspace_dir=workspace_dir,
            registry=registry,
            store=store,
        ),
        ListForgedTools(workflow_id=workflow_id, store=store),
        DeleteForgedTool(workflow_id=workflow_id, registry=registry, store=store),
        PythonEnvTool(workflow_id=workflow_id, workspace_dir=workspace_dir),
    ):
        try:
            _do_register(registry, _bind_tool_base(tool), None if core is None else True)
            authoring.append(tool.name)
        except Exception:  # noqa: BLE001
            logger.warning("forged tool 제작 도구 등록 실패: %s", tool.name, exc_info=True)
    if restored or authoring:
        logger.info(
            "agents/geny: 저장된 도구 %d개 복원 (%s) + 제작 도구 %d개",
            len(restored),
            ",".join(restored) or "-",
            len(authoring),
        )
    return {"restored": restored, "authoring": authoring}


def forged_tool_instances(
    *,
    workflow_id: str,
    workspace_dir: str,
    store: "ForgedToolSpecStore",
    registry: Any = None,
    sandboxed: bool = False,
) -> List[Any]:
    """CLI 브릿지용 — 복원 대상 + 제작 도구의 **인스턴스** 목록.

    CLI 경로는 클래스 맵으로 도구를 만들지만 forged tool 은 에이전트별 스펙을
    들고 있어야 하므로 인스턴스로 넘긴다.

    ``sandboxed`` — 스크립트가 러너 세션에 있으면 이 파드에서 파일 존재를
    확인할 수 없다. 그걸 "없음" 으로 읽으면 **모든 도구가 조용히 사라진다.**
    실제로 없는 경우는 호출 시점에 러너가 알려 준다 (그때 메시지가 더 정확하다).
    """
    if not workflow_id:
        return []
    out: List[Any] = []
    for spec in store.list():
        # 미검증 도구는 노출하지 않는다 (register_forged_tools 와 동일 방침).
        if not spec.enabled or not spec.verified:
            continue
        try:
            validate_spec(spec, workspace_dir, check_file=not sandboxed)
        except ForgedToolError:
            continue
        out.append(
            _bind_tool_base(ForgedScriptTool(spec, workspace_dir=workspace_dir, store=store))
        )
    from xgen_agent_runtime.host.python_env import PythonEnvTool

    out.append(
        _bind_tool_base(
            ForgeTool(
                workflow_id=workflow_id,
                workspace_dir=workspace_dir,
                registry=registry,
                store=store,
            )
        )
    )
    out.append(_bind_tool_base(ListForgedTools(workflow_id=workflow_id, store=store)))
    out.append(
        _bind_tool_base(
            DeleteForgedTool(
                workflow_id=workflow_id,
                registry=registry,
                store=store,
            )
        )
    )
    out.append(_bind_tool_base(PythonEnvTool(workflow_id=workflow_id, workspace_dir=workspace_dir)))
    return out
