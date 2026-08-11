"""XGeny 도구가 코드를 실행하는 곳 — ``xgen-workflow-sandbox`` 세션.

**컨테이너가 아니다.** ``docker`` 도, ``container_name`` 도, 호스트↔컨테이너
경로 변환도 없다. 에이전트를 태우는 서비스와 코드를 돌리는 서비스가 같은 절대
경로를 쓰기 때문에 (호스트가 두 루트를 같은 문자열로 맞춘다) **변환할 좌표계가
애초에 하나뿐이다.**

런타임은 :class:`XgenySandbox` 프로토콜만 안다. 그 뒤가 HTTP 인지 인프로세스인지
로컬 디렉터리인지는 호스트가 정한다 — 그래서 이 모듈은 stdlib 밖을 import 하지
않고, 테스트는 가짜 구현 하나로 파일/셸 도구 전부를 검증할 수 있다.

파일 읽기·쓰기가 :meth:`~XgenySandbox.read_bytes` / :meth:`write_bytes` 라는
**1급 연산**인 것이 중요하다. 이걸 셸 명령(``cat``, ``sh -c 'cat > …'``)으로
흉내내면 파일 하나 읽는 데 프로세스가 하나 뜨고, 실패가 "명령 실패"로 뭉개져
"파일이 없다"와 "권한이 없다"를 구분할 수 없게 된다.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "ExecResult",
    "SandboxError",
    "SandboxPathError",
    "XgenySandbox",
    "sandbox_path",
    "sandbox_root",
    "sb_read_bytes",
    "sb_run",
    "sb_write_bytes",
]


class SandboxError(RuntimeError):
    """샌드박스에 닿을 수 없거나 요청을 수행하지 못했다."""


class SandboxPathError(SandboxError):
    """세션 루트 밖을 가리키는 경로.

    가드가 여기 있는 이유: 도구마다 각자 막으면 새로 추가되는 도구가 매번
    빠뜨린다. 샌드박스로 나가는 모든 경로는 :func:`sandbox_path` 를 지난다.
    """


@dataclass(frozen=True)
class ExecResult:
    """명령 한 번의 결과. 바이트 그대로 — 디코딩은 부르는 쪽 몫이다."""

    rc: int
    stdout: bytes
    stderr: bytes

    @property
    def ok(self) -> bool:
        return self.rc == 0


@runtime_checkable
class XgenySandbox(Protocol):
    """에이전트 하나의 실행 세션.

    구현체는 :mod:`editor.geny_bridge.sandbox_mount`(xgen-workflow) 의 HTTP
    클라이언트다. 테스트는 같은 모양의 로컬 구현을 쓴다.
    """

    #: 세션의 작업 루트 — **절대 경로**. 이 밖으로는 나갈 수 없다.
    workdir: str

    async def ensure(self) -> None:
        """세션을 살아 있게 만든다. 멱등 — 몇 번 불러도 같다."""
        ...

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: Optional[str] = None,
        stdin: Optional[bytes] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout_s: float = 120.0,
    ) -> ExecResult:
        ...

    async def read_bytes(self, path: str) -> bytes:
        """없으면 :class:`FileNotFoundError`."""
        ...

    async def write_bytes(self, path: str, data: bytes) -> int:
        """상위 디렉터리는 알아서 만든다. 쓴 바이트 수를 돌려준다."""
        ...


# ── 경로 ──────────────────────────────────────────────────────────────


def sandbox_root(sandbox: Any) -> str:
    root = str(getattr(sandbox, "workdir", "") or "/workspace")
    return "/" + root.strip("/") if root != "/" else "/"


def sandbox_path(sandbox: Any, path: str, workdir: str = "") -> str:
    """도구가 준 경로 → 세션 안의 절대 경로.

    상대 경로는 ``workdir``(없으면 세션 루트) 기준으로 푼다. 결과가 루트 밖이면
    :class:`SandboxPathError` — ``..`` 나 절대경로로 세션을 빠져나가는 것을
    여기서 한 번에 막는다.

    ``workdir`` 은 보통 ``ToolContext.working_dir`` 이다. 호스트가 양쪽 루트를
    같은 문자열로 맞추므로 그 값은 세션 안에서도 그대로 유효하다 — 이것이
    변환 함수를 두지 않는 이유다.
    """
    root = sandbox_root(sandbox)
    base = str(workdir or "").strip() or root
    if not posixpath.isabs(base):
        base = posixpath.join(root, base)
    target = str(path or ".")
    if not posixpath.isabs(target):
        target = posixpath.join(base, target)
    resolved = posixpath.normpath(target)
    if resolved != root and not resolved.startswith(root.rstrip("/") + "/"):
        raise SandboxPathError(
            f"경로가 샌드박스 세션 밖을 가리킵니다: {path!r} → {resolved!r} (루트 {root!r})"
        )
    return resolved


def _cwd(sandbox: Any, workdir: str) -> str:
    """``exec`` 에 넘길 작업 디렉터리 — 항상 세션 안."""
    try:
        return sandbox_path(sandbox, ".", workdir)
    except SandboxPathError:
        # 세션과 무관한 workdir 이 들어왔다. chdir 실패로 모든 호출을 죽이느니
        # 루트에서 실행한다 (GAPT 시절 host-absolute workdir 이 exec 를 통째로
        # 죽였던 실패 모드를 되풀이하지 않는다).
        return sandbox_root(sandbox)


# ── 도구가 쓰는 3가지 ──────────────────────────────────────────────────


async def sb_run(
    sandbox: Any,
    command: str,
    *,
    workdir: str = "",
    env: Optional[Mapping[str, str]] = None,
    timeout_s: float = 120.0,
) -> Tuple[int, str, str]:
    """셸 명령 하나. ``(rc, stdout, stderr)`` — 문자열로 디코딩해서 준다."""
    await sandbox.ensure()
    result = await sandbox.exec(
        ["bash", "-lc", command],
        cwd=_cwd(sandbox, workdir),
        env=env,
        timeout_s=timeout_s,
    )
    return (
        result.rc,
        result.stdout.decode("utf-8", "replace"),
        result.stderr.decode("utf-8", "replace"),
    )


async def sb_read_bytes(sandbox: Any, path: str, *, workdir: str = "") -> bytes:
    await sandbox.ensure()
    return await sandbox.read_bytes(sandbox_path(sandbox, path, workdir))


async def sb_write_bytes(sandbox: Any, path: str, data: bytes, *, workdir: str = "") -> int:
    await sandbox.ensure()
    return await sandbox.write_bytes(sandbox_path(sandbox, path, workdir), data)
