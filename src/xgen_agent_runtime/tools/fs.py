"""도구가 파일을 만지는 **유일한 입구** — 파일시스템 포트.

왜 필요한가
-----------
파일 도구는 두 곳에서 돈다. 호스트가 실행 세션(``ToolContext.sandbox``)을
붙이면 **러너**에서, 아니면(라이브러리·``xgeny-cli``·테스트) **로컬**에서. 지금까지
그 선택을 도구가 **각자** 했다 — 12개 모듈이 ``if context.sandbox is not None:``
분기를 제각각 구현했고, 그 결과 같은 이름의 도구가 두 곳에서 다르게 동작했다:

* 경로가 허용 트리 밖이면 로컬은 ``PermissionError``, 러너는 ``SandboxPathError``
  (``RuntimeError``). 부르는 쪽이 백엔드마다 다른 예외를 잡아야 했다.
* Edit 은 "정확히 한 번 나오는가" 검사와 치환 로직이 **분기마다 두 벌** 있었다.
* Glob·Grep 은 러너에서는 셸(``find``/``grep``), 로컬에서는 ``Path.glob`` —
  서로 다른 구현이라 결과 순서·숨김 파일 처리가 같다는 보장이 없었다.

이 모듈은 그 선택을 **한 곳**(:func:`tool_fs`)으로 모으고, 두 백엔드가 **같은
의미**로 동작한다는 것을 계약 테스트로 묶는다(``tests/unit/test_tool_fs_contract.py``
— 같은 테스트를 두 백엔드에 돌린다).

무엇을 약속하나
---------------
* 경로 밖으로 나가면 :class:`FsAccessError` (``PermissionError`` 하위) — 백엔드 무관.
* 없는 파일 읽기는 ``FileNotFoundError`` — 백엔드 무관.
* ``write_bytes`` 는 상위 디렉터리를 만들고 쓴 바이트 수를 돌려준다.
* :meth:`~ToolFileSystem.materialize` 는 **로컬 경로가 필요한 엔진**(문서 변환기처럼
  파일명을 받는 라이브러리)을 위한 것이다. 러너 백엔드는 파일을 **파드 전용 임시
  디렉터리**로 가져온다 — 워크스페이스와 **같은 경로 문자열을 절대 쓰지 않는다.**
  같은 문자열을 쓰던 시절, 지난 턴의 파드 잔재가 러너의 새 파일을 조용히 가렸다
  ("방금 고쳤는데 옛날 내용이 나온다"). 그 버그가 구조적으로 생길 수 없게 한다.

이 단계(4.53.0)는 포트와 두 백엔드만 들인다 — **어떤 도구도 아직 바꾸지 않는다.**
도구 이관은 모듈 단위로 뒤따른다.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Protocol, runtime_checkable

__all__ = [
    "FsAccessError",
    "LocalFS",
    "RunnerFS",
    "ToolFileSystem",
    "tool_fs",
]


class FsAccessError(PermissionError):
    """경로가 허용된 트리 밖이거나, 읽기 전용 트리에 쓰려 했다.

    ``PermissionError`` 를 잇는다 — 로컬 경로를 쓰던 도구가 이미 이것을 잡고 있다.
    """


@runtime_checkable
class ToolFileSystem(Protocol):
    """도구가 보는 파일시스템 하나. 로컬이든 러너든 같은 약속을 지킨다."""

    #: ``"local"`` | ``"runner"`` — 로그·진단용. **동작을 이 값으로 가르지 말 것.**
    kind: str

    def resolve(self, path: str, *, write: bool = False) -> str:
        """허용 트리 안의 절대 경로. 밖이면 :class:`FsAccessError`, 빈 경로는 ``ValueError``."""
        ...

    async def read_bytes(self, path: str) -> bytes: ...

    async def write_bytes(self, path: str, data: bytes) -> int: ...

    async def exists(self, path: str) -> bool: ...

    def materialize(self, path: str) -> "AsyncIterator[Path]":
        """``async with fs.materialize(p) as local:`` — 로컬 경로로 읽을 수 있게 한다."""
        ...

    async def commit(self, local: Path, path: str) -> int:
        """로컬 파일(엔진의 산출물)을 워크스페이스의 ``path`` 로 들여놓는다."""
        ...

    async def search(self, req: dict) -> dict:
        """Glob·Grep. ``{"ok": bool, "text": str}`` — 두 백엔드가 **같은 코드**를 돌린다
        (``tools/built_in/_search.py``)."""
        ...


# ── 로컬 ──────────────────────────────────────────────────────────────


class LocalFS:
    """이 프로세스의 파일시스템. ``allowed_paths`` 가 있으면 그 안으로 가둔다."""

    kind = "local"

    def __init__(self, working_dir: str, allowed_paths: Optional[List[str]] = None) -> None:
        self.working_dir = working_dir or os.getcwd()
        self.allowed_paths = allowed_paths

    def resolve(self, path: str, *, write: bool = False) -> str:
        from xgen_agent_runtime.tools.built_in._path_guard import resolve_and_validate

        try:
            return str(resolve_and_validate(path, self.working_dir, self.allowed_paths))
        except PermissionError as e:
            raise FsAccessError(str(e)) from e

    async def read_bytes(self, path: str) -> bytes:
        target = Path(self.resolve(path))
        if target.is_dir():
            raise IsADirectoryError(str(target))
        return await asyncio.to_thread(target.read_bytes)

    async def write_bytes(self, path: str, data: bytes) -> int:
        target = Path(self.resolve(path, write=True))

        def _write() -> int:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return len(data)

        return await asyncio.to_thread(_write)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(self.resolve(path)).exists)

    @asynccontextmanager
    async def materialize(self, path: str) -> AsyncIterator[Path]:
        # 이미 로컬이다 — 진짜 파일을 그대로 준다(복사하면 엔진이 고친 내용이 사라진다).
        target = Path(self.resolve(path))
        if not target.exists():
            raise FileNotFoundError(str(target))
        yield target

    async def commit(self, local: Path, path: str) -> int:
        target = Path(self.resolve(path, write=True))
        if Path(local).resolve() == target:
            return target.stat().st_size

        def _copy() -> int:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)
            return target.stat().st_size

        return await asyncio.to_thread(_copy)

    async def search(self, req: dict) -> dict:
        from xgen_agent_runtime.tools.built_in import _search

        req = dict(req, roots=list(self.allowed_paths or []))
        return await asyncio.to_thread(_search.run, req)


# ── 러너 ──────────────────────────────────────────────────────────────


class RunnerFS:
    """실행 세션(``XgenySandbox``) 위의 파일시스템.

    읽기·쓰기는 세션의 1급 연산(``read_bytes``/``write_bytes``)을, 경로 가드는
    :func:`~xgen_agent_runtime.tools._xgeny_sandbox.sandbox_path` 를 그대로 쓴다 —
    새 의미를 만들지 않고 흩어져 있던 것을 한 이름 아래 모은다. 커넥터(사용자 PC)도
    같은 프로토콜로 붙으므로 여기를 지난다.
    """

    kind = "runner"

    def __init__(self, sandbox: Any, working_dir: str = "") -> None:
        self.sandbox = sandbox
        self.working_dir = working_dir or ""

    def resolve(self, path: str, *, write: bool = False) -> str:
        from xgen_agent_runtime.tools._xgeny_sandbox import SandboxPathError, sandbox_path

        if not path:
            raise ValueError("file_path must not be empty")
        try:
            return sandbox_path(self.sandbox, path, self.working_dir, write=write)
        except SandboxPathError as e:
            raise FsAccessError(str(e)) from e

    # 세션을 **먼저** 깨우고 경로를 푼다 — ``sb_read_bytes`` 와 같은 순서다. 붙지 않은
    # 세션의 ``workdir`` 은 예상값이라, 그걸로 먼저 풀면 러너의 실제 루트와 어긋날 수 있다.

    async def read_bytes(self, path: str) -> bytes:
        await self.sandbox.ensure()
        return await self.sandbox.read_bytes(self.resolve(path))

    async def write_bytes(self, path: str, data: bytes) -> int:
        await self.sandbox.ensure()
        return await self.sandbox.write_bytes(self.resolve(path, write=True), data)

    async def exists(self, path: str) -> bool:
        await self.sandbox.ensure()
        resolved = self.resolve(path)
        # **셸로 묻지 않는다.** 실제 세션(러너 HTTP 클라이언트·사용자 PC 커넥터)은
        # 둘 다 ``exists`` 를 1급으로 갖고 있고, 경로 해석을 루트를 아는 쪽에 맡긴다.
        # 예전에 ``test -f`` 를 셸로 돌렸더니 붙지 않은 세션이 루트를 추측해 세션
        # 밖을 보고 "없음" 이라 답했다(프로드 실증). 커넥터는 맥·윈도우 PC 이기도 해서
        # ``stat -c`` 같은 리눅스 문법은 거기서 통하지도 않는다.
        native = getattr(self.sandbox, "exists", None)
        if callable(native):
            return bool(await native(resolved))
        # 프로토콜의 필수 연산만 가진 세션 — 읽어 보는 것이 이식 가능한 유일한 길이다.
        try:
            await self.sandbox.read_bytes(resolved)
        except FileNotFoundError:
            return False
        except IsADirectoryError:
            return True
        return True

    @asynccontextmanager
    async def materialize(self, path: str) -> AsyncIterator[Path]:
        data = await self.read_bytes(path)  # 없으면 FileNotFoundError
        # 파드 전용 임시 디렉터리 — 워크스페이스와 같은 경로 문자열을 쓰지 않는다.
        scratch = Path(tempfile.mkdtemp(prefix="xgen-fs-"))
        try:
            local = scratch / (Path(path).name or "file")
            await asyncio.to_thread(local.write_bytes, data)
            yield local
        finally:
            await asyncio.to_thread(shutil.rmtree, scratch, True)

    async def commit(self, local: Path, path: str) -> int:
        data = await asyncio.to_thread(Path(local).read_bytes)
        return await self.write_bytes(path, data)

    async def search(self, req: dict) -> dict:
        """로컬과 **같은 파일**(``_search.py``)을 러너의 python3 로 실행한다.

        셸을 거치지 않는다 — 요청은 argv 의 JSON 한 덩어리다. 예전 러너 Glob 은 패턴을
        ``for f in {pattern}`` 으로 셸에 끼워 넣어서 ``$(…)`` 가 실행됐다.
        """
        import json

        from xgen_agent_runtime.tools._xgeny_sandbox import (
            sandbox_extra_roots,
            sandbox_root,
        )
        from xgen_agent_runtime.tools.built_in import _search

        await self.sandbox.ensure()
        roots = [sandbox_root(self.sandbox), *sandbox_extra_roots(self.sandbox)]
        payload = json.dumps(dict(req, roots=roots), ensure_ascii=False)
        source = Path(_search.__file__).read_text(encoding="utf-8")
        result = await self.sandbox.exec(["python3", "-c", source, payload], timeout_s=60.0)
        if result.rc == 127:
            return {
                "ok": False,
                "text": "Search is unavailable: python3 is not installed in this session.",
            }
        try:
            return json.loads(result.stdout.decode("utf-8", "replace"))
        except ValueError:
            err = result.stderr.decode("utf-8", "replace").strip()[-300:]
            return {"ok": False, "text": f"Search failed (exit {result.rc}): {err or 'no output'}"}


# ── 선택 ──────────────────────────────────────────────────────────────


def tool_fs(context: Any) -> ToolFileSystem:
    """이 호출의 파일시스템 — **백엔드를 고르는 유일한 자리.**

    호스트가 ``ToolContext.fs`` 를 직접 주입했으면 그것을 쓴다. 아니면 실행
    세션이 붙어 있는지로 정한다 — 붙어 있으면 러너, 아니면 로컬. 서버는 세션을
    **언제나** 붙인다(러너가 죽어도 "미부착 세션" 이 온다 — None 이 아니다).
    그래서 서버 턴이 로컬 파일시스템으로 조용히 떨어지는 길은 없다.
    """
    explicit = getattr(context, "fs", None)
    if explicit is not None:
        return explicit
    sandbox = getattr(context, "sandbox", None)
    working_dir = str(getattr(context, "working_dir", "") or "")
    if sandbox is not None:
        return RunnerFS(sandbox, working_dir)
    return LocalFS(working_dir, getattr(context, "allowed_paths", None))
