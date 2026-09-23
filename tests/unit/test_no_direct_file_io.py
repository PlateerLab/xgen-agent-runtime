"""내장 도구는 파일시스템 포트(``tools.fs``)로만 파일을 만진다 — AST 린트 (파일시스템 포트 3단계).

왜 이 테스트가 있나
-------------------
도구가 러너/로컬을 각자 고르던 동안, 같은 도구가 두 곳에서 다른 말을 했고(경로 탈출 예외,
이미지 판정, 성공 문구), 러너 Glob 은 셸 주입까지 있었다. 포트로 모은 뒤에도 **새 도구가
``open()``·``Path.write_text()`` 를 직접 부르면** 그 도구는 조용히 워크플로 파드의 파일시스템을
만진다 — 서버에서는 에이전트의 진짜 파일이 러너에 있는데도. 그 사고를 규칙으로 **기억**하는
대신 여기서 **검사**한다.

판정
----
포트와 세션의 파일 연산은 **async** 라 ``await fs.read_bytes(...)`` 처럼 불린다. pathlib 의 같은
이름(``read_bytes``·``write_text``·``exists`` …)은 sync 라 ``await`` 없이 불린다. 그래서 이
이름들은 **await 된 호출일 때만** 허용한다. 그 밖에 ``open(...)``, ``os.remove/rename/…``,
``os.path.exists/…``, ``shutil.*``(``which`` 제외)는 전부 위반이다.

허용 목록은 **이유와 함께** 모듈 단위로 둔다. 목록에 있는데 더 이상 위반이 없으면 테스트가
실패한다 — 목록은 줄어들기만 한다(래칫).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import xgen_agent_runtime.tools.built_in as _pkg

BUILT_IN = Path(_pkg.__file__).parent

#: 모듈 → 직접 파일 I/O 를 허용하는 이유. **새 항목을 더하기 전에 포트로 할 수 없는지 먼저 본다.**
ALLOWED: Dict[str, str] = {
    "workspace_tools.py": "로컬 작업 공간 ↔ 러너 세션 사이를 옮기는 것이 본업(SandboxFetch/Put) — 양쪽을 다 만진다",
    "_search.py": "Glob·Grep 의 구현 자체 — 로컬에선 import, 러너에선 이 소스가 python3 -c 로 그 자리에서 돈다",
    "_ssh_store.py": "SSH 서버 자격 증명 저장소 — 에이전트 작업 공간이 아니라 executor 저장소(storage_path)",
    "ssh_tools.py": "파드 거주 가족(paramiko) — 4단계에서 materialize/commit 으로",
    "audio_tools.py": "파드 거주 가족(STT 어댑터) — 4단계에서 materialize/commit 으로",
    "doc_tools.py": "파드 거주 가족(edit2docs) — 4단계에서 materialize/commit 으로",
    "worktree_tools.py": "로컬 git 체크아웃 전용(xgeny-cli). XGeny 서버 표면에 노출되지 않는다",
}

_PATH_METHODS = {
    "read_text", "write_text", "read_bytes", "write_bytes", "open",
    "unlink", "mkdir", "rmdir", "touch", "iterdir", "glob", "rglob",
    "exists", "is_file", "is_dir", "stat",
}
_OS_FUNCS = {"remove", "unlink", "rename", "replace", "makedirs", "mkdir", "rmdir",
             "listdir", "scandir", "walk", "chmod"}
_OS_PATH_FUNCS = {"exists", "isfile", "isdir", "getsize", "getmtime"}
_SHUTIL_OK = {"which"}


def _violations(tree: ast.AST) -> List[Tuple[int, str]]:
    awaited_calls = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Await)}
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "open":
                found.append((node.lineno, "open()"))
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                mod = f.value.id
                if mod == "os" and f.attr in _OS_FUNCS:
                    found.append((node.lineno, f"os.{f.attr}()"))
                elif mod == "shutil" and f.attr not in _SHUTIL_OK:
                    found.append((node.lineno, f"shutil.{f.attr}()"))
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Attribute)
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == "os"
                and f.value.attr == "path"
                and f.attr in _OS_PATH_FUNCS
            ):
                found.append((node.lineno, f"os.path.{f.attr}()"))
            if isinstance(f, ast.Attribute) and f.attr in _PATH_METHODS and id(node) not in awaited_calls:
                # os.* / 이미 위에서 센 것은 건너뛴다
                recv = f.value
                is_os_path = (
                    isinstance(recv, ast.Attribute)
                    and isinstance(recv.value, ast.Name)
                    and recv.value.id == "os"
                    and recv.attr == "path"
                )
                if not is_os_path and not (isinstance(recv, ast.Name) and recv.id in ("os", "shutil")):
                    found.append((node.lineno, f".{f.attr}() (await 없음 — pathlib)"))
    return sorted(set(found))


def _scan() -> Dict[str, List[Tuple[int, str]]]:
    out: Dict[str, List[Tuple[int, str]]] = {}
    for path in sorted(BUILT_IN.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _violations(tree)
        if hits:
            out[path.name] = hits
    return out


def test_no_built_in_tool_touches_files_behind_the_ports_back():
    offenders = {m: hits for m, hits in _scan().items() if m not in ALLOWED}
    detail = "\n".join(
        f"  {m}:{ln}  {what}" for m, hits in offenders.items() for ln, what in hits
    )
    assert not offenders, (
        "내장 도구가 파일시스템 포트(tools.fs.tool_fs)를 거치지 않고 파일을 만진다.\n"
        "서버에서는 에이전트의 파일이 러너에 있으므로 이 코드는 파드의 엉뚱한 파일을 본다.\n"
        f"{detail}\n"
        "→ `fs = tool_fs(context)` 후 `await fs.read_bytes/write_bytes/exists/…` 로 바꾸거나, "
        "정말 필요하면 ALLOWED 에 이유와 함께 추가한다."
    )


def test_the_allow_list_only_shrinks():
    """허용된 모듈이 더 이상 직접 I/O 를 하지 않으면 목록에서 빼야 한다(래칫)."""
    scanned = _scan()
    stale = [m for m in ALLOWED if m not in scanned and (BUILT_IN / m).exists()]
    gone = [m for m in ALLOWED if not (BUILT_IN / m).exists()]
    assert not stale, f"더 이상 직접 파일 I/O 가 없다 — ALLOWED 에서 지울 것: {stale}"
    assert not gone, f"없는 모듈이 허용 목록에 있다: {gone}"


def test_the_rule_itself():
    """규칙이 잡아야 할 것과 놓아야 할 것 — 린트가 조용히 무뎌지지 않게."""
    bad = ast.parse(
        "import os, shutil\n"
        "def f(p):\n"
        "    open(p)\n"
        "    p.write_text('x')\n"
        "    os.remove(p)\n"
        "    os.path.exists(p)\n"
        "    shutil.copy(p, p)\n"
    )
    assert len(_violations(bad)) == 5
    good = ast.parse(
        "import shutil\n"
        "async def f(fs, p):\n"
        "    await fs.read_bytes(p)\n"
        "    await fs.write_bytes(p, b'')\n"
        "    await fs.exists(p)\n"
        "    shutil.which('git')\n"
        "    'a'.replace('a', 'b')\n"
    )
    assert _violations(good) == []
