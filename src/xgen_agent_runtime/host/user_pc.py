"""사용자 PC 접속 — 대화에 연결된 기기 폴더에서 명령을 실행하고 파일을 옮기는 도구 ``UserPc``.

사용자가 대화에 폴더를 연결하면, 그 폴더가 있는 기기(데스크톱·모바일 앱, 웹 브라우저)에 대한
**접속**이 생긴다. 에이전트는 두 기계에서 일한다.

  sandbox    자기 작업 공간. 평소 도구(Bash·Read·Write·Edit·Glob·Grep)가 여기서 돈다.
  사용자 기기 이 도구 하나로만 닿는다. 명령은 연결된 폴더에서 시작하고(원격 셸에 그 폴더로
             들어간 것과 같다), 파일은 get·put 으로만 두 기계 사이를 오간다.

예전에는 기기마다 파일·셸·복사 도구 열 개 남짓을 따로 보였다(ReadFile·ListDir·SearchFiles·
Shell·CopyToWorkspace…). 동작이 sandbox 도구와 달라 모델이 두 번째 도구 체계를 익혀야 했고,
"어느 기계인가" 를 도구 이름과 안내문으로 거듭 설득해야 했다. 이제는 "어디서 실행하느냐" 하나다.

여기서 정하는 것:

  도구   :func:`build_user_pc_tool` — run·get·put·job 네 동작.
  판정   폴더 이름 해석, 꺼진 기기·명령을 받지 못하는 기기의 즉시 거절, 한 턴에서 응답하지 않은
         기기의 빠른 실패, 호출마다 시한. 모델의 판단에 맡기지 않고 이 코드가 지킨다.
  안내   :func:`turn_note` — 이번 턴에 연결된 폴더 목록(이름·기기·OS·셸·경로·켜짐). 사실만 적는다.

기기와 오가는 일(어느 기기 호출로 실행할지, 파일을 어떻게 옮길지)은 호스트가
:class:`UserPcConnection` 의 ``call`` 로 맡는다. 호스트가 접속을 주지 않으면 이 도구는 없다
(예전 기기 도구 규칙 그대로 — :mod:`xgen_agent_runtime.host.local_folders`).
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from xgen_agent_runtime.tools.base import (
    Tool,
    ToolCapabilities,
    ToolContext,
    ToolResult,
    build_tool,
    with_origin,
)

logger = logging.getLogger("xgen_agent_runtime.host.user_pc")

#: 모델이 보는 도구 이름.
TOOL_NAME = "UserPc"

#: run 이 끝나기를 기다리는 시간(초). 넘으면 명령은 작업(job)으로 계속 돌고 job_id 를 돌려준다.
DEFAULT_RUN_WAIT_S = 60
MAX_RUN_WAIT_S = 600
#: job 확인이 끝나기를 기다리는 시간(초).
DEFAULT_JOB_WAIT_S = 10
MAX_JOB_WAIT_S = 120
#: 파일 옮기기 한 번의 시한(초). 기기 쪽 한도(파일당 100MB, 합 1GB)를 옮기기에 충분한 값.
TRANSFER_TIMEOUT_S = 900
#: 호스트 호출에 얹는 여유 — 기기가 자기 시한에 걸려 실패 결과를 보낼 시간.
CALL_GRACE_S = 30

ACTIONS = ("run", "get", "put", "job")

#: 결과 글의 상한(자). Bash 와 같은 값.
MAX_OUTPUT_CHARS = 100_000


@dataclass(frozen=True)
class PcFolder:
    """대화에 연결된 기기 폴더 하나 — 모델이 ``folder`` 로 부르는 단위.

    ``name`` 은 이 대화 안에서 유일하다(여러 기기에 같은 이름이 있으면 호스트가 기기 이름을 붙여
    가른다). ``shell`` 은 그 기기에서 명령을 받는 셸의 이름이다. 빈 문자열이면 그 기기는 명령을
    받지 못하고 파일 옮기기(get·put)만 된다.
    """

    name: str
    path: str
    device_id: str
    device_name: str = ""
    platform: str = ""
    shell: str = ""
    #: 셸에 대해 덧붙일 사실 한 줄(예: "programs installed on the PC are not available; ...").
    shell_note: str = ""
    online: bool = True
    #: 이 턴을 보낸 화면이 그 기기가 아니다(웹·휴대폰·다른 PC 에서 보냈다).
    remote: bool = False

    @property
    def can_run(self) -> bool:
        return bool(self.shell)


#: 호스트가 기기에 동작 하나를 보낸다 — ``(폴더, 동작, 인자) → 결과``.
#:
#: 결과는 dict 하나다. 공통: ``ok``(bool), 실패면 ``error``(글)·``code``(짧은 표지).
#:   run·job : ``exit_code``·``stdout``·``stderr``·``cwd``, 아직 돌면 ``running`` 과 ``job_id``.
#:   get·put : ``files`` = ``[{"from", "to", "size"}]``, 옮기지 못한 것은 ``skipped``.
DeviceCall = Callable[[PcFolder, str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class UserPcConnection:
    """이번 턴의 사용자 기기 접속 — 호스트가 만든다."""

    folders: List[PcFolder]
    call: DeviceCall
    #: 한 턴 안에서 응답하지 않은 기기 — 같은 턴의 다음 호출은 기다리지 않는다.
    unresponsive: Dict[str, str] = field(default_factory=dict)


# ── 안내 ────────────────────────────────────────────────────────────

_PLATFORM_LABEL = {
    "darwin": "macOS",
    "mac": "macOS",
    "macos": "macOS",
    "win32": "Windows",
    "windows": "Windows",
    "linux": "Linux",
    "android": "Android",
    "ios": "iOS",
    "web": "web browser",
}


def platform_label(platform: Optional[str]) -> str:
    key = str(platform or "").strip().lower()
    return _PLATFORM_LABEL.get(key, key or "device")


def _folder_line(f: PcFolder) -> str:
    where = f'"{f.device_name}"' if f.device_name else "a device"
    if not f.online:
        # 꺼진 기기는 셸을 모른다(카탈로그가 없다) — 아는 사실만 적는다.
        return f'- "{f.name}": on {where} ({platform_label(f.platform)}), path {f.path}, offline'
    shell = f.shell or "no commands, file transfer only"
    line = (
        f'- "{f.name}": on {where} ({platform_label(f.platform)}, {shell}), path {f.path}, online'
    )
    if f.shell_note:
        line += f"; {f.shell_note}"
    return line


def turn_note(folders: Optional[Sequence[PcFolder]]) -> str:
    """이번 턴의 연결 폴더 — 매 턴 새로 쓰는 한 블록(기록에 남지 않는다). 폴더가 없으면 빈 문자열."""
    if not folders:
        return ""
    lines = [
        "# Folders on the user's devices",
        "Connected to this conversation. They are on the user's own devices, not in your sandbox; "
        f"{TOOL_NAME} reaches them.",
    ]
    lines += [_folder_line(f) for f in folders]
    if any(f.remote for f in folders):
        lines.append("The user is writing from another screen than the device(s) above.")
    return "\n".join(lines)


def no_folder_note() -> str:
    """앱에서 보낸 턴인데 연결 폴더가 없다 — 사실과 연결하는 법만."""
    return (
        "# Folders on the user's devices\n"
        "No folder on the user's devices is connected to this conversation. The user can connect "
        f"one with the [폴더] button in the chat header; {TOOL_NAME} then reaches it."
    )


# ── 판정 ────────────────────────────────────────────────────────────


def resolve_folder(folders: Sequence[PcFolder], requested: Any) -> PcFolder:
    """``folder`` 인자 → 폴더. 하나뿐이면 생략해도 된다. 못 찾으면 고를 수 있는 이름을 담아 던진다."""
    names = ", ".join(f'"{f.name}"' for f in folders)
    want = str(requested or "").strip()
    if not want:
        if len(folders) == 1:
            return folders[0]
        raise ValueError(f"More than one folder is connected; pass folder= one of {names}.")
    for f in folders:
        if f.name == want:
            return f
    lowered = want.lower()
    matches = [f for f in folders if f.name.lower() == lowered]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f'No connected folder named "{want}". Connected: {names}.')


def _slashes(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").rstrip("/")


def _is_absolute(path: str) -> bool:
    return path.startswith("/") or (len(path) > 1 and path[1] == ":")


def _rel(value: Any, folder: PcFolder) -> str:
    """폴더 기준 상대 경로로 정리한다.

    그 폴더의 절대 경로로 시작하면 상대로 바꿔 받는다(모델이 목록의 경로를 그대로 쓴 경우).
    폴더 밖(``..``, 다른 절대 경로)은 거절한다 — 기기도 호출마다 다시 막는다.
    """
    raw = _slashes(str(value or ""))
    if not raw or raw == ".":
        return ""
    if _is_absolute(raw):
        root = _slashes(folder.path)
        same = (
            raw.lower() == root.lower() if folder.platform in ("win32", "windows") else raw == root
        )
        if same:
            return ""
        prefix = root + "/"
        inside = (
            raw.lower().startswith(prefix.lower())
            if folder.platform in ("win32", "windows")
            else raw.startswith(prefix)
        )
        if not inside:
            raise ValueError(
                f'{value} is not inside the connected folder "{folder.name}" ({folder.path}); '
                "give a path relative to that folder."
            )
        raw = raw[len(prefix) :]
    norm = posixpath.normpath(raw)
    if norm == ".":
        return ""
    if norm == ".." or norm.startswith("../"):
        raise ValueError(f'{value} is outside the connected folder "{folder.name}".')
    return norm


def _int_in(value: Any, default: int, low: int, high: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(text)} chars total)"


def _head(folder: PcFolder) -> str:
    who = f'"{folder.device_name}"' if folder.device_name else platform_label(folder.platform)
    return f'[{TOOL_NAME} · {who} · folder "{folder.name}"]'


def _error(folder: Optional[PcFolder], message: str) -> ToolResult:
    head = _head(folder) + " " if folder is not None else ""
    return ToolResult(content=f"{head}Error: {message}", is_error=True)


def _render_run(folder: PcFolder, out: Dict[str, Any]) -> ToolResult:
    head = _head(folder)
    cwd = str(out.get("cwd") or "").strip()
    if cwd:
        head += f" cwd {cwd}"
    stdout = _clip(str(out.get("stdout") or ""))
    stderr = _clip(str(out.get("stderr") or ""))
    if out.get("running"):
        job = str(out.get("job_id") or "")
        parts = [f"{head} still running as job {job}"]
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        parts.append(f'Check or stop it with {TOOL_NAME}(action="job", job_id="{job}").')
        return ToolResult(content="\n".join(parts), metadata={"job_id": job, "running": True})
    code = out.get("exit_code")
    try:
        exit_code = int(code) if code is not None else 0
    except (TypeError, ValueError):
        exit_code = 1
    parts = [f"{head} exit {exit_code}"]
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    if not stdout and not stderr:
        parts.append("(no output)")
    return ToolResult(
        content="\n".join(parts),
        is_error=exit_code != 0,
        metadata={"exit_code": exit_code, "device_id": folder.device_id, "folder": folder.name},
    )


def _render_transfer(folder: PcFolder, action: str, out: Dict[str, Any]) -> ToolResult:
    files = [f for f in (out.get("files") or []) if isinstance(f, dict)]
    skipped = [s for s in (out.get("skipped") or []) if isinstance(s, (dict, str))]
    arrow = "device -> sandbox" if action == "get" else "sandbox -> device"
    lines = [f"{_head(folder)} {action} ({arrow}): {len(files)} file(s)"]
    for f in files[:200]:
        size = f.get("size")
        tail = f" ({size} bytes)" if isinstance(size, int) else ""
        lines.append(f"- {f.get('from', '')} -> {f.get('to', '')}{tail}")
    if len(files) > 200:
        lines.append(f"... {len(files) - 200} more")
    for s in skipped[:50]:
        lines.append(
            f"- not copied: {s if isinstance(s, str) else s.get('path', '')}"
            + ("" if isinstance(s, str) else f" ({s.get('reason', '')})")
        )
    return ToolResult(
        content="\n".join(lines),
        is_error=not files and bool(skipped),
        metadata={"files": files, "device_id": folder.device_id, "folder": folder.name},
    )


# ── 도구 ────────────────────────────────────────────────────────────

_DESCRIPTION = (
    "Work on the user's own device, inside a folder they connected to this conversation, like "
    "a remote shell session that starts in that folder. Your other tools (Bash, Read, Write, "
    "Edit, Glob, Grep) act on your sandbox, a different machine; files move between the two "
    "only with get and put. Actions: "
    "run (default): run `command` in that folder's shell (the connected folders list names it) "
    "and get the exit code and output; `cwd` is a subfolder to start in; a command still "
    "running after `timeout` seconds keeps running as a job. "
    "get: copy `path` (a file or folder in the connected folder) into your sandbox, into "
    "`sandbox_path` if given; the result lists the sandbox paths. "
    "put: copy `sandbox_path` from your sandbox to `path` in the connected folder "
    "(`overwrite` to replace an existing file). "
    "job: check a running command by `job_id`, or end it with `stop`. "
    "`folder` is the connected folder's name; it is required when more than one is connected."
)

_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": list(ACTIONS),
            "description": "run (default), get, put or job.",
        },
        "folder": {
            "type": "string",
            "description": "Name of the connected folder. Required when more than one is connected.",
        },
        "command": {
            "type": "string",
            "description": "run: the command line for that folder's shell.",
        },
        "cwd": {
            "type": "string",
            "description": "run: subfolder to start in, relative to the connected folder.",
        },
        "timeout": {
            "type": "integer",
            "description": (
                f"run: seconds to wait (default {DEFAULT_RUN_WAIT_S}, max {MAX_RUN_WAIT_S}); "
                "after that the command keeps running as a job."
            ),
        },
        "path": {
            "type": "string",
            "description": (
                "get: file or folder to copy, relative to the connected folder. "
                "put: destination relative to the connected folder (default: the folder itself)."
            ),
        },
        "sandbox_path": {
            "type": "string",
            "description": "put: file in your sandbox to copy. get: sandbox folder to copy into.",
        },
        "overwrite": {
            "type": "boolean",
            "description": "put: replace an existing file (default false).",
        },
        "job_id": {"type": "string", "description": "job: id from an earlier run."},
        "stop": {"type": "boolean", "description": "job: stop the command instead of checking it."},
    },
}


def _deadline(action: str, args: Dict[str, Any]) -> float:
    if action == "run":
        return float(args["wait_s"]) + CALL_GRACE_S
    if action == "job":
        return float(args["wait_s"]) + CALL_GRACE_S
    return float(TRANSFER_TIMEOUT_S) + CALL_GRACE_S


def _device_args(action: str, tool_input: Dict[str, Any], folder: PcFolder) -> Dict[str, Any]:
    """모델 입력 → 호스트에 넘길 인자. 잘못된 입력은 ValueError(모델에게 그대로 돌아간다)."""
    if action == "run":
        command = str(tool_input.get("command") or "")
        if not command.strip():
            raise ValueError("run needs `command`.")
        return {
            "command": command,
            "cwd": _rel(tool_input.get("cwd"), folder),
            "wait_s": _int_in(tool_input.get("timeout"), DEFAULT_RUN_WAIT_S, 1, MAX_RUN_WAIT_S),
        }
    if action == "job":
        job_id = str(tool_input.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("job needs `job_id` from an earlier run.")
        return {
            "job_id": job_id,
            "stop": bool(tool_input.get("stop")),
            "wait_s": DEFAULT_JOB_WAIT_S,
        }
    if action == "get":
        path = _rel(tool_input.get("path"), folder)
        if not path:
            raise ValueError("get needs `path` (a file or folder in the connected folder).")
        return {"path": path, "sandbox_path": str(tool_input.get("sandbox_path") or "").strip()}
    # put
    source = str(tool_input.get("sandbox_path") or "").strip()
    if not source:
        raise ValueError("put needs `sandbox_path` (the file in your sandbox).")
    return {
        "sandbox_path": source,
        "path": _rel(tool_input.get("path"), folder),
        "overwrite": bool(tool_input.get("overwrite")),
    }


def build_user_pc_tool(connection: UserPcConnection) -> Tool:
    """이번 턴의 ``UserPc`` 도구. 연결 폴더가 없으면 만들지 않는다(호출부가 거른다)."""

    async def _execute(tool_input: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool_input = dict(tool_input or {})
        action = str(tool_input.get("action") or "run").strip().lower() or "run"
        if action not in ACTIONS:
            return _error(None, f"Unknown action {action!r}; use one of {', '.join(ACTIONS)}.")
        try:
            folder = resolve_folder(connection.folders, tool_input.get("folder"))
        except ValueError as exc:
            return _error(None, str(exc))
        try:
            args = _device_args(action, tool_input, folder)
        except ValueError as exc:
            return _error(folder, str(exc))
        if not folder.online:
            return _error(
                folder,
                f"The device is not connected right now, so this folder cannot be reached. "
                f"It needs the XGEN app (or, for a web browser folder, the XGEN page) open and "
                f"signed in on {folder.device_name or 'that device'}.",
            )
        if action in ("run", "job") and not folder.can_run:
            return _error(
                folder,
                "This device does not take commands (its app is too old or has no shell); "
                "get and put still work.",
            )
        stuck = connection.unresponsive.get(folder.device_id)
        if stuck:
            return _error(folder, f"The device did not respond earlier in this turn ({stuck}).")
        try:
            out = await asyncio.wait_for(
                connection.call(folder, action, args), timeout=_deadline(action, args)
            )
        except asyncio.TimeoutError:
            reason = f"no response to {action} within {int(_deadline(action, args))}s"
            connection.unresponsive[folder.device_id] = reason
            logger.warning("UserPc: %s device=%s", reason, folder.device_id)
            return _error(folder, f"The device did not respond ({reason}).")
        except ValueError as exc:
            return _error(folder, str(exc))
        except Exception as exc:  # noqa: BLE001 — 기기 쪽 실패는 모델에게 사실로 돌아간다
            logger.warning("UserPc: %s failed: %s", action, exc)
            return _error(folder, str(exc) or exc.__class__.__name__)
        if not isinstance(out, dict):
            return _error(folder, "The device returned no result.")
        if not out.get("ok", False):
            if str(out.get("code") or "") == "NO_RESPONSE":
                connection.unresponsive[folder.device_id] = str(out.get("error") or "no response")
            return _error(folder, str(out.get("error") or "The device reported a failure."))
        if action in ("run", "job"):
            return _render_run(folder, out)
        return _render_transfer(folder, action, out)

    return with_origin(
        build_tool(
            name=TOOL_NAME,
            description=_DESCRIPTION,
            input_schema=_SCHEMA,
            execute=_execute,
            capabilities=ToolCapabilities(
                concurrency_safe=False,
                network_egress=True,
                interrupt="cancel",
            ),
        ),
        "device",
    )


#: 이 경로에서 기록 속 옛 기기 폴더 도구 호출이 평문으로 바뀔 때 붙는 이유.
RETIRE_REASON = "the user's device is reached with UserPc now"
NO_FOLDER_REASON = "no folder on the user's devices is connected now"


def without_folder_tools(tools: Sequence[Any]) -> List[Any]:
    """기기 카탈로그에서 폴더 도구(파일·셸·복사·열기…)와 옛 입구(LocalControl)를 뺀다.

    폴더에 닿는 일은 모두 UserPc 가 한다. 브라우저 조작·사용자가 붙인 로컬 MCP 서버처럼 폴더와
    무관한 기기 도구는 그대로 둔다.
    """
    from xgen_agent_runtime.host import local_folders as lf

    kept: List[Any] = []
    for tool in tools or []:
        name = lf._tool_name(tool)
        if lf.is_folder_tool(name) or lf.is_legacy_gate(name):
            continue
        kept.append(tool)
    return kept


def retired_calls_spec(*, has_folders: bool) -> Dict[str, Any]:
    """``SharedKeys.RETIRED_TOOL_CALLS`` 값 — 기록 속 옛 기기 폴더 도구 호출(그리고 연결 폴더가 없는
    턴이면 지난 UserPc 호출)을 요청 사본에서 평문 한 줄로 바꾼다."""
    from xgen_agent_runtime.host import local_folders as lf

    names = list(lf.retired_device_tool_names())
    if not has_folders:
        names.append(TOOL_NAME)
    return {"names": sorted(names), "reason": RETIRE_REASON if has_folders else NO_FOLDER_REASON}


def shared_folder_facts(folders: Sequence[PcFolder]) -> Optional[Dict[str, Any]]:
    """``local_folders.SHARED_FOLDERS_KEY`` 값 — sandbox 도구가 연결 폴더의 경로를 받았을 때
    (stages/s10_tool/second_machine) 그 경로가 어느 기기 폴더의 것인지 알려 주는 데 쓴다."""
    if not folders:
        return None
    return {
        "user_pc": True,
        "device": "device",
        "folders": [
            {"name": f.name, "path": f.path, "device": f.device_name, "platform": f.platform}
            for f in folders
        ],
    }


__all__ = [
    "ACTIONS",
    "DeviceCall",
    "PcFolder",
    "TOOL_NAME",
    "UserPcConnection",
    "build_user_pc_tool",
    "no_folder_note",
    "platform_label",
    "resolve_folder",
    "retired_calls_spec",
    "shared_folder_facts",
    "turn_note",
    "without_folder_tools",
]
