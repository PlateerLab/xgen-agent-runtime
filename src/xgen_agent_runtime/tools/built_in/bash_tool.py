"""BashTool — execute shell commands."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import shlex
import signal
import sys
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from xgen_agent_runtime.tools.base import (
    HOST_IS_EXECUTION_TARGET,
    Tool,
    ToolCapabilities,
    ToolContext,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Host env vars the model's shell is allowed to inherit (audit S3). The
# non-sandbox path used ``os.environ.copy()``, handing every backend
# secret (ANTHROPIC_API_KEY, GENY_AUTH_SECRET, DB URLs, …) to any command
# the model runs. We inherit only a benign base; the host injects anything
# the workload legitimately needs via ``ToolContext.env_vars``. Set
# ``GENY_BASH_INHERIT_ENV=1`` to restore the old full-inherit behavior for
# a fully-trusted single-tenant deployment.
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TERM",
        "TZ",
        "TMPDIR",
        "PWD",
        "HOSTNAME",
        "DISPLAY",
        "COLUMNS",
        "LINES",
    }
)

# Windows additions (desktop host — the connector sidecar runs this tool
# directly on the user's PC). A child spawned without ``SystemRoot`` fails
# to initialise Winsock/CRT, ``COMSPEC``/``PATHEXT`` are needed for the
# shell to resolve commands at all, and ``HOME`` is normally unset there
# (``USERPROFILE`` is the home). Kept as a local fallback table; the CLI
# runtime's authoritative Windows whitelist is reused when importable.
_SAFE_ENV_KEYS_WINDOWS_FALLBACK = frozenset(
    {
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "SYSTEMDRIVE",
        "USERNAME",
    }
)


def _windows_env_keys() -> FrozenSet[str]:
    """Windows whitelist — ``_cli_runtime``'s table when importable (one
    source of truth with the CLI subprocess env), else the local fallback."""
    try:
        from xgen_agent_runtime.llm_client._cli_runtime import _ENV_WHITELIST_WINDOWS

        return frozenset(_ENV_WHITELIST_WINDOWS) | _SAFE_ENV_KEYS_WINDOWS_FALLBACK
    except Exception:  # noqa: BLE001 — import cycle / layout drift: fall back
        return _SAFE_ENV_KEYS_WINDOWS_FALLBACK


def _scrubbed_env(
    extra: Optional[Mapping[str, str]],
    *,
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> Dict[str, str]:
    """Benign base env for the host-path subprocess.

    Platform-aware: on Windows (``platform == "win32"``) the process
    bootstrap variables are whitelisted too and matching is
    case-insensitive (``Path`` vs ``PATH``, ``SystemRoot`` vs
    ``SYSTEMROOT`` — the parent's spelling is preserved); ``HOME`` is
    mapped from ``USERPROFILE`` when unset so ``~``/``$HOME`` resolve. The
    ``environ``/``platform`` knobs exist for tests — production reads
    ``os.environ`` / ``sys.platform``.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform
    is_windows = plat == "win32"

    if str(source.get("GENY_BASH_INHERIT_ENV", "")).strip() in ("1", "true", "yes"):
        env: Dict[str, str] = dict(source)
    elif is_windows:
        allowed_ci = {k.upper() for k in (_SAFE_ENV_KEYS | _windows_env_keys())}
        env = {
            k: v
            for k, v in source.items()
            if k.upper() in allowed_ci or k.upper().startswith("LC_")
        }
        # PATH must exist under SOME spelling; only synthesise when absent.
        if not any(k.upper() == "PATH" for k in env):
            system_root = next((v for k, v in env.items() if k.upper() == "SYSTEMROOT"), "")
            if system_root:
                env["PATH"] = f"{system_root}\\System32;{system_root}"
        if not any(k.upper() == "HOME" for k in env):
            profile = next((v for k, v in env.items() if k.upper() == "USERPROFILE"), "")
            if profile:
                env["HOME"] = profile
    else:
        env = {k: v for k, v in source.items() if k in _SAFE_ENV_KEYS or k.startswith("LC_")}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    if extra:
        env.update(extra)
    return env


def _host_shell_argv(command: str, *, platform: Optional[str] = None) -> Optional[list]:
    """호스트 실행(샌드박스 없음)에서 명령을 돌릴 shell argv.

    Windows 는 bash 가 없다 — 커넥터 로컬 셸 도구와 동일하게 **PowerShell** 로 돈다
    (``powershell.exe -NoProfile -NonInteractive -Command <cmd>``). PowerShell 이 없으면
    (매우 드묾) ``cmd.exe /d /s /c`` 로 폴백. POSIX 는 None 을 돌려 기존 경로
    (``create_subprocess_shell`` = ``/bin/sh -c``)를 그대로 쓴다.

    반환 None → create_subprocess_shell(command) (POSIX).
    반환 [file, *args] → create_subprocess_exec(*argv) (Windows).
    """
    plat = sys.platform if platform is None else platform
    if plat != "win32":
        return None
    pwsh = _which_windows("powershell.exe") or _which_windows("pwsh.exe")
    if pwsh:
        return [pwsh, "-NoProfile", "-NonInteractive", "-Command", command]
    comspec = os.environ.get("ComSpec") or os.environ.get("COMSPEC") or "cmd.exe"
    return [comspec, "/d", "/s", "/c", command]


#: 그룹에 끝내라고(SIGTERM) 한 뒤 강제로 끝내기(SIGKILL)까지 기다리는 시간.
_KILL_GRACE_S = 2.0


#: Windows 프로세스 생성 플래그. ``subprocess`` 상수는 Windows 에만 있어 값으로 둔다.
_WINDOWS_NEW_GROUP = 0x00000200  # CREATE_NEW_PROCESS_GROUP
_WINDOWS_NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW


def _host_spawn_kwargs(*, platform: Optional[str] = None) -> Dict[str, Any]:
    """호스트 실행의 셸을 **자기 프로세스 그룹**으로 띄운다.

    Windows 는 콘솔 창도 만들지 않는다(``CREATE_NO_WINDOW``). 데스크톱 앱은 엔진을 창 없이 띄우므로 엔진에는
    콘솔이 없고, 콘솔 프로그램(PowerShell)을 그냥 띄우면 Windows 가 새 콘솔 창을 만들어 명령마다 창이
    깜빡인다. 출력은 파이프로 받으므로 창이 필요 없다.

    셸이 띄운 자식(``npm`` 이 띄운 ``node``, ``sleep``…)까지 한 그룹이 되어야 취소·시간 초과 때 한 번에
    끝낼 수 있다. 셸 하나만 죽이면 자식은 고아로 남아 계속 돈다 — 사용자가 [정지]를 눌렀는데 빌드나 개발
    서버가 이 PC 에서 계속 도는 것이다(2026-10-02 XD 실측: 취소한 ``sleep 30`` 이 남았다).
    """
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        return {"creationflags": _WINDOWS_NEW_GROUP | _WINDOWS_NO_WINDOW}
    return {"start_new_session": True}


async def _kill_process_tree(proc: Any, *, platform: Optional[str] = None) -> None:
    """셸과 그 자식들을 끝낸다. 이미 끝났으면 아무것도 하지 않는다. 실패는 삼킨다(정리는 턴을 깨지 않는다).

    POSIX: 그룹에 SIGTERM → 잠깐 기다림 → 남았으면 SIGKILL. Windows: ``taskkill /T /F`` 로 트리째.
    """
    if proc.returncode is not None:
        return
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/T",
                "/F",
                "/PID",
                str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=_WINDOWS_NO_WINDOW,
            )
            await asyncio.wait_for(killer.wait(), timeout=10)
        except Exception:  # noqa: BLE001 — taskkill 이 없거나 실패하면 셸만이라도
            pass
        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    else:
        for sig, grace in ((signal.SIGTERM, _KILL_GRACE_S), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            except OSError:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
                # 셸이 끝났어도 그룹에 남은 자식이 있을 수 있다 — SIGKILL 단계로 한 번 더 쓴다.
                if sig == signal.SIGKILL:
                    break
            except asyncio.TimeoutError:
                continue
            except Exception:  # noqa: BLE001
                break
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except Exception:  # noqa: BLE001
        pass


def _which_windows(name: str) -> Optional[str]:
    """PATH 에서 실행 파일을 찾는다(Windows). 없으면 None — shutil.which 얇은 래퍼."""
    try:
        import shutil

        return shutil.which(name)
    except Exception:  # noqa: BLE001
        return None


_DEFAULT_TIMEOUT_MS = 120_000  # 2 minutes

# Commands that try to leave a process running after the command returns.
# In the sandbox that never works: exec is request/response, and the
# session's shell isolation reaps the whole process namespace when the
# command exits. Before this guard the agent would run
# ``nohup npx serve … &``, wait out the timeout, and get a bare failure
# (2026-09-18). The right door for a long-running server is the app
# skill (AppCreate/AppPublish — the runner supervises the
# process); for long batch work, run it in the foreground with a larger
# timeout or use the job skill.
_DETACH_RE = re.compile(
    r"(?:^|[;&|(]\s*)(?:nohup|setsid|disown)\b"  # explicit detach verbs
    r"|(?<![&|])&\s*(?:$|[;)]|\n)"  # a single trailing '&' (not '&&', '|&')
)


def _detached_process_reason(command: str) -> Optional[str]:
    """Why a command that detaches a process cannot do what it intends here."""
    if not _DETACH_RE.search(command):
        return None
    return (
        "This command tries to leave a process running in the background "
        "(nohup / setsid / trailing &). In the sandbox that never survives the "
        "command: exec is request/response and the process namespace is reaped "
        "when the command exits, so the server or job would die immediately "
        "while this call waits out its timeout.\n"
        "- To serve an application: use the app skill — AppGuide, then "
        "AppCreate / AppPublish. The runner supervises that process and "
        "gives it an address.\n"
        "- To run long work: run it in the foreground with a larger `timeout`, or "
        "use the job skill (JobGuide) for work that must outlive this turn."
    )


_MAX_TIMEOUT_MS = 600_000  # 10 minutes
_MAX_OUTPUT = 100_000  # characters

# ── 셸 명령 분류 (BashTool.capabilities) ───────────────────────────────────────
# 실패-닫힘: 모든 구간이 아는 읽기 프로그램이고 쓰기 흔적이 없을 때만 읽기 전용이다.

#: 이보다 긴 명령은 분석하지 않고 읽기 전용이 아니라고 본다.
_CLASSIFY_MAX_CHARS = 20_000

#: 인자와 상관없이 읽기만 하는 프로그램(소문자).
_READ_ONLY_PROGRAMS = frozenset(
    {
        "ls",
        "dir",
        "cat",
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "ag",
        "ack",
        "wc",
        "echo",
        "printf",
        "pwd",
        "which",
        "whereis",
        "type",
        "printenv",
        "cal",
        "whoami",
        "id",
        "uname",
        "df",
        "du",
        "stat",
        "cut",
        "tr",
        "diff",
        "cmp",
        "comm",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "test",
        "[",
        "true",
        "false",
        "jq",
        "hexdump",
        "od",
        "strings",
        "column",
        "nl",
        "paste",
        "fold",
        "seq",
        "ps",
        "pgrep",
        "lsof",
        "free",
        "uptime",
        "nproc",
        "lscpu",
        "lsblk",
        "ping",
        "nslookup",
        "dig",
        "host",
        "traceroute",
        "netstat",
        "ss",
        "help",
        "tac",
        "rev",
        "expr",
        "bc",
        "sleep",
        "wait",
        "locale",
        "getconf",
        "lsb_release",
        "last",
        "w",
        "who",
        "zcat",
        "bzcat",
        "xzcat",
        "base64",
        "cd",
        "pushd",
        "popd",
        "export",
        "set",
        "unset",
        "get-childitem",
        "get-content",
        "get-item",
        "get-location",
        "select-string",
        "get-date",
        "get-process",
        "write-output",
        "write-host",
        "test-path",
        "resolve-path",
        "get-command",
        "measure-object",
        "select-object",
        "where-object",
        "format-table",
        "format-list",
        "out-string",
        "convertto-json",
        "convertfrom-json",
        "sort-object",
        "get-help",
        "get-member",
        "findstr",
        "where",
        "tasklist",
        "systeminfo",
        "ver",
        "ipconfig",
    }
)
_EGRESS_PROGRAMS = frozenset(
    {
        "curl",
        "wget",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "ping",
        "nslookup",
        "dig",
        "host",
        "traceroute",
        "nc",
        "netcat",
        "telnet",
        "invoke-webrequest",
        "invoke-restmethod",
        "iwr",
        "irm",
    }
)
_PKG_MANAGERS = frozenset(
    {
        "pip",
        "pip3",
        "npm",
        "pnpm",
        "yarn",
        "apt",
        "apt-get",
        "apk",
        "conda",
        "uv",
        "brew",
        "yum",
        "dnf",
        "cargo",
        "pipx",
        "gem",
        "go",
        "poetry",
        "choco",
        "winget",
    }
)
_PKG_READ_ONLY = frozenset(
    {
        "show",
        "list",
        "freeze",
        "search",
        "info",
        "ls",
        "view",
        "outdated",
        "why",
        "version",
        "--version",
        "-v",
        "-V",
        "help",
        "--help",
    }
)
#: 저장소(레지스트리)에 닿지 않는 하위 명령. 그 밖은 바깥 연결로 본다.
_PKG_LOCAL = frozenset(
    {"", "list", "ls", "freeze", "why", "env", "version", "--version", "-v", "-V", "help", "--help"}
)
_DOCKER_EGRESS = frozenset({"pull", "push", "login", "search", "build", "run"})
_GIT_READ_ONLY = frozenset(
    {
        "log",
        "status",
        "diff",
        "show",
        "rev-parse",
        "ls-files",
        "ls-tree",
        "blame",
        "describe",
        "shortlog",
        "grep",
        "cat-file",
        "rev-list",
        "name-rev",
        "var",
        "version",
        "--version",
        "check-ignore",
        "merge-base",
        "ls-remote",
        "count-objects",
        "whatchanged",
    }
)
_GIT_EGRESS = frozenset({"clone", "fetch", "pull", "push", "ls-remote", "submodule"})
_DOCKER_READ_ONLY = frozenset(
    {
        "ps",
        "images",
        "logs",
        "inspect",
        "version",
        "info",
        "stats",
        "top",
        "port",
        "diff",
        "history",
        "search",
        "events",
        "--version",
    }
)
_KUBECTL_READ_ONLY = frozenset(
    {
        "get",
        "describe",
        "logs",
        "top",
        "version",
        "cluster-info",
        "api-resources",
        "explain",
        "diff",
    }
)
_PYTHONS = frozenset({"python", "python3", "python2", "py"})
_INTERPRETERS = (
    frozenset(
        {
            "node",
            "ruby",
            "perl",
            "php",
            "bash",
            "sh",
            "zsh",
            "dash",
            "ksh",
            "fish",
            "pwsh",
            "powershell",
            "cmd",
            "rscript",
            "deno",
            "bun",
            "lua",
            "tclsh",
            "osascript",
        }
    )
    | _PYTHONS
)
_WRAPPERS = frozenset({"time", "nohup", "command", "exec", "busybox"})
#: 권한을 올리는 명령. 뒤의 명령과 상관없이 읽기 전용이 아니다.
_ELEVATE = frozenset({"sudo", "doas", "su", "runas", "gsudo"})
#: 파이썬 인라인 코드의 쓰기 · 실행 흔적.
_PY_WRITE_RE = re.compile(
    r"\bto_(?:excel|csv|json|parquet|pickle|sql|feather|hdf)\(|\.save(?:fig)?\(|\bwrite_(?:text|bytes)\(|\.write\("
    r"|\bopen\([^)]*[\"'][^\"']*[wax+]"
    r"|\bos\.(?:remove|unlink|rename|rmdir|removedirs|makedirs|mkdir|replace|chmod|chown|symlink|link|system|popen|exec\w*|spawn\w*|kill)\("
    r"|\bshutil\.|\bsubprocess\b|\bpathlib\b[^\n]*\.(?:write_text|write_bytes|unlink|rename|mkdir|touch|rmdir)\("
    r"|\.(?:write_text|write_bytes|unlink|touch|rmdir|mkdir|rename|replace)\(|\beval\(|\bexec\(|\b__import__\(",
    re.IGNORECASE,
)
#: 바깥으로 나가는 흔적(인라인 코드 · 명령 텍스트).
_NET_RE = re.compile(
    r"\brequests\.(?:get|post|put|delete|patch|head|request|session)\b|\burlopen\(|\burllib\.request\b|\bhttpx\."
    r"|\baiohttp\.|\bfetch\(|\bsocket\.",
    re.IGNORECASE,
)
_HEREDOC_RE = re.compile(
    r"<<-?\s*(['\"]?)(\w+)\1([^\n]*)\n(.*?)\n[ \t]*\2[ \t]*(?:\n|$)", re.DOTALL
)
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: 앞에 붙여도 실행할 코드를 바꾸지 않는 환경 변수(그 밖의 변수는 LD_PRELOAD · PAGER 처럼 다른 코드를 부를 수 있다).
_SAFE_ENV = frozenset(
    {"LANG", "LANGUAGE", "TZ", "NO_COLOR", "FORCE_COLOR", "CLICOLOR", "COLUMNS", "LINES", "TERM"}
)
_SAFE_SINKS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "nul", "$null"})


def _scan_shell(text: str) -> Tuple[List[str], bool]:
    """따옴표 밖의 연산자로 구간을 나누고, 치환 · 묶음 · 파일 리다이렉트 · 닫히지 않은 따옴표가 있으면 unsafe."""
    segs: List[str] = []
    buf: List[str] = []
    unsafe = False
    quote = ""
    i, n = 0, len(text)

    def flush() -> None:
        s = "".join(buf).strip()
        if s:
            segs.append(s)
        buf.clear()

    while i < n:
        c = text[i]
        if quote == "'":
            buf.append(c)
            quote = "" if c == "'" else quote
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            buf.append(text[i : i + 2])
            i += 2
            continue
        if quote == '"':
            if c == '"':
                quote = ""
            elif c == "`" or text.startswith("$(", i):
                unsafe = True
            buf.append(c)
            i += 1
            continue
        if c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "#" and (i == 0 or text[i - 1] in " \t\n;|&"):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "`" or c in "{}" or text.startswith("$(", i) or text.startswith("<(", i):
            unsafe = True
        if c == ">" or text.startswith("&>", i):
            j = i + (2 if c == "&" else 1)
            if j < n and text[j] in ">|":
                j += 1
            if c == ">" and j < n and text[j] == "&":  # >&2 · >&- 는 스트림 복제
                buf.append(text[i : j + 1])
                i = j + 1
                continue
            k = j
            while k < n and text[k] in " \t":
                k += 1
            m = re.match(r"[^\s;|&<>()]+", text[k:])
            target = (m.group(0) if m else "").strip("'\"").lower()
            if target not in _SAFE_SINKS:
                unsafe = True
            buf.append(text[i:k])
            i = k
            continue
        two = text[i : i + 2]
        if two in ("||", "&&", "|&"):
            flush()
            i += 2
            continue
        if c in "|;&\n":
            flush()
            i += 1
            continue
        buf.append(c)
        i += 1
    if quote:
        unsafe = True
    flush()
    return segs, unsafe


def _safe_assignment(word: str) -> bool:
    name = word.split("=", 1)[0]
    return name in _SAFE_ENV or name.startswith("LC_")


def _strip_wrappers(tokens: List[str]) -> Tuple[str, List[str]]:
    """앞의 환경 변수 · 감싸는 명령을 벗긴 프로그램(소문자 basename)과 인자."""
    t = list(tokens)
    while t:
        head = t[0]
        if _ASSIGN_RE.match(head):
            if not _safe_assignment(head):
                return "=", []
            t.pop(0)
            continue
        name = head.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if name.endswith(".exe"):
            name = name[:-4]
        if name in _WRAPPERS:
            t.pop(0)
            continue
        if name == "env":
            rest = t[1:]
            while rest and (rest[0].startswith("-") or _ASSIGN_RE.match(rest[0])):
                if _ASSIGN_RE.match(rest[0]) and not _safe_assignment(rest[0]):
                    return "=", []
                if rest[0] in ("-u", "--unset", "-C", "--chdir", "-S", "--split-string"):
                    rest = rest[2:]
                else:
                    rest = rest[1:]
            if not rest:
                return "env", []
            t = rest
            continue
        if name == "nice":
            t = t[1:]
            if t and t[0] == "-n":
                t = t[2:]
            elif t and re.fullmatch(r"-\d+", t[0]):
                t = t[1:]
            continue
        if name in ("timeout", "stdbuf"):
            t = t[1:]
            while t and t[0].startswith("-"):
                t = t[2:] if t[0] in ("-s", "--signal", "-k", "--kill-after") else t[1:]
            if name == "timeout" and t:
                t = t[1:]  # 시간
            continue
        return name, t[1:]
    return "", []


def _positional(args: List[str]) -> List[str]:
    return [a for a in args if not a.startswith("-")]


def _has_flag(args: List[str], *names: str, short: str = "") -> bool:
    """``--name`` / ``--name=…`` / 짧은 플래그 묶음(``-sSLo``) 안의 글자."""
    for a in args:
        if a == "--":
            break
        for nm in names:
            if a == nm or a.startswith(nm + "="):
                return True
        if (
            short
            and a.startswith("-")
            and not a.startswith("--")
            and any(ch in a[1:] for ch in short)
        ):
            return True
    return False


def _args_read_only(prog: str, args: List[str], stdin_code: str) -> bool:
    pos = _positional(args)
    if prog in _READ_ONLY_PROGRAMS:
        return True
    if prog == "env":
        return not args
    if prog in ("less", "more"):
        return not any(a.startswith("+") for a in args)
    if prog == "rg":
        return not _has_flag(args, "--pre")
    if prog == "file":
        return not _has_flag(args, "--compile", short="C")
    if prog == "xxd":
        return len(pos) <= 1
    if prog == "tree":
        return not _has_flag(args, short="o")
    if prog == "man":
        return not _has_flag(args, "--pager", "--html", short="PH")
    if prog == "date":
        return not _has_flag(args, "--set", short="s")
    if prog == "hostname":
        return not pos
    if prog in ("sort",):
        return not _has_flag(args, "--output", "--compress-program", short="o")
    if prog == "uniq":
        return len(pos) <= 1
    if prog == "iconv":
        return not _has_flag(args, "--output", short="o")
    if prog == "yq":
        return not _has_flag(args, "--inplace", short="i")
    if prog in ("awk", "gawk", "mawk", "nawk"):
        if _has_flag(args, "--file", "--inplace", short="f") or any(a == "-i" for a in args):
            return False
        program = pos[0] if pos else ""
        return not re.search(r"system\s*\(|>|\||getline|fflush|close\s*\(", program)
    if prog == "sed":
        if _has_flag(args, "--in-place", "--file", short="if"):
            return False
        script = " ".join(
            pos[:1] + [a for i, a in enumerate(args) if i and args[i - 1] in ("-e", "--expression")]
        )
        return not re.search(r"(?:^|[;}\s/])[wWe](?:\s|$)|/[gpIiMm0-9]*w\s", script)
    if prog == "find":
        return not any(
            a
            in (
                "-delete",
                "-exec",
                "-execdir",
                "-ok",
                "-okdir",
                "-fprint",
                "-fprint0",
                "-fprintf",
                "-fls",
            )
            for a in args
        )
    if prog == "tar":
        if _has_flag(
            args,
            "--delete",
            "--extract",
            "--get",
            "--create",
            "--append",
            "--update",
            "--concatenate",
            "--to-command",
            "--checkpoint-action",
            "--use-compress-program",
            "--info-script",
            "--new-volume-script",
            "--rsh-command",
            short="IF",
        ):
            return False
        mode = args[0].lstrip("-") if args and not args[0].startswith("--") else ""
        return ("t" in mode or _has_flag(args, "--list", short="t")) and not any(
            ch in mode for ch in "xcruAIF"
        )
    if prog in ("unzip",):
        return _has_flag(args, short="lZ") or "-l" in args
    if prog == "zipinfo":
        return True
    if prog == "git":
        sub = pos[0] if pos else ""
        rest = args[args.index(sub) + 1 :] if sub in args else []
        if _has_flag(args, "--output", "-o"):
            return False
        if sub == "branch":
            return not _positional(rest) or _has_flag(rest, "--list", short="l")
        if sub == "tag":
            return not _positional(rest) or _has_flag(rest, "--list", short="l")
        if sub == "remote":
            return not rest or rest[0] in ("-v", "--verbose", "show", "get-url")
        if sub == "config":
            return _has_flag(rest, "--get", "--get-all", "--get-regexp", "--list", short="l")
        if sub == "stash":
            return bool(rest) and rest[0] in ("list", "show")
        if sub == "reflog":
            return not rest or rest[0] == "show"
        if _has_flag(rest, "--ext-diff", "--textconv") or (
            sub == "grep" and _has_flag(rest, "--open-files-in-pager", short="O")
        ):
            return False
        return sub in _GIT_READ_ONLY
    if prog == "docker":
        sub = pos[0] if pos else ""
        if sub == "compose":
            return any(a in ("ps", "logs", "config", "ls", "images") for a in pos[1:2])
        return sub in _DOCKER_READ_ONLY
    if prog == "kubectl":
        sub = pos[0] if pos else ""
        if sub == "config":
            return any(
                a in ("view", "get-contexts", "current-context", "get-clusters") for a in pos[1:2]
            )
        return sub in _KUBECTL_READ_ONLY
    if prog in _PKG_MANAGERS:
        sub = pos[0] if pos else (args[0] if args else "")
        if prog == "go" and sub == "env":
            return not _has_flag(args, "-w", "-u")
        return sub in _PKG_READ_ONLY
    if prog == "curl":
        if _has_flag(
            args,
            "--output",
            "--remote-name",
            "--remote-name-all",
            "--upload-file",
            "--data",
            "--data-raw",
            "--data-binary",
            "--data-urlencode",
            "--json",
            "--form",
            "--config",
            "--cookie-jar",
            "--dump-header",
            "--trace",
            "--trace-ascii",
            "--stderr",
            short="oOTdFKcD",
        ):
            return False
        for i, a in enumerate(args):
            if a in ("-X", "--request") and i + 1 < len(args):
                if args[i + 1].upper() not in ("GET", "HEAD"):
                    return False
            elif a.startswith("-X") and len(a) > 2 and a[2:].upper() not in ("GET", "HEAD"):
                return False
        return True
    if prog == "wget":
        for i, a in enumerate(args):
            if a in ("-O", "--output-document") and i + 1 < len(args) and args[i + 1] == "-":
                return True
            if a in ("-O-", "-qO-", "--output-document=-", "--spider"):
                return True
        return False
    if prog in ("invoke-webrequest", "invoke-restmethod", "iwr", "irm"):
        low = [a.lower() for a in args]
        if "-outfile" in low or "-infile" in low or "-body" in low:
            return False
        for i, a in enumerate(low):
            if a == "-method" and i + 1 < len(low) and low[i + 1] not in ("get", "head"):
                return False
        return True
    if prog in _PYTHONS:
        if _has_flag(args, "--version", "--help", short="Vh") and not pos and "-c" not in args:
            return True
        if "-c" in args:
            i = args.index("-c")
            code = args[i + 1] if i + 1 < len(args) else ""
            return bool(code) and not _PY_WRITE_RE.search(code)
        if stdin_code and (not pos or pos == ["-"]):
            return not _PY_WRITE_RE.search(stdin_code)
        return False  # 스크립트 · 모듈 실행은 안을 모른다
    if prog in _INTERPRETERS:
        return _has_flag(args, "--version", "--help") and len(args) == 1
    return False


def _egress(prog: str, args: List[str], code: str) -> bool:
    pos = _positional(args)
    if prog in _EGRESS_PROGRAMS:
        return True
    if prog == "git":
        return bool(pos) and pos[0] in _GIT_EGRESS
    if prog in _PKG_MANAGERS:
        return (pos[0] if pos else (args[0] if args else "")) not in _PKG_LOCAL
    if prog == "docker":
        return bool(pos) and pos[0] in _DOCKER_EGRESS
    if prog == "kubectl":
        return not (pos and pos[0] == "config")
    if prog in _PYTHONS or prog in _INTERPRETERS:
        return bool(_NET_RE.search(" ".join(args) + "\n" + code))
    return False


@functools.lru_cache(maxsize=512)
def _classify(command: str) -> Tuple[bool, bool]:
    if len(command) > _CLASSIFY_MAX_CHARS:
        return False, bool(_NET_RE.search(command[:_CLASSIFY_MAX_CHARS]))
    bodies: List[str] = []

    def take(m: "re.Match[str]") -> str:
        bodies.append(m.group(4))
        # 같은 줄의 나머지(리다이렉트 · 파이프)는 남긴다
        return "<<" + m.group(2) + m.group(3) + "\n"

    text = _HEREDOC_RE.sub(take, command)
    stdin_code = "\n".join(bodies)
    segments, unsafe = _scan_shell(text)
    read_only = bool(segments) and not unsafe
    egress = False
    for seg in segments:
        try:
            tokens = shlex.split(seg, posix=True)
        except ValueError:
            read_only = False
            continue
        tokens = [t for t in tokens if not re.fullmatch(r"<<-?\w+|<<<?", t)]
        prog, args = _strip_wrappers(tokens)
        if not prog:
            continue
        egress = egress or _egress(prog, args, stdin_code)
        if prog in _ELEVATE or not _args_read_only(prog, args, stdin_code):
            read_only = False
    return read_only, egress


def classify_shell_command(command: str) -> ToolCapabilities:
    """읽기 전용 · 바깥 연결만 판정한다. 셸은 작업 디렉터리 · 환경을 공유하므로 병렬은 늘 끈다."""
    read_only, egress = _classify(str(command or ""))
    return ToolCapabilities(
        concurrency_safe=False,
        read_only=read_only,
        network_egress=egress,
    )


async def _finish_result(
    *,
    stdout: str,
    stderr: str,
    exit_code: int,
    sandboxed: bool,
    input: Dict[str, Any],
    context: ToolContext,
    stdout_total_bytes: int | None = None,
    stderr_total_bytes: int | None = None,
) -> ToolResult:
    """Shape command output and enforce an optional artifact contract."""
    if len(stdout) > _MAX_OUTPUT:
        suffix = f", {stdout_total_bytes} bytes total" if stdout_total_bytes is not None else ""
        stdout = stdout[:_MAX_OUTPUT] + f"\n\n... (truncated{suffix})"
    if len(stderr) > _MAX_OUTPUT:
        suffix = f", {stderr_total_bytes} bytes total" if stderr_total_bytes is not None else ""
        stderr = stderr[:_MAX_OUTPUT] + f"\n\n... (truncated{suffix})"
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    if exit_code != 0:
        parts.append(f"Exit code: {exit_code}")

    metadata: Dict[str, Any] = {"exit_code": exit_code, "sandboxed": sandboxed}
    if not sandboxed:
        metadata["execution_environment"] = "host"
    validation_failed = False
    contracts = input.get("artifact_contracts")
    if isinstance(contracts, list) and contracts:
        from xgen_agent_runtime.tools.built_in._artifact_contract import (
            validate_artifact_contracts,
        )

        report = await validate_artifact_contracts(contracts, context)
        parts.append(report.message)
        metadata["artifact_validation"] = report.metadata()
        validation_failed = not report.ok

    artifacts: Dict[str, Any] = {}
    if "artifact_validation" in metadata:
        artifacts["validated"] = metadata["artifact_validation"].get("checked", [])
    return ToolResult(
        content="\n".join(parts) if parts else "(no output)",
        is_error=exit_code != 0 or validation_failed,
        metadata=metadata,
        artifacts=artifacts,
    )


class BashTool(Tool):
    """Execute a bash command and return stdout/stderr.

    Commands run in the session's working directory with configurable
    timeout and environment variable injection.
    """

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return (
            "Run a shell command in your sandbox — your own isolated workspace on the "
            "server, where all your work runs. Commands start in your working folder; you "
            "can read and write there and install what you need (`pip install ...`, "
            "`npm install ...`). It is a Linux shell: use bash/sh syntax. It cannot reach "
            "the user's own devices; folders the user connected are reached with UserPc. "
            "Returns stdout, stderr, and exit code; a configurable timeout applies."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in milliseconds (default: {_DEFAULT_TIMEOUT_MS}, max: {_MAX_TIMEOUT_MS}).",
                    "minimum": 1000,
                    "maximum": _MAX_TIMEOUT_MS,
                },
                "artifact_contracts": {
                    "type": "array",
                    "maxItems": 8,
                    "description": (
                        "Optional deterministic checks run after a command. "
                        "Use workspace-relative paths. The runtime reopens each output "
                        "with a standard parser before Bash returns. Only JSON, CSV and "
                        "text outputs are checked; binary files (xlsx, docx, images) are skipped."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "format": {"type": "string", "enum": ["text", "json", "csv"]},
                            "columns": {"type": "array", "items": {"type": "string"}},
                            "allowed_values": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "unique_by": {"type": "array", "items": {"type": "string"}},
                            "exact_rows": {
                                "type": "integer",
                                "minimum": 0,
                                "description": (
                                    "Only a count the request itself states. A mismatch is "
                                    "reported as a note, not a failure."
                                ),
                            },
                            "min_rows": {"type": "integer", "minimum": 0},
                            "max_rows": {"type": "integer", "minimum": 0},
                            "required_keys": {
                                "type": "array",
                                "description": (
                                    "Keys required on a top-level JSON object or on every "
                                    "object in a top-level JSON array."
                                ),
                                "items": {"type": "string"},
                            },
                            "array_lengths": {
                                "type": "object",
                                "additionalProperties": {"type": "integer", "minimum": 0},
                            },
                            "required_strings": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "forbidden_strings": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["path", "format"],
                    },
                },
            },
            "required": ["command"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return classify_shell_command(str((input or {}).get("command") or ""))

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        command = input.get("command", "").strip()
        if not command:
            return ToolResult(content="command must not be empty", is_error=True)

        timeout_ms = min(input.get("timeout", _DEFAULT_TIMEOUT_MS), _MAX_TIMEOUT_MS)
        timeout_s = timeout_ms / 1000.0

        # Sandbox: run the command in the agent's Geny session instead
        # of on the host. Same output shaping as the host path below.
        if context.sandbox is not None:
            from xgen_agent_runtime.tools._geny_sandbox import sb_run

            detached = _detached_process_reason(command)
            if detached:
                # Don't spend the timeout to learn what we already know.
                return ToolResult(content=detached, is_error=True)

            try:
                exit_code, stdout, stderr = await sb_run(
                    context.sandbox,
                    command,
                    workdir=context.working_dir or "/workspace",
                    env=context.env_vars,
                    timeout_s=timeout_s,
                )
            except asyncio.TimeoutError:
                return ToolResult(content=f"Command timed out after {timeout_ms}ms", is_error=True)
            except Exception as e:  # noqa: BLE001
                return ToolResult(content=f"Sandbox exec failed: {e}", is_error=True)
            return await _finish_result(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                sandboxed=True,
                input=input,
                context=context,
            )

        # No sandbox attached — which of two very different situations is this?
        #
        #   (a) The host IS the execution target: the desktop connector's local
        #       turn runs on the user's own PC, where "no sandbox" is the whole
        #       point (path guard = allowed_paths). Hosts say so by setting
        #       ``extras[HOST_IS_EXECUTION_TARGET]``.
        #   (b) Anything else: a serving pod that was supposed to have a runner
        #       session. The command then quietly touches the pod and the files
        #       vanish with it — that must never pass silently.
        #
        # Warning on (a) too would be a false alarm every single local turn, and
        # it told the reader to go "check sandbox propagation" for something that
        # is working exactly as designed.
        if context.extras.get(HOST_IS_EXECUTION_TARGET):
            logger.debug("Bash executing on the host (local run — no sandbox by design)")
        else:
            logger.warning(
                "Bash executing on the HOST (no sandbox attached to ToolContext); "
                "command will run on the serving pod, not in an isolated session. "
                "This is a degraded path — check that the agent's sandbox session "
                "is being propagated into the tool dispatch context."
            )
        cwd = context.working_dir or None

        # Build a SCRUBBED environment (audit S3): a benign base +
        # host-injected env_vars, never the backend's full secret-bearing
        # os.environ.
        env = _scrubbed_env(context.env_vars)

        # Windows 호스트(커넥터 로컬)는 bash 가 없으므로 PowerShell 로 돈다 — 셸 선택은
        # _host_shell_argv 가 캡슐화한다(POSIX 는 None → 기존 /bin/sh 경로).
        shell_argv = _host_shell_argv(command)
        try:
            if shell_argv is not None:
                proc = await asyncio.create_subprocess_exec(
                    *shell_argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                    **_host_spawn_kwargs(),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                    **_host_spawn_kwargs(),
                )
        except OSError as e:
            return ToolResult(content=f"Failed to start process: {e}", is_error=True)

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            await _kill_process_tree(proc)
            return ToolResult(
                content=f"Command timed out after {timeout_ms}ms",
                is_error=True,
            )
        except asyncio.CancelledError:
            # 턴이 멈췄다(사용자 [정지]). 명령이 만든 프로세스를 남겨 두지 않는다.
            await _kill_process_tree(proc)
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0

        return await _finish_result(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            sandboxed=False,
            input=input,
            context=context,
            stdout_total_bytes=len(stdout_bytes),
            stderr_total_bytes=len(stderr_bytes),
        )
