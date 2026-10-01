"""BashTool — execute shell commands."""

from __future__ import annotations

import asyncio
import logging
import os
import re
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

# ── 명령의 성격 ────────────────────────────────────────────────────────
# ``capabilities(input)`` 이 명령마다 답한다(base.py 의 약속: "ls 는 read_only, rm 은 destructive"). 호출 사건을 받는
# 쪽(실행 기록 · 기억)이 셸 문자열을 저마다 다시 해석하지 않게 한다. **모르면 읽기 전용이 아니다**(실패-닫힘): 읽기
# 전용은 명령줄의 모든 구간이 아는 읽기 프로그램으로 시작하고 쓰기 흔적이 없을 때만이다.

#: 구간 머리의 환경 변수 · 감싸는 명령(sudo [-u x] · time · nohup · env · command · exec · nice). 이 뒤가 진짜 프로그램이다.
_SHELL_PREFIX_RE = re.compile(
    r"^(?:\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+|sudo(?:\s+-\S+(?:\s+\S+)?)*\s+|time\s+|nohup\s+"
    r"|env\s+|command\s+|exec\s+|nice(?:\s+-n\s*\S+)?\s+|busybox\s+))*"
)
#: 결과를 읽기만 하는 프로그램. 하위 명령이나 플래그에 따라 쓰는 것(git · find · sed · tar · docker …)은 따로 본다.
_READ_ONLY_PROGRAMS = frozenset({
    "ls", "dir", "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "ag", "ack", "wc", "echo", "printf",
    "pwd", "which", "whereis", "type", "env", "printenv", "date", "cal", "whoami", "id", "uname", "hostname", "df", "du",
    "stat", "file", "sort", "uniq", "cut", "awk", "gawk", "tr", "diff", "cmp", "comm", "md5sum", "sha1sum", "sha256sum",
    "basename", "dirname", "realpath", "readlink", "test", "[", "true", "false", "jq", "yq", "xxd", "hexdump", "od",
    "strings", "tree", "column", "nl", "paste", "fold", "seq", "ps", "pgrep", "lsof", "free", "uptime", "nproc", "lscpu",
    "lsblk", "mount", "ping", "nslookup", "dig", "host", "traceroute", "netstat", "ss", "ip", "ifconfig", "history",
    "man", "help", "tac", "rev", "expr", "bc", "sleep", "wait", "ulimit", "locale", "getconf", "lsb_release", "fdisk",
    "last", "w", "who", "zcat", "bzcat", "xzcat", "iconv", "base64", "sha512sum", "cd", "pushd", "popd", "export",
    "set", "unset", "alias", "ulimit", "umask", "shopt", "declare", "local", "readonly", "__heredoc__",
    "Get-ChildItem", "Get-Content", "Get-Item", "Get-Location", "Select-String", "Get-Date", "Get-Process", "Write-Output",
    "Write-Host", "Test-Path", "Resolve-Path", "Get-Command", "Measure-Object", "Select-Object", "Where-Object",
    "ForEach-Object", "Format-Table", "Format-List", "Out-String", "ConvertTo-Json", "ConvertFrom-Json", "Sort-Object",
    "Get-Help", "Get-Member", "findstr", "where", "tasklist", "systeminfo", "ver", "hostname", "ipconfig",
})
#: 바깥으로 나가는 프로그램(읽기 전용일 수 있다 - curl 로 보기만 하는 것).
_EGRESS_PROGRAMS = frozenset({"curl", "wget", "ssh", "scp", "sftp", "rsync", "ping", "nslookup", "dig", "host", "traceroute",
                              "nc", "netcat", "telnet", "Invoke-WebRequest", "Invoke-RestMethod", "iwr", "irm"})
_PKG_MANAGERS = frozenset({"pip", "pip3", "npm", "pnpm", "yarn", "apt", "apt-get", "apk", "conda", "uv", "brew", "yum",
                           "dnf", "cargo", "pipx", "gem", "go", "poetry", "choco", "winget"})
_PKG_READ_ONLY = frozenset({"show", "list", "freeze", "search", "info", "ls", "view", "outdated", "check", "env", "config",
                            "version", "--version", "-v", "-V", "help", "--help"})
_GIT_READ_ONLY = frozenset({"log", "status", "diff", "show", "rev-parse", "ls-files", "ls-tree", "blame", "describe",
                            "shortlog", "grep", "cat-file", "rev-list", "reflog", "name-rev", "var", "version", "--version",
                            "check-ignore", "merge-base", "ls-remote", "count-objects", "whatchanged"})
_DOCKER_READ_ONLY = frozenset({"ps", "images", "logs", "inspect", "version", "info", "stats", "top", "port", "diff",
                               "history", "search", "events", "--version"})
_KUBECTL_READ_ONLY = frozenset({"get", "describe", "logs", "top", "version", "cluster-info", "api-resources", "explain",
                                "config", "diff"})
_INTERPRETERS = frozenset({"python", "python3", "python2", "node", "ruby", "perl", "php", "bash", "sh", "zsh", "dash",
                           "pwsh", "powershell", "Rscript", "deno", "bun"})
#: 스크립트 안의 쓰기 흔적(인라인 코드에만 본다).
_INLINE_WRITE_RE = re.compile(
    r"\bto_(?:excel|csv|json|parquet|pickle|sql)\(|\.save(?:fig)?\(|\bwrite_(?:text|bytes)\(|\bopen\([^)]*[\"'][wax]b?\+?[\"']"
    r"|\bos\.(?:remove|unlink|rename|rmdir|makedirs|mkdir|replace)\(|\bshutil\.|\bfs\.(?:write|append|unlink|rm|rename|mkdir|copy)"
    r"|\bsubprocess\.|\bos\.system\(|\bexecSync\(|\bPath\([^)]*\)\.(?:write_text|write_bytes|unlink|rename|mkdir)\(",
    re.IGNORECASE,
)
#: 되돌릴 수 없는 삭제 · 덮어쓰기. 사용자 PC 의 셸(dex ``isDangerousShellCommand``)이 확인 창을 띄우는 묶음과 같은 범위
#: (재귀 삭제 · 절대 경로 삭제 · 디스크 포맷 · dd · 전원 · 재귀 권한 변경 · 포크 폭탄 · 강제 푸시 · curl | sh · sudo rm).
_SHELL_DESTRUCTIVE_RE = re.compile(
    r"\brm\s+(?:-[a-zA-Z]*[rf][a-zA-Z]*\s+)+"
    r"|\brm\s+/"
    r"|\bsudo\s+rm\b"
    r"|\bRemove-Item\b[^\n]*-Recurse|\brmdir\s+/s|\bdel\s+/[a-z]*[sf]"
    r"|\bdd\s+(?:[^\n|;&]*\s)?(?:of|if)="
    r"|\bch(?:mod|own)\s+-R\b"
    r"|>\s*/dev/(?:sd|nvme|disk|hd)"
    r"|:\s*\(\s*\)\s*\{\s*:\s*\|\s*:"
    r"|\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh)\b"
    r"|\bgit\s+(?:reset\s+--hard|clean\s+-[a-zA-Z]*f|push\b[^\n]*(?:--force|-f\b)|branch\s+-D|stash\s+drop|checkout\s+--\s)"
    r"|\b(?:DROP|TRUNCATE)\s+(?:TABLE|DATABASE|SCHEMA)\b|\bDELETE\s+FROM\b"
    r"|\b(?:shred|wipefs)\b|\brsync\b[^\n]*--delete|\bdocker\s+(?:rm|rmi|system\s+prune)\b|\bkubectl\s+delete\b"
    r"|\bfind\b[^\n]*(?:\s-delete\b|-exec\s+rm\b)|\bxargs\s+(?:-\S+\s+)*rm\b",
    re.IGNORECASE,
)
#: 전원 · 디스크 명령은 명령 자리에서만(``grep shutdown log`` 는 읽기다).
_DESTRUCTIVE_PROGRAMS = frozenset({"mkfs", "fdisk", "format", "format.com", "shutdown", "reboot", "halt", "poweroff",
                                   "shred", "wipefs", "diskpart"})
#: 바깥으로 나가는 흔적(프로그램 밖의 것): HTTP 클라이언트 호출 코드 · 원격 git · 설치.
_SHELL_EGRESS_RE = re.compile(
    r"\brequests\.(?:get|post|put|delete|patch|head)\(|\burlopen\(|\bhttpx\.|\bfetch\(|\baiohttp\."
    r"|\bgit\s+(?:clone|fetch|pull|push|ls-remote)\b",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(r"\|\||&&|\||;|\n|&(?!&)")
_REDIRECT_PATH_RE = re.compile(r"(?<![0-9&<>=-])>>?\s*(?!&|/dev/null\b|[0-9.]+(?:[\s'\")]|$))\S")
_REDIRECT_FD_RE = re.compile(r"\d*>&\d+|\d*>\s*/dev/null")


def _program(segment: str) -> Tuple[str, List[str]]:
    """구간의 프로그램 이름(경로 · 따옴표 벗김)과 인자."""
    body = _SHELL_PREFIX_RE.sub("", segment.strip())
    parts = body.split()
    if not parts:
        return "", []
    head = parts[0].strip("\"'")
    head = head.replace("\\", "/").rsplit("/", 1)[-1]
    if head.lower().endswith(".exe"):
        head = head[:-4]
    return head, parts[1:]


def _segment_read_only(segment: str) -> bool:
    prog, args = _program(segment)
    if not prog:
        return True
    flags = " ".join(args)
    if prog in ("xargs",):
        rest = [a for a in args if not a.startswith("-")]
        return _segment_read_only(" ".join(rest)) if rest else True
    if prog == "find":
        return not re.search(r"(?:^|\s)-(?:delete|exec|execdir|ok|okdir|fprint\w*)\b", flags)
    if prog == "sed":
        return not re.search(r"(?:^|\s)-[a-zA-Z]*i|(?:^|\s)--in-place", flags)
    if prog == "git":
        sub = next((a for a in args if not a.startswith("-")), "")
        if sub == "branch":
            return not re.search(r"(?:^|\s)-(?:[a-zA-Z]*[dDmM]|-delete|-move)\b", flags)
        if sub == "remote":
            return not re.search(r"\b(?:add|remove|rm|rename|set-url)\b", flags)
        if sub == "tag":
            return not re.search(r"(?:^|\s)-(?:[a-zA-Z]*[da])\b", flags) and not any(not a.startswith("-") for a in args[1:])
        if sub == "config":
            return "--get" in flags or "-l" in flags or "--list" in flags
        if sub == "stash":
            return any(a in ("list", "show") for a in args[1:])
        return sub in _GIT_READ_ONLY
    if prog in ("tar",):
        mode = args[0].lstrip("-") if args else ""
        return "t" in mode and not any(c in mode for c in "xc")
    if prog in ("unzip", "7z", "7za", "zipinfo"):
        return bool(re.search(r"(?:^|\s)-l\b|(?:^|\s)l\b|(?:^|\s)-Z\b", flags)) or prog == "zipinfo"
    if prog == "docker":
        sub = next((a for a in args if not a.startswith("-")), "")
        return sub in _DOCKER_READ_ONLY or (sub == "compose" and any(a in ("ps", "logs", "config") for a in args[1:]))
    if prog == "kubectl":
        return next((a for a in args if not a.startswith("-")), "") in _KUBECTL_READ_ONLY
    if prog in _PKG_MANAGERS:
        return next((a for a in args if not a.startswith("-")), "") in _PKG_READ_ONLY or (args and args[0] in _PKG_READ_ONLY)
    if prog in _EGRESS_PROGRAMS:
        return not re.search(r"(?:^|\s)(?:-o|-O|--output|--remote-name|-OutFile|-outfile)\b", flags) \
            and prog not in ("scp", "sftp", "rsync")
    if prog in _INTERPRETERS:
        inline = re.search(r"(?:^|\s)(?:-c|-e|-Command|-Comm?and)\s+(.*)$", flags, re.DOTALL)
        if inline:
            return not _INLINE_WRITE_RE.search(inline.group(1)) and not _SHELL_DESTRUCTIVE_RE.search(inline.group(1))
        return bool(re.search(r"(?:^|\s)(?:-V|--version|-h|--help)\b", flags))   # 스크립트 · 모듈 실행은 안을 모른다
    if prog == "python" or prog == "python3":
        return False
    return prog in _READ_ONLY_PROGRAMS


_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[ \t]*\n(.*?)\n[ \t]*\1[ \t]*(?:\n|$)", re.DOTALL)


def _split_heredocs(command: str) -> Tuple[str, List[str]]:
    """히어독 본문(``python3 - <<'EOF' … EOF``)을 떼어 낸다. 본문은 인라인 코드로 보고 명령줄은 따로 가른다."""
    bodies: List[str] = []

    def take(m: "re.Match[str]") -> str:
        bodies.append(m.group(2))
        return "\n"

    return _HEREDOC_RE.sub(take, command), bodies


def _segments(command: str) -> List[str]:
    text = _REDIRECT_FD_RE.sub(" ", command)
    return [s for s in (p.strip() for p in _SPLIT_RE.split(text)) if s]


def classify_shell_command(command: str) -> ToolCapabilities:
    """셸 명령 하나의 능력. 모든 구간이 아는 읽기 프로그램이고 파일로 쓰는 리다이렉트 · 되돌릴 수 없는 명령이 없을 때만
    읽기 전용이다. 모르는 프로그램 · 스크립트 실행은 읽기 전용이 아니다. 히어독으로 넘긴 스크립트는 그 본문을 본다."""
    raw = str(command or "")
    text, heredocs = _split_heredocs(raw)
    if heredocs:
        inline = "\n".join(heredocs)
        text = re.sub(r"(?<=\s)-(?=\s|$)", "-c __heredoc__", text, count=1)   # ``python3 -`` 는 인라인 코드 실행이다
        if _INLINE_WRITE_RE.search(inline) or _SHELL_DESTRUCTIVE_RE.search(inline):
            text += "\n__write__ > ./heredoc"   # 본문이 쓰면 명령도 쓴다
    segments = _segments(text)
    programs = [_program(s)[0] for s in segments]
    destructive = bool(_SHELL_DESTRUCTIVE_RE.search(_REDIRECT_FD_RE.sub(" ", text))) or any(
        p.lower() in _DESTRUCTIVE_PROGRAMS or p.lower().split(".", 1)[0] in _DESTRUCTIVE_PROGRAMS for p in programs)
    redirects = bool(_REDIRECT_PATH_RE.search(_REDIRECT_FD_RE.sub(" ", text)))
    read_only = bool(segments) and not destructive and not redirects and all(_segment_read_only(s) for s in segments)
    egress = bool(_SHELL_EGRESS_RE.search(text)) or any(p in _EGRESS_PROGRAMS for p in programs) or any(
        p in _PKG_MANAGERS and _program(s)[1] and _program(s)[1][0] in ("install", "add", "upgrade", "update")
        for p, s in zip(programs, segments))
    # 셸은 작업 디렉터리 · 환경 · 프로세스를 공유하므로 읽기 전용 명령도 다른 호출과 나란히 돌리지 않는다.
    return ToolCapabilities(
        concurrency_safe=False,
        read_only=read_only,
        destructive=destructive,
        idempotent=read_only,
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
            "the user's own devices — for folders the user connected, use the device tools. "
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
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
        except OSError as e:
            return ToolResult(content=f"Failed to start process: {e}", is_error=True)

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return ToolResult(
                content=f"Command timed out after {timeout_ms}ms",
                is_error=True,
            )

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
