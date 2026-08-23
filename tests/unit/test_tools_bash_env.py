"""BashTool host-path env scrub — platform-aware (desktop host on Windows)
+ description honest on both execution hosts."""

from __future__ import annotations

from xgen_agent_runtime.tools.built_in.bash_tool import (
    BashTool,
    _SAFE_ENV_KEYS_WINDOWS_FALLBACK,
    _scrubbed_env,
    _windows_env_keys,
)

_WIN_ENV = {
    "Path": r"C:\Windows\System32;C:\Python",
    "SystemRoot": r"C:\Windows",
    "windir": r"C:\Windows",
    "ComSpec": r"C:\Windows\System32\cmd.exe",
    "PATHEXT": ".COM;.EXE;.BAT;.CMD",
    "TEMP": r"C:\Users\u\AppData\Local\Temp",
    "TMP": r"C:\Users\u\AppData\Local\Temp",
    "USERPROFILE": r"C:\Users\u",
    "APPDATA": r"C:\Users\u\AppData\Roaming",
    "LOCALAPPDATA": r"C:\Users\u\AppData\Local",
    "ProgramData": r"C:\ProgramData",
    "HOMEDRIVE": "C:",
    "HOMEPATH": r"\Users\u",
    "ANTHROPIC_API_KEY": "sk-secret",
    "XGEN_TOKEN": "secret-token",
}


def test_windows_whitelist_includes_bootstrap_vars():
    keys = _windows_env_keys()
    for k in (
        "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE",
        "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "HOMEDRIVE", "HOMEPATH",
    ):
        assert k in keys
    # local fallback table is a subset (import-cycle safety net)
    assert _SAFE_ENV_KEYS_WINDOWS_FALLBACK <= keys


def test_windows_scrub_keeps_bootstrap_vars_case_insensitively_and_drops_secrets():
    env = _scrubbed_env(None, environ=_WIN_ENV, platform="win32")
    # parent's spelling preserved, matched case-insensitively
    assert env["Path"] == _WIN_ENV["Path"]
    assert env["SystemRoot"] == r"C:\Windows"
    assert env["ComSpec"].endswith("cmd.exe")
    assert env["windir"] == r"C:\Windows"
    for k in ("PATHEXT", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
              "ProgramData", "HOMEDRIVE", "HOMEPATH"):
        assert k in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "XGEN_TOKEN" not in env


def test_windows_scrub_maps_home_from_userprofile_when_unset():
    env = _scrubbed_env(None, environ=_WIN_ENV, platform="win32")
    assert env["HOME"] == r"C:\Users\u"


def test_windows_scrub_keeps_explicit_home():
    env = _scrubbed_env(None, environ={**_WIN_ENV, "HOME": r"D:\home"}, platform="win32")
    assert env["HOME"] == r"D:\home"


def test_windows_scrub_does_not_synthesise_path_when_present_in_any_case():
    env = _scrubbed_env(None, environ=_WIN_ENV, platform="win32")
    assert [k for k in env if k.upper() == "PATH"] == ["Path"]


def test_windows_scrub_synthesises_path_from_systemroot_when_missing():
    env = _scrubbed_env(None, environ={"SystemRoot": r"C:\Windows"}, platform="win32")
    assert env["PATH"] == r"C:\Windows\System32;C:\Windows"


def test_extra_env_overrides_on_windows():
    env = _scrubbed_env({"MY_VAR": "v", "HOME": "X"}, environ=_WIN_ENV, platform="win32")
    assert env["MY_VAR"] == "v" and env["HOME"] == "X"


def test_posix_scrub_unchanged():
    env = _scrubbed_env(None, environ={"HOME": "/h", "SECRET": "s", "LC_ALL": "C"}, platform="linux")
    assert env == {"HOME": "/h", "LC_ALL": "C", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    # Windows bootstrap names are NOT whitelisted on POSIX
    env = _scrubbed_env(None, environ={"SYSTEMROOT": "x"}, platform="linux")
    assert "SYSTEMROOT" not in env


def test_inherit_opt_in_applies_on_windows_too():
    env = _scrubbed_env(None, environ={**_WIN_ENV, "GENY_BASH_INHERIT_ENV": "1"}, platform="win32")
    assert env["ANTHROPIC_API_KEY"] == "sk-secret"


def test_description_is_true_on_both_hosts():
    desc = BashTool().description
    assert "separate from the server" not in desc  # the unconditional sandbox claim is gone
    assert "sandbox" in desc and "PC" in desc
    assert "prompt" in desc  # points at the environment prompt for which host applies
