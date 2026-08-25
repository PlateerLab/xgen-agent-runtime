"""호스트 Bash 도구의 shell 선택 — Windows 는 PowerShell(없으면 cmd), POSIX 는 /bin/sh.

회귀(2026-08-24 Windows 실기): 커넥터 로컬(sandbox 없음) Bash 가 Windows 에서
create_subprocess_shell(=cmd.exe)로 돌아 bash 문법 명령이 깨졌다. 이제 PowerShell 로 돈다.
"""
from __future__ import annotations


from xgen_agent_runtime.tools.built_in import bash_tool as bt


def test_posix_uses_default_shell_path(monkeypatch):
    assert bt._host_shell_argv("ls -la", platform="linux") is None
    assert bt._host_shell_argv("echo hi", platform="darwin") is None


def test_windows_prefers_powershell(monkeypatch):
    monkeypatch.setattr(bt, "_which_windows", lambda name: r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" if name == "powershell.exe" else None)
    argv = bt._host_shell_argv("Get-ChildItem", platform="win32")
    assert argv is not None
    assert argv[0].lower().endswith("powershell.exe")
    assert argv[1:] == ["-NoProfile", "-NonInteractive", "-Command", "Get-ChildItem"]


def test_windows_falls_back_to_cmd_when_no_powershell(monkeypatch):
    monkeypatch.setattr(bt, "_which_windows", lambda name: None)
    monkeypatch.setenv("ComSpec", r"C:\\Windows\\System32\\cmd.exe")
    argv = bt._host_shell_argv("dir", platform="win32")
    assert argv is not None
    assert argv[0].lower().endswith("cmd.exe")
    assert argv[1:] == ["/d", "/s", "/c", "dir"]


def test_description_mentions_powershell_on_windows():
    desc = bt.BashTool().description
    assert "PowerShell" in desc and "POSIX" in desc
