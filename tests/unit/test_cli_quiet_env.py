"""Claude Code CLI 는 필요한 트래픽만 낸다 — 자동 업데이트·텔레메트리·오류 보고를 끈다.

2026-09-23 안정성 감사 F18. 예전에는 ``DISABLE_AUTOUPDATER`` 하나만 줬다 — 폐쇄망에서
실행마다 텔레메트리·오류 보고 연결을 시도했고, 인터넷이 되는 곳에서도 서버가 보낼 이유가
없다. ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`` 가 그것들을 한 번에 끈다.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from xgen_agent_runtime.host import runner


@pytest.fixture
def captured(monkeypatch) -> Dict[str, Any]:
    import xgen_agent_runtime.llm_client.claude_code as cc

    seen: Dict[str, Any] = {}

    class _Capture:
        def __init__(self, **kw: Any) -> None:
            seen.clear()
            seen.update(kw)

    monkeypatch.setattr(cc, "ClaudeCodeCLIClient", _Capture)
    return seen


@pytest.mark.parametrize(
    "auth_mode, extra",
    [("api_key", {"api_key": "sk"}), ("setup_token", {"oauth_token": "tok"}), ("oauth", {})],
)
def test_every_auth_mode_is_quiet(captured, auth_mode, extra) -> None:
    runner.build_cli_client(auth_mode=auth_mode, **extra)
    env = captured["env_extras"]
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["DISABLE_AUTOUPDATER"] == "1"


def test_the_setup_token_still_rides_along(captured) -> None:
    runner.build_cli_client(auth_mode="setup_token", oauth_token="tok")
    assert captured["env_extras"]["CLAUDE_CODE_OAUTH_TOKEN"] == "tok"


def test_a_caller_cannot_accidentally_turn_it_back_on(captured) -> None:
    """호출자 extra_env 는 병합이지 덮어쓰기가 아니다(기존 값 우선)."""
    runner.build_cli_client(
        auth_mode="api_key", api_key="sk",
        extra_env={"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0", "X": "1"},
    )
    env = captured["env_extras"]
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1" and env["X"] == "1"


def test_the_shared_default_is_not_mutated(captured) -> None:
    runner.build_cli_client(auth_mode="setup_token", oauth_token="tok")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in runner.CLI_QUIET_ENV
