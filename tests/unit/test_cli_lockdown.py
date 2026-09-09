"""claude_code 백엔드는 **CLI 가 기본으로 주는 것을 전부 닫는다.**

무엇이 있었나 (2026-09-09 실측)
--------------------------------
에이전트가 아티팩트를 만들다가 CLI **번들 스킬**(dataviz)을 열었다. 그 스킬은
"차트를 그리기 전에 references/palette.md 를 먼저 읽어라" 고 말했고, 그 파일은
**CLI 가 도는 파드의 /tmp** 에 있었다. 우리 Read 는 러너 샌드박스로 가므로 닿을
수 없었고, 경로 가드가 정확히 막았다.

파일 실패는 첫 증상일 뿐이었다. 그 스킬들은 **다른 제품**을 설명한다 — Claude
Code 의 Artifact(HTML·window.claude.*·cdnjs). 우리 아티팩트는 React +
ArtifactSave + xgen.file() 이다. 즉 에이전트가 없는 시스템의 사용법을 따랐다.

왜 새고 있었나
--------------
네이티브 차단이 **손으로 적은 12종 열거**였다. 실측하니 CLI 는 27종을 광고했고,
15종이 새고 있었으며, 우리 목록의 6종은 이미 없는 이름이었다. 그리고 도구만이
아니었다 — 스킬 20종·슬래시커맨드 51종·auto-memory·CLAUDE.md 자동탐색이 전부
열려 있었다.

여기서 고정하는 것
------------------
1. 차단은 **상위집합**이다 — 알던 것을 전부 막는다(모르는 이름은 무해하다: 실측).
2. 스킬·커맨드 표면도 함께 닫는다 (``--disable-slash-commands``).
3. 맥락 자동주입을 끈다 (``CLAUDE_CODE_SIMPLE=1``) — **모든 인증 경로에서**.
4. 그래도 새면 **알린다** (:func:`native_tool_leaks`) — 조용한 드리프트가 이
   사고의 본체였다.
"""

from __future__ import annotations

from xgen_agent_runtime.host.runner import (
    CLI_NATIVE_TOOL_CATALOG,
    CLI_NATIVE_TOOLS_DENY,
    native_tool_leaks,
)

#: CLI 2.1.x 가 실제로 광고한 도구 — 고정점까지 반복해 얻은 실측값.
#: (26종 차단 → Glob·Grep 이 새로 나타남 → 그것까지 막아 0종)
MEASURED_2026_09_09 = (
    "Task", "Artifact", "Bash", "CronCreate", "CronDelete", "CronList",
    "DesignSync", "Edit", "EnterWorktree", "ExitWorktree", "Glob", "Grep",
    "ListAgents", "Monitor", "NotebookEdit", "PushNotification", "Read",
    "RemoteTrigger", "ReportFindings", "ScheduleWakeup", "SendMessage",
    "Skill", "TaskOutput", "TaskStop", "ToolSearch", "WebFetch", "WebSearch",
    "Workflow", "Write",
)


# ── 1. 상위집합이 실측을 덮는가 ───────────────────────────────────────
def test_실측한_도구를_하나도_빠짐없이_막는다():
    """이 목록이 뒤처지는 것이 사고의 원인이었다. 실측값이 기준선이다."""
    missing = [t for t in MEASURED_2026_09_09 if t not in CLI_NATIVE_TOOLS_DENY]
    assert not missing, f"차단 목록에서 빠진 도구: {missing}"


def test_사고의_주인공이_목록에_있다():
    """Skill 은 CLI 번들 스킬로 가는 문이었다. --disable-slash-commands 가
    목록까지 비우지만, 이름도 함께 막아 두 겹으로 닫는다."""
    assert "Skill" in CLI_NATIVE_TOOLS_DENY


def test_이름이_겹치는_것들이_막힌다():
    """우리도 같은 이름의 도구를 광고한다 — 둘이 같이 서면 어느 쪽이 돌았는지
    사후에 구분되지 않는다."""
    for name in ("ToolSearch", "Artifact", "Task", "Bash", "Read", "Write"):
        assert name in CLI_NATIVE_TOOLS_DENY, name


def test_옛_이름은_별칭으로_남는다():
    """xgen-workflow 가 옛 이름으로 import 한다. 두 레포가 어느 순서로 나가도
    ImportError 가 나지 않아야 한다 — 배포 순서를 계약으로 만들지 않는다."""
    assert CLI_NATIVE_TOOL_CATALOG is CLI_NATIVE_TOOLS_DENY


# ── 2. 누수 검산 ──────────────────────────────────────────────────────
def test_우리_브릿지는_누수가_아니다():
    """mcp__connector__* 는 남아야 정상이다 — 그게 우리가 의도한 유일한 표면이다."""
    assert native_tool_leaks(["mcp__connector__Read", "mcp__connector__Bash"]) == ()


def test_막은_것은_누수가_아니다():
    assert native_tool_leaks(list(MEASURED_2026_09_09)) == ()


def test_모르는_이름은_누수로_잡힌다():
    """CLI 가 도구를 늘리는 날, 이 한 줄이 유일한 신호다."""
    assert native_tool_leaks(["Bash", "SomeNewTool", "mcp__connector__X"]) == ("SomeNewTool",)


def test_빈_입력에도_깨지지_않는다():
    for empty in (None, (), [], ["", None]):
        assert native_tool_leaks(empty) == ()


# ── 3. argv — 스킬·커맨드 표면도 닫는가 ──────────────────────────────
def _argv(**kw):
    from xgen_agent_runtime.llm_client.translators._cli import claude_code_argv
    from xgen_agent_runtime.llm_client.types import APIRequest

    req = APIRequest(messages=[{"role": "user", "content": "hi"}], model="haiku", stream=True)
    return claude_code_argv(req, **kw)


def test_네이티브를_막으면_스킬_표면도_닫는다():
    """도구만 막는 것으로는 부족하다 — CLI 는 자기 번들 스킬을 세션에 광고하고,
    그 본문이 닿을 수 없는 파일을 가리킨다.

    실측: 이 플래그 하나로 skills 20→0, slash_commands 51→0, 그리고 도구 목록에서
    ``Skill`` 자체가 사라진다(27→26).
    """
    argv = _argv(disallow_tools=("Bash", "Read"))
    assert "--disable-slash-commands" in argv


def test_네이티브를_허용하면_스킬도_건드리지_않는다():
    """allow_local_tools 로 네이티브를 쓰기로 한 배포에서는 표면을 닫지 않는다 —
    한쪽만 닫으면 '도구는 있는데 스킬은 없는' 어중간한 상태가 된다."""
    argv = _argv()
    assert "--disable-slash-commands" not in argv


# ── 4. 맥락 제거 스위치는 인증을 건드리지 않는다 ─────────────────────
#
# 여기 있던 테스트가 **버그를 고정하고 있었다.** 4.18.0 은 "env 로 주면 맥락 제거만
# 얻고 인증은 그대로다" 라는 가정 위에 CLAUDE_CODE_SIMPLE=1 을 모든 인증 경로에
# 넣었고, 테스트는 그 가정이 아니라 그 **구현**을 확인했다(세 경로 전부에서 값이
# "1" 인가). 가정이 틀렸다는 것은 아무도 묻지 않았다.
#
# 실측(2026-09-09, CLI 2.1.236) — --bare 와 같은 스위치다:
#
#     CLAUDE_CODE_OAUTH_TOKEN 만           → 401 "OAuth access token is invalid"
#                                            (= 토큰을 읽었다)
#     CLAUDE_CODE_OAUTH_TOKEN + SIMPLE=1   → "Not logged in · Please run /login"
#                                            (= 토큰을 아예 안 읽는다)
#
# 그래서 프로드의 모든 대화가 authentication_failed 로 죽었고, **연결 테스트는
# 초록불이었다** — 테스트 경로(workflow _child_env)는 이 env 를 넣지 않는다.
def _client(auth, **kw):
    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient

    if auth == "api_key":
        kw.setdefault("api_key", "sk-test")
    return ClaudeCodeCLIClient(binary_path="/bin/true", auth_mode=auth, **kw)


def test_구독_인증에는_맥락_스위치를_켜지_않는다():
    """이 한 줄이 4.18.0 장애를 막는다."""
    for auth in ("oauth", "setup_token"):
        assert _client(auth)._env_extras().get("CLAUDE_CODE_SIMPLE") is None, auth


def test_호스트가_켜도_구독_인증에서는_걷어낸다():
    """설정 하나가 조용히 전 사용자를 로그아웃시키는 문을 남기지 않는다.
    그 문은 연결 테스트로 잡히지 않으므로, 여기서 닫는 수밖에 없다."""
    for auth in ("oauth", "setup_token", "auto"):
        client = _client(auth, env_extras={"CLAUDE_CODE_SIMPLE": "1"})
        assert client._env_extras().get("CLAUDE_CODE_SIMPLE") is None, auth


def test_api_key_경로는_bare_가_스위치를_소유한다():
    """api_key 채널에서는 --bare 가 이미 CLAUDE_CODE_SIMPLE=1 을 세운다
    (claude --help). 스위치는 하나고, 켜는 자리도 하나여야 한다."""
    assert "--bare" in _argv(bare_mode=True, auth_mode="api_key", has_api_key=True)


def test_argv_와_env_는_같은_판정을_쓴다():
    """문이 둘이면 한쪽만 고쳐진다 — 4.18.0 이 정확히 그랬다(argv 는 규칙을
    지켰고 env 만 어겼다)."""
    from xgen_agent_runtime.llm_client.translators._cli import resolve_auth_channel

    for auth, has_key in (("api_key", True), ("oauth", False), ("setup_token", False),
                          ("auto", False), ("auto", True)):
        channel = resolve_auth_channel(auth, has_key)
        argv = _argv(bare_mode=True, auth_mode=auth, has_api_key=has_key)
        client = _client(auth if auth != "auto" else "auto",
                         **({"api_key": "sk-test"} if has_key else {}))
        env_on = client._env_extras().get("CLAUDE_CODE_SIMPLE") == "1"
        assert ("--bare" in argv) == (channel == "api_key"), (auth, has_key)
        # env 가 argv 보다 넓으면 안 된다: --bare 가 안 붙는 턴에 스위치가 켜지면
        # 그 턴은 자격증명을 못 읽는다.
        assert not (env_on and "--bare" not in argv), (auth, has_key)


def test_구독_인증은_절대_bare_를_붙이지_않는다():
    for auth in ("oauth", "setup_token"):
        assert "--bare" not in _argv(bare_mode=True, auth_mode=auth, has_api_key=False)
