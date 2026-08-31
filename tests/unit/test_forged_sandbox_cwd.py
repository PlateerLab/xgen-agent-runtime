"""제작 도구는 **자기 workspace 에서** 돈다.

entrypoint 는 workspace 기준 상대 경로다. 실행 위치를 러너의 기본값에 맡기면,
그 기본값이 바뀌는 날 "도구는 등록됐는데 스크립트를 못 찾는다"가 되고 — 아무
로그도 그 이유를 말해 주지 않는다. 다른 모든 도구(sb_run)는 이미 세션 workdir 을
명시해서 넘긴다.
"""
import asyncio

from xgen_agent_runtime.host.forged_tools import ForgedToolSpec, _run_in_sandbox


class _Sandbox:
    workdir = "/data/agent-workspaces/wf_42"

    def __init__(self):
        self.calls = []

    async def exec(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return object()


def _spec(**over):
    base = {
        "name": "t", "description": "테스트 도구", "entrypoint": "tools/t.py", "runtime": "python3",
        "argv": [], "env": {}, "timeout_s": 30.0,
    }
    base.update(over)
    return ForgedToolSpec.from_dict(base)


def test_실행_위치는_에이전트의_workspace다():
    sb = _Sandbox()
    asyncio.run(_run_in_sandbox(sb, _spec(), b"{}"))
    _argv, kwargs = sb.calls[0]
    assert kwargs["cwd"] == "/data/agent-workspaces/wf_42", kwargs


def test_에이전트마다_제_자리에서_돈다():
    """workdir 이 다르면 실행 위치도 달라야 한다 — 공유 기본값이 아니다."""
    a, b = _Sandbox(), _Sandbox()
    b.workdir = "/data/agent-workspaces/wf_99"
    asyncio.run(_run_in_sandbox(a, _spec(), b"{}"))
    asyncio.run(_run_in_sandbox(b, _spec(), b"{}"))
    assert a.calls[0][1]["cwd"] != b.calls[0][1]["cwd"]


def test_로컬_폴백_env_id_는_러너로_안_넘어간다():
    sb = _Sandbox()
    asyncio.run(_run_in_sandbox(sb, _spec(env_id="local:abc"), b"{}"))
    assert "env_id" not in sb.calls[0][1]
    sb2 = _Sandbox()
    asyncio.run(_run_in_sandbox(sb2, _spec(env_id="sha256:abc"), b"{}"))
    assert sb2.calls[0][1]["env_id"] == "sha256:abc"
