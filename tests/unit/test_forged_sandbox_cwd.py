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


# ── 코드는 sandbox 에서만 돈다 ────────────────────────────────────────
#
# 예전엔 러너가 없을 때를 위한 로컬 폴백이 있었다. 그게 두 번째 세계를 만들었다:
# PythonEnv 는 workspace 안의 `pip install --target` 디렉터리에 설치하는데, 그
# 디렉터리를 PYTHONPATH 에 얹는 건 제작 도구뿐이라 Bash 로 테스트하는 에이전트는
# "설치했는데 못 찾는다" 를 반복했다(프로드 실증). 세계는 하나여야 한다.

def _ctx(sandbox):
    from xgen_agent_runtime.tools.base import ToolContext

    return ToolContext(session_id="i", working_dir="/w", allowed_paths=["/w"], sandbox=sandbox)


def test_세션이_없으면_도는_척하지_않는다(tmp_path):
    from xgen_agent_runtime.host.forged_tools import ForgedScriptTool, _bind_tool_base

    tool = _bind_tool_base(ForgedScriptTool(_spec(), workspace_dir=str(tmp_path), store=None))
    res = asyncio.run(tool.execute({}, _ctx(None)))
    assert res.is_error
    assert "sandbox" in res.content


def test_파이썬_환경도_세션_안에서만_관리된다(tmp_path):
    from xgen_agent_runtime.host.python_env import PythonEnvTool
    from xgen_agent_runtime.host.forged_tools import _bind_tool_base

    tool = _bind_tool_base(PythonEnvTool(workflow_id="wf", workspace_dir=str(tmp_path)))
    res = asyncio.run(tool.execute({"action": "install", "packages": ["httpx"]}, _ctx(None)))
    assert res.is_error
    assert "sandbox" in res.content


def test_로컬_폴백_기계는_남아_있지_않다():
    """죽은 폴백이 코드에 남아 있으면 다음 사람이 되살린다."""
    from xgen_agent_runtime.host import forged_tools as ft

    for gone in (
        "_session_local_env_dir", "_local_pythonpath", "_pip_install_target",
        "_ensure_local_env", "_tool_env_dir", "_pin_from_target", "_child_env",
    ):
        assert not hasattr(ft, gone), gone
