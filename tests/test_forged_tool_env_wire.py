"""도구는 **자기 환경**을 갖는다 — 부르는 사람의 세션 환경이 아니라.

회귀 배경: 승격 때 세션 환경을 병합하면 핀이 달라져 ``env_id`` 가 비워진다.
그 값을 다시 채우는 곳이 없으면 그 도구는 env 없이 = **부르는 에이전트의 세션
환경**에서 돈다. 원본 에이전트에서는 우연히 되고(그 패키지가 거기 있으니) 다른
에이전트에서는 ImportError 가 난다 — 공용 도구가 공용이 아니게 된다.
"""
from __future__ import annotations

import inspect

import pytest

from xgen_agent_runtime.host.forged_tools import ForgedToolSpec, _run_in_sandbox


class _Sandbox:
    workdir = "/w"

    def __init__(self, ensure_id: str = "resolved-env"):
        self.calls: list = []
        self._ensure_id = ensure_id

    async def ensure_env(self, deps):
        self.calls.append(("ensure_env", list(deps)))
        return (self._ensure_id, [f"{d}==1.0" for d in deps])

    async def exec(self, argv, **kw):
        self.calls.append(("exec", argv, kw))
        return type("R", (), {"rc": 0, "stdout": b"", "stderr": b""})()


def _spec(**over) -> ForgedToolSpec:
    base = dict(
        name="t", description="d", entrypoint="t.py", runtime="python3",
        input_schema={}, argv=[], env={}, timeout_s=10.0,
    )
    base.update(over)
    return ForgedToolSpec(**base)


@pytest.mark.asyncio
async def test_a_tool_env_is_sent_with_its_pins():
    sb = _Sandbox()
    spec = _spec(env_id="tool-env", dependencies=["pandas==2.2.3"])
    await _run_in_sandbox(sb, spec, b"{}")
    _, _argv, kw = next(c for c in sb.calls if c[0] == "exec")
    assert kw["env_id"] == "tool-env"
    # 핀이 없으면 러너가 그 환경을 재건하지 못한다 (아티팩트가 사라지면 끝).
    assert kw.get("env_packages") == ["pandas==2.2.3"]


@pytest.mark.asyncio
async def test_declared_deps_without_an_env_are_resolved_not_ignored():
    """env_id 가 비었는데 의존성이 있으면 **세션 환경으로 흘려보내지 않는다**."""
    sb = _Sandbox()
    spec = _spec(env_id="", dependencies=["pandas"])
    await _run_in_sandbox(sb, spec, b"{}")
    assert ("ensure_env", ["pandas"]) in sb.calls, "환경을 확정하지 않았다"
    _, _argv, kw = next(c for c in sb.calls if c[0] == "exec")
    assert kw["env_id"] == "resolved-env"


@pytest.mark.asyncio
async def test_a_tool_without_deps_uses_the_session_env():
    """의존성이 없으면 세션 환경 그대로 — 굳이 환경을 만들지 않는다."""
    sb = _Sandbox()
    await _run_in_sandbox(sb, _spec(env_id="", dependencies=[]), b"{}")
    assert not any(c[0] == "ensure_env" for c in sb.calls)
    _, _argv, kw = next(c for c in sb.calls if c[0] == "exec")
    assert "env_id" not in kw


@pytest.mark.asyncio
async def test_old_runners_do_not_get_the_new_argument():
    """구버전 러너 클라이언트는 env_packages 를 모른다 — 시그니처를 보고 넘긴다."""
    class _OldSandbox(_Sandbox):
        async def exec(self, argv, *, stdin=None, env=None, timeout_s=120.0, cwd=None, env_id=""):
            self.calls.append(("exec", argv, {"env_id": env_id}))
            return type("R", (), {"rc": 0, "stdout": b"", "stderr": b""})()

    sb = _OldSandbox()
    assert "env_packages" not in inspect.signature(sb.exec).parameters
    await _run_in_sandbox(sb, _spec(env_id="tool-env", dependencies=["pandas==2.2.3"]), b"{}")
    _, _argv, kw = next(c for c in sb.calls if c[0] == "exec")
    assert kw["env_id"] == "tool-env"
