"""SandboxExecTool — 에이전트가 작성한 스크립트를 자기 샌드박스 세션에서 돌린다.

샌드박스는 가짜다 — :class:`XgenySandbox` 모양을 그대로 만족하는 객체 하나.
monkeypatch 가 필요 없다는 점이 중요하다: 프로토콜이 좁으면 테스트가 실제
계약(도구↔스크립트 JSON 규약과 실패 경로)만 검증하게 된다.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from xgen_agent_runtime.tools._xgeny_sandbox import ExecResult
from xgen_agent_runtime.tools.base import ToolContext
from xgen_agent_runtime.tools.built_in.sandbox_exec_tool import SandboxExecTool


class _FakeSandbox:
    """``fn(argv, stdin) -> (rc, out, err)`` 로 결과를 정하는 세션."""

    workdir = "/workspace"

    def __init__(self, fn=None, *, raises: BaseException | None = None) -> None:
        self.ensured = 0
        self.calls: List[Tuple[Any, ...]] = []
        self._fn = fn or (lambda argv, stdin: (0, b"", b""))
        self._raises = raises

    async def ensure(self) -> None:
        self.ensured += 1

    async def exec(self, argv, *, cwd=None, stdin=None, env=None, timeout_s=120.0):
        self.calls.append((list(argv), cwd, stdin, timeout_s))
        if self._raises is not None:
            raise self._raises
        rc, out, err = self._fn(list(argv), stdin)
        return ExecResult(rc=rc, stdout=out, stderr=err)


def _spec(**over: Any) -> Dict[str, Any]:
    base = {
        "name": "echo_upper",
        "description": "uppercase the 'text' field",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "runtime": "python3",
        "entrypoint": "tools/echo_upper/main.py",
        "timeout_s": 5.0,
    }
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_success_passes_json_stdin_and_returns_stdout() -> None:
    def run(argv, input_bytes):
        # argv = [runtime, entrypoint]; input is JSON on stdin.
        assert argv == ["python3", "tools/echo_upper/main.py"]
        assert b'"hello"' in input_bytes
        return (0, b'{"result": "HELLO"}', b"")

    sb = _FakeSandbox(run)
    calls = sb.calls
    tool = SandboxExecTool.from_dict(_spec(), sandbox=sb)
    res = await tool.execute({"text": "hello"}, ToolContext())
    assert not res.is_error
    assert res.content == '{"result": "HELLO"}'
    assert calls and calls[0][1] == "/workspace"           # cwd
    assert calls[0][3] == 5.0                                # timeout passed through


@pytest.mark.asyncio
async def test_no_sandbox_is_error() -> None:
    tool = SandboxExecTool.from_dict(_spec())  # no sandbox attached
    res = await tool.execute({"text": "x"}, ToolContext())  # ctx.sandbox unset
    assert res.is_error and "no sandbox" in res.content.lower()


@pytest.mark.asyncio
async def test_uses_context_sandbox_when_tool_has_none() -> None:
    tool = SandboxExecTool.from_dict(_spec())  # no own sandbox
    sb = _FakeSandbox(lambda a, i: (0, b'{"ok": true}', b""))
    res = await tool.execute({}, ToolContext(sandbox=sb))
    assert not res.is_error and res.content == '{"ok": true}'


@pytest.mark.asyncio
async def test_nonzero_exit_is_error_with_stderr() -> None:
    tool = SandboxExecTool.from_dict(_spec(), sandbox=_FakeSandbox(lambda a, i: (2, b"", b"Traceback: boom")))
    res = await tool.execute({}, ToolContext())
    assert res.is_error
    assert "exit 2" in res.content and "boom" in res.content
    assert res.metadata["exit_code"] == 2


@pytest.mark.asyncio
async def test_json_error_payload_is_error() -> None:
    tool = SandboxExecTool.from_dict(_spec(), sandbox=_FakeSandbox(lambda a, i: (0, b'{"error": "bad input"}', b"")))
    res = await tool.execute({}, ToolContext())
    assert res.is_error and res.content == "bad input"


@pytest.mark.asyncio
async def test_timeout_is_error() -> None:
    sb = _FakeSandbox(raises=asyncio.TimeoutError())
    tool = SandboxExecTool.from_dict(_spec(timeout_s=3.0), sandbox=sb)
    res = await tool.execute({}, ToolContext())
    assert res.is_error and "timed out" in res.content and res.metadata["timeout"]


@pytest.mark.asyncio
async def test_plain_text_stdout_passed_through() -> None:
    tool = SandboxExecTool.from_dict(_spec(), sandbox=_FakeSandbox(lambda a, i: (0, b"not json, just text\n", b"")))
    res = await tool.execute({}, ToolContext())
    assert not res.is_error and res.content == "not json, just text"


def test_spec_roundtrip() -> None:
    spec = _spec(runtime="node", argv=["--flag"], network_egress=True, read_only=True)
    tool = SandboxExecTool.from_dict(spec)
    out = tool.to_dict()
    assert out["runtime"] == "node"
    assert out["argv"] == ["--flag"]
    assert out["network_egress"] is True and out["read_only"] is True
    # round-trips to an equivalent tool
    again = SandboxExecTool.from_dict(out)
    assert again.to_dict() == out
    assert tool.name == "echo_upper"


def test_capabilities_reflect_spec() -> None:
    ro = SandboxExecTool.from_dict(_spec(read_only=True, network_egress=True))
    caps = ro.capabilities({})
    assert caps.read_only and caps.concurrency_safe and caps.network_egress
    rw = SandboxExecTool.from_dict(_spec())
    assert not rw.capabilities({}).concurrency_safe       # fail-closed default


@pytest.mark.asyncio
async def test_exposed_from_public_packages() -> None:
    # built_in package export + XGeny sandbox primitives on the tools package.
    from xgen_agent_runtime.tools.built_in import SandboxExecTool as A
    from xgen_agent_runtime.tools import XgenySandbox, SandboxError, sb_run  # noqa: F401
    from xgen_agent_runtime.tools.built_in import BUILT_IN_TOOL_CLASSES

    assert A is SandboxExecTool
    # NOT a manifest-activated built-in.
    assert "SandboxExecTool" not in BUILT_IN_TOOL_CLASSES
    assert not any(v is SandboxExecTool for v in BUILT_IN_TOOL_CLASSES.values())
