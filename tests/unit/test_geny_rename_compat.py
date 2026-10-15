"""XGeny → Geny 개명(4.65.0) — 옛 이름으로 부르던 호스트가 새 런타임에서도 그대로 돈다.

호스트(xgen-workflow)는 런타임을 핀으로 받는다. 런타임이 먼저 올라가고 호스트가 나중에 따라오는
사이에도, 또 옛 배포 설정(XGENY_* env)이 남아 있어도 깨지지 않아야 한다.
"""

import importlib


def test_the_old_module_path_is_the_same_module():
    old = importlib.import_module("xgen_agent_runtime.tools._xgeny_sandbox")
    new = importlib.import_module("xgen_agent_runtime.tools._geny_sandbox")
    assert old is new, "사본이면 한쪽을 고쳐도 다른 쪽이 안 바뀐다"
    from xgen_agent_runtime.tools._xgeny_sandbox import ExecResult, SandboxError, sandbox_path  # noqa: F401


def test_the_old_protocol_name_still_imports():
    from xgen_agent_runtime.tools import GenySandbox, XgenySandbox

    assert XgenySandbox is GenySandbox


def test_the_image_budget_reads_the_old_env_name(monkeypatch):
    from xgen_agent_runtime.host import turn_executor as te

    monkeypatch.delenv("GENY_IMAGE_MAX_BYTES", raising=False)
    monkeypatch.setenv("XGENY_IMAGE_MAX_BYTES", "123")
    assert te._env_bytes("GENY_IMAGE_MAX_BYTES", 5) == 123
    monkeypatch.setenv("GENY_IMAGE_MAX_BYTES", "7")
    assert te._env_bytes("GENY_IMAGE_MAX_BYTES", 5) == 7, "새 이름이 이긴다"
    monkeypatch.delenv("GENY_IMAGE_MAX_BYTES")
    monkeypatch.delenv("XGENY_IMAGE_MAX_BYTES")
    assert te._env_bytes("GENY_IMAGE_MAX_BYTES", 5) == 5
    monkeypatch.setenv("GENY_IMAGE_MAX_BYTES", "")
    assert te._env_bytes("GENY_IMAGE_MAX_BYTES", 5) == 0, "빈 값은 예전처럼 0"
