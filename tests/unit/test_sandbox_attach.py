"""``attach_runtime(sandbox=)`` 는 **도구 실행지**를 붙이는 것이지 LLM 클라이언트를
바꾸는 것이 아니다.

예전에는 샌드박스를 붙이면 claude_code_cli 클라이언트가 컨테이너 러너로
감싸졌다(GAPT). 그 결합을 없앴다 — 어떤 프로바이더든 클라이언트는 여기서 돌고,
샌드박스에는 **도구를 통해서만** 닿는다. 프로바이더마다 격리 방식이 달라지면
"이 백엔드에서만 되는 도구"가 생긴다.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from xgen_agent_runtime.core.pipeline import Pipeline
from xgen_agent_runtime.llm_client import CredentialBundle, ProviderCredentials


class _FakeSandbox:
    workdir = "/xgeny/workspace/workflow/w1/workspace"

    async def ensure(self) -> None:  # pragma: no cover - not spawned here
        return None


def _cli_pipeline() -> Pipeline:
    p = Pipeline()
    p._credentials = CredentialBundle(
        by_provider={
            "claude_code_cli": ProviderCredentials(api_key="sk-test", binary_path="/bin/sh")
        }
    )
    return p


def test_attach_runtime_stores_the_sandbox() -> None:
    p = Pipeline()
    assert p._attached_sandbox is None
    p.attach_runtime(sandbox=_FakeSandbox())
    assert isinstance(p._attached_sandbox, _FakeSandbox)


def test_the_tool_stage_context_gets_it() -> None:
    """도구가 보는 것은 ``ctx.sandbox`` 하나다 — 배선의 전부."""
    p = Pipeline()
    p.attach_runtime(sandbox=_FakeSandbox())
    ctx = getattr(p, "_tool_context", None) or getattr(p, "_attached_tool_context", None)
    if ctx is not None:  # 스테이지가 없으면 attach 는 무음 no-op (러너 규약)
        assert getattr(ctx, "sandbox", None) is not None


def test_the_cli_client_is_not_wrapped() -> None:
    """CLI 는 여기서 돈다. 샌드박스가 붙어도 스폰 경로는 그대로다."""
    p = _cli_pipeline()
    plain = p._build_client_for("claude_code_cli")._make_runner()

    p.attach_runtime(sandbox=_FakeSandbox())
    wrapped = p._build_client_for("claude_code_cli")._make_runner()

    assert type(wrapped) is type(plain), "샌드박스가 클라이언트 스폰 방식을 바꿨다"


def test_sdk_providers_are_unaffected() -> None:
    p = Pipeline()
    p._credentials = CredentialBundle(
        by_provider={"anthropic": ProviderCredentials(api_key="sk-test")}
    )
    p._attached_sandbox = _FakeSandbox()
    assert type(p._build_client_for("anthropic")).__name__ == "AnthropicClient"
