"""BedrockClient — model-ID normalization + kwargs contract.

The traps this file pins (see bedrock.py docstring):
* plain Anthropic IDs / aliases must become region-scoped inference
  profiles; full Bedrock IDs must pass through untouched,
* the Opus-4.7 family gates (sampling drops) must still fire even though
  the wire model is a decorated Bedrock ID,
* AWS credentials map to the SDK's ``aws_*`` constructor kwargs and
  omitted keys stay omitted (boto3 default-chain deploys).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from xgen_agent_runtime.core.config import ModelConfig
from xgen_agent_runtime.llm_client.bedrock import (
    BedrockClient,
    core_model_id,
    to_bedrock_model_id,
)


def _request(client: BedrockClient, model: str, **model_kwargs):
    return client._build_request(
        model_config=ModelConfig(model=model, **model_kwargs),
        messages=[{"role": "user", "content": "hi"}],
        system="",
        tools=None,
        tool_choice=None,
        stream=False,
    )


class TestModelIdNormalization:
    def test_core_id_strips_geo_vendor_and_version(self):
        assert (
            core_model_id("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
            == "claude-sonnet-4-5-20250929"
        )
        assert (
            core_model_id("apac.anthropic.claude-3-7-sonnet-20250219-v1:0")
            == "claude-3-7-sonnet-20250219"
        )
        assert core_model_id("anthropic.claude-opus-4-7-v1:0") == "claude-opus-4-7"

    def test_core_id_passes_plain_ids_through(self):
        assert core_model_id("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_plain_id_becomes_region_scoped_profile(self):
        assert (
            to_bedrock_model_id("claude-sonnet-4-5-20250929", region="us-east-1")
            == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        assert to_bedrock_model_id("x", region="eu-west-1").startswith("eu.")
        assert to_bedrock_model_id("x", region="ap-northeast-2").startswith("apac.")

    def test_alias_resolves_before_decoration(self):
        out = to_bedrock_model_id("sonnet", region="us-east-1")
        assert out.startswith("us.anthropic.claude-sonnet-")

    def test_full_bedrock_id_passes_through(self):
        full = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert to_bedrock_model_id(full, region="us-east-1") == full


class TestBuildKwargs:
    def test_wire_model_is_bedrock_id(self):
        client = BedrockClient(aws_region="us-west-2")
        request = _request(client, "claude-sonnet-4-5-20250929")
        kwargs = client._build_kwargs(request)
        assert kwargs["model"] == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_family_gate_fires_on_decorated_id(self):
        """Opus-4.7 rejects sampling params; the gate matches on the
        canonical core even when the request carries a Bedrock ID."""
        client = BedrockClient(aws_region="us-east-1")
        request = _request(
            client, "us.anthropic.claude-opus-4-7-v1:0", temperature=0.5
        )
        kwargs = client._build_kwargs(request)
        assert "temperature" not in kwargs
        assert kwargs["model"] == "us.anthropic.claude-opus-4-7-v1:0"

    def test_sampling_kept_for_non_gated_family(self):
        client = BedrockClient(aws_region="us-east-1")
        request = _request(client, "claude-sonnet-4-5-20250929", temperature=0.5)
        kwargs = client._build_kwargs(request)
        assert kwargs.get("temperature") == 0.5


class TestClientConstruction:
    def test_sdk_kwargs_only_carry_explicit_credentials(self, monkeypatch):
        captured = {}

        class _FakeBedrock:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import anthropic as _anthropic

        monkeypatch.setattr(_anthropic, "AsyncAnthropicBedrock", _FakeBedrock, raising=False)

        client = BedrockClient(aws_region="eu-central-1")
        client._get_client()
        assert captured == {"aws_region": "eu-central-1"}

        captured.clear()
        client2 = BedrockClient(
            aws_region="us-east-1",
            aws_access_key_id="AKIA",
            aws_secret_access_key="s3cr3t",
            aws_session_token="tok",
        )
        client2._get_client()
        assert captured["aws_access_key"] == "AKIA"
        assert captured["aws_secret_key"] == "s3cr3t"
        assert captured["aws_session_token"] == "tok"


def test_bare_bedrock_id_is_promoted_to_inference_profile():
    """AWS ListFoundationModels 가 주는 bare ID — 승격 없이는 Claude 4.x 급이
    on-demand 거부로 죽는다 (프로파일 필수)."""
    assert (
        to_bedrock_model_id("anthropic.claude-sonnet-4-5-20250929-v1:0", region="ap-northeast-2")
        == "apac.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    # 버전 접미사 변형도 보존
    assert (
        to_bedrock_model_id("anthropic.claude-3-7-sonnet-20250219-v2:1", region="us-east-1")
        == "us.anthropic.claude-3-7-sonnet-20250219-v2:1"
    )


def test_geo_prefixed_and_arn_ids_pass_through():
    assert (
        to_bedrock_model_id("eu.anthropic.claude-opus-4-5-20251101-v1:0", region="us-east-1")
        == "eu.anthropic.claude-opus-4-5-20251101-v1:0"
    )
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc"
    assert to_bedrock_model_id(arn, region="ap-northeast-2") == arn


def test_wire_model_uses_promoted_id_for_bare_input():
    client = BedrockClient(aws_region="ap-northeast-2")
    from xgen_agent_runtime.llm_client.types import APIRequest

    req = APIRequest(
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
    )
    kwargs = client._build_kwargs(req)
    assert kwargs["model"] == "apac.anthropic.claude-sonnet-4-5-20250929-v1:0"
