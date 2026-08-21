"""AWS Bedrock client (Anthropic models over the Bedrock Messages API).

Extends :class:`AnthropicClient` — Bedrock speaks the same Messages API
through ``anthropic.AsyncAnthropicBedrock``, so request building, response
parsing, streaming, thinking translation and error classification are all
inherited. What this subclass owns:

* **Credentials** — AWS SigV4 (access key / secret / session token /
  profile / region) instead of an Anthropic API key. When no explicit
  keys are given the boto3 default chain applies (env, shared config,
  IRSA / instance role), so key-less deployments work.
* **Model IDs** — Bedrock rejects plain Anthropic IDs. Callers may pass
  either a full Bedrock ID (``anthropic.claude-…-v1:0`` or a cross-region
  inference profile ``us.anthropic.claude-…-v1:0``) which is used as-is,
  or a plain Anthropic ID / alias (``sonnet``) which is converted to the
  region-appropriate inference-profile ID.
* **Family gates** — the sampling/thinking prefix tables in
  ``anthropic.py`` match on *canonical* Anthropic IDs. We therefore build
  kwargs against the canonical core ID (so Opus-4.7-class gating still
  fires) and swap in the Bedrock ID at the end.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, Dict, Optional

from xgen_agent_runtime.llm_client.anthropic import (
    AnthropicClient,
    _resolve_anthropic_model,
)
from xgen_agent_runtime.llm_client.types import APIRequest

logger = logging.getLogger(__name__)

__all__ = ["BedrockClient", "core_model_id", "to_bedrock_model_id"]


#: Cross-region inference-profile geo prefixes Bedrock uses today.
_GEO_PREFIXES = ("us.", "eu.", "apac.", "global.", "jp.", "au.")

#: Trailing Bedrock version suffix, e.g. ``-v1:0`` / ``-v2:1``.
_VERSION_SUFFIX_RE = re.compile(r"-v\d+:\d+$")


def _region_geo(region: str) -> str:
    """AWS region → inference-profile geo prefix (best-effort)."""
    r = (region or "").lower()
    if r.startswith("us-") or r.startswith("ca-"):
        return "us"
    if r.startswith("eu-"):
        return "eu"
    if r.startswith("ap-"):
        return "apac"
    return "us"


def looks_like_bedrock_id(model: str) -> bool:
    """True iff *model* is already a Bedrock model / inference-profile ID."""
    if "anthropic." in model:
        return True
    return any(model.startswith(p) for p in _GEO_PREFIXES)


def _is_final_bedrock_id(model: str) -> bool:
    """True iff *model* must be sent to the wire untouched.

    Geo-prefixed inference-profile IDs (``us.anthropic.…``) and ARNs
    (application inference profiles / provisioned throughput) are final.
    Bare ``anthropic.claude-…-v1:0`` is NOT — Claude 4.x-class models reject
    on-demand invocation ("on-demand throughput isn't supported"), so bare
    IDs are promoted to the region's system inference profile. AWS 콘솔의
    ListFoundationModels 가 돌려주는 형태가 정확히 이 bare ID 라서, 관리자
    카탈로그 등록 경로가 그대로 이 함정을 밟는다.
    """
    if model.startswith("arn:"):
        return True
    return any(model.startswith(p) for p in _GEO_PREFIXES)


def core_model_id(model: str) -> str:
    """Strip Bedrock decorations down to the canonical Anthropic ID.

    ``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` →
    ``claude-sonnet-4-5-20250929``. Non-Bedrock IDs pass through
    unchanged. Used so the family prefix gates (sampling drops, adaptive
    thinking) and pricing lookups keep matching on Bedrock.
    """
    core = model
    for p in _GEO_PREFIXES:
        if core.startswith(p):
            core = core[len(p) :]
            break
    if core.startswith("anthropic."):
        core = core[len("anthropic.") :]
    return _VERSION_SUFFIX_RE.sub("", core)


def to_bedrock_model_id(model: str, *, region: str) -> str:
    """Canonical Anthropic ID (or alias) → Bedrock inference-profile ID.

    Full Bedrock IDs pass through untouched — the caller already chose
    (e.g. an admin catalog entry copied from the AWS console).
    """
    if _is_final_bedrock_id(model):
        return model
    if model.startswith("anthropic."):
        # bare Bedrock ID → 버전 접미사를 보존한 채 geo 프리픽스만 승격.
        return f"{_region_geo(region)}.{model}"
    canonical = _resolve_anthropic_model(model)
    return f"{_region_geo(region)}.anthropic.{canonical}-v1:0"


class BedrockClient(AnthropicClient):
    """Anthropic models served through AWS Bedrock."""

    provider = "bedrock"
    _sdk_module = "anthropic"

    def __init__(
        self,
        *,
        aws_region: str = "us-east-1",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
        aws_session_token: str = "",
        aws_profile: str = "",
        api_key: str = "",  # accepted for constructor symmetry; unused
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._aws_region = aws_region or "us-east-1"
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._aws_profile = aws_profile

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropicBedrock
            except ImportError as e:  # pragma: no cover — extra missing
                raise ImportError(
                    "Bedrock client requires the anthropic SDK's bedrock extra. "
                    "Install with: pip install 'anthropic[bedrock]'"
                ) from e

            kwargs: Dict[str, Any] = {"aws_region": self._aws_region}
            # Only pass explicit credentials — omitted keys let the boto3
            # default chain (env / shared config / IRSA / instance role)
            # resolve, which is the correct behaviour for key-less deploys.
            if self._aws_access_key_id:
                kwargs["aws_access_key"] = self._aws_access_key_id
            if self._aws_secret_access_key:
                kwargs["aws_secret_key"] = self._aws_secret_access_key
            if self._aws_session_token:
                kwargs["aws_session_token"] = self._aws_session_token
            if self._aws_profile:
                kwargs["aws_profile"] = self._aws_profile
            if self._base_url:
                kwargs["base_url"] = self._base_url  # VPC endpoint override
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = AsyncAnthropicBedrock(**kwargs)
        return self._client

    def _build_kwargs(self, request: APIRequest) -> Dict[str, Any]:
        # Build against the canonical core ID so the family gates
        # (Opus-4.7 sampling drops, adaptive-thinking migration) match,
        # then swap in the Bedrock ID for the wire.
        core = core_model_id(_resolve_anthropic_model(request.model))
        kwargs = super()._build_kwargs(replace(request, model=core))
        kwargs["model"] = to_bedrock_model_id(request.model, region=self._aws_region)
        return kwargs

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        """Bedrock has no cheap ``/v1/models`` — just build the client.

        Constructing ``AsyncAnthropicBedrock`` resolves the credential
        chain (profile / role lookups can touch disk); doing it here keeps
        that off the first token's critical path. No network round-trip:
        an unauthorized probe against Bedrock is neither cheap nor free.
        """
        try:
            self._get_client()
            return True
        except Exception:  # noqa: BLE001 — warmup is best-effort by contract
            logger.debug("bedrock: warmup failed", exc_info=True)
            return False
