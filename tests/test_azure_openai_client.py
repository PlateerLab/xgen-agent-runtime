"""Azure OpenAI — **붙여 넣은 무엇이든** 올바른 곳으로 간다.

Azure 는 표면이 둘이고 둘 다 살아 있다 (Microsoft Learn, 2026-09 확인):

  v1 (2025-08 GA)   base_url = https://<res>.openai.azure.com/openai/v1/
                    api-version 없음. model 자리에 **배포 이름**.
  deployments (구)  POST .../openai/deployments/<배포>/chat/completions
                    ?api-version=YYYY-MM-DD[-preview]

이 해석이 틀리면 증상은 우리 코드가 아니라 Azure 의 401/404 로 나타난다.
그때 사람이 볼 수 있는 것은 남의 오류 메시지뿐이므로, 해석은 여기서 못 박는다.
"""
from __future__ import annotations

import pytest

from xgen_agent_runtime.llm_client.azure_openai import AzureEndpoint, AzureOpenAIClient
from xgen_agent_runtime.llm_client.registry import ClientRegistry


class TestItReadsWhatPeoplePaste:
    """사람은 콘솔에서 본 것을 그대로 붙인다 — 거절하지 말고 읽어라."""

    def test_a_bare_resource_endpoint(self):
        e = AzureEndpoint("https://imcapital-agent-resource.openai.azure.com")
        assert e.resource == "https://imcapital-agent-resource.openai.azure.com"
        assert e.v1_base_url.endswith("/openai/v1/")
        assert e.deployment == "" and e.api_version == ""

    def test_a_trailing_slash_is_not_a_different_endpoint(self):
        assert AzureEndpoint("https://r.openai.azure.com/").resource == "https://r.openai.azure.com"

    def test_the_full_deployment_path_is_understood(self):
        """받아 온 설정이 딱 이 모양이었다."""
        e = AzureEndpoint(
            "https://imcapital-agent-resource.openai.azure.com"
            "/openai/deployments/gpt-5.4-mini/chat/completions?api-version=2024-02-15-preview"
        )
        assert e.resource == "https://imcapital-agent-resource.openai.azure.com"
        assert e.deployment == "gpt-5.4-mini"
        assert e.api_version == "2024-02-15-preview"

    def test_a_v1_base_url_is_not_doubled(self):
        e = AzureEndpoint("https://r.openai.azure.com/openai/v1/")
        assert e.resource == "https://r.openai.azure.com"
        assert e.v1_base_url == "https://r.openai.azure.com/openai/v1/"

    def test_the_foundry_hostname_is_the_same_thing(self):
        e = AzureEndpoint("https://r.services.ai.azure.com/openai/v1/")
        assert e.v1_base_url == "https://r.services.ai.azure.com/openai/v1/"

    def test_a_missing_scheme_is_forgiven(self):
        assert AzureEndpoint("r.openai.azure.com").resource == "https://r.openai.azure.com"

    def test_nothing_stays_nothing(self):
        e = AzureEndpoint("")
        assert e.resource == "" and e.v1_base_url == ""


class TestWhichSurfaceItUses:
    """규칙은 하나다: api_version 을 적으면 옛 경로, 안 적으면 v1."""

    def test_no_api_version_means_v1(self):
        c = AzureOpenAIClient(api_key="k", base_url="https://r.openai.azure.com")
        assert c.uses_legacy_deployment_path is False
        assert c.describe_target()["mode"] == "v1"

    def test_an_api_version_pins_the_old_path(self):
        c = AzureOpenAIClient(
            api_key="k", base_url="https://r.openai.azure.com", api_version="2024-02-15-preview",
        )
        assert c.uses_legacy_deployment_path is True
        assert c.describe_target()["mode"] == "deployments"

    def test_a_pasted_url_carries_its_own_version(self):
        """URL 에 ?api-version= 이 있으면 그 사람은 옛 경로를 쓰고 있는 것이다."""
        c = AzureOpenAIClient(
            api_key="k",
            base_url="https://r.openai.azure.com/openai/deployments/d/chat/completions"
                     "?api-version=2025-04-01-preview",
        )
        assert c.uses_legacy_deployment_path is True
        assert c.describe_target()["api_version"] == "2025-04-01-preview"

    def test_an_explicit_value_beats_the_pasted_one(self):
        """사람이 칸에 적은 것이 더 최근의 의도다."""
        c = AzureOpenAIClient(
            api_key="k",
            base_url="https://r.openai.azure.com/openai/deployments/old/chat/completions"
                     "?api-version=2024-02-15-preview",
            deployment="new",
            api_version="2025-04-01-preview",
        )
        t = c.describe_target()
        assert t["deployment"] == "new" and t["api_version"] == "2025-04-01-preview"


class TestAuthAndTargeting:
    def test_the_key_rides_the_header_azure_actually_reads(self):
        """키 인증의 정본은 ``api-key`` 헤더다 — Bearer 만 보내면 프록시에서 막힌다."""
        c = AzureOpenAIClient(api_key="secret", base_url="https://r.openai.azure.com")
        assert c._default_headers.get("api-key") == "secret"

    def test_a_caller_supplied_header_is_not_overwritten(self):
        c = AzureOpenAIClient(
            api_key="secret", base_url="https://r.openai.azure.com",
            default_headers={"api-key": "이미-있음"},
        )
        assert c._default_headers["api-key"] == "이미-있음"

    def test_the_model_becomes_the_deployment_name(self):
        """Azure 에서 model 자리는 **배포 이름**이다 — 모델 id 가 아니다."""
        from xgen_agent_runtime.llm_client.types import APIRequest

        c = AzureOpenAIClient(
            api_key="k", base_url="https://r.openai.azure.com", deployment="my-deploy",
        )
        kwargs = c._build_kwargs(APIRequest(model="gpt-4.1", messages=[{"role": "user", "content": "hi"}]))
        assert kwargs["model"] == "my-deploy"

    def test_without_a_deployment_the_model_name_is_used_as_is(self):
        """배포 이름을 모델명과 같게 지어 두는 것이 가장 흔하다."""
        from xgen_agent_runtime.llm_client.types import APIRequest

        c = AzureOpenAIClient(api_key="k", base_url="https://r.openai.azure.com")
        kwargs = c._build_kwargs(APIRequest(model="gpt-5.4-mini", messages=[{"role": "user", "content": "hi"}]))
        assert kwargs["model"] == "gpt-5.4-mini"

    def test_no_endpoint_says_so_plainly(self):
        c = AzureOpenAIClient(api_key="k", base_url=None)
        with pytest.raises(ValueError, match="리소스 엔드포인트"):
            c._get_client()


class TestItIsReachableByName:
    def test_every_name_people_use_lands_on_the_same_client(self):
        """설정에 무엇을 적든 같은 곳으로 간다 — 이름이 갈리면 조용히 안 붙는다."""
        for name in ("azure_openai", "azure", "azure_foundry"):
            assert ClientRegistry.get(name) is AzureOpenAIClient

    def test_the_host_can_build_it_with_credentials(self):
        from xgen_agent_runtime.host.runner import build_client

        c = build_client(
            "azure_openai", "k", "https://r.openai.azure.com",
            credentials={"api_version": "2025-04-01-preview", "deployment": "d"},
        )
        assert isinstance(c, AzureOpenAIClient)
        assert c.describe_target() == {
            "resource": "https://r.openai.azure.com",
            "deployment": "d",
            "api_version": "2025-04-01-preview",
            "mode": "deployments",
        }
