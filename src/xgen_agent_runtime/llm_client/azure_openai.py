"""Azure OpenAI (Microsoft Foundry) client.

무엇이 사실인가 (Microsoft Learn, 2026-09 확인)
-----------------------------------------------
Azure 는 **표면이 둘**이다. 둘 다 살아 있고, 어느 쪽을 쓰는지는 배포마다 다르다.

``v1`` (2025-08 GA — 권장)
    ``https://<resource>.openai.azure.com/openai/v1/`` 를 base_url 로 주고
    평범한 OpenAI 클라이언트를 쓴다. ``api-version`` 이 **필요 없다**.
    ``model`` 자리에는 **배포 이름**을 넣는다.
    ``https://<resource>.services.ai.azure.com/openai/v1/`` 도 같은 것이다.

``deployments`` (옛 방식 — 여전히 동작)
    ``POST https://<resource>.openai.azure.com/openai/deployments/<배포>/
    chat/completions?api-version=YYYY-MM-DD[-preview]``
    배포 이름이 **경로**에 들어가고 ``api-version`` 이 필수다.

인증은 둘 다 같다: 헤더 ``api-key: <키>`` (또는 Entra ID 의
``Authorization: Bearer``). v1 은 OpenAI 클라이언트가 보내는 Bearer 도 받는다.

왜 v1 을 기본으로 두나
---------------------
``api-version`` 은 한 번 적어 두면 **그대로 늙는다.** 받아 온 설정에 적혀 있던
``2024-02-15-preview`` 는 2년 넘게 묵은 값이고, 그 버전에 없는 파라미터
(``max_completion_tokens``·``parallel_tool_calls``·구조적 출력 등)를 쓰면 조용히
무시되거나 400 으로 튄다. v1 은 그 축을 아예 없앤다.

그래서 규칙은 하나다: **``api_version`` 을 적으면 옛 경로, 안 적으면 v1.**
사용자가 옛 방식을 써야만 하는 배포(구형 리소스·APIM 프록시)면 값을 적으면
되고, 그렇지 않으면 아무것도 안 적는 쪽이 최신을 따라간다.

붙여 넣기를 받아 준다
--------------------
사람은 콘솔에서 본 것을 그대로 붙인다 — 전체 경로,
``?api-version=`` 이 붙은 URL, 끝의 슬래시. 그걸 거절하는 대신 읽어서 나눈다.
거절하면 "왜 안 되지" 를 사용자가 풀어야 하고, 그 시간은 우리 것이 아니다.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlsplit, urlunsplit

from xgen_agent_runtime.llm_client.openai import OpenAIClient

logger = logging.getLogger(__name__)

#: ``/openai/deployments/<이름>/...`` 에서 배포 이름을 건져낸다.
_DEPLOYMENT_RE = re.compile(r"/openai/deployments/([^/?#]+)", re.IGNORECASE)

#: 붙여 넣은 URL 에서 잘라 낼 꼬리들 (자원 루트만 남긴다).
_TAIL_RE = re.compile(
    r"/openai(?:/v1)?(?:/deployments/[^/?#]+)?(?:/(?:chat/completions|responses|completions|embeddings))?/?$",
    re.IGNORECASE,
)


class AzureEndpoint:
    """붙여 넣은 무엇이든 → (자원 루트, 배포, api_version).

    분리해 둔 이유: 이 해석이 틀리면 증상이 "인증 실패" 나 "404" 로 나타나고,
    그때 사람이 보는 것은 우리 코드가 아니라 Azure 의 오류다. 그래서 여기만
    따로 시험할 수 있어야 한다.
    """

    __slots__ = ("resource", "deployment", "api_version")

    def __init__(self, raw: str) -> None:
        self.resource = ""
        self.deployment = ""
        self.api_version = ""
        text = (raw or "").strip()
        if not text:
            return
        if "://" not in text:
            text = f"https://{text}"
        parts = urlsplit(text)

        found = _DEPLOYMENT_RE.search(parts.path or "")
        if found:
            self.deployment = found.group(1)

        version = parse_qs(parts.query or "").get("api-version") or []
        if version:
            self.api_version = str(version[0]).strip()

        path = _TAIL_RE.sub("", parts.path or "")
        self.resource = urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))

    @property
    def v1_base_url(self) -> str:
        """OpenAI 클라이언트에 그대로 줄 v1 base_url."""
        return f"{self.resource}/openai/v1/" if self.resource else ""


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI — v1 기본, ``api_version`` 이 있으면 옛 deployment 경로.

    ``model`` 은 Azure 에서 **배포 이름**이다(모델 id 가 아니다). 배포를 따로
    적지 않으면 요청의 모델명을 배포 이름으로 쓴다 — 대부분의 배포가 그렇게
    이름 붙어 있고, 다르면 ``deployment`` 로 못을 박으면 된다.
    """

    provider = "azure_openai"
    _sdk_module = "openai"

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        *,
        api_version: str = "",
        deployment: str = "",
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        parsed = AzureEndpoint(base_url or "")
        # 명시한 값이 붙여 넣은 URL 에서 읽은 값을 이긴다 — 사람이 적은 것이
        # 더 최근의 의도다.
        self._azure_resource = parsed.resource
        self._azure_deployment = (deployment or parsed.deployment or "").strip()
        self._azure_api_version = (api_version or parsed.api_version or "").strip()

        headers = dict(default_headers or {})
        # v1 은 Bearer 도 받지만, 키 인증의 정본은 이 헤더다. 둘 다 보내면
        # 어느 쪽 리소스에서도 붙는다 (프록시/APIM 이 하나만 통과시키는 경우 포함).
        if api_key and "api-key" not in {k.lower() for k in headers}:
            headers["api-key"] = api_key

        super().__init__(
            api_key=api_key,
            base_url=parsed.v1_base_url or None,
            default_headers=headers or None,
            event_sink=event_sink,
        )

    # ── 진단 ──────────────────────────────────────────────────────────

    @property
    def uses_legacy_deployment_path(self) -> bool:
        return bool(self._azure_api_version)

    def describe_target(self) -> Dict[str, str]:
        """어디로 쏘는지 — 오류를 볼 사람에게 보여 줄 값."""
        return {
            "resource": self._azure_resource,
            "deployment": self._azure_deployment,
            "api_version": self._azure_api_version,
            "mode": "deployments" if self.uses_legacy_deployment_path else "v1",
        }

    # ── 클라이언트 ────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._azure_resource:
            raise ValueError(
                "Azure OpenAI 는 리소스 엔드포인트가 필요합니다 — "
                "예: https://<리소스이름>.openai.azure.com"
            )
        try:
            from openai import AsyncAzureOpenAI, AsyncOpenAI
        except ImportError as e:  # pragma: no cover — SDK 없는 환경
            raise ImportError(
                "Azure OpenAI client requires the 'openai' package. "
                "Install with: pip install xgen-agent-runtime[openai]"
            ) from e

        if self.uses_legacy_deployment_path:
            # 옛 경로: 배포 이름이 URL 에 들어가고 api-version 이 필수다.
            # SDK 의 Azure 클라이언트가 그 조립을 안다 — 우리가 손으로 만들지
            # 않는다(만들면 responses/embeddings 마다 다시 만들어야 한다).
            self._client = AsyncAzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_resource,
                api_version=self._azure_api_version,
                default_headers=self._default_headers or None,
            )
        else:
            kwargs: Dict[str, Any] = {
                "api_key": self._api_key or "EMPTY",
                "base_url": self._base_url,
            }
            if self._default_headers:
                kwargs["default_headers"] = self._default_headers
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    # ── 모델명 → 배포 이름 ────────────────────────────────────────────
    #
    # 가로채는 곳은 **한 곳**이다. 단발 호출과 스트리밍이 둘 다 여기로 모이므로
    # (``_send`` / ``create_message_stream`` → ``_build_kwargs``), 두 경로를
    # 따로 손보면 반드시 한쪽이 빠진다.

    def _build_kwargs(self, request: Any) -> Dict[str, Any]:
        kwargs = super()._build_kwargs(request)
        if self._azure_deployment:
            kwargs["model"] = self._azure_deployment
        return kwargs
