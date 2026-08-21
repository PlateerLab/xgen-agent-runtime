"""Google Vertex AI client (Gemini models on Vertex).

Extends :class:`GoogleClient` — the ``google-genai`` SDK speaks both the
Gemini API and Vertex through the same ``genai.Client``; only the
constructor differs. Request building, response parsing, streaming and
error classification are inherited (``_classify_api_core`` in the parent
was written for exactly the Vertex/grpc error surface).

Authentication (three supported channels, first match wins):

1. ``credentials_json`` — a service-account key JSON string. Parsed with
   ``google.oauth2.service_account`` (``google-auth``); the strictest and
   most portable channel for server deployments.
2. ``api_key`` — Vertex *express mode* API key.
3. Application Default Credentials — nothing passed; the SDK resolves
   ``GOOGLE_APPLICATION_CREDENTIALS`` / metadata-server identity.

``project`` is required for channels 1 and 3 (express-mode keys carry
their own project binding).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from xgen_agent_runtime.llm_client.google import GoogleClient

logger = logging.getLogger(__name__)

__all__ = ["VertexClient"]

#: Scope required by the Vertex AI generative endpoints.
_VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class VertexClient(GoogleClient):
    """Gemini models served through Google Vertex AI."""

    provider = "vertex"
    _sdk_module = "google.genai"

    def __init__(
        self,
        *,
        project: str = "",
        location: str = "us-central1",
        credentials_json: str = "",
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
    ) -> None:
        if not api_key and not project:
            # Express-mode keys are self-contained; every other channel
            # needs a project. Fail at construction (config error), not at
            # the first send (which would read as a model failure).
            raise ValueError("VertexClient requires project= (or an express-mode api_key=)")
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._project = project
        self._location = location or "us-central1"
        self._credentials_json = credentials_json

    def _load_sa_credentials(self) -> Any:
        try:
            from google.oauth2 import service_account
        except ImportError as e:  # pragma: no cover — dep missing
            raise ImportError(
                "Vertex service-account auth requires 'google-auth'. "
                "Install with: pip install google-auth"
            ) from e
        try:
            info = json.loads(self._credentials_json)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Vertex credentials_json is not valid JSON (expected a "
                "service-account key file's contents)"
            ) from e
        return service_account.Credentials.from_service_account_info(info, scopes=[_VERTEX_SCOPE])

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:  # pragma: no cover — dep missing
                raise ImportError(
                    "Vertex client requires the 'google-genai' package. "
                    "Install with: pip install google-genai"
                ) from e

            kwargs: Dict[str, Any] = {"vertexai": True}
            if self._credentials_json:
                kwargs["credentials"] = self._load_sa_credentials()
                kwargs["project"] = self._project
                kwargs["location"] = self._location
            elif self._api_key:
                # Express mode — the key binds its own project; passing
                # project/location alongside is rejected by the SDK.
                kwargs["api_key"] = self._api_key
            else:
                # Application Default Credentials.
                kwargs["project"] = self._project
                kwargs["location"] = self._location
            self._client = genai.Client(**kwargs)
        return self._client
