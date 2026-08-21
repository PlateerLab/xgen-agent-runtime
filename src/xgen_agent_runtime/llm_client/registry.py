"""Provider-name → client-class lookup with lazy imports.

Each adapter's vendor SDK is optional — ``AnthropicClient`` is the only
client whose SDK is a hard dependency of xgen-agent-runtime. Others are
lazily imported so a user installing only the anthropic extras is not
forced to pip-install ``google-genai`` or ``openai``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from xgen_agent_runtime.llm_client.base import BaseClient


class ClientRegistry:
    """Provider-name → client-class lookup."""

    _factories: Dict[str, Callable[[], Type[BaseClient]]] = {}

    @classmethod
    def register(cls, provider: str, factory: Callable[[], Type[BaseClient]]) -> None:
        cls._factories[provider] = factory

    @classmethod
    def get(cls, provider: str) -> Type[BaseClient]:
        if provider not in cls._factories:
            raise ValueError(
                f"Unknown LLM client provider: {provider!r}. Registered: {sorted(cls._factories)}"
            )
        return cls._factories[provider]()

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._factories)


def _anthropic_factory() -> Type[BaseClient]:
    from xgen_agent_runtime.llm_client.anthropic import AnthropicClient

    return AnthropicClient


def _openai_factory() -> Type[BaseClient]:
    try:
        from xgen_agent_runtime.llm_client.openai import OpenAIClient
    except ImportError as e:
        raise ImportError(
            "OpenAI client requires the 'openai' package. "
            "Install with: pip install xgen-agent-runtime[openai]"
        ) from e
    return OpenAIClient


def _google_factory() -> Type[BaseClient]:
    try:
        from xgen_agent_runtime.llm_client.google import GoogleClient
    except ImportError as e:
        raise ImportError(
            "Google client requires 'google-genai'. Install with: pip install xgen-agent-runtime[google]"
        ) from e
    return GoogleClient


def _vllm_factory() -> Type[BaseClient]:
    from xgen_agent_runtime.llm_client.vllm import VLLMClient

    return VLLMClient


def _claude_code_cli_factory() -> Type[BaseClient]:
    from xgen_agent_runtime.llm_client.claude_code import ClaudeCodeCLIClient

    return ClaudeCodeCLIClient


def _bedrock_factory() -> Type[BaseClient]:
    from xgen_agent_runtime.llm_client.bedrock import BedrockClient

    return BedrockClient


def _vertex_factory() -> Type[BaseClient]:
    try:
        from xgen_agent_runtime.llm_client.vertex import VertexClient
    except ImportError as e:
        raise ImportError(
            "Vertex client requires 'google-genai' (and 'google-auth' for "
            "service-account keys). Install with: pip install google-genai google-auth"
        ) from e
    return VertexClient


def _codex_cli_factory() -> Type[BaseClient]:
    from xgen_agent_runtime.llm_client.codex import CodexCLIClient

    return CodexCLIClient


def _profile_factory(provider_name: str) -> Callable[[], Type[BaseClient]]:
    """Factory for a profile-driven OpenAI-compatible local client.

    The client class is built from a :class:`ProviderProfile` and pulls the
    OpenAI client path — imported lazily here so merely registering
    ``ollama`` / ``lmstudio`` / ``custom`` costs no SDK import (same lazy
    contract as ``_openai_factory`` / ``_vllm_factory``).
    """

    def factory() -> Type[BaseClient]:
        from xgen_agent_runtime.llm_client.profiles import get_profiled_client_class

        return get_profiled_client_class(provider_name)

    return factory


ClientRegistry.register("anthropic", _anthropic_factory)
ClientRegistry.register("openai", _openai_factory)
ClientRegistry.register("google", _google_factory)
ClientRegistry.register("vllm", _vllm_factory)
ClientRegistry.register("claude_code_cli", _claude_code_cli_factory)
ClientRegistry.register("bedrock", _bedrock_factory)
ClientRegistry.register("vertex", _vertex_factory)
ClientRegistry.register("codex_cli", _codex_cli_factory)


# Branded local (OpenAI-compatible) providers, generated from profiles.
# Registered under their primary name and every alias (e.g. custom→local)
# so the manifest can pin any of them at stages[6].config["provider"].
def _register_profile_providers() -> None:
    from xgen_agent_runtime.llm_client.profiles import (
        BUILTIN_PROFILES,
    )

    for profile in BUILTIN_PROFILES:
        for name in profile.all_names():
            ClientRegistry.register(name, _profile_factory(name))


_register_profile_providers()
