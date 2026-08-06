"""Stage 6: API — default artifact."""

from xgen_agent_runtime.stages.s06_api.artifact.default.stage import APIStage
from xgen_agent_runtime.stages.s06_api.artifact.default.providers import (
    AnthropicProvider,
    MockProvider,
    RecordingProvider,
)
from xgen_agent_runtime.stages.s06_api.artifact.default.retry import (
    ExponentialBackoffRetry,
    NoRetry,
    RateLimitAwareRetry,
)
from xgen_agent_runtime.stages.s06_api.artifact.default.router import (
    AdaptiveModelRouter,
    PassthroughRouter,
)
from xgen_agent_runtime.stages.s06_api.artifact.default.tool_loop import (
    InternalAgenticLoop,
    PipelineToolLoop,
)

# Canonical alias
Stage = APIStage

__all__ = [
    "Stage",
    "APIStage",
    "AnthropicProvider",
    "MockProvider",
    "RecordingProvider",
    "ExponentialBackoffRetry",
    "NoRetry",
    "RateLimitAwareRetry",
    "AdaptiveModelRouter",
    "PassthroughRouter",
    "InternalAgenticLoop",
    "PipelineToolLoop",
]
