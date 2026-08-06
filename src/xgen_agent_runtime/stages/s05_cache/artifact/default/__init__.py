"""Stage 5: Cache — default artifact."""

from xgen_agent_runtime.stages.s05_cache.artifact.default.stage import CacheStage
from xgen_agent_runtime.stages.s05_cache.artifact.default.strategies import (
    NoCacheStrategy,
    SystemCacheStrategy,
    AggressiveCacheStrategy,
)

Stage = CacheStage

__all__ = [
    "Stage",
    "CacheStage",
    "NoCacheStrategy",
    "SystemCacheStrategy",
    "AggressiveCacheStrategy",
]
