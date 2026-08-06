"""Stage 5: Cache — prompt caching strategy management."""

from xgen_agent_runtime.stages.s05_cache.interface import CacheStrategy, EPHEMERAL_CACHE
from xgen_agent_runtime.stages.s05_cache.artifact.default import (
    Stage,
    CacheStage,
    NoCacheStrategy,
    SystemCacheStrategy,
    AggressiveCacheStrategy,
)

__all__ = [
    "Stage",
    "CacheStage",
    "CacheStrategy",
    "EPHEMERAL_CACHE",
    "NoCacheStrategy",
    "SystemCacheStrategy",
    "AggressiveCacheStrategy",
]
