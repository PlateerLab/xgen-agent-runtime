"""Core engine: Pipeline, Stage, Strategy, State, Config, Result, Errors, Mutation, Environment, Diff."""

from xgen_agent_runtime.core.errors import (
    ErrorCategory,
    GenyExecutorError,
    GuardRejectError,
    MutationError,
    MutationLocked,
    PipelineError,
    StageError,
)
from xgen_agent_runtime.core.stage import Stage, Strategy
from xgen_agent_runtime.core.state import CacheMetrics, PipelineState, TokenUsage
from xgen_agent_runtime.core.config import ModelConfig, PipelineConfig
from xgen_agent_runtime.core.result import PipelineResult
from xgen_agent_runtime.core.pipeline import Pipeline
from xgen_agent_runtime.core.schema import ConfigField, ConfigSchema
from xgen_agent_runtime.core.slot import StrategySlot
from xgen_agent_runtime.core.snapshot import PipelineSnapshot, StageSnapshot
from xgen_agent_runtime.core.mutation import (
    MutationKind,
    MutationRecord,
    MutationResult,
    PipelineMutator,
)
from xgen_agent_runtime.core.diff import DiffEntry, EnvironmentDiff
from xgen_agent_runtime.core.environment import (
    EnvironmentManifest,
    EnvironmentManager,
    EnvironmentMetadata,
    EnvironmentResolver,
    EnvironmentSanitizer,
    EnvironmentSummary,
    ToolsSnapshot,
)
from xgen_agent_runtime.core.presets import PipelinePresets, PresetInfo, PresetManager

__all__ = [
    # Engine
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineState",
    "Stage",
    "Strategy",
    "ModelConfig",
    "TokenUsage",
    "CacheMetrics",
    # Schema
    "ConfigField",
    "ConfigSchema",
    # Slot
    "StrategySlot",
    # Snapshot
    "PipelineSnapshot",
    "StageSnapshot",
    # Mutation
    "PipelineMutator",
    "MutationKind",
    "MutationRecord",
    "MutationResult",
    # Diff
    "DiffEntry",
    "EnvironmentDiff",
    # Environment
    "EnvironmentManifest",
    "EnvironmentManager",
    "EnvironmentMetadata",
    "EnvironmentResolver",
    "EnvironmentSanitizer",
    "EnvironmentSummary",
    "ToolsSnapshot",
    # Presets
    "PipelinePresets",
    "PresetInfo",
    "PresetManager",
    # Errors
    "ErrorCategory",
    "GenyExecutorError",
    "PipelineError",
    "StageError",
    "GuardRejectError",
    "MutationError",
    "MutationLocked",
]
