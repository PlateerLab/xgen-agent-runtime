"""xgen-agent-runtime: Harness-engineered agent pipeline library.

Usage:
    from xgen_agent_runtime import Pipeline, PipelineConfig
    from xgen_agent_runtime.stages.s01_input import InputStage
    from xgen_agent_runtime.stages.s06_api import APIStage, MockProvider
    from xgen_agent_runtime.stages.s09_parse import ParseStage
    from xgen_agent_runtime.stages.s21_yield import YieldStage

    pipeline = Pipeline(PipelineConfig(name="my-agent"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(api_key="..."))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Hello!")
"""

from xgen_agent_runtime.core.pipeline import Pipeline
from xgen_agent_runtime.core.config import PipelineConfig, ModelConfig, ModelOverrides
from xgen_agent_runtime.core.state import PipelineState, TokenUsage, CacheMetrics
from xgen_agent_runtime.core.result import PipelineResult
from xgen_agent_runtime.core.run_status import RunStatus, TerminationReason
from xgen_agent_runtime.core.continuation import CONTINUE_RUN, ContinuationInput
from xgen_agent_runtime.core.stage import Stage, Strategy, StageDescription, StrategyInfo
from xgen_agent_runtime.core.errors import (
    GenyExecutorError,
    PipelineError,
    StageError,
    GuardRejectError,
    APIError,
    ToolExecutionError,
    ErrorCategory,
    ExecutorErrorCode,
    MutationError,
    MutationLocked,
)
from xgen_agent_runtime.core.schema import ConfigField, ConfigSchema
from xgen_agent_runtime.core.slot import SlotChain, StrategySlot
from xgen_agent_runtime.core.snapshot import PipelineSnapshot, StageSnapshot
from xgen_agent_runtime.core.mutation import (
    PipelineMutator,
    MutationKind,
    MutationRecord,
    MutationResult,
)
from xgen_agent_runtime.core.builder import PipelineBuilder
from xgen_agent_runtime.core.presets import (
    PipelinePresets,
    PresetInfo,
    PresetManager,
    PresetRegistry,
    register_preset,
)
from xgen_agent_runtime.core.diff import DiffEntry, EnvironmentDiff
from xgen_agent_runtime.core.environment import (
    EnvironmentManifest,
    EnvironmentManager,
    EnvironmentMetadata,
    EnvironmentResolver,
    EnvironmentSanitizer,
    EnvironmentSummary,
    HostSelections,
    ManifestIssue,
    StageManifestEntry,
    ToolsSnapshot,
    validate_manifest,
)
from xgen_agent_runtime.core.manifest_factory import (
    PresetDescriptor,
    build_manifest,
    build_manifest_for,
    get_preset_descriptor,
    known_manifest_presets,
    preset_catalog,
)
from xgen_agent_runtime.core.artifact import (
    ArtifactInfo,
    create_stage,
    describe_artifact,
    get_artifact_map,
    list_artifacts,
    list_artifacts_with_meta,
)
from xgen_agent_runtime.core.introspection import (
    ChainIntrospection,
    IntrospectionUnsupported,
    SlotIntrospection,
    StageIntrospection,
    introspect_all,
    introspect_stage,
)
from xgen_agent_runtime.events import (
    EVENT_CATALOG_VERSION,
    EventBus,
    EventTypes,
    PipelineEvent,
    known_event_types,
)
from xgen_agent_runtime.llm_client import (
    APIRequest,
    APIResponse,
    BaseClient,
    ClaudeCodeCLIClient,
    ClientCapabilities,
    ClientRegistry,
    ConfigError,
    ContentBlock,
    CredentialBundle,
    ProviderCredentials,
)
from xgen_agent_runtime.memory import (
    GenyPresets,
    MemoryAwareRetriever,
    MemoryProviderFactory,
    ProviderDrivenStrategy,
)
from xgen_agent_runtime.memory.factory import provider_from_manifest_memory
from xgen_agent_runtime.stages.s12_agent.subagent_type import (
    ManifestSubagentPipelineFactory,
    SubAgentBuildContext,
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
    compile_subagent_descriptors,
    resolve_subagent_provider,
)

# Single source of truth: read the installed distribution version so
# ``__version__`` can never drift from ``pyproject.toml`` again.
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("xgen-agent-runtime")
except Exception:  # noqa: BLE001 — not installed (e.g. source checkout)
    __version__ = "0.0.0+local"

__all__ = [
    # Core
    "Pipeline",
    "PipelineConfig",
    "PipelineState",
    "RunStatus",
    "TerminationReason",
    "CONTINUE_RUN",
    "ContinuationInput",
    "PipelineResult",
    "ModelConfig",
    "ModelOverrides",
    "TokenUsage",
    "CacheMetrics",
    # Abstractions
    "Stage",
    "Strategy",
    "StageDescription",
    "StrategyInfo",
    # Builder & Presets
    "PipelineBuilder",
    "PipelinePresets",
    "PresetInfo",
    "PresetManager",
    "PresetRegistry",
    "register_preset",
    # Environment & Diff
    "EnvironmentManifest",
    "EnvironmentManager",
    "EnvironmentMetadata",
    "EnvironmentResolver",
    "EnvironmentSanitizer",
    "EnvironmentSummary",
    "HostSelections",
    "ManifestIssue",
    "StageManifestEntry",
    "ToolsSnapshot",
    "validate_manifest",
    "build_manifest",
    "build_manifest_for",
    "known_manifest_presets",
    "preset_catalog",
    "get_preset_descriptor",
    "PresetDescriptor",
    "DiffEntry",
    "EnvironmentDiff",
    # Artifact system
    "ArtifactInfo",
    "create_stage",
    "describe_artifact",
    "get_artifact_map",
    "list_artifacts",
    "list_artifacts_with_meta",
    # Introspection
    "ChainIntrospection",
    "IntrospectionUnsupported",
    "SlotIntrospection",
    "StageIntrospection",
    "introspect_all",
    "introspect_stage",
    # Events
    "EVENT_CATALOG_VERSION",
    "EventBus",
    "EventTypes",
    "PipelineEvent",
    "known_event_types",
    # LLM clients (unified)
    "APIRequest",
    "APIResponse",
    "BaseClient",
    "ClaudeCodeCLIClient",
    "ClientCapabilities",
    "ClientRegistry",
    "ConfigError",
    "ContentBlock",
    "CredentialBundle",
    "ProviderCredentials",
    # Errors
    "GenyExecutorError",
    "PipelineError",
    "StageError",
    "GuardRejectError",
    "APIError",
    "ToolExecutionError",
    "ErrorCategory",
    "ExecutorErrorCode",
    "MutationError",
    "MutationLocked",
    # Schema & Mutation
    "ConfigField",
    "ConfigSchema",
    "StrategySlot",
    "SlotChain",
    "PipelineSnapshot",
    "StageSnapshot",
    "PipelineMutator",
    "MutationKind",
    "MutationRecord",
    "MutationResult",
    # Memory plumbing (provider-driven)
    "MemoryAwareRetriever",
    "MemoryProviderFactory",
    "ProviderDrivenStrategy",
    "GenyPresets",
    "provider_from_manifest_memory",
    # Sub-agent types (manifest-expressible since 2.2.0 Wave 3)
    "ManifestSubagentPipelineFactory",
    "SubAgentBuildContext",
    "SubagentTypeDescriptor",
    "SubagentTypeOrchestrator",
    "SubagentTypeRegistry",
    "compile_subagent_descriptors",
    "resolve_subagent_provider",
]
