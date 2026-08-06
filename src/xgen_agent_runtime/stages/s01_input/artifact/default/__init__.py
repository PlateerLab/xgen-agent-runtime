"""Default artifact for Stage 1: Input."""

from xgen_agent_runtime.stages.s01_input.artifact.default.stage import InputStage
from xgen_agent_runtime.stages.s01_input.artifact.default.validators import (
    DefaultValidator,
    PassthroughValidator,
    StrictValidator,
    SchemaValidator,
)
from xgen_agent_runtime.stages.s01_input.artifact.default.normalizers import (
    DefaultNormalizer,
    MultimodalNormalizer,
)

# Convention: every artifact exports ``Stage``
Stage = InputStage

__all__ = [
    "Stage",
    "InputStage",
    "DefaultValidator",
    "PassthroughValidator",
    "StrictValidator",
    "SchemaValidator",
    "DefaultNormalizer",
    "MultimodalNormalizer",
]
