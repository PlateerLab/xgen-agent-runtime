"""Default artifact for Stage 14: Emit."""

from xgen_agent_runtime.stages.s17_emit.artifact.default.stage import EmitStage
from xgen_agent_runtime.stages.s17_emit.artifact.default.emitters import (
    TextEmitter,
    CallbackEmitter,
    VTuberEmitter,
    TTSEmitter,
)

Stage = EmitStage

__all__ = [
    "Stage",
    "EmitStage",
    "TextEmitter",
    "CallbackEmitter",
    "VTuberEmitter",
    "TTSEmitter",
]
