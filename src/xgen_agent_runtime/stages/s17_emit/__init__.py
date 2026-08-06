"""Stage 14: Emit — result output to external consumers."""

from xgen_agent_runtime.stages.s17_emit.stage import EmitStage
from xgen_agent_runtime.stages.s17_emit.emitters import (
    Emitter,
    TextEmitter,
    CallbackEmitter,
    VTuberEmitter,
    TTSEmitter,
    EmitterChain,
    EmitResult,
)
from xgen_agent_runtime.stages.s17_emit.types import OrderedEmitterChain

__all__ = [
    "EmitStage",
    "Emitter",
    "TextEmitter",
    "CallbackEmitter",
    "VTuberEmitter",
    "TTSEmitter",
    "EmitterChain",
    "OrderedEmitterChain",
    "EmitResult",
]
