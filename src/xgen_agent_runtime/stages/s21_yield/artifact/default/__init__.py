"""Default artifact for Stage 16: Yield."""

from xgen_agent_runtime.stages.s21_yield.artifact.default.stage import YieldStage
from xgen_agent_runtime.stages.s21_yield.artifact.default.formatters import (
    DefaultFormatter,
    StructuredFormatter,
    StreamingFormatter,
)
from xgen_agent_runtime.stages.s21_yield.artifact.default.multi_format import (
    MultiFormatFormatter,
    build_markdown,
    build_structured,
)

Stage = YieldStage

__all__ = [
    "Stage",
    "YieldStage",
    "DefaultFormatter",
    "StructuredFormatter",
    "StreamingFormatter",
    "MultiFormatFormatter",
    "build_markdown",
    "build_structured",
]
