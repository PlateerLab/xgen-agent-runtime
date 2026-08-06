"""Stage 16: Yield — final result packaging and return."""

from xgen_agent_runtime.stages.s21_yield.stage import YieldStage
from xgen_agent_runtime.stages.s21_yield.formatters import (
    ResultFormatter,
    DefaultFormatter,
    StructuredFormatter,
    StreamingFormatter,
    MultiFormatFormatter,
    build_markdown,
    build_structured,
)

__all__ = [
    "YieldStage",
    "ResultFormatter",
    "DefaultFormatter",
    "StructuredFormatter",
    "StreamingFormatter",
    "MultiFormatFormatter",
    "build_markdown",
    "build_structured",
]
