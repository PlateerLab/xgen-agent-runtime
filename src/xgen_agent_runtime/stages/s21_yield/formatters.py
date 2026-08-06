"""Result formatters — backward-compatible re-exports."""

from xgen_agent_runtime.stages.s21_yield.interface import ResultFormatter
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

__all__ = [
    "ResultFormatter",
    "DefaultFormatter",
    "StructuredFormatter",
    "StreamingFormatter",
    "MultiFormatFormatter",
    "build_markdown",
    "build_structured",
]
