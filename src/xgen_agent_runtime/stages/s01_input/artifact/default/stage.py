"""Default implementation of Stage 1: Input."""

from __future__ import annotations

from typing import Any, Dict, Optional

from xgen_agent_runtime.core.slot import StrategySlot
from xgen_agent_runtime.core.stage import Stage
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.core.errors import StageError
from xgen_agent_runtime.core.message_repair import repair_dangling_tool_calls
from xgen_agent_runtime.stages.s01_input.types import NormalizedInput
from xgen_agent_runtime.stages.s01_input.interface import InputValidator, InputNormalizer
from xgen_agent_runtime.stages.s01_input.artifact.default.validators import (
    DefaultValidator,
    PassthroughValidator,
    SchemaValidator,
    StrictValidator,
)
from xgen_agent_runtime.stages.s01_input.artifact.default.normalizers import (
    DefaultNormalizer,
    MultimodalNormalizer,
)


class InputStage(Stage[Any, NormalizedInput]):
    """Stage 1: Input — default artifact.

    Dual abstraction:
      - Level 2 validator: validates raw input
      - Level 2 normalizer: transforms to NormalizedInput
    """

    def __init__(
        self,
        validator: Optional[InputValidator] = None,
        normalizer: Optional[InputNormalizer] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "validator": StrategySlot(
                name="validator",
                strategy=validator or DefaultValidator(),
                registry={
                    "default": DefaultValidator,
                    "passthrough": PassthroughValidator,
                    "strict": StrictValidator,
                    "schema": SchemaValidator,
                },
                description="Raw input validation strategy",
            ),
            "normalizer": StrategySlot(
                name="normalizer",
                strategy=normalizer or DefaultNormalizer(),
                registry={
                    "default": DefaultNormalizer,
                    "multimodal": MultimodalNormalizer,
                },
                description="Input normalization strategy",
            ),
        }

    @property
    def _validator(self) -> InputValidator:
        return self._slots["validator"].strategy  # type: ignore[return-value]

    @property
    def _normalizer(self) -> InputNormalizer:
        return self._slots["normalizer"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "input"

    @property
    def order(self) -> int:
        return 1

    @property
    def category(self) -> str:
        return "ingress"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    async def execute(self, input: Any, state: PipelineState) -> NormalizedInput:
        # Validate
        error = self._validator.validate(input)
        if error:
            raise StageError(
                f"Input validation failed: {error}",
                stage_name=self.name,
                stage_order=self.order,
            )

        # Repair a history left dangling by an interrupted tool turn
        # BEFORE appending this turn's user message — otherwise the new
        # user message follows an unanswered assistant tool_use and every
        # request 400s (audit D4). Synthetic error results are inserted at
        # the required position; a clean history is untouched.
        repaired = repair_dangling_tool_calls(state.messages)
        if repaired:
            state.add_event("input.tool_calls_repaired", {"count": repaired})

        # Normalize
        normalized = self._normalizer.normalize(input)
        normalized.session_id = state.session_id

        # Add user message to state
        state.add_message("user", normalized.to_message_content())
        state.add_event("input.normalized", {"text_length": len(normalized.text)})

        return normalized
