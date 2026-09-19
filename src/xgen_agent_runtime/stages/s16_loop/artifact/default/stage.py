"""Default implementation of Stage 13: Loop."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from xgen_agent_runtime.core.schema import ConfigField, ConfigSchema
from xgen_agent_runtime.core.slot import StrategySlot
from xgen_agent_runtime.core.stage import Stage
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.stages.s16_loop.completion_review import CompletionReviewer
from xgen_agent_runtime.stages.s16_loop.interface import LoopController
from xgen_agent_runtime.stages.s16_loop.turn_budget import TurnInputBudget
from xgen_agent_runtime.stages.s16_loop.artifact.default.controllers import (
    BudgetAwareLoopController,
    MultiDimensionalBudgetController,
    SingleTurnController,
    StandardLoopController,
)


class LoopStage(Stage[Any, Any]):
    """Stage 13: Loop.

    Dual abstraction:
      - Level 2 controller: decides continue/complete/suspend/error/escalate
    """

    def __init__(
        self,
        controller: Optional[LoopController] = None,
        *,
        max_turns: Optional[int] = None,
        early_stop_on: Optional[List[str]] = None,
        completion_reviewers: Optional[List[CompletionReviewer]] = None,
        turn_input_budget: Optional[TurnInputBudget] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "controller": StrategySlot(
                name="controller",
                strategy=controller or StandardLoopController(max_turns=max_turns),
                registry={
                    "standard": StandardLoopController,
                    "single_turn": SingleTurnController,
                    "budget_aware": BudgetAwareLoopController,
                    # Phase 7 S7.7 — pluggable multi-dimensional
                    # budget. Dimensions arrive via constructor; the
                    # zero-arg slot-swap path produces an empty
                    # dimension list (acts like StandardLoopController).
                    "multi_dim_budget": MultiDimensionalBudgetController,
                },
                description="Loop decision strategy",
            ),
        }
        self._max_turns = max_turns
        self._early_stop_on: List[str] = list(early_stop_on or [])
        # 완료 직전 검토자들 (stages/s16_loop/completion_review.py). 하나라도
        # 메시지를 돌려주면 그걸 모델에게 보내고 한 바퀴 더 돈다. 검토자가
        # 스스로 "턴당 한 번" 을 지킨다 — 여기서는 순서대로 물어볼 뿐.
        self._completion_reviewers: List[CompletionReviewer] = list(completion_reviewers or [])

        # 턴 입력 토큰 예산 (stages/s16_loop/turn_budget.py). None 이면 없음.
        self._turn_input_budget: Optional[TurnInputBudget] = turn_input_budget

    def add_completion_reviewer(self, reviewer: CompletionReviewer) -> None:
        self._completion_reviewers.append(reviewer)

    def set_turn_input_budget(self, budget: Optional[TurnInputBudget]) -> None:
        self._turn_input_budget = budget

    @property
    def _controller(self) -> LoopController:
        return self._slots["controller"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "loop"

    @property
    def order(self) -> int:
        return 16

    @property
    def category(self) -> str:
        return "decision"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="loop",
            fields=[
                ConfigField(
                    name="max_turns",
                    type="integer",
                    label="Max Turns",
                    description="Hard cap on loop iterations. Blank to defer to state.max_iterations.",
                    default=0,
                    min_value=0,
                ),
                ConfigField(
                    name="early_stop_on",
                    type="array",
                    label="Early Stop Signals",
                    description="Completion signals that should abort the loop immediately.",
                    default=[],
                    item_type="string",
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_turns": self._max_turns or 0,
            "early_stop_on": list(self._early_stop_on),
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "max_turns" in config:
            value = int(config["max_turns"])
            self._max_turns = value if value > 0 else None
            controller = self._slots["controller"].strategy
            if self._controller_declares_max_turns(controller):
                # 2026-06-09 audit §2.1: the old hasattr('_max_turns')
                # poke silently skipped MultiDimensionalBudgetController
                # (it has no such attribute), so a manifest-level
                # max_turns was inert for exactly the controller Geny
                # prod runs. configure() is the contract now — the
                # controller decides what max_turns means for it.
                controller.configure({"max_turns": value})
            elif hasattr(controller, "_max_turns"):
                # Legacy fallback for host-supplied controllers that
                # predate the configure() contract.
                controller._max_turns = self._max_turns  # type: ignore[attr-defined]
        if "early_stop_on" in config:
            self._early_stop_on = list(config["early_stop_on"] or [])

    @staticmethod
    def _controller_declares_max_turns(controller: LoopController) -> bool:
        """True when the controller's config_schema() exposes ``max_turns``."""
        try:
            schema = controller.config_schema()
        except Exception:
            return False
        if schema is None:
            return False
        return any(getattr(f, "name", "") == "max_turns" for f in getattr(schema, "fields", []))

    async def execute(self, input: Any, state: PipelineState) -> Any:
        upstream = state.loop_decision
        if upstream in ("complete", "suspend", "error", "escalate"):
            decision = upstream
        elif self._early_stop_on and state.completion_signal in self._early_stop_on:
            decision = "complete"
        else:
            decision = self._controller.decide(state)

        if decision == "complete" and self._completion_reviewers:
            # 산출물 대조 등 — 완료를 한 번 미루고 검토 내용을 모델에게 보낸다.
            # suspend/error/escalate 는 건드리지 않는다 (마무리할 것이 없다).
            for reviewer in self._completion_reviewers:
                note = await reviewer.review(state)
                if note:
                    state.add_message("user", note)
                    state.completion_signal = None
                    state.completion_detail = None
                    decision = "continue"
                    break

        if self._turn_input_budget is not None:
            # 예산은 검토자 뒤에 — 마무리 응답이 온 뒤에는 검토자가 미뤄도 끝낸다.
            decision = self._turn_input_budget.apply(state, decision)

        state.loop_decision = decision

        state.add_event(
            f"loop.{decision}",
            {
                "iteration": state.iteration,
                "signal": state.completion_signal,
                "pending_tools": len(state.pending_tool_calls),
                "has_tool_results": bool(state.tool_results),
                "upstream_decision": upstream,
            },
        )

        state.tool_results = []
        return input
