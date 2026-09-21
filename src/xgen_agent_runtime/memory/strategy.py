"""``ProviderDrivenStrategy`` — provider-driven Stage 18 strategy.

Replaces the legacy ``GenyMemoryStrategy`` (host-manager duck-type)
with a generic implementation that talks to a ``MemoryProvider``
directly. The strategy is intentionally minimal — it only wires
``provider.record_turn`` for newly-appended messages. Heavy lifting
(record_execution / reflect / promote) is handled by ``MemoryStage``
itself (see ``stages/s18_memory/artifact/default/stage.py``); the
strategy's job is to make sure every user/assistant message lands in
STM via the provider before that stage's terminal logic runs.

The legacy ``GenyMemoryStrategy`` and its ``ReflectionResolver``
helper are deleted in this cycle (D5 — "즉시 폐기"). Hosts that need
custom reflection plumbing should implement a callback through
``MemoryHooks.should_reflect`` + ``state.metadata['needs_reflection']``
and run their own reflection job out-of-band, OR use the executor's
native reflection path via the ``MemoryStage`` provider hooks.
"""

from __future__ import annotations

import logging
from typing import Optional

from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.memory.provider import MemoryProvider, Turn
from xgen_agent_runtime.stages.s18_memory.interface import MemoryUpdateStrategy

logger = logging.getLogger(__name__)


_RECORDED_KEY = "memory.provider_strategy_recorded_idx"
#: ``MemoryStage._drive_provider`` 의 워터마크. 같은 messages 접두부를 세는 두 번째 자라,
#: 둘을 따로 보면 같은 메시지가 두 번 STM 에 들어간다(4.38.0 에서 합침). 문자열로 둔다 —
#: memory 층이 stage 층을 import 하지 않는다.
_STAGE_RECORDED_KEY = "memory.last_recorded_idx"


def _recorded_upto(state: PipelineState) -> int:
    """이미 기록된 messages 접두부 길이 — 두 워터마크 중 큰 값(중복 기록 방지)."""
    best = 0
    for key in (_RECORDED_KEY, _STAGE_RECORDED_KEY):
        try:
            value = int(state.metadata.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        best = max(best, value)
    return max(0, min(best, len(state.messages)))


class ProviderDrivenStrategy(MemoryUpdateStrategy):
    """Per-turn provider drive strategy.

    For every newly-appended message in ``state.messages`` since the
    last invocation, calls ``provider.stm().append(Turn.from_state_message(msg))``.
    This is a thin wrapper because ``MemoryStage._drive_provider``
    already handles the same duty when ``provider`` is attached to
    the stage; ``ProviderDrivenStrategy`` exists so the strategy slot
    of ``MemoryStage`` is non-empty even when the host wants the
    provider to be the sole STM authority. With this strategy + a
    provider attached, the legacy ``AppendOnlyStrategy`` /
    ``ReflectiveStrategy`` paths can stay unused.

    Construction takes only the provider — every piece of policy
    (should_reflect, should_auto_promote, importance gate) lives on
    ``provider.set_hooks(MemoryHooks(...))`` and is consumed by
    ``MemoryStage`` directly.
    """

    def __init__(self, provider: Optional[MemoryProvider] = None) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return "provider_driven"

    @property
    def description(self) -> str:
        return "Drive provider.record_turn per appended message; let MemoryStage own reflection / promotion."

    def attach_provider(self, provider: MemoryProvider) -> None:
        """Late-bind the provider after construction.

        ``default_manifest`` constructs the strategy before the host
        has built its provider; the host then calls this method (or
        ``MemoryStage.provider = ...``) once the provider exists.
        """
        self._provider = provider

    async def update(self, state: PipelineState) -> None:
        provider = self._provider
        if provider is None:
            return
        last_recorded = _recorded_upto(state)
        new_msgs = state.messages[last_recorded:]
        if not new_msgs:
            return

        recorded = 0
        for msg in new_msgs:
            try:
                turn = Turn.from_state_message(msg)
            except Exception:  # noqa: BLE001
                logger.debug("provider_driven: Turn.from_state_message failed", exc_info=True)
                continue
            try:
                await provider.record_turn(turn)
                recorded += 1
            except Exception:  # noqa: BLE001
                logger.debug("provider_driven: record_turn failed", exc_info=True)
                continue

        state.metadata[_RECORDED_KEY] = len(state.messages)
        state.metadata[_STAGE_RECORDED_KEY] = len(state.messages)
        if recorded:
            try:
                state.add_event(
                    "memory.provider_recorded",
                    {"count": recorded, "total_messages": len(state.messages)},
                )
            except Exception:  # noqa: BLE001
                pass


__all__ = ["ProviderDrivenStrategy"]
