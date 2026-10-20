"""Count real dispatches instead of cumulative state/usage observations."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xgen_agent_runtime.core.errors import APIError, ErrorCategory
from xgen_agent_runtime.host.execution_scope import ExecutionBoundaryError, RunBudget, bind_run_guard
from xgen_agent_runtime.stages.s06_api.artifact.default.stage import APIStage


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_retry_cannot_dispatch_past_host_model_limit(streaming):
    class Budget(RunBudget):
        calls = 0

        def before_model_call(self):
            super().before_model_call()
            if self.calls >= 1:
                raise ExecutionBoundaryError("run_model_budget_exceeded")
            self.calls += 1

    stage = APIStage()
    retry = SimpleNamespace(max_retries=3, should_retry=lambda *a: True, get_delay=lambda *a: 0)
    # StrategySlot's implementation can be replaced without touching provider logic.
    stage._slots["retry"].strategy = retry
    failure = APIError("fixture", category=ErrorCategory.UNKNOWN)
    provider = AsyncMock(side_effect=failure)
    client = SimpleNamespace(create_message=provider)
    stage._call_kwargs = lambda *a, **k: {}
    stage._apply_timeout_kwarg = lambda *a, **k: None
    stage._call_streaming = provider
    budget = Budget()
    pipeline = SimpleNamespace(get_stage=lambda index: stage if index == 6 else None)
    bind_run_guard(pipeline, SimpleNamespace(run_budget=budget, cancellation_scope=None))
    state = SimpleNamespace(add_event=lambda *a, **k: None, shared={})
    method = stage._call_streaming_with_retry if streaming else stage._call_with_retry
    with pytest.raises(ExecutionBoundaryError, match="run_model_budget_exceeded"):
        await method(client, None, state)
    assert provider.await_count == budget.calls == 1


def test_cancelled_scope_denies_dispatch_before_counting():
    from xgen_agent_runtime.host.execution_scope import CancellationScope

    budget = RunBudget()
    stage = SimpleNamespace()
    pipeline = SimpleNamespace(get_stage=lambda index: stage if index == 6 else None)
    scope = CancellationScope(cancel_check=lambda: True)
    bind_run_guard(pipeline, SimpleNamespace(run_budget=budget, cancellation_scope=scope))
    with pytest.raises(ExecutionBoundaryError, match="run_cancelled"):
        stage._host_model_boundary()
