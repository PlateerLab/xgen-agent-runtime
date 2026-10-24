"""Run-wide tool accounting and cancellation shared by parallel/nested calls."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import threading
import time
from typing import Any, Callable, Optional


class ExecutionBoundaryError(RuntimeError):
    pass


class RunBudget:
    """One run's tool calls, result bytes, usage and time.

    Time is either a fixed wall budget (``max_wall_seconds`` from creation, the default) or, with
    ``max_idle_seconds`` set, an **idle** budget: the run ends only after that long with nothing
    happening (no model call, no tool call, nothing in flight); ``max_wall_seconds`` is then the hard
    ceiling. A host serving long agent turns uses idle mode. A fixed five minutes cut a turn that was
    working, a long build or a run of tool calls, in the middle (2026-10-24 review).

    ``max_result_bytes`` bounds one result; ``max_total_result_bytes`` bounds all of the run's results
    together (the same value when not given, as before).
    """

    def __init__(
        self,
        *,
        max_calls: int = 1000,
        max_active: int = 16,
        max_result_bytes: int = 16 * 1024 * 1024,
        max_wall_seconds: float = 300,
        max_cost: float = 0,
        max_tokens: int = 0,
        max_idle_seconds: float = 0,
        max_total_result_bytes: Optional[int] = None,
    ) -> None:
        if (
            min(max_calls, max_active, max_result_bytes) <= 0
            or not math.isfinite(max_wall_seconds)
            or max_wall_seconds <= 0
            or not math.isfinite(max_idle_seconds)
            or max_idle_seconds < 0
        ):
            raise ValueError("invalid run budget")
        self.max_calls, self.max_active, self.max_result_bytes = (
            max_calls,
            max_active,
            max_result_bytes,
        )
        self.max_total_result_bytes = (
            max_result_bytes if max_total_result_bytes is None else int(max_total_result_bytes)
        )
        if self.max_total_result_bytes <= 0:
            raise ValueError("invalid run budget")
        now = time.monotonic()
        self._hard_deadline = now + max_wall_seconds
        self.max_idle_seconds = float(max_idle_seconds)
        self._last_activity = now
        self.models_active = 0
        if (
            not math.isfinite(max_cost)
            or max_cost < 0
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 0
        ):
            raise ValueError("invalid run usage budget")
        self.max_cost, self.max_tokens = max_cost, max_tokens
        self.calls = self.active = self.result_bytes = self.tokens = 0
        self.cost = 0.0
        self._usage_seen: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    @property
    def idle_mode(self) -> bool:
        return self.max_idle_seconds > 0

    @property
    def deadline(self) -> float:
        """When the run ends if nothing more happens. Work in flight keeps an idle-mode run open until
        the hard ceiling."""
        if not self.idle_mode:
            return self._hard_deadline
        if self.active > 0 or self.models_active > 0:
            return self._hard_deadline
        return min(self._hard_deadline, self._last_activity + self.max_idle_seconds)

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def idle_seconds(self) -> float:
        """Seconds since anything happened in this run (0 while a tool or model call is in flight)."""
        if self.active > 0 or self.models_active > 0:
            return 0.0
        return max(0.0, time.monotonic() - self._last_activity)

    def touch(self) -> None:
        """Record activity (an idle-mode run stays open)."""
        self._last_activity = time.monotonic()

    def before_model_call(self) -> None:
        """Host port invoked before every provider dispatch, including retries."""
        if self.expired():
            raise ExecutionBoundaryError("run_budget_exceeded")
        self.touch()

    def begin_model_call(self) -> None:
        """A provider call is in flight (a long answer or a CLI turn is activity, not idleness)."""
        with self._lock:
            self.models_active += 1
            self._last_activity = time.monotonic()

    def end_model_call(self) -> None:
        with self._lock:
            self.models_active = max(0, self.models_active - 1)
            self._last_activity = time.monotonic()

    def reserve(self) -> None:
        with self._lock:
            if self.expired() or self.calls >= self.max_calls or self.active >= self.max_active:
                raise ExecutionBoundaryError("run_budget_exceeded")
            self.calls += 1
            self.active += 1
            self._last_activity = time.monotonic()

    def release(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            self._last_activity = time.monotonic()

    def charge_result(self, value: Any) -> None:
        # A bounded traversal prevents cyclic/deep results becoming a bypass.
        from xgen_agent_runtime.permission.approval import canonical_digest

        try:
            canonical_digest(value, max_bytes=self.max_result_bytes)
            size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except Exception:
            raise ExecutionBoundaryError("tool_result_invalid") from None
        with self._lock:
            self.result_bytes += size
            self._last_activity = time.monotonic()
            if self.result_bytes > self.max_total_result_bytes or self.expired():
                raise ExecutionBoundaryError("run_budget_exceeded")

    def charge_usage(self, *, cost: float = 0, tokens: int = 0) -> None:
        if (
            not math.isfinite(cost)
            or cost < 0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
        ):
            raise ExecutionBoundaryError("run_usage_invalid")
        with self._lock:
            self.cost += cost
            self.tokens += tokens
            self._last_activity = time.monotonic()
            if (self.max_cost > 0 and self.cost > self.max_cost) or (
                self.max_tokens > 0 and self.tokens > self.max_tokens
            ):
                raise ExecutionBoundaryError("run_budget_exceeded")

    def reconcile_usage(self, key: str, *, cost: float, tokens: int) -> None:
        """Charge cumulative host usage once per run, including continuation."""
        if not math.isfinite(cost) or cost < 0 or not isinstance(tokens, int) or tokens < 0:
            raise ExecutionBoundaryError("run_usage_invalid")
        with self._lock:
            old_tokens, old_cost = self._usage_seen.get(key, (0, 0.0))
            if tokens < old_tokens or cost < old_cost:
                raise ExecutionBoundaryError("run_usage_invalid")
            self.tokens += tokens - old_tokens
            self.cost += cost - old_cost
            self._usage_seen[key] = (tokens, cost)
            self._last_activity = time.monotonic()
            if (self.max_cost > 0 and self.cost >= self.max_cost) or (
                self.max_tokens > 0 and self.tokens >= self.max_tokens
            ):
                raise ExecutionBoundaryError("run_budget_exceeded")


def bind_run_guard(pipeline: Any, context: Any) -> None:
    """Extend existing model guards with the same parent ledger/deadline."""
    if context is None or (context.run_budget is None and context.cancellation_scope is None):
        return
    from xgen_agent_runtime.stages.s04_guard.interface import Guard
    from xgen_agent_runtime.stages.s04_guard.types import GuardResult

    def enforce(state: Any) -> None:
        if context.cancellation_scope is not None:
            context.cancellation_scope.check()
        if context.run_budget is not None:
            expired = getattr(context.run_budget, "expired", None)
            if expired() if callable(expired) else time.monotonic() >= context.run_budget.deadline:
                raise ExecutionBoundaryError("run_budget_exceeded")
            usage = getattr(state, "turn_token_usage", []) or []
            tokens = sum(
                int(u.input_tokens or 0)
                + int(u.output_tokens or 0)
                + int(getattr(u, "cache_read_input_tokens", 0) or 0)
                + int(getattr(u, "cache_creation_input_tokens", 0) or 0)
                for u in usage
            )
            context.run_budget.reconcile_usage(
                str(getattr(state, "_run_id", "") or id(state)),
                tokens=tokens,
                cost=float(getattr(state, "total_cost_usd", 0) or 0),
            )

    class RunGuard(Guard):
        name = "host_run_boundary"

        def check(self, state: Any) -> Any:
            try:
                enforce(state)
                return GuardResult(passed=True, guard_name=self.name)
            except ExecutionBoundaryError as exc:
                return GuardResult(
                    passed=False, guard_name=self.name, message=str(exc), action="reject"
                )

    stage = pipeline.get_stage(4)
    if stage is not None and callable(getattr(stage, "add_guard", None)):
        stage.add_guard(RunGuard())
    token_stage = pipeline.get_stage(7)
    if token_stage is not None:
        # Charge the final response too, before tool dispatch, emit or persist.
        token_stage._host_usage_boundary = enforce
    api_stage = pipeline.get_stage(6)
    if api_stage is not None:

        def before_model_call() -> None:
            if context.cancellation_scope is not None:
                context.cancellation_scope.check()
            if context.run_budget is not None:
                context.run_budget.before_model_call()

        api_stage._host_model_boundary = before_model_call
        budget = context.run_budget
        if budget is not None and callable(getattr(budget, "begin_model_call", None)):
            # A provider call in flight is activity: an idle-mode run must not end under a long answer
            # or a CLI turn that is working (its tool calls come back through the bridge meanwhile).
            api_stage._host_model_activity = (budget.begin_model_call, budget.end_model_call)


class CancellationScope:
    def __init__(
        self,
        *,
        deadline: Optional[float] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        cleanup_timeout: float = 5,
    ) -> None:
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("invalid deadline")
        self.deadline, self.cancel_check = deadline, cancel_check
        self.cleanup_timeout = max(0.1, min(float(cleanup_timeout), 30))
        self.cancelled = False
        self.cleanup_pending = False
        self._cleanups: list[Callable[[], Any]] = []

    def check(self) -> None:
        try:
            cancelled = self.cancel_check is not None and bool(self.cancel_check())
        except Exception:
            cancelled = True
        if (
            self.cancelled
            or cancelled
            or (self.deadline is not None and time.monotonic() >= self.deadline)
        ):
            self.cancelled = True
            raise ExecutionBoundaryError("run_cancelled")

    def add_cleanup(self, callback: Callable[[], Any]) -> Callable[[], None]:
        self._cleanups.append(callback)
        return lambda: self._cleanups.remove(callback) if callback in self._cleanups else None

    async def cleanup(self) -> None:
        callbacks, self._cleanups = self._cleanups, []
        for callback in callbacks:
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, self.cleanup_timeout)
            except Exception:
                # Never expose backend stderr/credentials through cancellation.
                pass

    async def run(self, operation: Any, *, timeout: float = 0) -> Any:
        task = asyncio.ensure_future(operation)
        started = time.monotonic()
        try:
            while not task.done():
                self.check()
                if timeout > 0 and time.monotonic() - started >= timeout:
                    raise asyncio.TimeoutError
                await asyncio.wait({task}, timeout=0.05)
            self.check()
            return task.result()
        except BaseException:
            self.cancelled = True
            task.cancel()
            done, pending = await asyncio.wait({task}, timeout=self.cleanup_timeout)
            self.cleanup_pending = bool(pending)
            for completed in done:
                try:
                    completed.result()
                except BaseException:
                    pass
            await self.cleanup()
            if pending:
                # Discard all late completions. The scope remains closed; the
                # host must not reuse this backend until cleanup is confirmed.
                task.add_done_callback(
                    lambda future: future.exception() if not future.cancelled() else None
                )
            raise
