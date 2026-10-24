"""A turn that is working is not cut by a clock; one with nothing happening is (2026-10-24, 4.102.0).

A fixed five-minute run budget and a CLI limit counted from the start stopped agent turns in the
middle of a long build or a run of tool calls. Idle mode counts time with nothing happening instead.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from xgen_agent_runtime.host.execution_scope import ExecutionBoundaryError, RunBudget


def test_an_idle_budget_stays_open_while_a_tool_or_a_model_call_is_in_flight():
    budget = RunBudget(max_wall_seconds=30, max_idle_seconds=0.2)
    budget.reserve()  # a tool starts
    time.sleep(0.3)
    assert not budget.expired(), "a running tool is activity"
    budget.release()
    budget.begin_model_call()
    time.sleep(0.3)
    assert not budget.expired(), "a provider call in flight is activity"
    budget.end_model_call()
    assert not budget.expired()
    time.sleep(0.3)
    assert budget.expired(), "nothing happening for the idle limit ends the run"
    with pytest.raises(ExecutionBoundaryError):
        budget.before_model_call()


def test_activity_keeps_an_idle_budget_open_past_the_old_wall_and_the_ceiling_still_holds():
    budget = RunBudget(max_wall_seconds=0.6, max_idle_seconds=0.25)
    for _ in range(3):
        time.sleep(0.15)
        budget.before_model_call()
    assert not budget.expired(), "0.45s of steady activity is more than one idle limit"
    time.sleep(0.2)
    budget.touch()
    assert budget.expired(), "the hard ceiling still ends the run"


def test_the_default_budget_is_still_a_fixed_wall_and_total_results_have_their_own_cap():
    fixed = RunBudget(max_wall_seconds=0.1)
    fixed.reserve()
    time.sleep(0.15)
    assert fixed.expired(), "without idle mode nothing changes"

    budget = RunBudget(max_result_bytes=64, max_total_result_bytes=200)
    for _ in range(4):
        budget.charge_result("x" * 40)  # each result is within the per-result cap
    with pytest.raises(ExecutionBoundaryError):
        budget.charge_result("x" * 40)


def _spawn(code: str):
    return asyncio.create_subprocess_exec(
        sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )


async def _lines(code: str, **kw):
    from xgen_agent_runtime.llm_client._cli_runtime import _aiter_lines

    proc = await _spawn(code)
    out = []
    try:
        async for line in _aiter_lines(proc.stdout, start_t=time.monotonic(), proc=proc, **kw):
            out.append(line.strip())
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
    return out


async def test_a_cli_that_keeps_printing_runs_past_its_time_limit():
    code = "import time\nfor i in range(6):\n    print(i, flush=True); time.sleep(0.15)"
    assert await _lines(code, timeout_s=0.5) == [b"0", b"1", b"2", b"3", b"4", b"5"]


async def test_a_quiet_cli_waits_for_a_running_tool_and_stops_when_nothing_happens():
    from xgen_agent_runtime.llm_client._cli_runtime import CLITimeout

    code = "import time\nprint('a', flush=True); time.sleep(0.8); print('b', flush=True)"
    assert await _lines(code, timeout_s=0.4, idle_probe=lambda: 0.0) == [b"a", b"b"]
    with pytest.raises(CLITimeout, match="no output"):
        await _lines(code, timeout_s=0.4)


def test_the_tool_surface_knows_when_its_tools_are_busy():
    from xgen_agent_runtime.host.tool_surface import TurnToolSurface
    from xgen_agent_runtime.tools import ToolRegistry

    surface = TurnToolSurface(registry=ToolRegistry(), tool_context=None, state=None)
    surface._activity(+1)
    time.sleep(0.05)
    assert surface.idle_seconds() == 0.0
    surface._activity(-1)
    time.sleep(0.05)
    assert surface.idle_seconds() >= 0.05


def test_the_run_guard_tells_the_model_stage_when_a_call_is_in_flight():
    from types import SimpleNamespace

    from xgen_agent_runtime.host.execution_scope import bind_run_guard

    budget = RunBudget(max_idle_seconds=60)
    api_stage = SimpleNamespace()
    pipeline = SimpleNamespace(get_stage=lambda order: api_stage if order == 6 else None)
    bind_run_guard(pipeline, SimpleNamespace(run_budget=budget, cancellation_scope=None))
    begin, end = api_stage._host_model_activity
    begin()
    assert budget.models_active == 1 and budget.idle_seconds() == 0.0
    end()
    assert budget.models_active == 0
