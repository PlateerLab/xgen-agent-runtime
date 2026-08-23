"""Host agent_event builder tests (v3.6.1).

``_tool_call_event`` / ``_tool_end_event`` feed the xgen SSE ``tool``
frames — chat UIs pair start/result rows by ``run_id`` (the engine's
``tool_use_id``). Missing run_id forces consumers into name-based
guess-pairing; these tests seal the additive key.
"""

from __future__ import annotations

from xgen_agent_runtime.host.runner import _tool_call_event, _tool_end_event


def test_tool_call_event_carries_run_id():
    event = _tool_call_event("DiskFree", {"path": "/"}, run_id="toolu_abc")
    assert event["run_id"] == "toolu_abc"
    assert event["type"] == "tool_call"
    assert event["tool_name"] == "DiskFree"


def test_tool_call_event_omits_empty_run_id():
    event = _tool_call_event("DiskFree", {}, run_id=None)
    assert "run_id" not in event
    event = _tool_call_event("DiskFree", {}, run_id="")
    assert "run_id" not in event


def test_tool_end_event_carries_run_id_and_result():
    event = _tool_end_event("DiskFree", "452.1GB", duration_ms=39, run_id="toolu_abc")
    assert event["run_id"] == "toolu_abc"
    assert event["type"] == "tool_result"
    assert event["result"] == "452.1GB"
    assert event["duration_ms"] == 39


def test_tool_end_event_error_carries_run_id():
    event = _tool_end_event("DiskFree", "boom", is_error=True, run_id="toolu_abc")
    assert event["type"] == "tool_error"
    assert event["error"] == "boom"
    assert event["run_id"] == "toolu_abc"


def test_tool_end_event_omits_empty_run_id():
    event = _tool_end_event("DiskFree", "ok")
    assert "run_id" not in event
