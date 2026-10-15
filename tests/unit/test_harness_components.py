"""하네스 장치 작동 요약 — 턴 usage 의 ``harness`` 계약.

* 사건 수는 턴 단위: 연속 슬라이스(events 비움)를 넘어 이어지고, 새 턴에서 비운다.
* 작동한 장치만(0 회 제외) 장치 이름으로 묶여 나온다.
* 빠른 경로 판정은 state.shared 에 있을 때 그대로 싣는다.
* 아무것도 없으면 usage 에 ``harness`` 키가 없다(기존 소비자 shape 유지).
"""

from __future__ import annotations

from types import SimpleNamespace

from xgen_agent_runtime import PipelineState
from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.core.state import TokenUsage
from xgen_agent_runtime.events.catalog import EventTypes
from xgen_agent_runtime.host import runner
from xgen_agent_runtime.host.harness_components import COMPONENTS, harness_summary


def _catalog_names() -> set:
    return {v for k, v in vars(EventTypes).items() if k.isupper() and isinstance(v, str)}


def test_every_component_event_is_in_the_catalog() -> None:
    known = _catalog_names()
    for name, (_layer, events) in COMPONENTS.items():
        for e in events:
            assert e in known, f"{name}: {e} 는 사건 목록에 없다"


def test_counts_survive_continuation_slices_and_reset_per_turn() -> None:
    state = PipelineState()
    state.begin_turn()
    state.add_event("tool.repeat_failure", {})
    state.begin_continuation_slice()
    assert state.events == []
    state.add_event("tool.repeat_blocked", {})
    state.add_event("loop.completion_review", {})
    assert harness_summary(state) == {"components": {"repeat_guard": 2, "completion_review": 1}}

    state.begin_turn()
    assert harness_summary(state) is None


def test_fast_path_verdict_is_reported() -> None:
    state = PipelineState()
    state.shared[SharedKeys.WORKSPACE_FAST_PATH] = {
        "active": False,
        "reason": "too_many_files",
        "file_count": 0,
        "total_bytes": 0,
    }
    assert harness_summary(state) == {
        "fast_path": {"active": False, "reason": "too_many_files", "file_count": 0, "total_bytes": 0}
    }


def test_turn_usage_carries_harness_only_when_something_fired() -> None:
    pipeline = SimpleNamespace(_resolved_provider_name=lambda _s: "vllm")
    state = PipelineState()
    state.turn_token_usage = [TokenUsage(input_tokens=100, output_tokens=10)]
    assert "harness" not in runner.turn_usage(pipeline, state)

    state.add_event("context.pruned", {})
    assert runner.turn_usage(pipeline, state)["harness"] == {"components": {"context_prune": 1}}
