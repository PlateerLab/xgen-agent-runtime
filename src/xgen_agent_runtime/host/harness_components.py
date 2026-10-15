"""하네스 장치 목록 — 턴마다 어떤 장치가 몇 번 작동했는지 요약한다.

가드·복구·검증·맥락 장치는 단계 코드에 흩어져 있어 "이 장치가 실사용에서 얼마나
자주 켜지나" 를 볼 곳이 없었다. 각 장치가 이미 내는 사건(events/catalog.py)을
장치 이름으로 묶어 턴 usage 페이로드(``harness``)에 싣는다 — 호스트는 그대로
트레이스에 남기고, 주간 집계는 그 기록만 읽는다.

새 장치를 붙이면 여기에 한 줄을 더한다. 사건을 내지 않는 장치는 아직 셀 수 없다
(llm_client 사건은 state 를 거치지 않는다).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from xgen_agent_runtime.core.shared_keys import SharedKeys

#: 장치 이름 → (층위, 그 장치가 작동했다는 사건들)
COMPONENTS: Mapping[str, Tuple[str, Tuple[str, ...]]] = {
    "repeat_guard": (
        "middleware",
        ("tool.repeat_failure", "tool.repeat_blocked", "tool.same_result"),
    ),
    "second_machine": ("middleware", ("tool.not_in_sandbox",)),
    "user_denied": ("middleware", ("tool.user_denied",)),
    "message_repair": ("recovery", ("input.tool_calls_repaired",)),
    "api_retry": ("recovery", ("api.retry", "api.stream_restart")),
    "repeat_stop": ("recovery", ("loop.repeat_stop",)),
    "turn_budget": ("recovery", ("loop.turn_budget",)),
    "completion_review": ("verification", ("loop.completion_review",)),
    "context_prune": ("context", ("context.pruned",)),
    "context_compact": ("context", ("context.compacted",)),
}


def harness_summary(state: Any) -> Optional[Dict[str, Any]]:
    """이번 턴에 작동한 장치(0 회는 빼고)와 작업 폴더 빠른 경로 판정. 아무것도 없으면 None."""
    counts = getattr(state, "_turn_event_counts", None) or {}
    fired: Dict[str, int] = {}
    for name, (_layer, events) in COMPONENTS.items():
        n = sum(int(counts.get(e, 0)) for e in events)
        if n:
            fired[name] = n
    out: Dict[str, Any] = {}
    if fired:
        out["components"] = fired
    shared = getattr(state, "shared", None) or {}
    fast_path = shared.get(SharedKeys.WORKSPACE_FAST_PATH)
    if isinstance(fast_path, dict):
        out["fast_path"] = dict(fast_path)
    return out or None
