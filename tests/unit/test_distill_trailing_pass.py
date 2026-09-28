"""증류가 도는 중에 들어온 요청은 버리지 않고 끝난 뒤 한 번 더 돈다.

예전엔 건너뛰었다 — 최근 대화(STM)는 세션별이라, 앞 턴의 증류가 도는 사이 끝난 세션 마지막 턴
(대개 사용자의 교정·지시)은 영영 사실로 추출되지 않았다.
"""

import threading
import time
from types import SimpleNamespace

from xgen_agent_runtime.host import distill


def test_request_during_a_pass_runs_one_trailing_pass_with_the_latest_spec(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    seen = []

    def fake_run(spec):
        seen.append(spec.interaction_id)
        if len(seen) == 1:
            started.set()
            release.wait(5)
        return {"facts_changes": 0, "digest": False, "daily": False, "evergreen": False}

    monkeypatch.setattr(distill, "run_distillation", fake_run)
    monkeypatch.setattr(distill, "_load_state", lambda wf, host: {})
    monkeypatch.setattr(distill, "_save_state", lambda wf, state, host: None)

    def spec(iid):
        return SimpleNamespace(workflow_id="wf-trailing", interaction_id=iid, host=None)

    assert distill.launch_distillation(spec("turn-1")) is True
    assert started.wait(5)
    assert distill.launch_distillation(spec("turn-2")) is False  # 맡겨 둠
    assert distill.launch_distillation(spec("turn-3")) is False  # 마지막 요청만 남는다
    release.set()

    deadline = time.time() + 5
    while time.time() < deadline:
        with distill._inflight_lock:
            if "wf-trailing" not in distill._inflight:
                break
        time.sleep(0.02)
    assert seen == ["turn-1", "turn-3"]
    assert "wf-trailing" not in distill._pending
