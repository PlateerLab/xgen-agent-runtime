"""실행 엔진 선택점 — 턴 조립은 엔진과 무관하고, 실행 코어만 설정으로 고른다(4.76.0).

* 기본은 기존 21-stage 엔진이다(설정 없음 = 무변화).
* ``XGEN_HARNESS_ENGINE=rsi`` 면 xgen_rsi 의 RSI 하네스가 같은 계획(TurnPlan)으로 돈다.
* xgen_rsi 가 없으면 기존 엔진으로 돌아간다(경고 한 번) — 설정만 먼저 켜도 턴이 깨지지 않는다.
* 테스트용 kwargs ``harness_engine`` 은 호스트로 넘기기 전에 뺀다.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from xgen_agent_runtime.host import runner as runner_mod
from xgen_agent_runtime.host import turn_executor as te
from xgen_agent_runtime.host.turn_executor import (
    ENGINE_PIPELINE21,
    ENGINE_RSI,
    ENGINE_SETTING,
    SystemPromptParts,
    select_engine,
)
from tests.test_host_turn_executor_gates import _FakeHost


@pytest.fixture
def capture(monkeypatch):
    seen = {}

    def _fake_build_pipeline(**kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(runner_mod, "build_pipeline", _fake_build_pipeline)
    monkeypatch.setattr(runner_mod, "run_turn", lambda *a, **k: seen.setdefault("run_kwargs", dict(k)) and "done")
    monkeypatch.setattr(runner_mod, "stream_turn", lambda *a, **k: iter([]))
    return seen


class _SettingHost:
    def __init__(self, value):
        self._value = value

    def setting(self, name, default=""):
        return self._value if name == ENGINE_SETTING else default


def test_default_engine_is_the_pipeline() -> None:
    assert select_engine(_SettingHost(None), {}) == ENGINE_PIPELINE21
    assert select_engine(SimpleNamespace(), {}) == ENGINE_PIPELINE21
    assert select_engine(_SettingHost("  RSI "), {}) == ENGINE_RSI
    assert select_engine(_SettingHost("something-else"), {}) == ENGINE_PIPELINE21


def test_kwargs_override_is_popped_before_the_host_sees_it() -> None:
    kwargs = {"harness_engine": "rsi", "text": "hi"}
    assert select_engine(_SettingHost(None), kwargs) == ENGINE_RSI
    assert "harness_engine" not in kwargs


def test_system_prompt_parts_join_is_the_same_string() -> None:
    sp = SystemPromptParts("base")
    sp.add("jobs", "\n\nJOBS")
    final = sp.add("efficiency", "\n\nEFF")
    assert final == "base\n\nJOBS\n\nEFF" == sp.text
    assert [p for p, _ in sp.parts] == ["base", "jobs", "efficiency"]


def test_rsi_setting_without_the_package_falls_back_to_the_pipeline(capture, monkeypatch, caplog) -> None:
    monkeypatch.setitem(sys.modules, "xgen_rsi.kernel.executor", None)  # import 실패를 흉내낸다
    host = _FakeHost(memory=False)
    monkeypatch.setattr(host, "setting", lambda name, default="": "rsi" if name == ENGINE_SETTING else default)
    te.AgentTurnExecutor().run(host, text="hi", provider="openai", streaming=False)
    assert capture.get("provider") == "openai"  # 기존 build_pipeline 이 불렸다(인자를 잡았다)
    assert "run_kwargs" in capture
    assert "xgen_rsi" in caplog.text


def test_plan_carries_the_system_parts_and_pipeline_kwargs(capture) -> None:
    host = _FakeHost(memory=False)
    plan = te.assemble_turn(host, {"text": "hi", "provider": "openai"}, {})
    assert isinstance(plan, te.TurnPlan)
    assert "".join(t for _, t in plan.system_parts) == plan.system_prompt
    assert plan.pipeline_kwargs["system_prompt"] == plan.system_prompt
    assert plan.pipeline_kwargs["provider"] == "openai"
