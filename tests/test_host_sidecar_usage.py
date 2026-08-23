"""sidecar v2 프로토콜 추가 계약 — ``usage`` 1급 이벤트 + cancel/done 레이스.

* runner.stream_turn 의 ``{"type": "usage", "data": {...}}`` 청크는 ``meta`` 로 감싸지
  않고 ``usage`` 이벤트로 그대로 올라간다 (커넥터 TurnReport.usage → report-turn).
* cancel 이 관측된 턴은 스트림이 자연 종료해도 ``done`` 이 아니라 ``cancelled`` 로 닫는다.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import pytest

from xgen_agent_runtime.host import sidecar

_USAGE = {
    "input_tokens": 12,
    "output_tokens": 3,
    "cache_read_tokens": 0,
    "cache_creation_tokens": None,
    "total_cost_usd": 0.001,
    "model": "m",
    "provider": "claude_code_cli",
}


class _FakeExecutor:
    items: List[Any] = []
    result: Any = None  # non-stream 반환값(str)

    def run(self, host: Any, **kwargs: Any) -> Any:
        if not kwargs.get("streaming", True):
            return _FakeExecutor.result

        def _gen() -> Iterator[Any]:
            for it in _FakeExecutor.items:
                if callable(it):
                    it()
                    continue
                yield it

        return _gen()


@pytest.fixture
def fake_executor(monkeypatch, tmp_path):
    import xgen_agent_runtime.host.turn_executor as te

    monkeypatch.setattr(te, "AgentTurnExecutor", _FakeExecutor)
    _FakeExecutor.items = []
    _FakeExecutor.result = None
    return str(tmp_path / "ws")


def _req(ws: str, **opts: Any) -> Dict[str, Any]:
    return {
        "workspace_dir": ws,
        "provider": "claude_code",
        "text": "hello",
        "context": {"api_keys": {}, "settings": {}},
        "options": {"interaction_id": "i1", "workflow_id": "wf", **opts},
    }


def test_normalize_usage_is_first_class_event() -> None:
    ev = sidecar._normalize_event({"type": "usage", "data": dict(_USAGE)})
    assert ev == {"type": "usage", "data": _USAGE}
    # data 가 dict 가 아니면 기존 meta 강등 유지
    assert sidecar._normalize_event({"type": "usage", "data": "x"})["type"] == "meta"


def test_run_turn_request_emits_usage_before_done(fake_executor) -> None:
    _FakeExecutor.items = ["안", "녕", {"type": "usage", "data": dict(_USAGE)}]
    events = list(sidecar.run_turn_request(_req(fake_executor)))
    assert [e["type"] for e in events] == ["started", "chunk", "chunk", "usage", "done"]
    assert events[3]["data"] == _USAGE
    assert events[-1]["text"] == "안녕"  # usage 는 텍스트에 섞이지 않는다


def test_protocol_doc_mentions_usage_event() -> None:
    assert '{"type": "usage"' in (sidecar.__doc__ or "")


def test_cancel_observed_after_last_chunk_emits_cancelled_not_done(fake_executor) -> None:
    flag = {"v": False}

    def _flip() -> None:
        flag["v"] = True

    # 마지막 청크 뒤(스트림 종료 직전)에 cancel — 기존엔 done 으로 닫혔다(레이스).
    _FakeExecutor.items = ["a", "b", _flip]
    events = list(sidecar.run_turn_request(_req(fake_executor), cancel_check=lambda: flag["v"]))
    kinds = [e["type"] for e in events]
    assert kinds == ["started", "chunk", "chunk", "cancelled"]
    assert "done" not in kinds


def test_cancel_without_request_still_done(fake_executor) -> None:
    _FakeExecutor.items = ["a"]
    events = list(sidecar.run_turn_request(_req(fake_executor), cancel_check=lambda: False))
    assert [e["type"] for e in events] == ["started", "chunk", "done"]


def test_non_stream_cancel_observed_emits_cancelled(fake_executor) -> None:
    _FakeExecutor.result = "full text"
    events = list(
        sidecar.run_turn_request(_req(fake_executor, streaming=False), cancel_check=lambda: True)
    )
    assert [e["type"] for e in events] == ["started", "cancelled"]
    events2 = list(
        sidecar.run_turn_request(_req(fake_executor, streaming=False), cancel_check=lambda: False)
    )
    assert [e["type"] for e in events2] == ["started", "done"] and events2[-1]["text"] == "full text"
