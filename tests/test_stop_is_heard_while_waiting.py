"""정지는 **기다리는 동안에도** 들린다.

프로드 실증: [정지]를 눌러도 화면이 계속 흘렀다. 확인이 **이벤트와 이벤트
사이에서만** 일어났기 때문이다 — 긴 도구 실행이나 긴 모델 대기 중에는 아무도
듣지 않았고, 그 도구가 끝나야 비로소 멈췄다. 도구 하나가 30초면 정지도 30초
뒤에 듣는다.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from xgen_agent_runtime.host.runner import _CancelRequested, _next_event


def _loop():
    return asyncio.new_event_loop()


async def _slow(seconds: float):
    await asyncio.sleep(seconds)
    yield "늦게 온 이벤트"


def test_a_stop_during_a_long_wait_is_heard_quickly():
    loop = _loop()
    agen = _slow(30.0).__aiter__()          # 30초짜리 도구를 흉내낸다
    stop = {"v": False}

    def cancel_check() -> bool:
        return stop["v"]

    async def _press_stop_soon():
        await asyncio.sleep(0.3)
        stop["v"] = True

    loop.create_task(_press_stop_soon())
    started = time.monotonic()
    with pytest.raises(_CancelRequested):
        _next_event(loop, agen, cancel_check)
    waited = time.monotonic() - started

    assert waited < 3.0, (
        f"정지를 듣는 데 {waited:.1f}초 걸렸다 — 도구가 끝날 때까지 기다린 것이다"
    )
    loop.close()


def test_events_still_arrive_normally():
    """정지가 없으면 예전과 똑같이 그냥 기다린다."""
    loop = _loop()

    async def _quick():
        yield "첫 이벤트"

    agen = _quick().__aiter__()
    assert _next_event(loop, agen, lambda: False) == "첫 이벤트"
    with pytest.raises(StopAsyncIteration):
        _next_event(loop, agen, lambda: False)
    loop.close()


def test_without_a_cancel_hook_it_behaves_exactly_as_before():
    loop = _loop()

    async def _quick():
        yield "값"

    agen = _quick().__aiter__()
    assert _next_event(loop, agen, None) == "값"
    loop.close()


def test_a_broken_cancel_check_does_not_kill_the_turn():
    """확인이 터져도 턴은 계속된다 — 관측이 실행을 죽이면 안 된다."""
    loop = _loop()

    async def _slowish():
        await asyncio.sleep(0.5)
        yield "값"

    def boom() -> bool:
        raise RuntimeError("확인 실패")

    agen = _slowish().__aiter__()
    assert _next_event(loop, agen, boom) == "값"
    loop.close()
