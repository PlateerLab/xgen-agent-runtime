"""호스트의 선택 훅 ``turn_notes()`` 는 폴더 안내 뒤 ``SharedKeys.TURN_NOTES`` 로만 실린다."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.host import runner as runner_mod

from tests.test_host_local_folders import _DeviceHost, _state
from tests.test_host_turn_executor_gates import _FakeHost, _run

NOTE = "[기억] 같은 질문의 답이 기억에 있습니다. 이 답으로 바로 답하고 도구를 부르지 마세요."


@pytest.fixture
def capture(monkeypatch):
    """build_pipeline 인자와 스트림에 넘어간 입력 · state 를 잡는다."""
    seen: Dict[str, Any] = {}

    def _fake_build_pipeline(**kw):
        seen.update(kw)
        return object()

    def _fake_stream_turn(*args, **kwargs):
        seen["stream_input"] = args[1]
        seen["stream_state"] = args[2]
        return iter([])

    monkeypatch.setattr(runner_mod, "build_pipeline", _fake_build_pipeline)
    monkeypatch.setattr(runner_mod, "stream_turn", _fake_stream_turn)
    return seen


class _NotesHost(_FakeHost):
    def __init__(self, notes: Any, **kw: Any) -> None:
        super().__init__(**kw)
        self._notes = notes

    def turn_notes(self):
        if isinstance(self._notes, Exception):
            raise self._notes
        return self._notes


class _NotesDeviceHost(_DeviceHost):
    def turn_notes(self):
        return [NOTE]


def test_host_notes_ride_the_turn_notes_not_the_request(capture) -> None:
    _run(_NotesHost([NOTE]), capture)
    assert _state(capture).shared[SharedKeys.TURN_NOTES] == [NOTE]
    assert NOTE not in str(capture["stream_input"])


def test_absent_hook_adds_nothing(capture) -> None:
    _run(_FakeHost(), capture)
    assert SharedKeys.TURN_NOTES not in _state(capture).shared


def test_failing_hook_is_ignored_and_the_turn_runs(capture) -> None:
    _run(_NotesHost(RuntimeError("vault down")), capture)
    assert SharedKeys.TURN_NOTES not in _state(capture).shared


def test_async_hook_is_ignored_without_an_unawaited_coroutine(capture, recwarn) -> None:
    class _AsyncHost(_FakeHost):
        async def turn_notes(self):
            return [NOTE]

    _run(_AsyncHost(), capture)
    assert SharedKeys.TURN_NOTES not in _state(capture).shared
    assert not [w for w in recwarn if "never awaited" in str(w.message)]


def test_a_single_string_is_one_note_and_blank_items_are_dropped(capture) -> None:
    _run(_NotesHost(NOTE), capture)
    assert _state(capture).shared[SharedKeys.TURN_NOTES] == [NOTE]
    _run(_NotesHost(["", "  ", NOTE, None]), capture)
    assert _state(capture).shared[SharedKeys.TURN_NOTES] == [NOTE]


def test_host_notes_come_after_the_folder_note(capture) -> None:
    _run(_NotesDeviceHost(), capture, local_folders=[])
    notes = _state(capture).shared[SharedKeys.TURN_NOTES]
    assert "No folder on the user's device" in notes[0]
    assert notes[-1] == NOTE


def test_cli_backend_gets_the_host_notes_too(capture) -> None:
    _run(_NotesDeviceHost(platform="win32"), capture, provider="claude_code", client_surface="connector",
         local_folders=[{"path": "C:\\work\\docs"}])
    notes = _state(capture).shared[SharedKeys.TURN_NOTES]
    assert notes[-1] == NOTE
    assert any("- docs:" in n for n in notes)
