"""Early cancellation without an execution row must not poison the next turn."""
import threading

from xgen_agent_runtime.host.cancel_context import clear_cancel, is_cancelled, request_cancel


def test_new_turn_checker_is_not_poisoned_by_previous_interaction_stop():
    interaction = 'same-conversation-early-stop'
    previous = threading.Event()
    current = threading.Event()
    previous.set()
    request_cancel(interaction)  # first turn stopped before response_io_id existed
    try:
        assert is_cancelled(interaction, cancel_check=previous.is_set)
        for io_id in (None, 123, 124):
            assert not is_cancelled(interaction, io_id, cancel_check=current.is_set)
        current.set()
        assert is_cancelled(interaction, 124, cancel_check=current.is_set)
        assert is_cancelled(interaction, 124)  # legacy callers are unchanged
    finally:
        clear_cancel(interaction)


def test_per_turn_cancellation_works_without_a_conversation_id():
    assert is_cancelled(None, cancel_check=lambda: True)
    assert not is_cancelled(None, cancel_check=lambda: False)
