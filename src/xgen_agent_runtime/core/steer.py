"""Mid-turn steering — messages the user adds to a RUNNING turn.

While an agent turn runs, the user may send more instructions ("also check
the tests", "use the other file"). Instead of waiting for the turn to end or
stopping it, the host offers the message to the turn's :class:`SteerInbox`;
the pipeline takes it at the next loop boundary (after the current model call
and its tools) and the model sees it on its next call.

Threading: the host offers from its own thread / event loop, the pipeline
takes from the turn's private loop. Everything here is guarded by one lock;
callbacks run in the TAKING thread (the turn's), so hosts must hop back to
their own loop themselves (e.g. ``loop.call_soon_threadsafe``).

Delivery contract (one terminal outcome per message):

* ``delivered=True``  — the message is in the conversation; the model sees it.
* ``delivered=False`` — the turn could not take it any more (it was already
  finishing, failed or was stopped). The host should send it as the next turn.

The pipeline closes the inbox in the same locked step that decides "the loop
is done", so a message can never slip in between the last check and the end
of the turn without being rejected.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

#: How the injected text is framed for the model. The user's own words follow
#: verbatim; the frame only tells the model this arrived mid-task.
STEER_FRAME = (
    "[The user sent this message while you were working. Take it into account and continue.]"
)


@dataclass
class SteerMessage:
    """One message offered to a running turn."""

    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    client_msg_id: Optional[str] = None
    #: Free-form host data echoed back in callbacks / events (e.g. user id).
    meta: Dict[str, Any] = field(default_factory=dict)
    delivered: Optional[bool] = field(default=None, init=False)
    _done: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def wait(self, timeout: Optional[float] = None) -> Optional[bool]:
        """Block until delivered/rejected; ``None`` on timeout."""
        self._done.wait(timeout)
        return self.delivered


class SteerInbox:
    """Thread-safe mid-turn input channel for ONE turn.

    ``on_resolve(message, delivered)`` — optional, called once per message
    when it is delivered into the conversation or rejected.
    """

    def __init__(self, on_resolve: Optional[Callable[[SteerMessage, bool], None]] = None) -> None:
        self._lock = threading.Lock()
        self._queue: Deque[SteerMessage] = deque()
        self._open = True
        self._on_resolve = on_resolve
        #: Set by the runner to the turn's cancel check. A turn being stopped
        #: must not take messages: the model would never see them, so they go
        #: back to the host (as the next turn) instead of looking delivered.
        self.stop_check: Optional[Callable[[], bool]] = None

    def stopping(self) -> bool:
        try:
            return bool(self.stop_check is not None and self.stop_check())
        except Exception:  # noqa: BLE001 - a broken check must not block delivery
            return False

    @property
    def open(self) -> bool:
        with self._lock:
            return self._open

    def offer(self, message: SteerMessage | str, **kwargs: Any) -> Optional[SteerMessage]:
        """Queue a message for the running turn.

        Returns the queued :class:`SteerMessage`, or ``None`` when the turn no
        longer takes input (closed) or the text is empty — send it as the next
        turn instead.
        """
        msg = (
            message
            if isinstance(message, SteerMessage)
            else SteerMessage(text=str(message), **kwargs)
        )
        if not msg.text.strip():
            return None
        with self._lock:
            if not self._open:
                return None
            self._queue.append(msg)
        return msg

    def take(self) -> List[SteerMessage]:
        """Take everything queued so far (the inbox stays open)."""
        with self._lock:
            out = list(self._queue)
            self._queue.clear()
        return out

    def take_or_close(self) -> List[SteerMessage]:
        """Atomically: take the queued messages, or — if there are none —
        close the inbox. Used where the turn is about to finish: either there
        is more to do, or nothing can arrive any more."""
        with self._lock:
            if self._queue:
                out = list(self._queue)
                self._queue.clear()
                return out
            self._open = False
            return []

    def close(self) -> List[SteerMessage]:
        """Stop taking input; reject (``delivered=False``) whatever is queued."""
        with self._lock:
            self._open = False
            left = list(self._queue)
            self._queue.clear()
        for msg in left:
            self.resolve(msg, False)
        return left

    def resolve(self, message: SteerMessage, delivered: bool) -> None:
        """Record the outcome once and notify the host."""
        with self._lock:
            if message.delivered is not None:
                return
            message.delivered = bool(delivered)
        message._done.set()
        if self._on_resolve is not None:
            try:
                self._on_resolve(message, bool(delivered))
            except Exception:  # noqa: BLE001 - a host callback must never break the turn
                pass


def steer_text(messages: List[SteerMessage]) -> str:
    """The text block the model receives for one or more steered messages."""
    body = "\n\n".join(m.text.strip() for m in messages if m.text.strip())
    return f"{STEER_FRAME}\n{body}"


def append_steer(state_messages: List[Dict[str, Any]], text: str) -> None:
    """Put ``text`` into the conversation as the user's next words.

    After a tool round the last message is the user-role ``tool_result``
    message; a second user message in a row is rejected by some providers,
    so the text is added to that message as a text block (translators split
    tool results and text correctly for every provider). Otherwise it is a
    new user message.
    """
    last = state_messages[-1] if state_messages else None
    if last is not None and last.get("role") == "user":
        content = last.get("content")
        if isinstance(content, list):
            last["content"] = [*content, {"type": "text", "text": text}]
        else:
            prev = "" if content is None else str(content)
            last["content"] = f"{prev}\n\n{text}" if prev else text
        return
    state_messages.append({"role": "user", "content": text})
