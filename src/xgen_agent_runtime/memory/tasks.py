"""TASK memory: the work a person asked for, kept as its own memory layer (2026-10-23).

Facts say who the user is; the transcript says what was said. Neither says *what we are in the middle
of*. A request that takes several steps or turns ("write the quarterly report", "migrate the jobs")
used to live only in the transcript. Once the window slid past the turns where it was set up, or a
compaction folded them into a summary, the agent lost the goal, the plan, the decisions and the files,
and a "continue" had nothing to continue from.

A task is a record the agent and the host keep up to date while the work goes on:

* **Declared by the model** with the ``Task`` tool (open, update, switch, close). The system prompt
  carries the *task board* every turn: this conversation's current task in full and the other open
  tasks in one line each, so a short follow-up ("continue", "fix that") lands on the right task
  without any search.
* **Completed by the host** without a model call: every turn worked on the current task is added to
  it as an event (request, answer, tools, files, outcome); TodoWrite's list becomes its plan; a
  compaction's summary is kept on it; and when the model did not open a task but the turn clearly
  was multi-step work (a plan of two or more steps not done, or a turn that ended before it
  finished), the host opens one from the request, marked as opened by the host.
* **Recalled** through the board (no search involved), ``TaskRecall`` (the full record, with the
  turns that are no longer in context) and the memory search (the store renders each task to a note).

Tasks belong to a person (the store is scoped by the host, e.g. agent and user) and span
conversations: a task opened in one conversation can be picked up in another.

Everything here is store-agnostic: :class:`TaskStore` is the contract a host implements (XGEN keeps
tasks in Postgres tables), :class:`InMemoryTaskStore` is the reference implementation.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "BOARD_KEY",
    "CURRENT_KEY",
    "OPEN_STATUSES",
    "STATUSES",
    "Task",
    "TaskEvent",
    "TaskStore",
    "InMemoryTaskStore",
    "load_board",
    "render_board",
    "render_card",
    "record_turn",
    "plan_from_todos",
    "provider_tasks",
    "task_note",
    "tool_calls_in",
    "turn_messages",
]

#: ``state.shared`` slot: the rendered task board for this turn (Stage 3 shows it every iteration).
BOARD_KEY = "memory.task_board"
#: ``state.shared`` slot: id of this conversation's current task ("" when none).
CURRENT_KEY = "memory.task_current"

#: active = being worked on · blocked = waiting on someone/something · open = not worked on now ·
#: done = goal met · dropped = abandoned.
STATUSES = ("active", "blocked", "open", "done", "dropped")
OPEN_STATUSES = ("active", "blocked", "open")
PLAN_STATUSES = ("pending", "in_progress", "done")

#: Notes the store renders tasks into, so memory search finds them.
TASK_NOTE_CATEGORY = "tasks"

#: Board sizes. The board is a share of the model's window, not a fixed number (see load_board).
OTHER_TASKS_SHOWN = 8
RECENT_EVENTS_ON_CARD = 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short(text: Any, limit: int) -> str:
    """One line, at most ``limit`` chars, with an ellipsis when cut (titles and one-line entries)."""
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: max(1, limit - 1)] + "…"


def _excerpt(text: Any, limit: int) -> str:
    """A body cut to ``limit`` chars says that it is an excerpt."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} …[excerpt: {limit:,} of {len(text):,} chars]"


@dataclass
class Task:
    id: str
    title: str
    goal: str = ""
    done_when: str = ""
    status: str = "active"
    plan: List[Dict[str, str]] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    next_step: str = ""
    project: str = ""
    #: "model" (the agent opened it with the Task tool) or "host" (opened from a turn's signals).
    origin: str = "model"
    #: The conversation where this is the current task ("" when it is current nowhere).
    focus_session: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "done_when": self.done_when,
            "status": self.status,
            "plan": [dict(step) for step in self.plan],
            "decisions": list(self.decisions),
            "files": list(self.files),
            "next_step": self.next_step,
            "project": self.project,
            "origin": self.origin,
            "focus_session": self.focus_session,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        def _list(key: str) -> list:
            value = data.get(key)
            return list(value) if isinstance(value, (list, tuple)) else []

        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or ""),
            goal=str(data.get("goal") or ""),
            done_when=str(data.get("done_when") or ""),
            status=str(data.get("status") or "active"),
            plan=[
                {"text": str(s.get("text") or ""), "status": str(s.get("status") or "pending")}
                for s in _list("plan")
                if isinstance(s, dict)
            ],
            decisions=[str(d) for d in _list("decisions")],
            files=[str(f) for f in _list("files")],
            next_step=str(data.get("next_step") or ""),
            project=str(data.get("project") or ""),
            origin=str(data.get("origin") or "model"),
            focus_session=str(data.get("focus_session") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class TaskEvent:
    """Something that happened on a task. ``kind``: opened · updated · switched · turn · plan ·
    checkpoint · note · closed."""

    task_id: str
    kind: str
    summary: str = ""
    session_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    at: str = ""
    seq: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "summary": self.summary,
            "session_id": self.session_id,
            "data": dict(self.data),
            "at": self.at,
            "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskEvent":
        return cls(
            task_id=str(data.get("task_id") or ""),
            kind=str(data.get("kind") or "note"),
            summary=str(data.get("summary") or ""),
            session_id=str(data.get("session_id") or ""),
            data=dict(data.get("data") or {}),
            at=str(data.get("at") or ""),
            seq=int(data.get("seq") or 0),
        )


@runtime_checkable
class TaskStore(Protocol):
    """Tasks of one person (the host scopes the store). Every method is async and never raises for
    a missing task: it returns ``None`` / an empty list."""

    async def create(self, task: Task) -> Task: ...

    async def get(self, task_id: str) -> Optional[Task]: ...

    async def save(self, task: Task) -> Task:
        """Write every field of ``task`` (the store stamps ``updated_at``)."""
        ...

    async def list(
        self, *, statuses: Optional[Sequence[str]] = None, limit: int = 50
    ) -> List[Task]:
        """Most recently updated first."""
        ...

    async def count(self, *, statuses: Optional[Sequence[str]] = None) -> int: ...

    async def current(self, session_id: str) -> Optional[Task]:
        """The open task whose ``focus_session`` is this conversation (the newest if several)."""
        ...

    async def focus(self, task_id: str, session_id: str) -> Optional[Task]:
        """Make ``task_id`` the current task of ``session_id`` (and no other task current there)."""
        ...

    async def append_event(self, event: TaskEvent) -> TaskEvent: ...

    async def events(
        self, task_id: str, *, limit: int = 20, before_seq: int = 0
    ) -> List[TaskEvent]:
        """Newest first; ``before_seq`` pages back (0 = from the newest)."""
        ...

    async def count_events(self, task_id: str) -> int: ...


def provider_tasks(provider: Any) -> Optional["TaskStore"]:
    """The provider's TASK layer, or None (the layer is optional)."""
    getter = getattr(provider, "tasks", None)
    if not callable(getter):
        return None
    try:
        store = getter()
    except Exception:  # noqa: BLE001 (a provider without the layer)
        return None
    return store if store is not None else None


def new_task_id() -> str:
    return "T-" + uuid.uuid4().hex[:8]


class InMemoryTaskStore:
    """Reference :class:`TaskStore` (tests, hosts without a database)."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._events: Dict[str, List[TaskEvent]] = {}
        self._seq = 0
        self._lock = asyncio.Lock()

    async def create(self, task: Task) -> Task:
        async with self._lock:
            if not task.id:
                task.id = new_task_id()
            stamp = _now()
            task.created_at = task.created_at or stamp
            task.updated_at = stamp
            self._tasks[task.id] = Task.from_dict(task.to_dict())
            return Task.from_dict(task.to_dict())

    async def get(self, task_id: str) -> Optional[Task]:
        task = self._tasks.get(str(task_id or ""))
        return Task.from_dict(task.to_dict()) if task else None

    async def save(self, task: Task) -> Task:
        async with self._lock:
            task.updated_at = _now()
            self._tasks[task.id] = Task.from_dict(task.to_dict())
            return Task.from_dict(task.to_dict())

    async def list(
        self, *, statuses: Optional[Sequence[str]] = None, limit: int = 50
    ) -> List[Task]:
        rows = [t for t in self._tasks.values() if not statuses or t.status in statuses]
        rows.sort(key=lambda t: (t.updated_at, t.created_at), reverse=True)
        return [Task.from_dict(t.to_dict()) for t in rows[: max(0, limit)]]

    async def count(self, *, statuses: Optional[Sequence[str]] = None) -> int:
        return sum(1 for t in self._tasks.values() if not statuses or t.status in statuses)

    async def current(self, session_id: str) -> Optional[Task]:
        if not session_id:
            return None
        rows = [
            t
            for t in self._tasks.values()
            if t.focus_session == session_id and t.status in OPEN_STATUSES
        ]
        if not rows:
            return None
        rows.sort(key=lambda t: t.updated_at, reverse=True)
        return Task.from_dict(rows[0].to_dict())

    async def focus(self, task_id: str, session_id: str) -> Optional[Task]:
        async with self._lock:
            target = self._tasks.get(task_id)
            if target is None:
                return None
            for task in self._tasks.values():
                if task.focus_session == session_id and task.id != task_id:
                    task.focus_session = ""
            target.focus_session = session_id
            target.updated_at = _now()
            return Task.from_dict(target.to_dict())

    async def append_event(self, event: TaskEvent) -> TaskEvent:
        async with self._lock:
            self._seq += 1
            event.seq = self._seq
            event.at = event.at or _now()
            self._events.setdefault(event.task_id, []).append(TaskEvent.from_dict(event.to_dict()))
            return TaskEvent.from_dict(event.to_dict())

    async def events(
        self, task_id: str, *, limit: int = 20, before_seq: int = 0
    ) -> List[TaskEvent]:
        rows = [e for e in self._events.get(task_id, []) if not before_seq or e.seq < before_seq]
        rows.sort(key=lambda e: e.seq, reverse=True)
        return [TaskEvent.from_dict(e.to_dict()) for e in rows[: max(0, limit)]]

    async def count_events(self, task_id: str) -> int:
        return len(self._events.get(task_id, []))


# ── rendering ──────────────────────────────────────────────────────────────

_PLAN_MARK = {"done": "[done]", "in_progress": "[in progress]", "pending": "[ ]"}


def _day(stamp: str) -> str:
    return (stamp or "")[:16].replace("T", " ")


def _event_line(event: TaskEvent, session_id: str) -> str:
    where = (
        "this conversation"
        if event.session_id and event.session_id == session_id
        else (f"conversation {event.session_id}" if event.session_id else "")
    )
    head = f"{_day(event.at)} {event.kind}"
    if where:
        head += f" ({where})"
    return f"- {head}: {_short(event.summary, 240)}" if event.summary else f"- {head}"


def render_card(
    task: Task,
    events: Sequence[TaskEvent] = (),
    *,
    session_id: str = "",
    total_events: int = 0,
    max_chars: int = 0,
) -> str:
    """A task as the agent reads it. ``events`` newest first. Nothing is cut silently: past
    ``max_chars`` the oldest list entries give way to a marker naming TaskRecall."""
    opener = "opened by the host from a turn" if task.origin == "host" else "opened by you"
    lines = [
        f"[{task.id}] {task.title} · {task.status}",
        f"({opener} {_day(task.created_at)}, last updated {_day(task.updated_at)})",
    ]
    if task.project:
        lines.append(f"Project: {task.project}")
    if task.goal:
        lines.append(f"Goal: {task.goal}")
    if task.done_when:
        lines.append(f"Done when: {task.done_when}")

    def _section(title: str, items: List[str]) -> List[str]:
        return [f"{title}:"] + items if items else []

    plan = [
        f"  {i}. {_PLAN_MARK.get(step.get('status', 'pending'), '[ ]')} {step.get('text', '')}"
        for i, step in enumerate(task.plan, start=1)
    ]
    decisions = [f"  - {d}" for d in task.decisions]
    files = [f"  - {f}" for f in task.files]
    recent = [_event_line(e, session_id) for e in reversed(list(events)[:RECENT_EVENTS_ON_CARD])]
    body_fixed = list(lines)
    tail = [f"Next step: {task.next_step}"] if task.next_step else []
    more_events = max(0, (total_events or len(events)) - len(recent))

    def _assemble(dec: List[str], fil: List[str], rec: List[str], cut_note: str) -> str:
        parts = list(body_fixed)
        parts += _section("Plan", plan)
        parts += _section("Decisions", dec)
        parts += _section("Files", fil)
        parts += tail
        if rec:
            head = "Recent work (oldest first)"
            if more_events:
                head += f'; {more_events} earlier event(s) in TaskRecall(task_id="{task.id}", view="history")'
            parts += [head + ":"] + rec
        if cut_note:
            parts.append(cut_note)
        return "\n".join(parts)

    text = _assemble(decisions, files, recent, "")
    if not max_chars or len(text) <= max_chars:
        return text
    # Over budget: keep the newest decisions/files/events, name what was left out and where it is.
    dec, fil, rec = list(decisions), list(files), list(recent)
    left = {"decisions": 0, "files": 0, "events": 0}
    while len(text) > max_chars and (dec or fil or rec):
        if rec and len(rec) >= max(len(dec), len(fil)):
            rec.pop(0)
            left["events"] += 1
        elif dec and len(dec) >= len(fil):
            dec.pop(0)
            left["decisions"] += 1
        elif fil:
            fil.pop(0)
            left["files"] += 1
        what = ", ".join(f"{n} older {k}" for k, n in left.items() if n)
        note = (
            f'[{what} left out here to fit; TaskRecall(task_id="{task.id}") shows the whole task]'
        )
        text = _assemble(dec, fil, rec, note)
    return text


_BOARD_GUIDE = (
    "A task is work that takes more than one step or turn. If this message continues the current "
    'task, keep working on it and keep it up to date with Task(action="update") as you go (plan '
    'progress, decisions, files, next step). If it starts new multi-step work, Task(action="open"). '
    'If it returns to another open task, Task(action="switch", task_id=...). When the goal is met, '
    'Task(action="close"). A one-step question needs no task. TaskRecall(task_id=...) shows a '
    "task's whole record, including turns that are no longer in your context."
)


def render_board(
    current: Optional[Task],
    current_events: Sequence[TaskEvent],
    others: Sequence[Task],
    *,
    session_id: str = "",
    total_events: int = 0,
    total_open: int = 0,
    max_chars: int = 0,
) -> str:
    parts = ["# Tasks"]
    if current is not None:
        card_room = max(0, max_chars - 1200) if max_chars else 0
        parts.append(
            "## Current task in this conversation\n"
            + render_card(
                current,
                current_events,
                session_id=session_id,
                total_events=total_events,
                max_chars=card_room,
            )
        )
    else:
        parts.append("No task is current in this conversation.")
    shown = [t for t in others if current is None or t.id != current.id][:OTHER_TASKS_SHOWN]
    if shown:
        rows = []
        for t in shown:
            row = (
                f"- [{t.id}] {_short(t.title, 80)} · {t.status} · last updated {_day(t.updated_at)}"
            )
            if t.next_step:
                row += f" · next: {_short(t.next_step, 120)}"
            rows.append(row)
        more = max(0, total_open - len(shown) - (1 if current is not None else 0))
        if more:
            rows.append(f'(+{more} more open task(s): Task(action="list"))')
        parts.append("## Other open tasks\n" + "\n".join(rows))
    parts.append(_BOARD_GUIDE)
    return "\n\n".join(parts)


async def load_board(
    store: TaskStore, session_id: str, *, max_chars: int = 0
) -> Tuple[str, Optional[Task]]:
    """(board text, current task or None) for the start of a turn."""
    current = await store.current(session_id)
    events: List[TaskEvent] = []
    total_events = 0
    if current is not None:
        events = await store.events(current.id, limit=RECENT_EVENTS_ON_CARD)
        total_events = await store.count_events(current.id)
    others = await store.list(statuses=OPEN_STATUSES, limit=OTHER_TASKS_SHOWN + 1)
    total_open = await store.count(statuses=OPEN_STATUSES)
    board = render_board(
        current,
        events,
        others,
        session_id=session_id,
        total_events=total_events,
        total_open=total_open,
        max_chars=max_chars,
    )
    return board, current


def task_note(task: Task, events: Sequence[TaskEvent] = ()) -> Tuple[str, str, str]:
    """(filename, title, body) of the note that makes a task findable by memory search."""
    body = render_card(task, events, total_events=len(events))
    return f"task-{task.id}.md", f"Task: {task.title}", body


# ── host side: what a turn did to the task ─────────────────────────────────


def turn_messages(messages: Sequence[Any]) -> List[Any]:
    """This turn's messages: from the last message that opens a turn (the person's words, not a tool
    result or a runtime directive) to the end."""
    from xgen_agent_runtime.memory.transcript import opens_turn

    msgs = list(messages or [])
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, dict) and opens_turn(str(m.get("role") or ""), m.get("content")):
            return msgs[i:]
    return msgs


def tool_calls_in(messages: Sequence[Any]) -> int:
    count = 0
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            count += sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
    return count


def plan_from_todos(todos: Any) -> List[Dict[str, str]]:
    """TodoWrite's list (``executor.todos``) as a task plan."""
    out: List[Dict[str, str]] = []
    for item in todos or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content") or "").strip()
        if not text:
            continue
        status = str(item.get("status") or "pending")
        out.append({"text": text, "status": "done" if status == "completed" else status})
    return out


def _unfinished_plan(plan: Sequence[Dict[str, str]]) -> bool:
    return len(plan) >= 2 and any(step.get("status") != "done" for step in plan)


def _merge_files(existing: Sequence[str], new: Iterable[str]) -> List[str]:
    out = list(existing)
    for path in new:
        path = str(path or "").strip()
        if path and path not in out:
            out.append(path)
    return out


async def record_turn(
    store: TaskStore,
    *,
    session_id: str,
    request: str,
    answer: str,
    outcome: str,
    tool_calls: int = 0,
    tool_failures: int = 0,
    files: Sequence[str] = (),
    todos: Any = None,
    current_id: str = "",
    task_tool_used: bool = False,
) -> Optional[Task]:
    """Add what this turn did to the conversation's current task (no model call).

    ``outcome``: ok · partial · failed · stopped (the execution card's words). A turn is recorded on
    the current task when it did work (used tools) or the agent touched the task (``Task`` tool):
    a quick unrelated question in the middle of a task is not the task's history. When there is no
    current task and the turn was clearly multi-step work, opens one from the request (origin
    "host"): a plan of two or more steps not all done, or a turn that ended before it finished
    after using tools. Returns the task the turn was recorded on, or None.
    """
    task = await store.current(session_id)
    if task is None and current_id:
        task = await store.get(current_id)
        if task is not None and task.status not in OPEN_STATUSES:
            task = None
    if task is not None and tool_calls <= 0 and not task_tool_used:
        return None
    plan = plan_from_todos(todos)
    unfinished = outcome in ("failed", "stopped") and tool_calls > 0
    if task is None:
        if not (_unfinished_plan(plan) or unfinished):
            return None
        first = next((ln.strip() for ln in str(request or "").splitlines() if ln.strip()), "")
        if not first:
            return None
        task = await store.create(
            Task(
                id=new_task_id(),
                title=_short(first, 80),
                goal=_excerpt(request, 1_000),
                status="active",
                plan=plan,
                origin="host",
                focus_session=session_id,
            )
        )
        await store.focus(task.id, session_id)
        await store.append_event(
            TaskEvent(
                task_id=task.id,
                kind="opened",
                session_id=session_id,
                summary="opened by the host: "
                + (
                    "the turn planned several steps"
                    if _unfinished_plan(plan)
                    else f"the turn {outcome} before it finished"
                ),
            )
        )
    if plan:
        task.plan = plan
    task.files = _merge_files(task.files, files)
    if task.status == "open":
        task.status = "active"  # worked on in this turn
    task = await store.save(task)  # also stamps updated_at: the task was worked on
    request_line = _short(request, 160)
    answer_line = _short(answer, 240)
    summary = f'"{request_line}"'
    if answer_line:
        summary += f" → {answer_line}"
    await store.append_event(
        TaskEvent(
            task_id=task.id,
            kind="turn",
            session_id=session_id,
            summary=summary,
            data={
                "request": _excerpt(request, 2_000),
                "answer": _excerpt(answer, 2_000),
                "outcome": outcome,
                "tool_calls": int(tool_calls),
                "tool_failures": int(tool_failures),
                "files": list(files)[:50],
            },
        )
    )
    return task
