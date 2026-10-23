"""The ``Task`` and ``TaskRecall`` tools over a provider's TASK layer (memory.tasks, 2026-10-23).

``Task`` is how the agent says what it is working on: open a task for multi-step work, keep it up
to date (plan progress, decisions, files, next step), switch to another open task, close it when
the goal is met. Each change rebuilds the task board the next model call reads, so the agent sees
its own update at once, and renders the task into a note so memory search finds it.

``TaskRecall`` reads a task's whole record: the card without any cut, its history page by page, and
one turn's request and answer, with the turn word for word when the conversation's record can be
read (this conversation: ConversationHistory; another one: the provider's ``session_stm``).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.memory.tasks import (
    BOARD_KEY,
    CURRENT_KEY,
    OPEN_STATUSES,
    PLAN_STATUSES,
    Task,
    TaskEvent,
    load_board,
    new_task_id,
    provider_tasks,
    render_card,
    task_note,
)

__all__ = [
    "TASK_TOOL_NAME",
    "TASK_RECALL_TOOL_NAME",
    "board_budget_chars",
    "build_task_tools",
    "sync_task_note",
]

TASK_TOOL_NAME = "Task"
TASK_RECALL_TOOL_NAME = "TaskRecall"
#: The board's share of the model's window (tokens), at least this many chars.
BOARD_WINDOW_SHARE = 0.05
BOARD_MIN_CHARS = 3_000
HISTORY_PAGE = 20


def board_budget_chars(state: Any) -> int:
    """How much of the window the task board may take: a share of it, never a fixed small number."""
    try:
        from xgen_agent_runtime.core.compaction import window_of
        from xgen_agent_runtime.core.text_tokens import chars_for_tokens

        return max(BOARD_MIN_CHARS, chars_for_tokens(int(window_of(state) * BOARD_WINDOW_SHARE)))
    except Exception:  # noqa: BLE001
        return BOARD_MIN_CHARS


def _session(ctx: Any) -> str:
    state = getattr(ctx, "state_view", None)
    return str(getattr(ctx, "session_id", "") or getattr(state, "session_id", "") or "")


def _plan(value: Any) -> Optional[List[Dict[str, str]]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = [line for line in value.splitlines() if line.strip()]
    if not isinstance(value, list):
        return None
    out: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
            status = str(item.get("status") or "pending").strip().lower()
        else:
            text, status = str(item or "").strip(), "pending"
        if status == "completed":
            status = "done"
        if text:
            out.append({"text": text, "status": status if status in PLAN_STATUSES else "pending"})
    return out


def _strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v or "").strip()]
    return []


async def sync_task_note(provider: Any, store: Any, task: Task) -> None:
    """The task as a note in category ``tasks`` (memory search, the memory screen). Best-effort.

    Tasks belong to one person; a provider whose notes are shared by everyone using the agent says
    ``task_notes = False`` and gets no task notes (its tasks are found through the board and
    ``Task(action="list", query=...)`` instead)."""
    if not getattr(provider, "task_notes", True):
        return
    try:
        from xgen_agent_runtime.memory.provider import NoteDraft, NotePatch

        events = await store.events(task.id, limit=50)
        filename, title, body = task_note(task, events)
        notes = provider.notes()
        existing = await notes.read(filename)
        if existing is None:
            await notes.write(
                NoteDraft(
                    title=title,
                    body=body,
                    category="tasks",
                    tags=["task", task.status],
                    filename=filename,
                    frontmatter={"task_id": task.id, "status": task.status},
                )
            )
        else:
            await notes.update(filename, NotePatch(body=body, tags=["task", task.status]))
    except Exception:  # noqa: BLE001 (the table is the record; the note is a view of it)
        return


def build_task_tools(provider: Any) -> List[Any]:
    """The two tools, or none when the provider has no TASK layer."""
    store = provider_tasks(provider)
    if store is None:
        return []

    from xgen_agent_runtime.tools import ToolResult, build_tool
    from xgen_agent_runtime.tools.base import ToolCapabilities

    def _err(message: str) -> ToolResult:
        return ToolResult(content=message, is_error=True)

    async def _board(ctx: Any, session_id: str) -> Dict[str, Any]:
        state = getattr(ctx, "state_view", None)
        board, current = await load_board(store, session_id, max_chars=board_budget_chars(state))
        return {BOARD_KEY: board, CURRENT_KEY: current.id if current else ""}

    async def _done(ctx: Any, session_id: str, task: Task, message: str) -> ToolResult:
        await sync_task_note(provider, store, task)
        events = await store.events(task.id, limit=4)
        card = render_card(
            task, events, session_id=session_id, total_events=await store.count_events(task.id)
        )
        return ToolResult(
            content=f"{message}\n\n{card}", state_mutations=await _board(ctx, session_id)
        )

    async def _resolve(task_id: str, session_id: str) -> Optional[Task]:
        if task_id:
            return await store.get(task_id)
        return await store.current(session_id)

    async def _leave_current(session_id: str, keep: str) -> None:
        """The task that was current here stops being worked on (unless it waits on something)."""
        prev = await store.current(session_id)
        if prev is not None and prev.id != keep and prev.status == "active":
            prev.status = "open"
            await store.save(prev)
            await store.append_event(
                TaskEvent(
                    task_id=prev.id, kind="switched", session_id=session_id, summary="set aside"
                )
            )
            await sync_task_note(provider, store, prev)

    async def _task(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        session_id = _session(ctx)
        action = str(tool_input.get("action") or "").strip().lower()
        task_id = str(tool_input.get("task_id") or "").strip()

        if action == "open":
            title = str(tool_input.get("title") or "").strip()
            if not title:
                return _err("title is required to open a task")
            await _leave_current(session_id, keep="")
            task = await store.create(
                Task(
                    id=new_task_id(),
                    title=title,
                    goal=str(tool_input.get("goal") or "").strip(),
                    done_when=str(tool_input.get("done_when") or "").strip(),
                    status="active",
                    plan=_plan(tool_input.get("plan")) or [],
                    next_step=str(tool_input.get("next_step") or "").strip(),
                    project=str(tool_input.get("project") or "").strip(),
                    origin="model",
                    focus_session=session_id,
                )
            )
            task = await store.focus(task.id, session_id) or task
            await store.append_event(
                TaskEvent(task_id=task.id, kind="opened", session_id=session_id, summary=title)
            )
            return await _done(
                ctx,
                session_id,
                task,
                f"Opened {task.id}; it is the current task in this conversation.",
            )

        if action == "list":
            include_done = bool(tool_input.get("include_done"))
            query = str(tool_input.get("query") or "").strip().lower()
            statuses = None if include_done or query else OPEN_STATUSES
            rows = await store.list(statuses=statuses, limit=500 if query else 100)
            total = await store.count(statuses=statuses)
            if query:
                # Words in the title, goal, decisions, files or next step ("last week's report").
                words = query.split()
                rows = [
                    t
                    for t in rows
                    if all(
                        w
                        in " ".join(
                            [t.title, t.goal, t.next_step, " ".join(t.decisions), " ".join(t.files)]
                        ).lower()
                        for w in words
                    )
                ]
                total = len(rows)
            if not rows:
                if query:
                    return ToolResult(
                        content=f"No task matches {query!r} (open and closed tasks were searched)."
                    )
                return ToolResult(content="No tasks." if include_done else "No open tasks.")
            current = await store.current(session_id)
            lines = [f"[{len(rows)} of {total} task(s), most recently updated first]"]
            for t in rows:
                mark = " (current here)" if current is not None and t.id == current.id else ""
                line = f"- [{t.id}] {t.title} · {t.status}{mark} · last updated {t.updated_at[:16].replace('T', ' ')}"
                if t.next_step:
                    line += f" · next: {t.next_step}"
                lines.append(line)
            return ToolResult(content="\n".join(lines))

        task = await _resolve(task_id, session_id)
        if task is None:
            return _err(
                f"No task {task_id!r}."
                if task_id
                else "No task is current in this conversation: pass task_id, or open one."
            )

        if action == "switch":
            await _leave_current(session_id, keep=task.id)
            if task.status in ("open", "done", "dropped"):
                reopened = task.status in ("done", "dropped")
                task.status = "active"
                task = await store.save(task)
                if reopened:
                    await store.append_event(
                        TaskEvent(
                            task_id=task.id,
                            kind="updated",
                            session_id=session_id,
                            summary="reopened",
                        )
                    )
            task = await store.focus(task.id, session_id) or task
            await store.append_event(
                TaskEvent(
                    task_id=task.id, kind="switched", session_id=session_id, summary="picked up"
                )
            )
            return await _done(
                ctx, session_id, task, f"{task.id} is now the current task in this conversation."
            )

        if action == "update":
            changed: List[str] = []
            for key in ("title", "goal", "done_when", "next_step", "project"):
                if key in tool_input and tool_input.get(key) is not None:
                    value = str(tool_input.get(key) or "").strip()
                    if value != getattr(task, key):
                        setattr(task, key, value)
                        changed.append(key)
            status = str(tool_input.get("status") or "").strip().lower()
            if status:
                if status not in ("active", "blocked", "open"):
                    return _err(
                        "status for update is active, blocked or open (use action='close' to finish)"
                    )
                if status != task.status:
                    task.status = status
                    changed.append(f"status={status}")
            plan = _plan(tool_input.get("plan"))
            if plan is not None:
                task.plan = plan
                changed.append("plan")
            step = tool_input.get("step")
            if isinstance(step, dict):
                try:
                    n = int(step.get("n") or 0)
                except (TypeError, ValueError):
                    n = 0
                step_status = str(step.get("status") or "").strip().lower()
                step_status = "done" if step_status == "completed" else step_status
                if not 1 <= n <= len(task.plan) or step_status not in PLAN_STATUSES:
                    return _err(
                        f"step needs n between 1 and {len(task.plan)} and a status ({', '.join(PLAN_STATUSES)})"
                    )
                task.plan[n - 1]["status"] = step_status
                changed.append(f"step {n}={step_status}")
            for decision in _strings(tool_input.get("decision")):
                task.decisions.append(decision)
                changed.append("decision")
            for path in _strings(tool_input.get("files")):
                if path not in task.files:
                    task.files.append(path)
                    changed.append("files")
            note = str(tool_input.get("note") or "").strip()
            if not changed and not note:
                return _err(
                    "nothing to update: give a field to change, a plan step, a decision, files or a note"
                )
            task = await store.save(task)
            summary = ", ".join(dict.fromkeys(changed))
            if note:
                summary = f"{summary}; {note}" if summary else note
            await store.append_event(
                TaskEvent(task_id=task.id, kind="updated", session_id=session_id, summary=summary)
            )
            return await _done(ctx, session_id, task, f"Updated {task.id}: {summary}.")

        if action == "close":
            outcome = str(tool_input.get("outcome") or "done").strip().lower()
            if outcome not in ("done", "dropped"):
                return _err("outcome is done or dropped")
            task.status = outcome
            task.next_step = ""
            summary = str(tool_input.get("summary") or "").strip()
            task = await store.save(task)
            await store.append_event(
                TaskEvent(
                    task_id=task.id,
                    kind="closed",
                    session_id=session_id,
                    summary=f"{outcome}: {summary}" if summary else outcome,
                )
            )
            return await _done(ctx, session_id, task, f"Closed {task.id} as {outcome}.")

        return _err("action is one of open, update, switch, close, list")

    async def _recall(tool_input: Dict[str, Any], ctx: Any) -> ToolResult:
        session_id = _session(ctx)
        task_id = str(tool_input.get("task_id") or "").strip()
        task = await _resolve(task_id, session_id)
        if task is None:
            return ToolResult(
                content=f"No task {task_id!r}."
                if task_id
                else "No task is current in this conversation: pass task_id (Task(action='list') lists them).",
                is_error=True,
            )
        view = str(tool_input.get("view") or "card").strip().lower()
        total = await store.count_events(task.id)
        if view == "card":
            events = await store.events(task.id, limit=10)
            return ToolResult(
                content=render_card(task, events, session_id=session_id, total_events=total)
            )
        if view == "history":
            try:
                before = int(tool_input.get("before_seq") or 0)
            except (TypeError, ValueError):
                before = 0
            page = await store.events(task.id, limit=HISTORY_PAGE, before_seq=before)
            if not page:
                return ToolResult(
                    content=f"[{task.id}: no events{' before ' + str(before) if before else ''}; {total} in all]"
                )
            lines = [f"[{task.id} history: {len(page)} event(s), newest first, of {total}]"]
            for e in page:
                where = (
                    "this conversation"
                    if e.session_id == session_id
                    else f"conversation {e.session_id}"
                )
                lines.append(
                    f"- #{e.seq} {e.at[:16].replace('T', ' ')} {e.kind} ({where}): {e.summary}"
                )
            oldest = page[-1].seq
            if len(page) == HISTORY_PAGE:
                lines.append(
                    f'(older: TaskRecall(task_id="{task.id}", view="history", before_seq={oldest}))'
                )
            lines.append(
                f'(one turn in full: TaskRecall(task_id="{task.id}", view="turn", seq=<#>))'
            )
            return ToolResult(content="\n".join(lines))
        if view == "turn":
            try:
                seq = int(tool_input.get("seq") or 0)
            except (TypeError, ValueError):
                seq = 0
            match = None
            before = 0
            while match is None:
                page = await store.events(task.id, limit=100, before_seq=before)
                if not page:
                    break
                match = next((e for e in page if e.seq == seq), None)
                before = page[-1].seq
            if match is None:
                return ToolResult(content=f"{task.id} has no event #{seq}.", is_error=True)
            data = match.data or {}
            parts = [
                f"[{task.id} #{match.seq} {match.kind} at {match.at}, conversation {match.session_id}]"
            ]
            if data.get("request"):
                parts.append(f"Request:\n{data['request']}")
            if data.get("answer"):
                parts.append(f"Answer:\n{data['answer']}")
            facts = {
                k: data[k] for k in ("outcome", "tool_calls", "tool_failures", "files") if k in data
            }
            if facts:
                parts.append("Facts: " + json.dumps(facts, ensure_ascii=False))
            if match.kind == "checkpoint" and data.get("summary"):
                parts.append(f"Checkpoint summary:\n{data['summary']}")
            parts.append(await _original_turn(provider, match, session_id))
            return ToolResult(content="\n\n".join(p for p in parts if p))
        return ToolResult(content="view is card, history or turn", is_error=True)

    tools = [
        build_tool(
            name=TASK_TOOL_NAME,
            description=(
                "Keep track of work that takes more than one step or turn. action='open' (title, goal, "
                "done_when, plan) starts a task and makes it current in this conversation; "
                "action='update' records progress: a plan step done (step={n, status}), a new plan, a "
                "decision, files, the next step, status active/blocked/open, or a note; "
                "action='switch' (task_id) picks up another task; action='close' (outcome done or "
                "dropped, summary) finishes it; action='list' lists open tasks (query finds any task "
                "by its words). The current task is "
                "shown to you every turn, so keep it accurate. One-step questions need no task."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "update", "switch", "close", "list"],
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Default: this conversation's current task.",
                    },
                    "title": {"type": "string"},
                    "goal": {
                        "type": "string",
                        "description": "What the user wants, in their terms.",
                    },
                    "done_when": {
                        "type": "string",
                        "description": "How you and the user will know it is done.",
                    },
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "status": {"type": "string", "enum": list(PLAN_STATUSES)},
                            },
                            "required": ["text"],
                        },
                        "description": "The whole plan (replaces the current one).",
                    },
                    "step": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Plan step number (1 = first).",
                            },
                            "status": {"type": "string", "enum": list(PLAN_STATUSES)},
                        },
                    },
                    "decision": {
                        "type": "string",
                        "description": "A decision made, with its reason.",
                    },
                    "files": {"type": "array", "items": {"type": "string"}},
                    "next_step": {"type": "string"},
                    "project": {"type": "string"},
                    "status": {"type": "string", "enum": ["active", "blocked", "open"]},
                    "note": {"type": "string"},
                    "outcome": {"type": "string", "enum": ["done", "dropped"]},
                    "summary": {"type": "string", "description": "For close: what was delivered."},
                    "include_done": {
                        "type": "boolean",
                        "description": "For list: include closed tasks.",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "For list: words to find in a task, open or closed (a task from an "
                            "earlier conversation)."
                        ),
                    },
                },
                "required": ["action"],
            },
            execute=_task,
            capabilities=ToolCapabilities(concurrency_safe=False),
        ),
        build_tool(
            name=TASK_RECALL_TOOL_NAME,
            description=(
                "Read a task's whole record: view='card' (goal, plan, decisions, files, next step, "
                "recent work), view='history' (every event, newest first, paged with before_seq), "
                "view='turn' with seq (one turn's request and answer, and that turn word for word when "
                "its conversation can be read). Use it when the task's earlier turns are no longer in "
                "your context, before relying on what you remember of them."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Default: this conversation's current task.",
                    },
                    "view": {"type": "string", "enum": ["card", "history", "turn"]},
                    "before_seq": {"type": "integer"},
                    "seq": {"type": "integer"},
                },
            },
            execute=_recall,
            capabilities=ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True),
        ),
    ]
    return tools


async def _original_turn(provider: Any, event: TaskEvent, session_id: str) -> str:
    """The turn word for word, or how to read it."""
    request = str((event.data or {}).get("request") or "")
    first = next((ln.strip() for ln in request.splitlines() if ln.strip()), "")[:60]
    if not first:
        return ""
    if event.session_id and event.session_id == session_id:
        return (
            'Word for word: ConversationHistory(view="search", query='
            + json.dumps(first, ensure_ascii=False)
            + ') then view="turn".'
        )
    reader = getattr(provider, "session_stm", None)
    if not callable(reader) or not event.session_id:
        return "This turn's conversation cannot be read from here; the request and answer above are excerpts."
    try:
        from xgen_agent_runtime.host.memory_tools import _render_turn
        from xgen_agent_runtime.memory.transcript import _blocks, group_logical_turns

        stm = reader(event.session_id)
        rows = list(await stm.recent(5000)) if stm is not None else []
        for turn in group_logical_turns(rows, 1_000_000):
            msgs = getattr(turn, "messages", []) or []
            user = next(
                (
                    " ".join(
                        b.get("text", "")
                        for b in _blocks(getattr(m, "content", ""))
                        if b.get("type") == "text"
                    )
                    for m in msgs
                    if str(getattr(m, "role", "")) == "user"
                ),
                "",
            )
            if first and first in user:
                body = _render_turn(turn)
                limit = 40_000
                if len(body) > limit:
                    body = (
                        body[:limit] + f"\n…[turn cut here: {limit:,} of {len(body):,} chars shown]"
                    )
                return f"The turn word for word (conversation {event.session_id}):\n{body}"
    except Exception:  # noqa: BLE001
        return "This turn's conversation could not be read just now; the request and answer above are excerpts."
    return "This turn was not found in its conversation's record; the request and answer above are excerpts."
