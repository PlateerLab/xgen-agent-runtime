"""OpenAI Codex CLI wire translation.

Codex-side sibling of ``_cli.py`` (which is Claude Code specific).
Covers three concerns:

* :func:`codex_argv` — canonical :class:`APIRequest` → ``codex exec``
  argument vector (JSONL output mode, model, sandbox, MCP overrides,
  session resume, structured-output schema).
* :class:`CodexEventAccumulator` — JSONL event stream → canonical
  streaming events + final :class:`APIResponse`. Mirrors the interface
  the CLI client reads from ``StreamJsonAccumulator``: ``feed()``,
  ``finalize()``, ``unknown_line_count`` / ``malformed_line_count`` /
  ``first_unknown_type``.
* :func:`parse_codex_output_to_response` — one-shot stdout → response.

Wire tolerance: the Codex JSONL vocabulary has drifted between releases
(``item.completed``-style thread events vs the older ``{"id", "msg"}``
envelopes). Both shapes are handled; anything else is counted and
reported through the same unknown-wire telemetry channel the Claude
backend uses — tolerate in prod, fail under ``strict_wire`` canaries.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from xgen_agent_runtime.core.state import TokenUsage
from xgen_agent_runtime.llm_client.translators._cli import (
    flatten_messages_to_prompt,
)
from xgen_agent_runtime.llm_client.types import APIRequest, APIResponse, ContentBlock

logger = logging.getLogger(__name__)

__all__ = [
    "codex_argv",
    "codex_mcp_overrides",
    "CodexEventAccumulator",
    "parse_codex_output_to_response",
    "flatten_messages_to_prompt",
]


def _toml_string(value: str) -> str:
    """Quote *value* as a TOML basic string (JSON escaping is a subset)."""
    return json.dumps(str(value), ensure_ascii=False)


def codex_mcp_overrides(mcp_config: Any) -> List[str]:
    """``{"mcpServers": {...}}`` → repeated ``-c mcp_servers.*`` overrides.

    Codex reads MCP servers from ``$CODEX_HOME/config.toml``; ``-c`` lets
    us inject them per-invocation **without** redirecting ``CODEX_HOME``
    (which would orphan the user's ChatGPT login stored in
    ``$CODEX_HOME/auth.json``). Values must be valid TOML — strings are
    JSON-quoted (a subset of TOML), arrays/tables composed explicitly.
    """
    if not isinstance(mcp_config, dict):
        return []
    servers = mcp_config.get("mcpServers")
    if not isinstance(servers, dict):
        return []
    args: List[str] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name))
        command = spec.get("command")
        if command:
            args += ["-c", f"mcp_servers.{safe}.command={_toml_string(command)}"]
        srv_args = spec.get("args")
        if isinstance(srv_args, (list, tuple)) and srv_args:
            rendered = ", ".join(_toml_string(a) for a in srv_args)
            args += ["-c", f"mcp_servers.{safe}.args=[{rendered}]"]
        env = spec.get("env")
        if isinstance(env, dict) and env:
            rendered = ", ".join(f"{k} = {_toml_string(v)}" for k, v in env.items() if k)
            args += ["-c", f"mcp_servers.{safe}.env={{{rendered}}}"]
    return args


def codex_argv(
    request: APIRequest,
    *,
    sandbox_mode: str = "workspace-write",
    bypass_sandbox: bool = False,
    mcp_config: Any = None,
    output_schema_path: str = "",
    extra_args: Iterable[str] = (),
) -> List[str]:
    """Build the ``codex`` argument vector for one canonical request.

    The prompt itself is NOT placed on argv — it travels over stdin
    (``-`` positional), because flattened histories routinely exceed the
    kernel's argv size limit. Session continuity uses the ``exec resume``
    subcommand with the thread id captured from a prior turn's
    ``thread.started`` event.
    """
    hint = request.session_hint or {}
    resume_id = str(hint.get("session_id") or "") if hint.get("resume") else ""

    argv: List[str] = ["exec"]
    if resume_id:
        argv += ["resume", resume_id]
    argv += ["--json", "--skip-git-repo-check"]

    if request.model:
        argv += ["-m", str(request.model)]

    if bypass_sandbox:
        # The host runs the CLI inside its own sandbox (container runner) —
        # double-sandboxing breaks tool I/O the same way Claude's --bare
        # path does. Never the default.
        argv += ["--dangerously-bypass-approvals-and-sandbox"]
    else:
        argv += ["--sandbox", sandbox_mode]

    # Reasoning effort — canonical ``thinking`` maps onto Codex's config
    # knob when the caller pinned an effort; absent that, model defaults.
    thinking = request.thinking or {}
    effort = thinking.get("effort") if isinstance(thinking, dict) else None
    if effort:
        argv += ["-c", f"model_reasoning_effort={_toml_string(effort)}"]

    if output_schema_path:
        argv += ["--output-schema", output_schema_path]

    argv += codex_mcp_overrides(mcp_config)
    argv += list(extra_args)
    # Prompt over stdin.
    argv += ["-"]
    return argv


# ── event accumulation ───────────────────────────────────────────────


def _item_of(line_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    item = line_obj.get("item")
    return item if isinstance(item, dict) else None


def _item_type(item: Dict[str, Any]) -> str:
    return str(item.get("item_type") or item.get("type") or "")


class CodexEventAccumulator:
    """Codex JSONL → canonical streaming events + final APIResponse.

    Interface-compatible with ``StreamJsonAccumulator`` where the CLI
    client reads it: ``feed(line_obj) -> Iterable[event]``,
    ``finalize() -> APIResponse`` and the three wire-drift counters.

    Codex executes its own tools inside the subprocess (same contract as
    Claude Code). Its tool items (``command_execution`` /
    ``mcp_tool_call`` / ``file_change`` / ``web_search``) never become
    response *content*, but — exactly like the Claude CLI accumulator
    (``_cli.py``: ``tool_use`` on the assistant block, ``tool_result``
    on the ``user`` envelope) — they ARE surfaced as canonical
    streaming events so Stage 6 emits ``api.tool_use`` /
    ``api.cli_tool_call`` / ``api.tool_result`` (``source="cli"``) and
    hosts show a tool timeline for Codex too:

    * ``item.started``   → ``{"type": "tool_use", "id", "name", "input"}``
    * ``item.completed`` → ``{"type": "tool_result", "tool_use_id",
      "content", "is_error"}`` (preceded by the ``tool_use`` when no
      ``item.started`` announced the item — older CLIs skip it)
    * ``item.updated``   → silent (progressive ``aggregated_output``)

    Tool names reuse the Claude CLI vocabulary where a 1:1 exists
    (``Bash``, ``WebSearch``), ``mcp__<server>__<tool>`` for MCP calls
    and ``ApplyPatch`` for file changes, so UI consumers need no
    per-backend branch. ``todo_list`` / ``patch_apply`` stay silent.
    """

    #: item types we recognise but deliberately do not surface.
    _KNOWN_SILENT_ITEMS = frozenset(
        {
            "todo_list",
            "patch_apply",
        }
    )
    #: tool-ish item types → canonical tool name (see class docstring).
    _TOOL_ITEM_NAMES: Dict[str, str] = {
        "command_execution": "Bash",
        "mcp_tool_call": "",  # composed per item: mcp__<server>__<tool>
        "file_change": "ApplyPatch",
        "web_search": "WebSearch",
    }
    #: top-level event types that carry no response content.
    _KNOWN_SILENT_EVENTS = frozenset(
        {
            "thread.started",
            "turn.started",
            "item.updated",
            "session.created",
        }
    )

    def __init__(self, *, model: str = "", cli_version: str = "") -> None:
        self._model = model
        self._cli_version = cli_version
        self._text_parts: List[str] = []
        self._thinking_parts: List[str] = []
        self._usage: Optional[TokenUsage] = None
        self._session_id: str = ""
        self._structured: Any = None
        # Tool items announced via ``item.started`` (id → name) so the
        # matching ``item.completed`` emits only the ``tool_result``; an
        # item that completes without a start gets both events.
        self._open_tools: Dict[str, str] = {}
        self._anon_tool_seq = 0
        self.unknown_line_count = 0
        self.malformed_line_count = 0
        self.first_unknown_type: Optional[str] = None

    # ── helpers ──────────────────────────────────────────────────────

    def _note_unknown(self, type_name: str) -> None:
        self.unknown_line_count += 1
        if self.first_unknown_type is None:
            self.first_unknown_type = type_name or "<missing>"

    def _usage_from(self, payload: Dict[str, Any]) -> TokenUsage:
        def _i(key: str) -> int:
            try:
                return int(payload.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        cached = _i("cached_input_tokens") or _i("cache_read_input_tokens")
        # Codex (OpenAI accounting) includes cached tokens inside
        # input_tokens; keep the raw figure — total_tokens stays
        # provider-honest (see TokenUsage docstring).
        return TokenUsage(
            input_tokens=_i("input_tokens"),
            output_tokens=_i("output_tokens"),
            cache_read_input_tokens=cached,
        )

    # ── tool items → canonical tool_use / tool_result ────────────────

    def _tool_name(self, itype: str, item: Dict[str, Any]) -> str:
        if itype == "mcp_tool_call":
            server = str(item.get("server") or "").strip()
            tool = str(item.get("tool") or "").strip()
            if server and tool:
                return f"mcp__{server}__{tool}"
            return tool or server or "mcp_tool_call"
        return self._TOOL_ITEM_NAMES.get(itype) or itype

    @staticmethod
    def _tool_input(itype: str, item: Dict[str, Any]) -> Dict[str, Any]:
        if itype == "command_execution":
            return {"command": str(item.get("command") or "")}
        if itype == "mcp_tool_call":
            args = item.get("arguments")
            return dict(args) if isinstance(args, dict) else {"arguments": args}
        if itype == "file_change":
            changes = item.get("changes")
            return {"changes": list(changes) if isinstance(changes, list) else []}
        if itype == "web_search":
            return {"query": str(item.get("query") or "")}
        return {}

    @staticmethod
    def _tool_outcome(itype: str, item: Dict[str, Any]) -> tuple[Any, bool]:
        """(content, is_error) for a completed tool item."""
        status = str(item.get("status") or "").lower()
        failed = status in ("failed", "declined", "error")
        if itype == "command_execution":
            output = str(item.get("aggregated_output") or "")
            exit_code = item.get("exit_code")
            try:
                code = int(exit_code) if exit_code is not None else 0
            except (TypeError, ValueError):
                code = 0
            if code != 0:
                output = (output + "\n" if output else "") + f"Exit code: {code}"
            return output, failed or code != 0
        if itype == "mcp_tool_call":
            error = item.get("error")
            if error:
                msg = error.get("message") if isinstance(error, dict) else error
                return str(msg or "mcp tool call failed"), True
            result = item.get("result")
            if isinstance(result, dict) and isinstance(result.get("content"), list):
                # MCP CallToolResult — keep the content blocks (text-bearing).
                return result.get("content"), failed
            return result if result is not None else "", failed
        if itype == "file_change":
            changes = item.get("changes")
            lines = []
            if isinstance(changes, list):
                for ch in changes:
                    if isinstance(ch, dict):
                        lines.append(f"{ch.get('kind') or 'update'}: {ch.get('path') or ''}")
            return "\n".join(lines), failed
        if itype == "web_search":
            return "", failed
        return "", failed

    def _tool_id(self, item: Dict[str, Any]) -> str:
        raw_id = str(item.get("id") or "").strip()
        if raw_id:
            return raw_id
        self._anon_tool_seq += 1
        return f"codex_item_{self._anon_tool_seq}"

    def _tool_started(self, itype: str, item: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        tool_id = self._tool_id(item)
        name = self._tool_name(itype, item)
        self._open_tools[tool_id] = name
        yield {
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": self._tool_input(itype, item),
        }

    def _tool_completed(self, itype: str, item: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raw_id = str(item.get("id") or "").strip()
        if not raw_id or raw_id not in self._open_tools:
            # No ``item.started`` seen (older CLIs / anonymous items) —
            # announce the call first so the pair stays intact.
            events = list(self._tool_started(itype, item))
            tool_id = events[0]["id"]
            yield from events
        else:
            tool_id = raw_id
        self._open_tools.pop(tool_id, None)
        content, is_error = self._tool_outcome(itype, item)
        yield {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content,
            "is_error": bool(is_error),
        }

    # ── the interface the client consumes ────────────────────────────

    def feed(self, line_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        if "__malformed__" in line_obj:
            self.malformed_line_count += 1
            return
        etype = str(line_obj.get("type") or "")

        # Legacy envelope: {"id": "...", "msg": {"type": ..., ...}}
        if not etype and isinstance(line_obj.get("msg"), dict):
            msg = line_obj["msg"]
            mtype = str(msg.get("type") or "")
            if mtype == "agent_message":
                text = str(msg.get("message") or "")
                if text:
                    self._text_parts.append(text)
                    yield {"type": "text_delta", "text": text}
                return
            if mtype in ("agent_reasoning", "reasoning"):
                text = str(msg.get("text") or msg.get("message") or "")
                if text:
                    self._thinking_parts.append(text)
                    yield {"type": "thinking_delta", "text": text}
                return
            if mtype == "token_count":
                self._usage = self._usage_from(msg)
                return
            if mtype in (
                "session_configured",
                "task_started",
                "task_complete",
                "exec_command_begin",
                "exec_command_end",
                "mcp_tool_call_begin",
                "mcp_tool_call_end",
            ):
                if mtype == "session_configured":
                    self._session_id = str(msg.get("session_id") or "")
                return
            self._note_unknown(f"msg.{mtype}")
            return

        if etype == "thread.started":
            self._session_id = str(line_obj.get("thread_id") or "")
            return
        if etype in self._KNOWN_SILENT_EVENTS:
            return
        if etype == "item.started":
            item = _item_of(line_obj) or {}
            itype = _item_type(item)
            if itype in self._TOOL_ITEM_NAMES:
                yield from self._tool_started(itype, item)
            # every other item kind is only interesting once completed
            return
        if etype == "item.completed":
            item = _item_of(line_obj) or {}
            itype = _item_type(item)
            if itype in self._TOOL_ITEM_NAMES:
                yield from self._tool_completed(itype, item)
                return
            if itype in ("agent_message", "assistant_message"):
                text = str(item.get("text") or "")
                if text:
                    self._text_parts.append(text)
                    yield {"type": "text_delta", "text": text}
                return
            if itype == "reasoning":
                text = str(item.get("text") or "")
                if text:
                    self._thinking_parts.append(text)
                    yield {"type": "thinking_delta", "text": text}
                return
            if itype in self._KNOWN_SILENT_ITEMS:
                return
            if itype == "error":
                # surfaced by the client (it watches for this before feed)
                return
            self._note_unknown(f"item.{itype}")
            return
        if etype in ("turn.completed", "turn.complete"):
            usage = line_obj.get("usage")
            if isinstance(usage, dict):
                self._usage = self._usage_from(usage)
            return
        if etype in ("turn.failed",):
            # client raises on the paired error frame; nothing to emit
            return
        self._note_unknown(etype)

    def finalize(self) -> APIResponse:
        content: List[ContentBlock] = []
        thinking = "".join(self._thinking_parts)
        if thinking:
            content.append(ContentBlock(type="thinking", thinking_text=thinking))
        text = "\n".join(p for p in self._text_parts if p)
        content.append(ContentBlock(type="text", text=text))

        raw: Dict[str, Any] = {
            "unknown_line_count": self.unknown_line_count,
            "malformed_line_count": self.malformed_line_count,
            "first_unknown_type": self.first_unknown_type,
        }
        if self._cli_version:
            raw["cli_version"] = self._cli_version
        if self._session_id:
            raw["session_id"] = self._session_id
        if self._structured is None and text:
            # A schema-constrained run answers with the JSON document as
            # its final message; surface it on the structured channel too.
            stripped = text.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    raw["structured_output"] = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

        return APIResponse(
            content=content,
            stop_reason="end_turn",
            usage=self._usage or TokenUsage(),
            model=self._model,
            raw=raw,
        )


def parse_codex_output_to_response(
    stdout: bytes, *, model: str = "", cli_version: str = ""
) -> APIResponse:
    """One-shot ``codex exec --json`` stdout → APIResponse.

    The one-shot wire is the same JSONL stream, just fully buffered —
    reuse the accumulator so both paths share one vocabulary.
    """
    from xgen_agent_runtime.llm_client._cli_runtime import parse_stream_json_line

    accum = CodexEventAccumulator(model=model, cli_version=cli_version)
    for raw_line in stdout.splitlines():
        line_obj = parse_stream_json_line(raw_line)
        if line_obj is None:
            continue
        for _event in accum.feed(line_obj):
            pass
    return accum.finalize()
