"""Codex CLI wire translation — argv builder + event accumulator."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from xgen_agent_runtime.llm_client.translators._codex import (
    CodexEventAccumulator,
    codex_argv,
    codex_mcp_overrides,
    parse_codex_output_to_response,
)
from xgen_agent_runtime.llm_client.types import APIRequest


def _request(**kwargs) -> APIRequest:
    base = dict(model="gpt-5.3-codex", messages=[{"role": "user", "content": "hi"}])
    base.update(kwargs)
    return APIRequest(**base)


class TestArgv:
    def test_basic_shape(self):
        argv = codex_argv(_request())
        assert argv[0] == "exec"
        assert "--json" in argv
        assert "--skip-git-repo-check" in argv
        assert argv[-1] == "-"  # prompt travels over stdin, never argv
        i = argv.index("-m")
        assert argv[i + 1] == "gpt-5.3-codex"
        i = argv.index("--sandbox")
        assert argv[i + 1] == "workspace-write"

    def test_resume_uses_the_subcommand(self):
        argv = codex_argv(
            _request(session_hint={"session_id": "thr_1", "resume": True})
        )
        assert argv[:3] == ["exec", "resume", "thr_1"]

    def test_bypass_replaces_sandbox_flag(self):
        argv = codex_argv(_request(), bypass_sandbox=True)
        assert "--dangerously-bypass-approvals-and-sandbox" in argv
        assert "--sandbox" not in argv

    def test_output_schema_flag(self):
        argv = codex_argv(_request(), output_schema_path="/tmp/s.json")
        i = argv.index("--output-schema")
        assert argv[i + 1] == "/tmp/s.json"

    def test_reasoning_effort_from_thinking(self):
        argv = codex_argv(_request(thinking={"type": "adaptive", "effort": "high"}))
        assert '-c' in argv
        assert 'model_reasoning_effort="high"' in argv


class TestMcpOverrides:
    def test_servers_become_toml_config_overrides(self):
        overrides = codex_mcp_overrides(
            {
                "mcpServers": {
                    "connector": {
                        "command": "python3",
                        "args": ["/srv/bridge.py", "--flag"],
                        "env": {"TOKEN": "t1"},
                    }
                }
            }
        )
        joined = " ".join(overrides)
        assert 'mcp_servers.connector.command="python3"' in joined
        assert 'mcp_servers.connector.args=["/srv/bridge.py", "--flag"]' in joined
        assert 'mcp_servers.connector.env={TOKEN = "t1"}' in joined

    def test_hostile_server_name_is_sanitized(self):
        overrides = codex_mcp_overrides(
            {"mcpServers": {'x"]\ninject': {"command": "c"}}}
        )
        joined = " ".join(overrides)
        assert "\n" not in joined and '"]' not in joined.split("=")[0]

    def test_non_dict_inputs_yield_nothing(self):
        assert codex_mcp_overrides(None) == []
        assert codex_mcp_overrides({"mcpServers": "nope"}) == []


class TestAccumulator:
    def test_item_events_build_the_response(self):
        accum = CodexEventAccumulator(model="gpt-5.3-codex", cli_version="1.0.0")
        events = []
        for line in (
            {"type": "thread.started", "thread_id": "thr_9"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"item_type": "reasoning", "text": "think"}},
            {"type": "item.completed", "item": {"item_type": "command_execution", "command": "ls"}},
            {"type": "item.completed", "item": {"item_type": "agent_message", "text": "answer"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 3}},
        ):
            events.extend(accum.feed(line))

        # command_execution is a CLI-internal tool item: surfaced as a
        # tool_use/tool_result pair (audit #26), never as response content.
        assert [e["type"] for e in events] == [
            "thinking_delta",
            "tool_use",
            "tool_result",
            "text_delta",
        ]
        response = accum.finalize()
        assert response.text == "answer"
        assert [b.type for b in response.content] == ["thinking", "text"]
        assert response.usage.input_tokens == 10
        assert response.usage.cache_read_input_tokens == 4
        assert response.usage.output_tokens == 3
        assert response.raw["session_id"] == "thr_9"
        assert accum.unknown_line_count == 0

    def test_legacy_msg_envelope_is_understood(self):
        accum = CodexEventAccumulator(model="m")
        events = list(
            accum.feed({"id": "0", "msg": {"type": "agent_message", "message": "hello"}})
        )
        assert events == [{"type": "text_delta", "text": "hello"}]
        assert accum.finalize().text == "hello"

    def test_unknown_and_malformed_are_counted_not_raised(self):
        accum = CodexEventAccumulator(model="m")
        list(accum.feed({"type": "totally.new.event"}))
        list(accum.feed({"__malformed__": "garbage"}))
        assert accum.unknown_line_count == 1
        assert accum.malformed_line_count == 1
        assert accum.first_unknown_type == "totally.new.event"

    def test_json_answer_is_surfaced_on_the_structured_channel(self):
        accum = CodexEventAccumulator(model="m")
        list(
            accum.feed(
                {"type": "item.completed", "item": {"item_type": "agent_message", "text": '{"a": 1}'}}
            )
        )
        response = accum.finalize()
        assert response.structured == {"a": 1}


def test_oneshot_parse_shares_the_vocabulary():
    stdout = b"\n".join(
        [
            b'{"type":"thread.started","thread_id":"t1"}',
            b'{"type":"item.completed","item":{"item_type":"agent_message","text":"done"}}',
            b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}',
        ]
    )
    response = parse_codex_output_to_response(stdout, model="m", cli_version="v")
    assert response.text == "done"
    assert response.usage.output_tokens == 2
    assert response.raw["cli_version"] == "v"


# ---------------------------------------------------------------------------
# audit #26 — CLI-internal tool items → canonical tool_use / tool_result
# (same event shapes the Claude CLI accumulator emits, so Stage 6 produces
# api.cli_tool_call / api.tool_result with source="cli" for Codex too)
# ---------------------------------------------------------------------------

# Recorded-style ``codex exec --json`` frames (codex-cli ≥ 0.40 vocabulary:
# ``item.{started,completed}`` with a flattened ``{"id", "type", ...}`` item).
_CMD_STARTED = {
    "type": "item.started",
    "item": {
        "id": "item_1",
        "type": "command_execution",
        "command": "bash -lc 'ls /tmp/x'",
        "aggregated_output": "",
        "exit_code": None,
        "status": "in_progress",
    },
}
_CMD_UPDATED = {
    "type": "item.updated",
    "item": {
        "id": "item_1",
        "type": "command_execution",
        "command": "bash -lc 'ls /tmp/x'",
        "aggregated_output": "a.txt\n",
        "exit_code": None,
        "status": "in_progress",
    },
}
_CMD_COMPLETED = {
    "type": "item.completed",
    "item": {
        "id": "item_1",
        "type": "command_execution",
        "command": "bash -lc 'ls /tmp/x'",
        "aggregated_output": "a.txt\nb.txt\n",
        "exit_code": 0,
        "status": "completed",
    },
}
_MCP_STARTED = {
    "type": "item.started",
    "item": {
        "id": "item_2",
        "type": "mcp_tool_call",
        "server": "connector",
        "tool": "memory_search",
        "arguments": {"query": "x"},
        "status": "in_progress",
    },
}
_MCP_COMPLETED = {
    "type": "item.completed",
    "item": {
        "id": "item_2",
        "type": "mcp_tool_call",
        "server": "connector",
        "tool": "memory_search",
        "arguments": {"query": "x"},
        "result": {"content": [{"type": "text", "text": "hit"}]},
        "status": "completed",
    },
}


class TestToolItemsSurfaceAsCanonicalEvents:
    def test_command_execution_pair(self):
        accum = CodexEventAccumulator(model="m")
        events = []
        for line in (_CMD_STARTED, _CMD_UPDATED, _CMD_COMPLETED):
            events.extend(accum.feed(line))
        assert events == [
            {
                "type": "tool_use",
                "id": "item_1",
                "name": "Bash",
                "input": {"command": "bash -lc 'ls /tmp/x'"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "item_1",
                "content": "a.txt\nb.txt\n",
                "is_error": False,
            },
        ]
        assert accum.unknown_line_count == 0

    def test_failed_command_is_error_with_exit_code(self):
        accum = CodexEventAccumulator(model="m")
        failed = dict(_CMD_COMPLETED)
        failed["item"] = {**_CMD_COMPLETED["item"], "aggregated_output": "nope", "exit_code": 2, "status": "failed"}
        events = list(accum.feed(_CMD_STARTED)) + list(accum.feed(failed))
        result = events[-1]
        assert result["type"] == "tool_result" and result["is_error"] is True
        assert result["content"] == "nope\nExit code: 2"

    def test_mcp_tool_call_uses_claude_mcp_naming_and_keeps_result_blocks(self):
        accum = CodexEventAccumulator(model="m")
        events = list(accum.feed(_MCP_STARTED)) + list(accum.feed(_MCP_COMPLETED))
        assert events[0] == {
            "type": "tool_use",
            "id": "item_2",
            "name": "mcp__connector__memory_search",
            "input": {"query": "x"},
        }
        assert events[1] == {
            "type": "tool_result",
            "tool_use_id": "item_2",
            "content": [{"type": "text", "text": "hit"}],
            "is_error": False,
        }

    def test_mcp_error_becomes_error_result(self):
        accum = CodexEventAccumulator(model="m")
        failed = {
            "type": "item.completed",
            "item": {**_MCP_STARTED["item"], "error": {"message": "boom"}, "status": "failed"},
        }
        events = list(accum.feed(_MCP_STARTED)) + list(accum.feed(failed))
        assert events[-1] == {
            "type": "tool_result",
            "tool_use_id": "item_2",
            "content": "boom",
            "is_error": True,
        }

    def test_file_change_and_web_search(self):
        accum = CodexEventAccumulator(model="m")
        events = list(
            accum.feed(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_3",
                        "type": "file_change",
                        "changes": [{"path": "a.py", "kind": "update"}, {"path": "b.py", "kind": "add"}],
                        "status": "completed",
                    },
                }
            )
        )
        events += list(
            accum.feed(
                {"type": "item.completed", "item": {"id": "item_4", "type": "web_search", "query": "codex"}}
            )
        )
        assert [e["type"] for e in events] == ["tool_use", "tool_result", "tool_use", "tool_result"]
        assert events[0]["name"] == "ApplyPatch"
        assert events[0]["input"] == {"changes": [{"path": "a.py", "kind": "update"}, {"path": "b.py", "kind": "add"}]}
        assert events[1]["content"] == "update: a.py\nadd: b.py"
        assert events[2] == {"type": "tool_use", "id": "item_4", "name": "WebSearch", "input": {"query": "codex"}}
        assert events[3]["tool_use_id"] == "item_4" and events[3]["is_error"] is False

    def test_completed_without_started_synthesises_the_pair(self):
        """Older CLIs emit only item.completed (and sometimes no id): the
        tool_use is announced right before the tool_result so consumers
        pairing by id stay intact."""
        accum = CodexEventAccumulator(model="m")
        events = list(
            accum.feed(
                {"type": "item.completed", "item": {"item_type": "command_execution", "command": "ls"}}
            )
        )
        assert [e["type"] for e in events] == ["tool_use", "tool_result"]
        assert events[0]["name"] == "Bash" and events[0]["input"] == {"command": "ls"}
        assert events[0]["id"] == events[1]["tool_use_id"] == "codex_item_1"
        assert events[1]["is_error"] is False

    def test_other_items_stay_silent_and_known(self):
        accum = CodexEventAccumulator(model="m")
        assert list(accum.feed({"type": "item.started", "item": {"id": "i", "type": "agent_message"}})) == []
        assert list(accum.feed({"type": "item.started", "item": {"id": "i", "type": "todo_list"}})) == []
        assert list(accum.feed({"type": "item.completed", "item": {"id": "i", "type": "todo_list", "items": []}})) == []
        assert accum.unknown_line_count == 0

    def test_tool_events_never_enter_the_response_content(self):
        accum = CodexEventAccumulator(model="m")
        for line in (_CMD_STARTED, _CMD_COMPLETED, _MCP_STARTED, _MCP_COMPLETED):
            list(accum.feed(line))
        list(accum.feed({"type": "item.completed", "item": {"id": "x", "type": "agent_message", "text": "done"}}))
        response = accum.finalize()
        assert [b.type for b in response.content] == ["text"]
        assert response.text == "done"
