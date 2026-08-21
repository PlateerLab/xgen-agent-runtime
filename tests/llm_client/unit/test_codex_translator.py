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

        assert [e["type"] for e in events] == ["thinking_delta", "text_delta"]
        response = accum.finalize()
        assert response.text == "answer"
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
