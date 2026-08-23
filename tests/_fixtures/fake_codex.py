#!/usr/bin/env python3
"""Fake ``codex`` binary for CLI-backend tests (stdlib only).

Mirror of ``fake_claude.py``'s contract: the --version handshake is
answered BEFORE scenario dispatch (the client probes it on every first
call), then ``FAKE_CODEX_SCENARIO`` selects the wire behaviour.

Scenarios:
  ok_stream   read stdin fully, then emit a normal JSONL turn.
  auth_fail   exit 1 with the CLI's not-logged-in phrase on stderr.
  echo_argv   emit one agent_message whose text is the JSON argv + stdin,
              so tests can assert exactly what reached the binary.
  ok_stream_tools
              a turn with CLI-internal tool items (command_execution +
              mcp_tool_call, item.started/item.completed pairs) — drives
              the tool_use/tool_result surfacing path.
"""

import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--version"]:
        print("codex-cli 1.0.0-fake")
        return 0

    scenario = os.environ.get("FAKE_CODEX_SCENARIO", "ok_stream")
    stdin_text = ""
    if argv and argv[-1] == "-":
        stdin_text = sys.stdin.read()

    if scenario == "auth_fail":
        print("Error: Not logged in. Please run `codex login`.", file=sys.stderr)
        return 1

    if scenario == "echo_argv":
        payload = json.dumps({"argv": argv, "stdin": stdin_text}, ensure_ascii=False)
        print(json.dumps({"type": "thread.started", "thread_id": "thr_echo"}))
        print(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"item_type": "agent_message", "text": payload},
                }
            )
        )
        print(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        )
        return 0

    if scenario == "ok_stream_tools":
        # Recorded-style ``codex exec --json`` turn with a shell command
        # and an MCP call executed inside the CLI (item.started →
        # item.completed pairs), then the final message.
        print(json.dumps({"type": "thread.started", "thread_id": "thr_tools"}))
        print(json.dumps({"type": "turn.started"}))
        print(
            json.dumps(
                {
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
            )
        )
        print(
            json.dumps(
                {
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
            )
        )
        print(
            json.dumps(
                {
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
            )
        )
        print(
            json.dumps(
                {
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
            )
        )
        print(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "item_3", "type": "agent_message", "text": "two files"},
                }
            )
        )
        print(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                }
            )
        )
        return 0

    # ok_stream (default)
    print(json.dumps({"type": "thread.started", "thread_id": "thr_1"}))
    print(json.dumps({"type": "turn.started"}))
    print(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"item_type": "reasoning", "text": "pondering"},
            }
        )
    )
    print(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"item_type": "agent_message", "text": "fake codex answer"},
            }
        )
    )
    print(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 12, "cached_input_tokens": 2, "output_tokens": 5},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
