"""대화에 연결된 사용자 기기 폴더 — 표면·안내·기록 정리 규칙 (host.local_folders).

폴더는 대화에 붙는다. 연결된 폴더가 있는 대화에서만 폴더 도구가 보이고, 해제하면 다음
턴부터 사라지며, 그 사이 기록에 남은 옛 호출은 평문으로 바뀌어 모델이 다시 부르지 않는다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from xgen_agent_runtime.core.message_repair import (
    normalize_messages_for_request,
    retire_tool_calls_by_name,
)
from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host import local_folders as lf
from xgen_agent_runtime.host.runner import _system_builder
from xgen_agent_runtime.stages.s03_system.artifact.default.builders import TurnNotesBlock
from xgen_agent_runtime.stages.s06_api.artifact.default.stage import APIStage


def _t(name: str) -> Dict[str, str]:
    return {"name": name}


def _names(tools: List[Dict[str, str]]) -> List[str]:
    return [t["name"] for t in tools]


# ── 요청 필드 ─────────────────────────────────────────────────────


def test_absent_field_means_an_old_client_not_an_empty_list():
    assert lf.parse_local_folders(None) is None
    assert lf.parse_local_folders([]) == []


def test_folders_parse_from_dicts_strings_and_json_and_dedupe_by_path():
    got = lf.parse_local_folders(
        [
            {"id": "a", "name": "proj", "path": "/Users/me/proj"},
            {"path": "C:\\work\\docs"},
            "/Users/me/proj",  # 같은 경로 — 한 번만
            {"name": "no path"},  # 경로 없음 — 버린다
        ]
    )
    assert [(f.id, f.name, f.path) for f in got] == [
        ("a", "proj", "/Users/me/proj"),
        ("C:\\work\\docs", "docs", "C:\\work\\docs"),
    ]
    assert [f.path for f in lf.parse_local_folders('[{"path": "/Notes"}]')] == ["/Notes"]
    assert lf.parse_local_folders("not json") == []


# ── 이름 규칙 ─────────────────────────────────────────────────────


def test_device_tool_names_are_read_with_or_without_the_cli_bridge_prefix():
    assert lf.device_tool("mcp_local_Shell") == ("local", "Shell")
    assert lf.device_tool("mcp__connector__mcp_mobile_ReadFile") == ("mobile", "ReadFile")
    assert lf.device_tool("mcp_github_search") is None
    assert lf.device_tool("Bash") is None


def test_only_folder_tools_are_folder_bound_per_app():
    assert lf.is_folder_tool("mcp_local_ReadFile")
    assert lf.is_folder_tool("mcp_local_Shell")
    assert lf.is_folder_tool("mcp_local_Notify")  # 데스크톱은 PC 조작 전부가 폴더에 묶인다
    assert lf.is_folder_tool("mcp_mobile_DeleteFile")
    assert not lf.is_folder_tool("mcp_mobile_Notify")  # 모바일 알림은 도구 그룹이 정한다
    assert not lf.is_folder_tool("mcp_mobile_Shell")  # 휴대폰엔 터미널이 없다
    assert not lf.is_folder_tool("mcp_local_BrowserNavigate")
    assert not lf.is_folder_tool("mcp_local_McpAddServer")
    assert not lf.is_folder_tool("mcp_local_LocalControl")  # 옛 입구는 따로
    assert lf.is_legacy_gate("mcp_local_LocalControl")


# ── 표면 ──────────────────────────────────────────────────────────


_CATALOG = [
    _t("mcp_local_LocalControl"),
    _t("mcp_local_ReadFile"),
    _t("mcp_local_Shell"),
    _t("mcp_local_BrowserNavigate"),
    _t("mcp_local_McpListServers"),
    _t("mcp_github_search"),
]


def test_old_client_keeps_its_catalog_untouched():
    assert _names(lf.filter_device_tools(_CATALOG, None)) == _names(_CATALOG)


def test_no_folder_drops_folder_tools_and_the_old_gate_but_keeps_the_rest():
    kept = _names(lf.filter_device_tools(_CATALOG, []))
    assert kept == ["mcp_local_BrowserNavigate", "mcp_local_McpListServers", "mcp_github_search"]


def test_connected_folders_keep_folder_tools_and_drop_only_the_old_gate():
    folders = lf.parse_local_folders([{"path": "/Users/me/proj"}])
    kept = _names(lf.filter_device_tools(_CATALOG, folders))
    assert "mcp_local_LocalControl" not in kept
    assert {"mcp_local_ReadFile", "mcp_local_Shell", "mcp_local_BrowserNavigate"} <= set(kept)


def test_folder_tools_go_on_the_first_screen_only_when_a_folder_is_connected():
    folders = lf.parse_local_folders(["/p"])
    assert lf.folder_tool_is_turn_one("mcp_local_ReadFile", folders)
    assert not lf.folder_tool_is_turn_one("mcp_local_ReadFile", [])
    assert not lf.folder_tool_is_turn_one("mcp_local_ReadFile", None)
    assert not lf.folder_tool_is_turn_one("mcp_local_BrowserNavigate", folders)


# ── 턴 안내 ───────────────────────────────────────────────────────


def test_old_client_gets_no_note():
    assert lf.turn_note(None) == ""


def test_no_folder_note_says_there_are_no_device_tools_and_how_to_connect():
    note = lf.turn_note([])
    assert "No folder on the user's device is connected" in note
    assert "history only" in note
    assert "[폴더 연결]" in note


def test_connected_note_lists_folders_and_the_terminal_on_desktop():
    folders = lf.parse_local_folders([{"name": "proj", "path": "/Users/me/proj"}])
    note = lf.turn_note(
        folders, available_tools=["mcp_local_ReadFile", "mcp_local_Shell"], platform="darwin"
    )
    assert "- proj: /Users/me/proj" in note
    assert "user's Mac" in note
    assert "mcp_local_*" in note
    assert "Shell and ShellJob" in note
    assert "not in your server sandbox" in note


def test_mobile_note_says_there_is_no_terminal():
    folders = lf.parse_local_folders([{"name": "Notes", "path": "/Notes"}])
    note = lf.turn_note(
        folders, available_tools=["mcp_mobile_ReadFile", "mcp_mobile_ListDir"], platform="android"
    )
    assert "Android phone" in note
    assert "mcp_mobile_*" in note
    assert "no terminal" in note


def test_connected_but_unreachable_app_is_said_plainly():
    folders = lf.parse_local_folders(["/p"])
    note = lf.turn_note(folders, available_tools=["mcp_local_BrowserNavigate"], platform="win32")
    assert "- p: /p" in note
    assert "not reachable right now" in note


# ── 기록 정리 ─────────────────────────────────────────────────────


def _history() -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "read my notes"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "reading"},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "mcp_local_ReadFile",
                    "input": {"path": "/Users/me/proj/a.md"},
                },
                {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "# notes\nbody"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "a b"},
            ],
        },
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "now what?"},
    ]


def test_retired_calls_become_plain_text_pairs_and_other_calls_stay():
    history = _history()
    out = lf.retire_device_tool_calls(history)
    assert len(out) == len(history)
    call = out[1]["content"]
    assert call[1]["type"] == "text"
    assert call[1]["text"].startswith("[earlier call, device folder no longer connected]")
    assert "mcp_local_ReadFile" in call[1]["text"] and "a.md" in call[1]["text"]
    assert call[2]["type"] == "tool_use"  # 서버 sandbox 도구는 그대로
    result = out[2]["content"]
    assert result[0]["type"] == "text" and "# notes body" in result[0]["text"]
    assert result[1]["type"] == "tool_result"
    # 원본은 그대로(요청 사본만 바뀐다)
    assert history[1]["content"][1]["type"] == "tool_use"
    # 짝이 맞아 요청 검증을 통과한다 — 합성 결과가 끼어들지 않는다
    normalized = normalize_messages_for_request(out)
    assert [b.get("type") for b in normalized[2]["content"]] == ["text", "tool_result"]


def test_cli_bridge_names_are_retired_too():
    history = _history()
    history[1]["content"][1]["name"] = "mcp__connector__mcp_local_ReadFile"
    out = lf.retire_device_tool_calls(history)
    assert out[1]["content"][1]["type"] == "text"


def test_malformed_spec_leaves_history_alone():
    history = _history()
    assert retire_tool_calls_by_name(history, None) == history
    assert retire_tool_calls_by_name(history, {"names": []}) == history


def test_stage6_retires_only_when_the_turn_says_so():
    stage = APIStage()
    state = PipelineState()
    state.messages = _history()
    cfg = stage.resolve_model_config(state)

    sent = stage._call_kwargs(cfg, state)["messages"]
    assert sent[1]["content"][1]["type"] == "tool_use"

    state.shared[SharedKeys.RETIRED_TOOL_CALLS] = lf.retired_calls_spec()
    sent = stage._call_kwargs(cfg, state)["messages"]
    assert sent[1]["content"][1]["type"] == "text"
    assert state.messages[1]["content"][1]["type"] == "tool_use"  # 기록은 그대로


# ── 턴 안내 블록 ──────────────────────────────────────────────────


def test_turn_notes_block_is_volatile_and_renders_the_notes():
    block = TurnNotesBlock()
    state = PipelineState()
    assert block.volatile
    assert block.render(state) == ""
    state.shared[SharedKeys.TURN_NOTES] = ["# A\none", "", "# B\ntwo"]
    assert block.render(state) == "# A\none\n\n# B\ntwo"


def test_default_system_builder_puts_turn_notes_in_the_volatile_tail():
    builder = _system_builder("base prompt")
    state = PipelineState()
    state.shared[SharedKeys.TURN_NOTES] = ["# Folders on the user's device\n- p: /p"]
    parts = builder.build_parts(state)
    notes = [p for p in parts if p["name"] == "turn_notes"]
    assert notes and notes[0]["volatile"] is True
    assert "- p: /p" in notes[0]["text"]
    assert parts[0]["volatile"] is False  # base 는 캐시 접두에 남는다
