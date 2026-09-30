from __future__ import annotations

import asyncio

from xgen_agent_runtime.core.shared_keys import SharedKeys
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.host.workspace_fast_path import (
    FAST_PATH_MAX_DIRECTORIES,
    FAST_PATH_MAX_FILES,
    SNAPSHOT_MAX_BYTES,
    flag_enabled,
    prepare_workspace_fast_path,
    register_snapshot_witnesses,
)
from xgen_agent_runtime.tools.base import ToolContext


def _prepare(tmp_path, text, *, attachments=(), enabled=True):
    context = ToolContext(working_dir=str(tmp_path), allowed_paths=[str(tmp_path)])
    return asyncio.run(
        prepare_workspace_fast_path(text, attachments, context, enabled=enabled)
    )


def test_flag_parser_does_not_treat_false_strings_as_true() -> None:
    assert flag_enabled(True) is True
    assert flag_enabled("yes") is True
    assert flag_enabled("false") is False
    assert flag_enabled(2) is False


def test_disabled_fast_path_does_not_touch_the_filesystem() -> None:
    plan = asyncio.run(
        prepare_workspace_fast_path("hello", (), object(), enabled=False)
    )
    assert plan.text == "hello"
    assert plan.active is False
    assert plan.reason == "disabled"


def test_small_complete_workspace_with_explicit_root_is_injected(tmp_path) -> None:
    (tmp_path / "INPUT").write_text("alpha", encoding="utf-8")
    (tmp_path / "data.oddity").write_text("beta", encoding="utf-8")

    plan = _prepare(tmp_path, f"Convert everything in {tmp_path} and save the result")

    assert plan.active is True
    assert plan.reason == "eligible"
    assert plan.included_paths == ("INPUT", "data.oddity")
    assert '"path":"INPUT"' in plan.text
    assert '"content":"beta"' in plan.text
    assert "normally only tool action MUST be exactly one Bash" in plan.text
    assert "not in a separate call" in plan.text
    assert "just to display the outputs again" in plan.text
    assert "standard serializer" in plan.text
    assert "structured data outputs such as JSON or CSV" in plan.text
    assert "this output must not include/mention X" in plan.text
    assert "Do not convert code or behavioral prohibitions" in plan.text


def test_relative_file_reference_is_a_generic_route_signal(tmp_path) -> None:
    (tmp_path / "records").write_text("one", encoding="utf-8")
    plan = _prepare(tmp_path, "Transform `records` into the requested output")
    assert plan.active is True


def test_structured_attachment_path_is_a_route_signal(tmp_path) -> None:
    (tmp_path / "records").write_text("one", encoding="utf-8")
    plan = _prepare(
        tmp_path,
        "Transform the attached input",
        attachments=({"workspace_path": "records"},),
    )
    assert plan.active is True


def test_plain_chat_does_not_route_even_when_a_small_workspace_exists(tmp_path) -> None:
    (tmp_path / "records").write_text("one", encoding="utf-8")
    plan = _prepare(tmp_path, "How are you today?")
    assert plan.active is False
    assert plan.reason == "path_not_referenced"
    assert plan.text == "How are you today?"


def test_workspace_root_prefix_is_not_an_exact_path_reference(tmp_path) -> None:
    (tmp_path / "records").write_text("one", encoding="utf-8")
    plan = _prepare(tmp_path, f"Use {tmp_path}-other instead")
    assert plan.active is False
    assert plan.reason == "path_not_referenced"


def test_empty_workspace_never_routes(tmp_path) -> None:
    plan = _prepare(tmp_path, f"Work in {tmp_path}")
    assert plan.active is False
    assert plan.reason == "empty_workspace"


def test_binary_large_and_many_file_workspaces_fall_back(tmp_path) -> None:
    binary = tmp_path / "binary"
    binary.write_bytes(b"a\x00b")
    assert _prepare(tmp_path, f"Use {binary}").reason == "non_text_input"
    binary.unlink()

    large = tmp_path / "large"
    large.write_bytes(b"x" * (SNAPSHOT_MAX_BYTES + 1))
    assert _prepare(tmp_path, f"Use {large}").reason == "workspace_over_budget"
    large.unlink()

    for index in range(FAST_PATH_MAX_FILES + 1):
        (tmp_path / str(index)).write_text(str(index), encoding="utf-8")
    assert _prepare(tmp_path, f"Use {tmp_path}").reason == "too_many_files"


def test_deeply_partitioned_workspace_falls_back_structurally(tmp_path) -> None:
    for index in range(FAST_PATH_MAX_DIRECTORIES + 1):
        (tmp_path / f"dir-{index}").mkdir()
    assert _prepare(tmp_path, f"Use {tmp_path}").reason == "too_many_directories"


def test_a_long_detailed_request_still_takes_the_fast_path(tmp_path) -> None:
    """요청 길이는 게이트가 아니다 — 작은 폴더에 긴 명세를 준 과제가 가장 많이 이득을 봤다."""
    (tmp_path / "input").write_text("x", encoding="utf-8")
    plan = _prepare(tmp_path, f"{tmp_path} " + ("rule. " * 2000))
    assert plan.active is True
    assert plan.reason == "eligible"


def test_snapshot_witnesses_use_the_canonical_executor_key(tmp_path) -> None:
    (tmp_path / "input").write_text("x", encoding="utf-8")
    plan = _prepare(tmp_path, f"Update {tmp_path}/input")
    state = PipelineState(session_id="s")

    register_snapshot_witnesses(state, plan)

    assert state.shared[SharedKeys.FILE_WITNESSED] == [
        "input",
        f"{tmp_path}/input",
    ]
    assert "file.witnessed" not in state.shared
