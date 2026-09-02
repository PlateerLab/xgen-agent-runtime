"""Host rollout path isolation and bounded retention."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xgen_agent_runtime.host.rollouts import (
    allocate_rollout_path,
    prune_rollout_files,
    rollout_directory,
)


def test_allocate_rollout_path_is_unique_and_hides_unsafe_interaction_id(tmp_path: Path) -> None:
    interaction_id = "../../customer/대화\x00secret"

    first = allocate_rollout_path(tmp_path, interaction_id)
    second = allocate_rollout_path(tmp_path, interaction_id)

    assert first.parent == rollout_directory(tmp_path)
    assert second.parent == first.parent
    assert first != second
    assert first.name.startswith("rollout-") and first.suffix == ".jsonl"
    assert "customer" not in first.name and "대화" not in first.name
    assert not first.exists(), "allocation must not create an empty artifact"


def test_rollout_directory_resolves_storage_root_before_appending_layout(tmp_path: Path) -> None:
    alias = tmp_path / "nested" / ".."

    assert rollout_directory(alias) == tmp_path.resolve() / "executor" / "rollouts"


def test_prune_rollouts_keeps_newest_generated_files_only(tmp_path: Path) -> None:
    root = rollout_directory(tmp_path)
    root.mkdir(parents=True)
    files = []
    for index in range(5):
        path = root / f"rollout-20260101T00000000000{index}Z-id-{index}.jsonl"
        path.write_text(f"{index}\n", encoding="utf-8")
        os.utime(path, ns=(index + 1, index + 1))
        files.append(path)
    unrelated = root / "manual.jsonl"
    unrelated.write_text("keep", encoding="utf-8")

    assert prune_rollout_files(root, keep_last=2) == 3
    assert [path.exists() for path in files] == [False, False, False, True, True]
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_prune_rollouts_never_follows_symlink(tmp_path: Path) -> None:
    root = rollout_directory(tmp_path)
    root.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside", encoding="utf-8")
    link = root / "rollout-0000-link.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - restricted Windows environments
        pytest.skip(f"symlinks unavailable: {exc}")
    current = root / "rollout-9999-current.jsonl"
    current.write_text("current", encoding="utf-8")

    assert prune_rollout_files(root, keep_last=1) == 0
    assert link.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"


def test_prune_rollouts_handles_missing_directory_and_rejects_invalid_limit(
    tmp_path: Path,
) -> None:
    assert prune_rollout_files(tmp_path / "missing", keep_last=10) == 0
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_rollout_files(tmp_path, keep_last=0)
