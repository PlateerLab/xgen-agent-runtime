"""Host-owned paths and retention for opt-in pipeline rollout files.

The engine only knows how to write to a ``rollout_recorder`` supplied through
the existing session runtime.  This module keeps product storage policy out of
the engine: one collision-free JSONL file per turn, under the workflow's
existing executor storage root, with a fixed file-count cap.

Raw interaction IDs are hashed instead of embedded in filenames.  Besides
preventing path traversal, this avoids leaking user-controlled identifiers in
directory listings while retaining a stable value for operational grouping.
"""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROLLOUT_KEEP_LAST = 100
_ROLLOUT_PREFIX = "rollout-"
_ROLLOUT_SUFFIX = ".jsonl"


def rollout_directory(storage_root: str | os.PathLike[str]) -> Path:
    """Return the dedicated rollout directory below a workflow storage root."""
    return Path(storage_root).resolve(strict=False) / "executor" / "rollouts"


def allocate_rollout_path(
    storage_root: str | os.PathLike[str], interaction_id: str
) -> Path:
    """Allocate a unique, component-safe path for one host turn.

    The function only chooses a path; directory and file creation remain owned
    by :class:`~xgen_agent_runtime.core.rollout_recorder.RolloutRecorder` so a
    configured-but-never-started turn leaves no empty artifacts behind.
    """
    digest = hashlib.sha256(
        str(interaction_id).encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:16]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    nonce = uuid.uuid4().hex
    return rollout_directory(storage_root) / (
        f"{_ROLLOUT_PREFIX}{timestamp}-{digest}-{nonce}{_ROLLOUT_SUFFIX}"
    )


def prune_rollout_files(directory: str | os.PathLike[str], *, keep_last: int) -> int:
    """Best-effort removal of old host-created rollouts; return removed count.

    Only regular, non-symlink files matching our private filename prefix are
    candidates.  The directory is fsynced after deletion where supported so
    retention changes survive a host crash just like recorder writes do.
    """
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1 (got {keep_last})")

    root = Path(directory)
    try:
        entries: list[tuple[int, str, Path]] = []
        with os.scandir(root) as scan:
            for entry in scan:
                if not (
                    entry.name.startswith(_ROLLOUT_PREFIX)
                    and entry.name.endswith(_ROLLOUT_SUFFIX)
                    and entry.is_file(follow_symlinks=False)
                ):
                    continue
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    entries.append((info.st_mtime_ns, entry.name, Path(entry.path)))
    except (FileNotFoundError, NotADirectoryError):
        return 0

    entries.sort()
    removed = 0
    for _, _, stale in entries[: max(0, len(entries) - keep_last)]:
        try:
            stale.unlink()
            removed += 1
        except OSError:
            # Another completed turn may prune the same shared directory.
            continue
    if removed:
        _fsync_directory(root)
    return removed


def _fsync_directory(directory: Path) -> None:
    """Persist removed directory entries on filesystems that support it."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            os.fsync(fd)
    finally:
        os.close(fd)
