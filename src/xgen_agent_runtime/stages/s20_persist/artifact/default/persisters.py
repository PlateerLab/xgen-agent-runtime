"""Default persisters for Stage 20 (S9b.5)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from stat import S_ISDIR
from threading import Lock
from typing import Any, Dict, List, Optional

from xgen_agent_runtime.core.schema import ConfigField, ConfigSchema
from xgen_agent_runtime.core.state import PipelineState
from xgen_agent_runtime.stages.s20_persist.interface import Persister
from xgen_agent_runtime.stages.s20_persist.types import CheckpointRecord


class NoPersister(Persister):
    """Default. Writes nothing — kept so the stage is a zero-cost no-op
    until a host opts in."""

    @property
    def name(self) -> str:
        return "no_persist"

    @property
    def description(self) -> str:
        return "No-op persister"

    async def write(self, record: CheckpointRecord, state: PipelineState) -> None:
        return None


class FilePersister(Persister):
    """JSON-file checkpoint persister.

    One file per checkpoint, named ``<checkpoint_id>.json``. Files are
    grouped under ``base_dir/<session_id>/`` so listing by session is
    cheap. Writes are atomic via tempfile + ``os.replace``; the
    directory is created on first :meth:`write`.

    Hosts that need encryption or stronger durability should plug their
    own :class:`Persister` — this implementation is plaintext-on-disk.
    """

    DEFAULT_BASE_DIR = ".geny/checkpoints"
    _SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@+-]{0,119}\Z")
    _WINDOWS_RESERVED = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }

    def __init__(self, base_dir: str | os.PathLike[str] = DEFAULT_BASE_DIR) -> None:
        # Default lets the registry instantiate via cls() during a
        # manifest swap; configure() then overrides with the manifest's
        # actual base_dir.
        self._base = Path(base_dir)
        self._lock = Lock()

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "JSON-file checkpoint persister (one file per checkpoint)"

    @property
    def base_dir(self) -> Path:
        return self._base

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="file",
            fields=[
                ConfigField(
                    name="base_dir",
                    type="string",
                    label="Base directory",
                    description="Filesystem root for checkpoint files. Per-session subfolders are created automatically.",
                    default=cls.DEFAULT_BASE_DIR,
                    required=True,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        base = config.get("base_dir")
        if isinstance(base, str) and base.strip():
            self._base = Path(base)

    def get_config(self) -> Dict[str, Any]:
        return {"base_dir": str(self._base)}

    def _path_for(self, session_id: str, checkpoint_id: str) -> Path:
        checkpoint = self._storage_component(checkpoint_id, label="checkpoint")
        return self._ensure_contained(self._session_dir(session_id) / f"{checkpoint}.json")

    def _session_dir(self, session_id: str) -> Path:
        bucket = self._storage_component(session_id, label="session") if session_id else "_unknown"
        raw = self._base / bucket
        if raw.is_symlink():
            raise ValueError(f"checkpoint session directory must not be a symlink: {raw}")
        return self._ensure_contained(raw)

    @classmethod
    def _storage_component(cls, value: str, *, label: str) -> str:
        """Keep ordinary IDs readable and hash unsafe/oversized components.

        Hashing rather than rejecting preserves the public free-form ID
        contract while ensuring separators, dot components, NULs, Unicode,
        and platform-specific path syntax can never alter directory layout.
        The leading underscore is outside the safe-input grammar, preventing
        a raw ID from colliding with an encoded component.
        """
        text = str(value)
        windows_stem = text.split(".", 1)[0].upper()
        if (
            cls._SAFE_COMPONENT.fullmatch(text)
            and text not in {".", ".."}
            and not text.endswith(".")
            and windows_stem not in cls._WINDOWS_RESERVED
        ):
            return text
        digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
        return f"_{label}_{digest}"

    def _ensure_contained(self, path: Path) -> Path:
        root = self._base.resolve(strict=False)
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"checkpoint path escapes base directory: {path}") from exc
        return resolved

    async def write(self, record: CheckpointRecord, state: PipelineState) -> None:
        # Run blocking IO in a thread so we don't stall the event loop.
        await asyncio.to_thread(self._write_sync, record)

    #: Retention: keep at most this many checkpoint files per session. A
    #: long-lived session accumulated 1,482 files (367 MB) in production —
    #: checkpoints are resume points, not an archive; the recent tail is all
    #: recovery ever needs.
    KEEP_LAST = 100

    def _write_sync(self, record: CheckpointRecord) -> None:
        path = self._path_for(record.session_id, record.checkpoint_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                text=True,
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(record.to_dict(), fh, sort_keys=True)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
                self._fsync_directory(path.parent)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            self._prune_sync(path.parent)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Persist a rename's directory entry where the platform supports it."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(directory, flags)
        except OSError:
            # Windows and some filesystems do not permit opening directories.
            return
        try:
            if S_ISDIR(os.fstat(fd).st_mode):
                os.fsync(fd)
        finally:
            os.close(fd)

    def _prune_sync(self, session_dir: Path) -> None:
        """Drop the oldest checkpoints beyond KEEP_LAST (by mtime). Called
        under the lock, right after a successful write, so retention rides
        the write path — no background job, no unbounded growth."""
        try:
            files = sorted(
                (path for path in session_dir.glob("*.json") if not path.is_symlink()),
                key=lambda q: q.stat().st_mtime,
            )
            removed = False
            for stale in files[: max(0, len(files) - self.KEEP_LAST)]:
                try:
                    stale.unlink()
                    removed = True
                except OSError:
                    continue
            if removed:
                self._fsync_directory(session_dir)
        except OSError:
            pass  # retention is best-effort; the write itself succeeded

    async def read(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        return await asyncio.to_thread(self._read_sync, checkpoint_id)

    def _read_sync(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        checkpoint = self._storage_component(checkpoint_id, label="checkpoint")
        with self._lock:
            for session_dir in sorted(self._base.glob("*")):
                if session_dir.is_symlink() or not session_dir.is_dir():
                    continue
                raw_candidate = session_dir / f"{checkpoint}.json"
                if raw_candidate.is_symlink():
                    continue
                candidate = self._ensure_contained(raw_candidate)
                if candidate.exists():
                    return self._record_from_path(candidate)
        return None

    def _record_from_path(self, path: Path) -> CheckpointRecord:
        if path.is_symlink():
            raise ValueError(f"checkpoint file must not be a symlink: {path}")
        path = self._ensure_contained(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        from datetime import datetime

        created = data.get("created_at")
        created_dt = datetime.fromisoformat(created) if isinstance(created, str) else None
        kwargs: Dict[str, Any] = {
            "checkpoint_id": str(data.get("checkpoint_id", "")),
            "session_id": str(data.get("session_id", "")),
            "iteration": int(data.get("iteration", 0)),
            "payload": dict(data.get("payload") or {}),
        }
        if created_dt is not None:
            kwargs["created_at"] = created_dt
        return CheckpointRecord(**kwargs)

    async def list_checkpoints(self, session_id: str = "") -> List[CheckpointRecord]:
        return await asyncio.to_thread(self._list_sync, session_id)

    def _list_sync(self, session_id: str) -> List[CheckpointRecord]:
        with self._lock:
            session_dir = self._session_dir(session_id) if session_id else None
            paths: List[Path] = []
            if session_dir is not None:
                if session_dir.exists():
                    paths.extend(sorted(session_dir.glob("*.json")))
            else:
                for sd in sorted(self._base.glob("*")):
                    if sd.is_dir() and not sd.is_symlink():
                        paths.extend(
                            sorted(path for path in sd.glob("*.json") if not path.is_symlink())
                        )
            out: List[CheckpointRecord] = []
            for p in paths:
                try:
                    out.append(self._record_from_path(p))
                except Exception:  # noqa: BLE001 — best-effort listing
                    continue
            return out


__all__ = [
    "FilePersister",
    "NoPersister",
]
