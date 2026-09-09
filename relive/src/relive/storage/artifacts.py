"""Write-once JSON artifacts; interrupted temporary writes are never readable.

Stage events are separate, content-addressed JSON records instead of a shared
JSONL file. Thus a process interruption cannot truncate an earlier event.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator

from relive.types import to_dict


def canonical_json(value: Any) -> str:
    return json.dumps(to_dict(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{stable_hash(value)[:24]}"


class ArtifactError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", namespace) or any(
            part in {"", ".", ".."} for part in namespace.split("/")
        ):
            raise ArtifactError("Unsafe artifact namespace")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or key in {".", ".."}:
            raise ArtifactError("Unsafe artifact key")
        path = (self.root / namespace / f"{key}.json").resolve()
        if not path.is_relative_to(self.root):
            raise ArtifactError("Artifact path escapes store")
        return path

    def put_json(self, namespace: str, key: str, payload: Any) -> str:
        path = self.path(namespace, key)
        body = (canonical_json(payload) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Atomic create-if-absent: never silently overwrite history.
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != body:
                    raise ArtifactError(f"Immutable artifact collision: {path}") from None
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(temporary)
        return str(path)

    def get_json(self, namespace_or_reference: str, key: str | None = None) -> Any:
        path = self.path(namespace_or_reference, key) if key is not None else Path(namespace_or_reference).resolve()
        if not path.is_relative_to(self.root) or path.suffix != ".json":
            raise ArtifactError("Artifact reference outside this store")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ArtifactError(f"Invalid or incomplete JSON artifact: {path}") from exc

    def append_event(self, stage: str, payload: Any) -> str:
        return self.put_json(f"events/{stage}", stable_hash(payload), payload)

    def iter_events(self, stage: str | None = None) -> Iterator[Any]:
        directory = self.root / "events" if stage is None else self.path(f"events/{stage}", "unused").parent
        for path in sorted(directory.rglob("*.json")):
            yield self.get_json(str(path))

    @contextlib.contextmanager
    def run_lock(self):
        """Exclusive nonblocking lock, automatically released even after death."""
        with (self.root / ".run.lock").open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ArtifactError("Another process holds this run directory") from None
            try:
                yield self
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
