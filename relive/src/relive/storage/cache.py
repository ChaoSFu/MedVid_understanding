"""Content-addressed raw inference, bounded retries and deterministic replay.

Certificate policies are not inference inputs. Every other request field,
including frame order, source paths and frame bytes, is included in identity.
Raw attempts are write-once. Completed indexes are validated against those
attempts; interrupted indexes can be reconstructed without another model call.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
import fcntl
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Any

from relive.backends.base import Backend, BackendError
from relive.config import _plain
from relive.types import ExecutionStatus
from .artifacts import ArtifactError, ArtifactStore, stable_hash


CACHE_VERSION = "relive-inference-cache-v1"
POLICY_FIELDS = {"policy", "policy_name", "policy_version", "certificate_policy"}


class BudgetExceeded(RuntimeError):
    """No further logical inference attempt may be admitted for this sample."""


class Budget:
    """Per-sample accounting, with identical logical costs on cached replay.

``calls`` counts distinct (request, technical attempt) pairs. ``new_calls``
counts newly executed attempts; ``cache_hits`` counts distinct replayed
requests. Repeated access to the same result within a sample costs nothing.
Latency counts newly executed work, so an all-cache replay has zero latency.
"""

    def __init__(self, max_calls: int):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.max_calls = max_calls
        self.calls = 0
        self.new_calls = 0
        self.cache_hits = 0
        self.latency_seconds = 0.0
        self._seen: set[tuple[str, int]] = set()
        self._cached_keys: set[str] = set()

    @property
    def remaining(self) -> int:
        return self.max_calls - self.calls

    def snapshot(self) -> dict[str, Any]:
        return {"max_calls": self.max_calls, "calls": self.calls,
                "new_calls": self.new_calls, "cache_hits": self.cache_hits,
                "latency_seconds": self.latency_seconds, "remaining": self.remaining}

    def _admit(self, key: str, indexes: list[int], *, cached: bool) -> None:
        unseen = {(key, index) for index in indexes} - self._seen
        if len(unseen) > self.remaining:
            raise BudgetExceeded("INFERENCE_CALL_BUDGET_EXHAUSTED")
        self.calls += len(unseen)
        self._seen.update(unseen)
        if cached:
            if unseen and key not in self._cached_keys:
                self._cached_keys.add(key)
                self.cache_hits += 1
        else:
            self.new_calls += len(unseen)


def _safe_failure(exc: Exception) -> str:
    # Exception strings can include API keys, HTTP bodies or request reprs.
    # Only adapter-owned diagnostic codes are eligible for persisted errors.
    if isinstance(exc, BackendError):
        code = str(exc)
        allowed = {"MOCK_FIXTURE_ERROR", "UNSUPPORTED_MOCK_STAGE", "HTTP_REDIRECT_REFUSED",
                   "UNSUPPORTED_SOURCE_IMAGE_FORMAT", "ANIMATED_FRAME_NOT_ALLOWED",
                   "FRAME_IDENTITY_LENGTH_MISMATCH", "CREDENTIAL_ENV_UNSET",
                   "TRANSPORT_FAILURE", "INVALID_COMPLETION_ENVELOPE"}
        if code in allowed or re.fullmatch(r"HTTP_[1-5][0-9]{2}", code):
            return code
        return "BACKEND_ERROR"
    return "UNEXPECTED_BACKEND_ERROR"


class CachedInference:
    def __init__(self, backend: Backend, store: ArtifactStore):
        self.backend = backend
        self.store = store

    def request_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("Inference request must be an object")
        _plain(request, "inference request")
        for field in ("stage", "prompt", "prompt_version"):
            if not isinstance(request.get(field), str) or not request[field].strip():
                raise ValueError(f"Inference request requires nonempty {field}")
        paths, ids = request.get("image_paths", []), request.get("frame_ids", [])
        if not isinstance(paths, list) or not isinstance(ids, list) or len(paths) != len(ids):
            raise ValueError("image_paths and frame_ids must be aligned lists")
        if any(not isinstance(item, str) or not item for item in [*paths, *ids]) or len(set(ids)) != len(ids):
            raise ValueError("Frame paths and unique identities must be nonempty strings")
        frames = []
        for order, (frame_id, source_path) in enumerate(zip(ids, paths)):
            path = Path(source_path).expanduser().resolve()
            data = path.read_bytes()
            frames.append({"frame_id": frame_id, "order": order, "source_path": source_path,
                           "resolved_path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                           "byte_count": len(data)})
        inference_request = {k: deepcopy(v) for k, v in request.items() if k not in POLICY_FIELDS}
        return {"cache_version": CACHE_VERSION, "backend": self.backend.fingerprint(),
                "request": inference_request, "frames": frames,
                "prompt_sha256": hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest()}

    def cache_key(self, request: dict[str, Any]) -> str:
        return stable_hash(self.request_identity(request))

    @contextlib.contextmanager
    def _lock(self, key: str):
        directory = self.store.root / "cache_locks"
        directory.mkdir(exist_ok=True)
        with (directory / f"{key}.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _load_attempts(self, key: str, identity: dict[str, Any]) -> list[tuple[str, dict]]:
        directory = self.store.path(f"raw/{key}", "unused").parent
        by_index: dict[int, tuple[str, dict]] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                record = self.store.get_json(str(path))
                if not isinstance(record, dict) or stable_hash(record) != path.stem:
                    continue
                index = record.get("attempt_index")
                if record.get("cache_key") != key or record.get("identity") != identity:
                    continue
                if type(index) is not int or index < 0 or record.get("cache_version") != CACHE_VERSION:
                    continue
                status = record.get("execution_status")
                if status not in {ExecutionStatus.OK.value, ExecutionStatus.INFERENCE_ERROR.value}:
                    continue
                if status == ExecutionStatus.OK.value and (not isinstance(record.get("raw_text"), str) or record.get("failure_reason") is not None or record.get("retryable") is not False):
                    continue
                if status == ExecutionStatus.INFERENCE_ERROR.value and (record.get("raw_text") is not None or not isinstance(record.get("failure_reason"), str)):
                    continue
                latency = record.get("latency_seconds")
                if type(latency) not in {int, float} or not math.isfinite(latency) or latency < 0 or type(record.get("retryable")) is not bool:
                    continue
                if record.get("synthetic") is not self.backend.synthetic:
                    continue
            except (ArtifactError, FileNotFoundError, TypeError, ValueError):
                continue
            if index in by_index:
                raise ArtifactError("CONFLICTING_IMMUTABLE_ATTEMPTS")
            by_index[index] = (str(path), record)
        if sorted(by_index) != list(range(len(by_index))):
            raise ArtifactError("INCOMPLETE_RAW_ATTEMPT_CHAIN")
        result = [by_index[index] for index in sorted(by_index)]
        for _, record in result[:-1]:
            if record["execution_status"] == ExecutionStatus.OK.value or not record["retryable"]:
                raise ArtifactError("ATTEMPT_AFTER_TERMINAL_RAW_RESULT")
        return result

    def _complete(self, key: str, attempts: list[tuple[str, dict]]) -> str:
        manifest = {"cache_version": CACHE_VERSION, "cache_key": key, "complete": True,
                    "synthetic": self.backend.synthetic,
                    "attempts": [{"raw_response_ref": ref, "sha256": stable_hash(record)} for ref, record in attempts],
                    "terminal_raw_response_ref": attempts[-1][0]}
        # The content hash prevents replacing even a corrupted/incomplete index.
        # Every replay is rebuilt from validated raw records, never an unchecked flag.
        return self.store.put_json(f"cache/{key}", stable_hash(manifest), manifest)

    def call(self, request: dict[str, Any], budget: Budget) -> dict[str, Any]:
        identity = self.request_identity(request)
        key = stable_hash(identity)
        maximum_attempts = self.backend.config.get("max_retries", 0) + 1
        if type(maximum_attempts) is not int or not 1 <= maximum_attempts <= 4:
            raise ValueError("At most three technical retries are supported")
        with self._lock(key):
            attempts = self._load_attempts(key, identity)
            if len(attempts) > maximum_attempts:
                raise ArtifactError("RAW_ATTEMPTS_EXCEED_FROZEN_RETRY_LIMIT")
            previous_count = len(attempts)
            budget._admit(key, list(range(previous_count)), cached=True)
            latency = 0.0
            while not attempts or (attempts[-1][1]["retryable"] and len(attempts) < maximum_attempts):
                index = len(attempts)
                budget._admit(key, [index], cached=False)
                if index:
                    time.sleep(self.backend.config.get("retry_backoff_seconds", 0))
                started = time.monotonic()
                status, raw, failure, retryable = ExecutionStatus.OK.value, None, None, False
                try:
                    raw = self.backend.infer(deepcopy(identity["request"]))
                    if not isinstance(raw, str):
                        raise BackendError("INVALID_COMPLETION_ENVELOPE")
                except Exception as exc:
                    status, raw = ExecutionStatus.INFERENCE_ERROR.value, None
                    failure = _safe_failure(exc)
                    retryable = isinstance(exc, BackendError) and exc.retryable is True
                elapsed = time.monotonic() - started
                latency += elapsed
                budget.latency_seconds += elapsed
                record = {"cache_version": CACHE_VERSION, "cache_key": key, "identity": identity,
                          "attempt_index": index, "execution_status": status, "raw_text": raw,
                          "failure_reason": failure, "retryable": retryable,
                          "latency_seconds": elapsed, "synthetic": self.backend.synthetic}
                ref = self.store.put_json(f"raw/{key}", stable_hash(record), record)
                attempts.append((ref, record))
            self._complete(key, attempts)
            ref, last = attempts[-1]
            return {"execution_status": last["execution_status"], "raw_response_ref": ref,
                    "raw_text": last["raw_text"], "cache_hit": len(attempts) == previous_count,
                    "latency_seconds": latency, "failure_reason": last["failure_reason"],
                    "cache_key": key, "attempts": [
                        {"raw_response_ref": attempt_ref, "attempt_index": record["attempt_index"],
                         "execution_status": record["execution_status"], "failure_reason": record["failure_reason"],
                         "retryable": record["retryable"], "cache_hit": index < previous_count,
                         "latency_seconds": record["latency_seconds"]}
                        for index, (attempt_ref, record) in enumerate(attempts)]}
