"""Closed runtime schema and source declarations, independent of evaluation.

The hash-bound sidecar is an auditable source attestation, not a proof that its
author has never seen hidden annotations. Curators must review upstream sources.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from relive.types import Claim, Frame, RuntimeSample

SCHEMA_VERSION = "relive-runtime-v1"
FIELD_SOURCES = {
    "question": "public_question",
    "target_claim": "user_query",
    "required_claims": "user_query",
    "frames": "public_frames",
    "metadata": "public_metadata",
}
# Generic core-runner capability only.  This is intentionally independent of
# dataset adapters: inclusion here never asserts MedVidU native-task support.
SUPPORTED_TASKS = frozenset({"claim_verification", "action_qa"})
_SAMPLE_FIELDS = {"sample_id", "task", "question", "frames", "video_path", "target_claim", "required_claims", "metadata"}
_FRAME_FIELDS = {"frame_id", "path", "order", "timestamp", "timestamp_source", "source_reference"}
_CLAIM_FIELDS = {"claim_id", "text", "entity", "action", "target", "time_scope"}
_METADATA_FIELDS = {
    "dataset_name", "query_timestamps", "unresolved_relations", "question_scope",
    "runtime_adapter", "source_qa_type", "source_record_sha256", "nonofficial_protocol",
}
_TIMESTAMP_SOURCES = {"explicit_public_metadata", "decoder_pts", "public_frame_manifest", "synthetic", "explicit"}


class RuntimeInputError(ValueError):
    status = "INVALID_INPUT"


def _object(value: Any, allowed: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeInputError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise RuntimeInputError(f"{label} contains non-runtime fields: {sorted(unknown)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeInputError(f"{label} must be a non-empty string")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RuntimeInputError(f"{label} must be a finite non-negative number")
    return float(value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    """Reject ambiguous JSON before the closed schema sees a overwritten key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeInputError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json(payload: str) -> Any:
    return json.loads(payload, object_pairs_hook=_unique_object)


def _claim(value: Any, default_id: str, frame_ids: set[str]) -> Claim:
    if isinstance(value, str):
        return Claim(default_id, _text(value, "claim"), source="runtime_user_query", required_for_question=True)
    obj = _object(value, _CLAIM_FIELDS, "claim")
    scope = _object(obj.get("time_scope", {}), {"frame_ids", "start", "end", "timestamp_source", "source_reference"}, "claim.time_scope")
    if "frame_ids" in scope:
        if not isinstance(scope["frame_ids"], list) or not all(isinstance(x, str) and x in frame_ids for x in scope["frame_ids"]):
            raise RuntimeInputError("claim.time_scope.frame_ids must reference supplied frames")
        if len(scope["frame_ids"]) != len(set(scope["frame_ids"])):
            raise RuntimeInputError("claim.time_scope.frame_ids must be unique")
    for key in ("timestamp_source", "source_reference"):
        if key in scope:
            _text(scope[key], f"claim.time_scope.{key}")
    if ("start" in scope) != ("end" in scope):
        raise RuntimeInputError("claim time interval needs both start and end")
    if "start" in scope:
        start, end = _number(scope["start"], "claim start"), _number(scope["end"], "claim end")
        if end < start or scope.get("timestamp_source") != "public_query" or not scope.get("source_reference"):
            raise RuntimeInputError("claim time scope requires an ordered interval with public_query source and reference")
    elif "timestamp_source" in scope or "source_reference" in scope:
        raise RuntimeInputError("claim time source requires an explicit start/end interval")
    for key in ("entity", "action", "target"):
        if obj.get(key) is not None:
            _text(obj[key], f"claim.{key}")
    return Claim(
        claim_id=_text(obj.get("claim_id", default_id), "claim_id"), text=_text(obj.get("text"), "claim.text"),
        entity=obj.get("entity"), action=obj.get("action"), target=obj.get("target"), time_scope=dict(scope),
        source="runtime_user_query", required_for_question=True,
    )


def _parse_sample(value: Any, base: Path, provenance: dict) -> RuntimeSample:
    obj = _object(value, _SAMPLE_FIELDS, "sample")
    sample_id = _text(obj.get("sample_id"), "sample_id")
    task = _text(obj.get("task"), "task")
    question = _text(obj.get("question"), "question")
    if "frames" in obj and "video_path" in obj:
        raise RuntimeInputError("provide frames or video_path, not both")
    if "video_path" in obj:
        raise RuntimeInputError("video_path decoding is not enabled in ReliVE-v1; provide decoded frames with decoder PTS provenance or without timestamps")
    rows = obj.get("frames")
    if not isinstance(rows, list) or not rows:
        raise RuntimeInputError("frames must be a non-empty list")
    frames = []
    for row in rows:
        f = _object(row, _FRAME_FIELDS, "frame")
        frame_id, path = _text(f.get("frame_id"), "frame_id"), _text(f.get("path"), "frame.path")
        if not isinstance(f.get("order"), int) or isinstance(f["order"], bool) or f["order"] < 0:
            raise RuntimeInputError("frame.order must be an explicit non-negative integer")
        timestamp = f.get("timestamp")
        source, reference = f.get("timestamp_source"), f.get("source_reference")
        if source is not None:
            _text(source, "timestamp_source")
        if reference is not None:
            _text(reference, "source_reference")
        if timestamp is not None:
            timestamp = _number(timestamp, "timestamp")
            if source not in _TIMESTAMP_SOURCES or not reference:
                raise RuntimeInputError("timestamp requires a supported timestamp_source and source_reference")
            if source == "synthetic" and provenance["source_kind"] != "synthetic":
                raise RuntimeInputError("synthetic timestamps require synthetic source_kind")
        elif source is not None:
            raise RuntimeInputError("timestamp_source cannot appear without timestamp")
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = base / resolved
        if not resolved.is_file():
            raise RuntimeInputError(f"frame path does not exist: {resolved}")
        frames.append(Frame(frame_id, str(resolved.resolve()), f["order"], timestamp, source, reference or path))
    ids = [f.frame_id for f in frames]
    orders = [f.order for f in frames]
    if len(set(ids)) != len(ids) or len(set(orders)) != len(orders):
        raise RuntimeInputError("frame_id and frame.order must be unique within a sample")
    if orders != sorted(orders):
        raise RuntimeInputError("frames must appear in their explicit original order")
    times = [f.timestamp for f in frames if f.timestamp is not None]
    if times != sorted(times):
        raise RuntimeInputError("timestamps must respect original frame order")
    meta = _object(obj.get("metadata", {}), _METADATA_FIELDS, "metadata")
    if "dataset_name" in meta:
        _text(meta["dataset_name"], "dataset_name")
    if "query_timestamps" in meta:
        if not isinstance(meta["query_timestamps"], list):
            raise RuntimeInputError("query_timestamps must be a list of public query times")
        for t in meta["query_timestamps"]:
            _number(t, "query_timestamp")
    if "unresolved_relations" in meta and (not isinstance(meta["unresolved_relations"], list) or not all(isinstance(x, str) and x.strip() for x in meta["unresolved_relations"])):
        raise RuntimeInputError("unresolved_relations must be a list of relation descriptions")
    if "question_scope" in meta:
        _text(meta["question_scope"], "question_scope")
        if meta["question_scope"] not in {"single_action", "required_claims"}:
            raise RuntimeInputError("unsupported question_scope")
    for key in ("runtime_adapter", "source_qa_type"):
        if key in meta:
            _text(meta[key], key)
    if "source_record_sha256" in meta:
        if not isinstance(meta["source_record_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", meta["source_record_sha256"]):
            raise RuntimeInputError("source_record_sha256 must be a SHA-256 digest")
    if "nonofficial_protocol" in meta and type(meta["nonofficial_protocol"]) is not bool:
        raise RuntimeInputError("nonofficial_protocol must be boolean")
    target = _claim(obj["target_claim"], f"{sample_id}:target", set(ids)) if obj.get("target_claim") is not None else None
    required_raw = obj.get("required_claims", [])
    if not isinstance(required_raw, list):
        raise RuntimeInputError("required_claims must be a list")
    required = tuple(_claim(c, f"{sample_id}:required:{i}", set(ids)) for i, c in enumerate(required_raw))
    if len({c.claim_id for c in required}) != len(required):
        raise RuntimeInputError("required claim IDs must be unique")
    if target is not None and any(c.claim_id == target.claim_id and c != target for c in required):
        raise RuntimeInputError("target and required claim IDs cannot identify different claims")
    if task == "claim_verification" and target is None:
        raise RuntimeInputError("claim_verification requires target_claim")
    if task == "action_qa" and not required:
        raise RuntimeInputError("action_qa requires nonempty required_claims before inference")
    if meta.get("question_scope") == "required_claims" and not required:
        raise RuntimeInputError("required_claims question scope requires explicit claims")
    return RuntimeSample(sample_id, task, question, tuple(frames), target, required, dict(meta), dict(provenance))


def load_runtime(path: str | Path) -> list[RuntimeSample]:
    """Load a source-attested JSONL without importing annotation loaders.

    Unsupported task strings are retained so the runner can emit UNSUPPORTED.
    """
    source = Path(path).expanduser().resolve()
    sidecar = Path(str(source) + ".provenance.json")
    try:
        declaration = _json(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeInputError(f"runtime requires a readable source provenance sidecar: {sidecar}") from exc
    declaration = _object(declaration, {"schema_version", "source_kind", "runtime_sha256", "field_sources"}, "source provenance")
    if declaration.get("schema_version") != SCHEMA_VERSION or not isinstance(declaration.get("source_kind"), str) or declaration["source_kind"] not in {"synthetic", "public_runtime"}:
        raise RuntimeInputError("runtime source must declare relive-runtime-v1 and synthetic or public_runtime origin")
    if declaration.get("field_sources") != FIELD_SOURCES:
        raise RuntimeInputError("source field mapping must exactly match the public runtime source contract")
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise RuntimeInputError(f"cannot read runtime: {source}") from exc
    fingerprint = hashlib.sha256(payload).hexdigest()
    if declaration.get("runtime_sha256") != fingerprint:
        raise RuntimeInputError("runtime file does not match source provenance SHA-256")
    provenance = dict(declaration, runtime_path=str(source), source_sidecar=str(sidecar), loader="relive.data.schemas.load_runtime", source_attestation_limit="declaration_and_hash_are_not_independent_proof_of_origin")
    samples = []
    try:
        lines = payload.decode("utf-8").splitlines()
        for index, line in enumerate(lines, 1):
            if line.strip():
                samples.append(_parse_sample(_json(line), source.parent, dict(provenance, runtime_line=index)))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeInputError(f"invalid UTF-8 JSONL runtime: {exc}") from exc
    if not samples:
        raise RuntimeInputError("runtime contains no samples")
    if len({s.sample_id for s in samples}) != len(samples):
        raise RuntimeInputError("sample_id must be unique")
    return samples
