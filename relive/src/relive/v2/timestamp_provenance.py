"""Deterministic, GT-isolated public timestamp provenance for ReliVE-v2 TAL.

A timestamp is accepted only with a provenance sidecar.  This module never
infers time from sample IDs, filenames, frame order, frame count, or FPS that
has not been explicitly documented in a versioned public metadata source.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import IDENTITY_FIELDS, TALSelectionError, strict_json_loads, strict_jsonl

TIMESTAMP_MANIFEST_FORMAT = "relive-v2-tal-public-per-frame-timestamps-v1"
TIMESTAMP_PROVENANCE_FORMAT = "relive-v2-tal-timestamp-provenance-v1"
TIMESTAMP_AUDIT_FORMAT = "relive-v2-tal-timestamp-source-audit-v1"
DOCUMENTED_SOURCE_FORMAT = "relive-v2-tal-documented-frame-index-fps-source-v1"
DECODER_MAP_FORMAT = "relive-v2-tal-decoder-frame-map-v1"
ALLOWED_SOURCE_TYPES = frozenset({"DECODER_PTS", "PUBLIC_PER_FRAME_TIMESTAMP", "DOCUMENTED_SOURCE_FRAME_INDEX_AND_FPS", "VERSIONED_DATASET_TIMEBASE_ADAPTER"})
TIME_ORIGINS = frozenset({"CLIP_LOCAL_ZERO", "SOURCE_VIDEO_ABSOLUTE"})
ROW_FIELDS = frozenset({"source_record_index", "sample_id", "public_record_sha256", "question_sha256", "frame_order", "source_frame_reference", "timestamp_seconds", "timestamp_unit"})
PROVENANCE_FIELDS = frozenset({"format", "timestamp_manifest_sha256", "source_type", "dataset_name", "sample_id", "source_record_index", "public_record_sha256", "question_sha256", "time_origin", "timestamp_unit", "mapping_method", "adapter_version", "source_reference", "source_reference_sha256", "frame_count", "gt_used", "assistant_or_gt_values_accessed", "model_calls_made", "backend_loaded", "cache_opened", "certificate_created", "new_verified_count", "certificate_status", "public_media_projection_sha256", "frame_sha256"})


class TimestampProvenanceError(ValueError):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_bytes().decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise TimestampProvenanceError(f"{code}_INVALID") from exc
    if not isinstance(value, dict):
        raise TimestampProvenanceError(f"{code}_OBJECT_REQUIRED")
    if path.read_bytes() != (canonical_json(value) + "\n").encode("utf-8"):
        raise TimestampProvenanceError(f"{code}_NONCANONICAL_BYTES")
    return value


def _strict_rows(path: Path, code: str) -> tuple[dict[str, Any], ...]:
    try:
        rows = strict_jsonl(path, error_code=code)
        raw = path.read_bytes()
    except (OSError, TALSelectionError) as exc:
        raise TimestampProvenanceError(f"{code}_INVALID") from exc
    if raw != b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows):
        raise TimestampProvenanceError(f"{code}_NONCANONICAL_BYTES")
    return rows


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TimestampProvenanceError(code)
    return float(value)


def _identity_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(value[name] for name in IDENTITY_FIELDS)
    except KeyError as exc:
        raise TimestampProvenanceError("TIMESTAMP_IDENTITY_MISSING") from exc


def validate_timestamp_pair(*, timestamp_manifest_path: Path, provenance_path: Path, identity: Mapping[str, Any], frames: list[Mapping[str, Any]]) -> tuple[list[float], dict[str, Any], str, str]:
    """Validate a timestamp JSONL plus immutable provenance sidecar.

    `frames` is a logical, ordered Stage-2 projection.  No sorting or dedupe is
    performed: every logical frame must occur exactly once.
    """
    rows = _strict_rows(timestamp_manifest_path, "PUBLIC_TIMESTAMP_MANIFEST")
    provenance = _strict_object(provenance_path, "PUBLIC_TIMESTAMP_PROVENANCE")
    if not PROVENANCE_FIELDS.issubset(provenance) or provenance.get("format") != TIMESTAMP_PROVENANCE_FORMAT:
        raise TimestampProvenanceError("TIMESTAMP_PROVENANCE_SCHEMA_INVALID")
    manifest_sha, provenance_sha = sha256_path(timestamp_manifest_path), sha256_path(provenance_path)
    if provenance["timestamp_manifest_sha256"] != manifest_sha:
        raise TimestampProvenanceError("TIMESTAMP_MANIFEST_SHA_MISMATCH")
    if provenance["source_type"] not in ALLOWED_SOURCE_TYPES or provenance["time_origin"] not in TIME_ORIGINS or provenance["timestamp_unit"] != "seconds":
        raise TimestampProvenanceError("TIMESTAMP_PROVENANCE_SOURCE_INVALID")
    if not isinstance(provenance["mapping_method"], str) or not provenance["mapping_method"].strip() or not isinstance(provenance["adapter_version"], str) or not provenance["adapter_version"].strip():
        raise TimestampProvenanceError("TIMESTAMP_PROVENANCE_MAPPING_INVALID")
    if not isinstance(provenance["source_reference"], str) or not provenance["source_reference"]:
        raise TimestampProvenanceError("TIMESTAMP_PROVENANCE_REFERENCE_INVALID")
    source = Path(provenance["source_reference"])
    try:
        source_sha = sha256_path(source)
    except OSError as exc:
        raise TimestampProvenanceError("TIMESTAMP_SOURCE_REFERENCE_UNREADABLE") from exc
    if provenance["source_reference_sha256"] != source_sha:
        raise TimestampProvenanceError("TIMESTAMP_SOURCE_REFERENCE_SHA_MISMATCH")
    if any(provenance[field] != identity[field] for field in IDENTITY_FIELDS) or provenance["frame_count"] != len(frames):
        raise TimestampProvenanceError("TIMESTAMP_PROVENANCE_BINDING_MISMATCH")
    if not isinstance(provenance["public_media_projection_sha256"], str) or len(provenance["public_media_projection_sha256"]) != 64 or not isinstance(provenance["frame_sha256"], list) or provenance["frame_sha256"] != [frame.get("frame_sha256") for frame in frames]:
        raise TimestampProvenanceError("TIMESTAMP_PROVENANCE_MEDIA_BINDING_MISMATCH")
    if provenance["gt_used"] is not False or provenance["assistant_or_gt_values_accessed"] is not False or provenance["model_calls_made"] != 0 or provenance["backend_loaded"] is not False or provenance["cache_opened"] is not False or provenance["certificate_created"] is not False or provenance["new_verified_count"] != 0 or provenance["certificate_status"] != "NOT_APPLICABLE":
        raise TimestampProvenanceError("TIMESTAMP_PROVENANCE_ISOLATION_INVALID")
    if len(rows) != len(frames):
        raise TimestampProvenanceError("TIMEBASE_MANIFEST_FRAME_COUNT_MISMATCH")
    timestamps: list[float] = []
    for order, (row, frame) in enumerate(zip(rows, frames)):
        if set(row) != ROW_FIELDS or row["timestamp_unit"] != "seconds":
            raise TimestampProvenanceError("TIMEBASE_MANIFEST_SCHEMA_INVALID")
        if _identity_key(row) != _identity_key(identity) or row["frame_order"] != order or row["source_frame_reference"] != frame["source_frame_reference"]:
            raise TimestampProvenanceError("TIMEBASE_MANIFEST_BINDING_MISMATCH")
        timestamp = _finite(row["timestamp_seconds"], "TIMEBASE_TIMESTAMP_NONFINITE")
        if timestamp < 0:
            raise TimestampProvenanceError("TIMEBASE_TIMESTAMP_NEGATIVE")
        timestamps.append(timestamp)
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise TimestampProvenanceError("TIMEBASE_TIMESTAMP_NONMONOTONIC")
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        raise TimestampProvenanceError("TIMEBASE_DURATION_NOT_POSITIVE")
    return timestamps, provenance, manifest_sha, provenance_sha


def _write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise TimestampProvenanceError("IMMUTABLE_OUTPUT_EXISTS")
    path.write_bytes(payload)


def _rows_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _audit(*, status: str, created: bool, reason: str | None, **extra: Any) -> dict[str, Any]:
    payload = {"format": TIMESTAMP_AUDIT_FORMAT, "status": status, "timestamp_manifest_created": created, "reason_code": reason,
               "gt_used": False, "assistant_or_gt_values_accessed": False, "model_calls_made": 0, "backend_loaded": False,
               "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", **extra}
    payload["audit_content_sha256"] = stable_hash(payload)
    return payload


def _documented_rows(source_path: Path, projections: tuple[Mapping[str, Any], ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _strict_object(source_path, "DOCUMENTED_TIMEBASE_SOURCE")
    required = {"format", "adapter_version", "dataset_name", "timestamp_unit", "time_origin", "mapping_method", "records"}
    if set(source) != required or source["format"] != DOCUMENTED_SOURCE_FORMAT or source["timestamp_unit"] != "seconds" or source["time_origin"] not in TIME_ORIGINS or not isinstance(source["adapter_version"], str) or not source["adapter_version"] or not isinstance(source["mapping_method"], str) or not source["mapping_method"]:
        raise TimestampProvenanceError("DOCUMENTED_TIMEBASE_SOURCE_SCHEMA_INVALID")
    if not isinstance(source["records"], list):
        raise TimestampProvenanceError("DOCUMENTED_TIMEBASE_SOURCE_RECORDS_INVALID")
    by_key = {_identity_key(item): item for item in source["records"] if isinstance(item, dict)}
    if len(by_key) != len(source["records"]):
        raise TimestampProvenanceError("DOCUMENTED_TIMEBASE_SOURCE_DUPLICATE_OR_INVALID")
    outputs, provenance = [], []
    for projection in projections:
        key = _identity_key(projection)
        record = by_key.get(key)
        if record is None:
            raise TimestampProvenanceError("DOCUMENTED_TIMEBASE_SOURCE_IDENTITY_MISSING")
        needed = set(IDENTITY_FIELDS) | {"frame_indices", "fps", "source_frame_references"}
        if set(record) != needed:
            raise TimestampProvenanceError("DOCUMENTED_TIMEBASE_SOURCE_RECORD_SCHEMA_INVALID")
        refs, indices, frames = record["source_frame_references"], record["frame_indices"], projection["frames"]
        fps = _finite(record["fps"], "DOCUMENTED_TIMEBASE_FPS_INVALID")
        if fps <= 0 or not isinstance(indices, list) or not isinstance(refs, list) or len(indices) != len(frames) or refs != [frame["source_frame_reference"] for frame in frames] or any(type(value) is not int or value < 0 for value in indices) or any(right < left for left, right in zip(indices, indices[1:])) or indices[-1] <= indices[0]:
            raise TimestampProvenanceError("DOCUMENTED_TIMEBASE_FRAME_INDEX_INVALID")
        times = [(index - indices[0]) / fps for index in indices]
        rows = [{**{field: projection[field] for field in IDENTITY_FIELDS}, "frame_order": order, "source_frame_reference": ref, "timestamp_seconds": value, "timestamp_unit": "seconds"} for order, (ref, value) in enumerate(zip(refs, times))]
        outputs.extend(rows)
        provenance.append({"identity": key, "dataset_name": source["dataset_name"], "source_type": "DOCUMENTED_SOURCE_FRAME_INDEX_AND_FPS", "time_origin": source["time_origin"], "timestamp_unit": "seconds", "mapping_method": source["mapping_method"], "adapter_version": source["adapter_version"], "frame_count": len(rows)})
    return outputs, provenance


def _ffprobe_pts(video_path: Path) -> list[float]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video_path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        parsed = json.loads(completed.stdout)
        values = [float(frame["best_effort_timestamp_time"]) for frame in parsed["frames"]]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TimestampProvenanceError("DECODER_PTS_UNAVAILABLE") from exc
    if not values or any(not math.isfinite(item) for item in values):
        raise TimestampProvenanceError("DECODER_PTS_INVALID")
    return values


def _decoder_rows(video_path: Path, mapping_path: Path, projections: tuple[Mapping[str, Any], ...], pts_reader: Callable[[Path], list[float]] = _ffprobe_pts) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapping = _strict_object(mapping_path, "DECODER_FRAME_MAP")
    if set(mapping) != {"format", "time_origin", "records"} or mapping["format"] != DECODER_MAP_FORMAT or mapping["time_origin"] not in TIME_ORIGINS or not isinstance(mapping["records"], list):
        raise TimestampProvenanceError("DECODER_FRAME_MAP_SCHEMA_INVALID")
    pts = pts_reader(video_path)
    by_key = {_identity_key(item): item for item in mapping["records"] if isinstance(item, dict)}
    if len(by_key) != len(mapping["records"]):
        raise TimestampProvenanceError("DECODER_FRAME_MAP_DUPLICATE_OR_INVALID")
    outputs, provenance = [], []
    for projection in projections:
        record = by_key.get(_identity_key(projection))
        if record is None or set(record) != set(IDENTITY_FIELDS) | {"source_frame_references", "decoder_frame_indices"}:
            raise TimestampProvenanceError("DECODER_FRAME_MAP_BINDING_MISMATCH")
        indices, frames = record["decoder_frame_indices"], projection["frames"]
        refs = [frame["source_frame_reference"] for frame in frames]
        if not isinstance(indices, list) or record["source_frame_references"] != refs or len(indices) != len(frames) or any(type(index) is not int or index < 0 or index >= len(pts) for index in indices) or any(right <= left for left, right in zip(indices, indices[1:])):
            raise TimestampProvenanceError("DECODER_FRAME_MAP_NOT_UNIQUE")
        selected = [pts[index] for index in indices]
        if any(right < left for left, right in zip(selected, selected[1:])):
            raise TimestampProvenanceError("DECODER_PTS_NONMONOTONIC")
        values = selected if mapping["time_origin"] == "SOURCE_VIDEO_ABSOLUTE" else [value - selected[0] for value in selected]
        if values[-1] <= values[0]:
            raise TimestampProvenanceError("DECODER_PTS_DURATION_INVALID")
        outputs.extend([{**{field: projection[field] for field in IDENTITY_FIELDS}, "frame_order": order, "source_frame_reference": ref, "timestamp_seconds": value, "timestamp_unit": "seconds"} for order, (ref, value) in enumerate(zip(refs, values))])
        provenance.append({"identity": _identity_key(projection), "dataset_name": None, "source_type": "DECODER_PTS", "time_origin": mapping["time_origin"], "timestamp_unit": "seconds", "mapping_method": "explicit_decoder_frame_index_map", "adapter_version": "relive-v2-decoder-pts-v1", "frame_count": len(frames)})
    return outputs, provenance


def export_public_timestamps(*, media_projection_path: Path, output_dir: Path, documented_source_path: Path | None = None, public_video_path: Path | None = None, decoder_frame_map_path: Path | None = None, pts_reader: Callable[[Path], list[float]] = _ffprobe_pts) -> dict[str, Any]:
    """Export deterministic public timestamps from exactly one registered source."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TimestampProvenanceError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    projections = _strict_rows(media_projection_path, "PUBLIC_MEDIA_PROJECTION")
    if not projections:
        raise TimestampProvenanceError("PUBLIC_MEDIA_PROJECTION_EMPTY")
    use_doc = documented_source_path is not None
    use_pts = public_video_path is not None or decoder_frame_map_path is not None
    if use_doc and use_pts:
        raise TimestampProvenanceError("TIMESTAMP_SOURCE_AMBIGUOUS")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not use_doc and not use_pts:
        audit = _audit(status="UNRESOLVED_TIMEBASE_SOURCE", created=False, reason="NO_TRUSTED_TIMESTAMP_SOURCE", public_media_projection_sha256=sha256_path(media_projection_path))
        _write(output_dir / "v2_tal_timestamp_source_audit.json", (canonical_json(audit) + "\n").encode("utf-8"))
        return audit
    if use_doc:
        rows, details = _documented_rows(documented_source_path, projections)
        source_path = documented_source_path
    else:
        if public_video_path is None or decoder_frame_map_path is None:
            raise TimestampProvenanceError("DECODER_PTS_INPUT_PAIR_REQUIRED")
        rows, details = _decoder_rows(public_video_path, decoder_frame_map_path, projections, pts_reader)
        source_path = public_video_path
    manifest_path = output_dir / "public_per_frame_timestamps.jsonl"
    _write(manifest_path, _rows_bytes(rows))
    manifest_sha = sha256_path(manifest_path)
    provenance_rows = []
    cursor = 0
    for detail, projection in zip(details, projections):
        count = detail["frame_count"]
        case_rows = rows[cursor:cursor + count]; cursor += count
        identity = {field: projection[field] for field in IDENTITY_FIELDS}
        # One sidecar per exported source/case is unavoidable for multi-case input; use a deterministic directory index.
        sidecar = {"format": TIMESTAMP_PROVENANCE_FORMAT, "timestamp_manifest_sha256": manifest_sha, "source_type": detail["source_type"], "dataset_name": detail["dataset_name"], **identity, "time_origin": detail["time_origin"], "timestamp_unit": "seconds", "mapping_method": detail["mapping_method"], "adapter_version": detail["adapter_version"], "source_reference": str(source_path), "source_reference_sha256": sha256_path(source_path), "frame_count": len(case_rows), "gt_used": False, "assistant_or_gt_values_accessed": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "public_media_projection_sha256": projection["public_media_projection_sha256"], "frame_sha256": [frame["frame_sha256"] for frame in projection["frames"]]}
        provenance_rows.append(sidecar)
    # Stage 2 currently freezes one selected record; for a cohort write deterministic per-record sidecars and index.
    if len(provenance_rows) != 1:
        raise TimestampProvenanceError("TIMESTAMP_EXPORT_REQUIRES_SINGLE_PUBLIC_RECORD")
    sidecar_path = output_dir / "public_per_frame_timestamps.provenance.json"
    _write(sidecar_path, (canonical_json(provenance_rows[0]) + "\n").encode("utf-8"))
    audit = _audit(status="PASS", created=True, reason=None, public_media_projection_sha256=sha256_path(media_projection_path), timestamp_manifest_sha256=manifest_sha, timestamp_provenance_sha256=sha256_path(sidecar_path), source_type=provenance_rows[0]["source_type"], source_reference_sha256=provenance_rows[0]["source_reference_sha256"], frame_count=len(rows))
    _write(output_dir / "v2_tal_timestamp_source_audit.json", (canonical_json(audit) + "\n").encode("utf-8"))
    return audit


def validate_export(output_dir: Path) -> dict[str, Any]:
    audit = _strict_object(output_dir / "v2_tal_timestamp_source_audit.json", "TIMESTAMP_AUDIT")
    if audit.get("format") != TIMESTAMP_AUDIT_FORMAT or audit.get("audit_content_sha256") != stable_hash({key: value for key, value in audit.items() if key != "audit_content_sha256"}):
        raise TimestampProvenanceError("TIMESTAMP_AUDIT_BINDING_MISMATCH")
    if audit.get("timestamp_manifest_created") is False:
        if audit.get("status") != "UNRESOLVED_TIMEBASE_SOURCE":
            raise TimestampProvenanceError("TIMESTAMP_AUDIT_SCHEMA_INVALID")
        return {"status": "PASS", "timestamp_manifest_created": False}
    manifest, sidecar = output_dir / "public_per_frame_timestamps.jsonl", output_dir / "public_per_frame_timestamps.provenance.json"
    rows = _strict_rows(manifest, "PUBLIC_TIMESTAMP_MANIFEST")
    if not rows:
        raise TimestampProvenanceError("PUBLIC_TIMESTAMP_MANIFEST_EMPTY")
    identity = {field: rows[0][field] for field in IDENTITY_FIELDS}
    provenance = _strict_object(sidecar, "PUBLIC_TIMESTAMP_PROVENANCE")
    frames = [{"source_frame_reference": row["source_frame_reference"], "frame_sha256": value} for row, value in zip(rows, provenance.get("frame_sha256", []))]
    validate_timestamp_pair(timestamp_manifest_path=manifest, provenance_path=sidecar, identity=identity, frames=frames)
    return {"status": "PASS", "timestamp_manifest_created": True, "frame_count": len(rows)}
