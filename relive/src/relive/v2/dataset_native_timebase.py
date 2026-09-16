"""Strict NurViD ``frames_2fps`` dataset-native TAL timebase adapter.

This adapter reads only a narrow public metadata allowlist.  It neither opens
human/assistant answers nor any annotation/GT field, and is deliberately not a
generic filename/FPS inference mechanism.
"""
from __future__ import annotations
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib, math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import IDENTITY_FIELDS, TALSelectionError, strict_json_loads
from .video_index import FrameRootMapper, VideoIndexError, _canonical_object, _load_selection, _load_requirements, _sha, _write, _jsonl
from .timestamp_provenance import TIMESTAMP_AUDIT_FORMAT, TIMESTAMP_MANIFEST_FORMAT, TIMESTAMP_PROVENANCE_FORMAT

ADAPTER_VERSION = "relive-v2-nurvid-frames-2fps-timebase-v1"
FRAME_BANK_LAYOUT = "frames_2fps"
FRAME_BANK_RATE = Decimal("2")
INDEX_BASE = 1
ALLOWED = frozenset({"sampled_video_frames", "video", "id", "train", "data_source", "dataset_name", "qa_type", "metadata"})
METADATA_ALLOWED = frozenset({"video_id", "fps", "input_video_start_time", "input_video_end_time"})

class DatasetNativeTimebaseError(ValueError): pass

def _decimal(value: Any, code: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise DatasetNativeTimebaseError(code)
    try: result = Decimal(str(value))
    except Exception as exc: raise DatasetNativeTimebaseError(code) from exc
    if not result.is_finite(): raise DatasetNativeTimebaseError(code)
    return result

def _safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    # Intentionally call get only for fields in this allowlist.
    if not isinstance(row, Mapping): raise DatasetNativeTimebaseError("SOURCE_RECORD_INVALID")
    values = {key: row.get(key) for key in ALLOWED}
    if values["dataset_name"] != "NurViD" or values["data_source"] != "NurViD":
        raise DatasetNativeTimebaseError("UNSUPPORTED_FRAME_BANK_LAYOUT")
    if not isinstance(values["id"], str) or not values["id"] or not isinstance(values["metadata"], Mapping): raise DatasetNativeTimebaseError("SOURCE_PUBLIC_METADATA_INVALID")
    metadata = {key: values["metadata"].get(key) for key in METADATA_ALLOWED}
    if set(metadata) != METADATA_ALLOWED or not isinstance(metadata["video_id"], str) or not metadata["video_id"]: raise DatasetNativeTimebaseError("SOURCE_PUBLIC_METADATA_INVALID")
    video, refs = values["video"], values["sampled_video_frames"]
    if not isinstance(video, list) or not isinstance(refs, list) or not video or len(video) != len(refs): raise DatasetNativeTimebaseError("FRAME_REFERENCE_COUNT_MISMATCH")
    if any(type(ref) is not int or ref < 1 for ref in refs): raise DatasetNativeTimebaseError("FRAME_REFERENCE_INVALID")
    if any(right < left for left, right in zip(refs, refs[1:])): raise DatasetNativeTimebaseError("NON_MONOTONIC_SOURCE_REFERENCE")
    if any(not isinstance(path, str) or not path for path in video): raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH")
    start, end, fps = _decimal(metadata["input_video_start_time"], "SOURCE_TIME_METADATA_INVALID"), _decimal(metadata["input_video_end_time"], "SOURCE_TIME_METADATA_INVALID"), _decimal(metadata["fps"], "SOURCE_TIME_METADATA_INVALID")
    if end <= start or fps <= 0 or fps > FRAME_BANK_RATE: raise DatasetNativeTimebaseError("SOURCE_TIME_METADATA_INVALID")
    return {"id": values["id"], "video": video, "refs": refs, "metadata": metadata, "start": start, "end": end, "fps": fps}

def _path_ref(path: str, video_id: str) -> int:
    parts = PurePosixPath(path).parts
    try: parent = parts[-2]; name = parts[-1]
    except IndexError as exc: raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH") from exc
    if parent != video_id or "frames_2fps" not in parts or not name.endswith(".jpg") or len(name) != 10 or not name[:-4].isdigit(): raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH")
    if parent != video_id: raise DatasetNativeTimebaseError("VIDEO_ID_PATH_MISMATCH")
    return int(name[:-4])

def _expected_bounds(start: Decimal, end: Decimal) -> tuple[int, int]:
    first = int((start * FRAME_BANK_RATE).to_integral_value(rounding=ROUND_CEILING)) + 1
    last = int((end * FRAME_BANK_RATE).to_integral_value(rounding=ROUND_FLOOR)) + 1
    return first, last

def _source_seconds(reference: int) -> Decimal: return (Decimal(reference) - Decimal(INDEX_BASE)) / FRAME_BANK_RATE

def _audit_record(record: Mapping[str, Any], mapper: FrameRootMapper | None, *, require_files: bool) -> dict[str, Any]:
    row = _safe_row(record); refs, paths, md = row["refs"], row["video"], row["metadata"]
    first, last = _expected_bounds(row["start"], row["end"])
    if refs[0] != first or refs[-1] != last: raise DatasetNativeTimebaseError("CLIP_BOUNDARY_GRID_MISMATCH")
    parsed = [_path_ref(path, md["video_id"]) for path in paths]
    if parsed != refs: raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH")
    if require_files:
        if mapper is None: raise DatasetNativeTimebaseError("FRAME_ROOT_REQUIRED")
        # One-based evidence must be present per concrete bank directory.
        first_path = PurePosixPath(paths[0]).parent / "000001.jpg"
        zero_path = PurePosixPath(paths[0]).parent / "000000.jpg"
        try: one_exists = mapper.materialize(str(first_path)).is_file()
        except VideoIndexError: one_exists = False
        try: zero_exists = mapper.materialize(str(zero_path)).is_file()
        except VideoIndexError: zero_exists = False
        if not one_exists or zero_exists: raise DatasetNativeTimebaseError("FRAME_BANK_INDEX_BASE_UNPROVEN")
        for path in paths: mapper.materialize(path)
    source = [_source_seconds(ref) for ref in refs]
    clip = [value - row["start"] for value in source]
    if any(value < row["start"] or value > row["end"] for value in source) or any(value < 0 for value in clip): raise DatasetNativeTimebaseError("SOURCE_TIMESTAMP_OUTSIDE_CLIP")
    if any(right < left for left, right in zip(source, source[1:])) or any(right < left for left, right in zip(clip, clip[1:])): raise DatasetNativeTimebaseError("NON_MONOTONIC_SOURCE_REFERENCE")
    if row["end"] - source[-1] >= Decimal(1) / FRAME_BANK_RATE: raise DatasetNativeTimebaseError("CLIP_BOUNDARY_GRID_MISMATCH")
    return {"record_id": row["id"], "video_id": md["video_id"], "frame_count": len(refs), "duplicate_logical_frame_count": len(refs) - len(set(refs)), "expected_first_reference": first, "expected_last_reference": last, "source_timestamp_seconds": [str(value) for value in source], "clip_timestamp_seconds": [str(value) for value in clip], "start": str(row["start"]), "end": str(row["end"]), "refs": refs, "paths": paths}

def _load_dataset(path: Path) -> list[Any]:
    try: value = strict_json_loads(path.read_bytes().decode("utf-8"), error_code="SOURCE_DATASET")
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc: raise DatasetNativeTimebaseError("SOURCE_DATASET_UNREADABLE") from exc
    if not isinstance(value, list): raise DatasetNativeTimebaseError("SOURCE_DATASET_INVALID")
    return value

def audit_dataset_native_timebase(*, dataset_json: Path, frame_root: Path, source_prefix: str, output_dir: Path, dataset_name: str = "NurViD", data_source: str = "NurViD", frame_bank_layout: str = FRAME_BANK_LAYOUT) -> dict[str, Any]:
    if dataset_name != "NurViD" or data_source != "NurViD" or frame_bank_layout != FRAME_BANK_LAYOUT: raise DatasetNativeTimebaseError("UNSUPPORTED_FRAME_BANK_LAYOUT")
    if output_dir.exists() and any(output_dir.iterdir()): raise DatasetNativeTimebaseError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    mapper = FrameRootMapper.create(source_prefix, frame_root); dataset = _load_dataset(dataset_json)
    checked, failures = [], []
    for index, record in enumerate(dataset):
        # Read only selected top-level fields inside _audit_record; all other row values remain untouched.
        if isinstance(record, Mapping) and record.get("dataset_name") == dataset_name and record.get("data_source") == data_source:
            try: checked.append({"source_record_index": index, "status": "PASS", **_audit_record(record, mapper, require_files=True)})
            except DatasetNativeTimebaseError as exc: failures.append({"source_record_index": index, "status": "UNRESOLVED", "reason_code": str(exc)})
    status = "PASS" if checked and not failures else "UNRESOLVED_TIMEBASE_SOURCE"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"format": "relive-v2-nurvid-dataset-native-timebase-audit-v1", "status": status, "timebase_status": "RESOLVED_DATASET_NATIVE" if status == "PASS" else "UNRESOLVED_TIMEBASE_SOURCE", "timebase_provenance_tier": "DATASET_INTERNAL_DERIVED", "frame_bank_layout": FRAME_BANK_LAYOUT, "frame_bank_rate_hz": 2, "frame_bank_index_base": 1, "timestamp_origin": "CLIP_START", "timestamp_resolution_seconds": 0.5, "original_video_pts_verified": False, "external_original_video_opened": False, "public_metadata_projection_sha256": None, "records_checked": len(checked) + len(failures), "records_passed": len(checked), "failure_count": len(failures), "gt_used": False, "assistant_or_gt_values_accessed": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    records_path = output_dir / "v2_nurvid_dataset_native_timebase_records.jsonl"
    _write(records_path, _jsonl(checked + failures))
    report["public_metadata_projection_sha256"] = _sha(records_path)
    report["audit_content_sha256"] = stable_hash(report)
    _write(output_dir / "v2_nurvid_dataset_native_timebase_audit.json", (canonical_json(report)+"\n").encode())
    return report

def export_dataset_native_timebase(*, dataset_json: Path, selection_manifest: Path, requirement_freeze_dir: Path, media_audit_dir: Path, frame_root: Path, source_prefix: str, output_dir: Path, frame_bank_layout: str = FRAME_BANK_LAYOUT) -> dict[str, Any]:
    if frame_bank_layout != FRAME_BANK_LAYOUT: raise DatasetNativeTimebaseError("UNSUPPORTED_FRAME_BANK_LAYOUT")
    if output_dir.exists() and any(output_dir.iterdir()): raise DatasetNativeTimebaseError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    req, _ = _load_requirements(requirement_freeze_dir); selection, _ = _load_selection(selection_manifest, req)
    projection_path = media_audit_dir / "v2_public_media_projection.jsonl"
    from .video_index import _strict_rows
    projections = _strict_rows(projection_path, code="MEDIA_PROJECTION")
    if len(projections) != len(selection["items"]): raise DatasetNativeTimebaseError("SELECTED_IDENTITY_BINDING_MISMATCH")
    dataset = _load_dataset(dataset_json); mapper = FrameRootMapper.create(source_prefix, frame_root)
    rows: list[dict[str, Any]] = []; sidecars=[]
    for selected, projection in zip(selection["items"], projections):
        if any(projection[key] != selected[key] for key in IDENTITY_FIELDS): raise DatasetNativeTimebaseError("SELECTED_IDENTITY_BINDING_MISMATCH")
        index = selected["source_record_index"]
        if type(index) is not int or index < 0 or index >= len(dataset): raise DatasetNativeTimebaseError("SELECTED_IDENTITY_BINDING_MISMATCH")
        proof = _audit_record(dataset[index], mapper, require_files=True)
        if not selected["sample_id"].startswith("medvidu:" + proof["record_id"] + ":"): raise DatasetNativeTimebaseError("SELECTED_IDENTITY_BINDING_MISMATCH")
        frames = projection["frames"]
        if [frame["source_frame_reference"] for frame in frames] != proof["paths"] or [int(Path(path).stem) for path in proof["paths"]] != proof["refs"]: raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH")
        if len(frames) != len(proof["refs"]): raise DatasetNativeTimebaseError("FRAME_REFERENCE_COUNT_MISMATCH")
        case_rows=[]
        for order, (frame, ref, clip) in enumerate(zip(frames, proof["refs"], proof["clip_timestamp_seconds"])):
            case_rows.append({**{key:selected[key] for key in IDENTITY_FIELDS}, "frame_order": order, "source_frame_reference": frame["source_frame_reference"], "timestamp_seconds": float(Decimal(clip)), "timestamp_unit":"seconds"})
        rows.extend(case_rows)
        sidecars.append((selected, projection, proof, case_rows))
    if len(sidecars) != 1: raise DatasetNativeTimebaseError("TIMESTAMP_EXPORT_REQUIRES_SINGLE_PUBLIC_RECORD")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, projection, proof, case_rows = sidecars[0]
    public_projection = output_dir / "v2_nurvid_public_metadata_projection.jsonl"
    _write(public_projection, _jsonl([{**{key:selected[key] for key in IDENTITY_FIELDS}, "record_id": proof["record_id"], "video_id": proof["video_id"], "source_frame_references": proof["refs"], "input_video_start_time": proof["start"], "input_video_end_time": proof["end"], "frame_bank_layout": FRAME_BANK_LAYOUT, "frame_bank_rate_hz": 2, "frame_bank_index_base": 1}]))
    manifest = output_dir / "public_per_frame_timestamps.jsonl"; _write(manifest, _jsonl(rows)); manifest_sha=_sha(manifest)
    provenance={"format": TIMESTAMP_PROVENANCE_FORMAT, "timestamp_manifest_sha256": manifest_sha, "source_type":"VERSIONED_DATASET_TIMEBASE_ADAPTER", "dataset_name":"NurViD", **{key:selected[key] for key in IDENTITY_FIELDS}, "time_origin":"CLIP_LOCAL_ZERO", "timestamp_origin":"CLIP_START", "timestamp_unit":"seconds", "mapping_method":"clip=(source_frame_reference-1)/2-input_video_start_time", "adapter_version":ADAPTER_VERSION, "source_reference":str(public_projection), "source_reference_sha256":_sha(public_projection), "frame_count":len(case_rows), "gt_used":False, "assistant_or_gt_values_accessed":False, "model_calls_made":0, "backend_loaded":False, "cache_opened":False, "certificate_created":False, "new_verified_count":0, "certificate_status":"NOT_APPLICABLE", "public_media_projection_sha256":projection["public_media_projection_sha256"], "frame_sha256":[frame["frame_sha256"] for frame in projection["frames"]], "source_timestamp_seconds":proof["source_timestamp_seconds"], "input_video_start_time":proof["start"], "input_video_end_time":proof["end"], "frame_bank_rate_hz":2, "frame_bank_index_base":1, "derivation_formula":"(reference-1)/2; clip=source-input_video_start_time", "timestamp_resolution_seconds":0.5, "original_video_pts_verified":False, "external_original_video_opened":False, "duplicate_logical_frame_count":proof["duplicate_logical_frame_count"]}
    sidecar=output_dir / "public_per_frame_timestamps.provenance.json"; _write(sidecar,(canonical_json(provenance)+"\n").encode())
    audit={"format":TIMESTAMP_AUDIT_FORMAT,"status":"PASS","timestamp_manifest_created":True,"timebase_status":"RESOLVED_DATASET_NATIVE","timebase_provenance_tier":"DATASET_INTERNAL_DERIVED","timestamp_manifest_sha256":manifest_sha,"timestamp_provenance_sha256":_sha(sidecar),"public_metadata_projection_sha256":_sha(public_projection),"frame_count":len(rows),"duplicate_logical_frame_count":proof["duplicate_logical_frame_count"],"gt_used":False,"assistant_or_gt_values_accessed":False,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE"}; audit["audit_content_sha256"]=stable_hash(audit); _write(output_dir/"v2_tal_timestamp_source_audit.json",(canonical_json(audit)+"\n").encode())
    return audit
