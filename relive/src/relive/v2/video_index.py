"""GT-isolated public-media audit and immutable TAL VideoIndex freeze.

No temporal event is localized here.  A resolved timebase means only that a
public, per-frame seconds source was validated for a selected public record.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from PIL import Image

from relive.storage.artifacts import canonical_json, stable_hash
from .requirement_freeze import RequirementFreezeError, validate_requirement_freeze_artifacts
from .task_selection import IDENTITY_FIELDS, TALSelectionError, strict_json_loads, strict_jsonl
from .timestamp_provenance import ALLOWED_SOURCE_TYPES, TimestampProvenanceError, validate_timestamp_pair

VIDEO_INDEX_FORMAT = "relive-v2-tal-video-index-freeze-v1"
TIMEBASE_POLICY_FORMAT = "relive-v2-tal-timebase-policy-v1"
TIMEBASE_POLICY_VERSION = "relive-v2-tal-timebase-policy-v1"
TIMEBASE_ADAPTER_VERSION = "relive-v2-public-timestamp-provenance-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN = ("answer", "reference_answer", "assistant_answer", "temporal_gt", "temporal_span", "start_time", "end_time",
              "timestamp_gt", "bbox", "mask", "region", "roi", "struc_info", "rc_info", "evaluation", "model", "certificate")
_ARTIFACTS = ("v2_public_media_projection.jsonl", "v2_video_index.jsonl", "v2_video_index_unresolved.jsonl")


class VideoIndexError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes().decode("utf-8")
        value = strict_json_loads(raw, error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise VideoIndexError(f"{code}_INVALID") from exc
    if not isinstance(value, dict):
        raise VideoIndexError(f"{code}_OBJECT_REQUIRED")
    return value


def _canonical_object(path: Path, *, code: str) -> dict[str, Any]:
    value = _strict_object(path, code=code)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise VideoIndexError(f"{code}_UNREADABLE") from exc
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise VideoIndexError(f"{code}_NONCANONICAL_BYTES")
    return value


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _strict_rows(path: Path, *, code: str, allow_empty: bool = False) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise VideoIndexError(f"{code}_UNREADABLE") from exc
    if not raw and allow_empty:
        return ()
    try:
        rows = strict_jsonl(path, error_code=code)
    except TALSelectionError as exc:
        raise VideoIndexError(f"{code}_INVALID") from exc
    if raw != _jsonl(list(rows)):
        raise VideoIndexError(f"{code}_NONCANONICAL_BYTES")
    return rows


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(any(token in str(key).casefold() for token in _FORBIDDEN) or _contains_forbidden(nested)
                   for key, nested in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    return False


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VideoIndexError(code)
    return float(value)


def _public_question(conversations: Any) -> str:
    if not isinstance(conversations, list):
        raise VideoIndexError("SOURCE_CONVERSATIONS_INVALID")
    humans = []
    for turn in conversations:
        if not isinstance(turn, Mapping):
            raise VideoIndexError("SOURCE_CONVERSATION_TURN_INVALID")
        # Deliberately read role before any value. Non-human values remain unread.
        if turn.get("from") == "human":
            value = turn.get("value")
            if not isinstance(value, str) or not value.strip():
                raise VideoIndexError("SOURCE_HUMAN_QUESTION_INVALID")
            humans.append(value)
    if len(humans) != 1:
        raise VideoIndexError("SOURCE_HUMAN_QUESTION_COUNT_INVALID")
    return humans[0].removeprefix("<video>").lstrip("\r\n ").strip()


def _source_identity(row: Mapping[str, Any], index: int) -> tuple[dict[str, Any], tuple[str, ...], tuple[Any, ...]]:
    """Use only whitelisted source values; never enumerate or propagate raw row."""
    source_id, qa_type = row.get("id"), row.get("qa_type")
    if not isinstance(source_id, str) or not source_id.strip() or not isinstance(qa_type, str) or not qa_type.strip():
        raise VideoIndexError("SOURCE_PUBLIC_IDENTITY_INVALID")
    question = _public_question(row.get("conversations"))
    video, sampled = row.get("video"), row.get("sampled_video_frames")
    if not isinstance(video, list) or not video or not isinstance(sampled, list) or len(video) != len(sampled):
        raise VideoIndexError("SOURCE_PUBLIC_MEDIA_INVALID")
    if any(not isinstance(item, str) or not item.strip() for item in video):
        raise VideoIndexError("SOURCE_FRAME_REFERENCE_INVALID")
    video = [item.strip() for item in video]
    # sampled references are allowed only to establish the public pairing shape.
    # Their values are neither emitted as timestamps nor used for ordering.
    for value in sampled:
        _finite(value, "SOURCE_SAMPLED_REFERENCE_INVALID")
    dataset = row.get("dataset_name")
    if dataset is not None and (not isinstance(dataset, str) or not dataset.strip()):
        raise VideoIndexError("SOURCE_DATASET_NAME_INVALID")
    dataset = dataset.strip() if dataset is not None else None
    public_identity = {"source_record_index": index, "source_id": source_id.strip(), "qa_type": qa_type.strip(),
                       "question": question, "video": list(video), "sampled_frame_count": len(sampled), "dataset_name": dataset}
    public_sha = hashlib.sha256(canonical_json(public_identity).encode("utf-8")).hexdigest()
    identity = {"source_record_index": index, "sample_id": f"medvidu:{source_id.strip()}:{public_sha[:16]}",
                "public_record_sha256": public_sha, "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "qa_type": qa_type.strip(), "question": question, "dataset_name": dataset}
    return identity, tuple(video), tuple(sampled)


def _load_selected_source_records(source_json: Path, selection: dict[str, Any]) -> dict[tuple[Any, ...], tuple[dict[str, Any], tuple[str, ...]]]:
    """Open a mixed source container, while accessing only public whitelisted values."""
    try:
        raw = strict_json_loads(source_json.read_bytes().decode("utf-8"), error_code="SOURCE_CONTAINER")
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise VideoIndexError("SOURCE_CONTAINER_UNREADABLE") from exc
    if not isinstance(raw, list):
        raise VideoIndexError("SOURCE_CONTAINER_ARRAY_REQUIRED")
    fields = ("source_record_index", "sample_id", "public_record_sha256", "question_sha256")
    wanted = {tuple(item[key] for key in fields) for item in selection["items"]}
    output = {}
    for index in sorted({item[0] for item in wanted}):
        if type(index) is not int or index < 0 or index >= len(raw) or not isinstance(raw[index], Mapping):
            raise VideoIndexError("SOURCE_RECORD_INDEX_INVALID")
        identity, references, _ = _source_identity(raw[index], index)
        key = tuple(identity[key] for key in fields)
        if key not in wanted:
            raise VideoIndexError("PUBLIC_RECORD_IDENTITY_MISMATCH")
        output[key] = (identity, references)
    if set(output) != wanted:
        raise VideoIndexError("SELECTION_PUBLIC_IDENTITY_MISMATCH")
    return output


@dataclass(frozen=True)
class FrameRootMapper:
    source_prefix: PurePosixPath
    root: Path
    resolved_root: Path

    @classmethod
    def create(cls, source_prefix: str, frame_root: Path) -> "FrameRootMapper":
        prefix = PurePosixPath(source_prefix)
        if not prefix.is_absolute() or prefix == PurePosixPath("/"):
            raise VideoIndexError("SOURCE_PREFIX_INVALID")
        try:
            resolved = frame_root.expanduser().resolve(strict=True)
        except OSError as exc:
            raise VideoIndexError("FRAME_ROOT_UNREADABLE") from exc
        if not resolved.is_dir():
            raise VideoIndexError("FRAME_ROOT_INVALID")
        return cls(prefix, frame_root, resolved)

    def materialize(self, source_reference: str) -> Path:
        source = PurePosixPath(source_reference)
        if not source.is_absolute() or any(part in {".", ".."} for part in source.parts):
            raise VideoIndexError("SOURCE_FRAME_REFERENCE_PATH_INVALID")
        try:
            relative = source.relative_to(self.source_prefix)
        except ValueError as exc:
            raise VideoIndexError("SOURCE_FRAME_REFERENCE_OUTSIDE_PREFIX") from exc
        candidate = self.resolved_root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.resolved_root)
        except (OSError, ValueError) as exc:
            raise VideoIndexError("FRAME_PATH_ESCAPE_OR_MISSING") from exc
        if not resolved.is_file():
            raise VideoIndexError("FRAME_PATH_NOT_FILE")
        return resolved


def _frame_row(order: int, source_reference: str, mapper: FrameRootMapper) -> dict[str, Any]:
    resolved = mapper.materialize(source_reference)
    try:
        with Image.open(resolved) as image:
            image.verify()
        with Image.open(resolved) as image:
            width, height, image_format = image.width, image.height, image.format
    except Exception as exc:
        raise VideoIndexError("FRAME_IMAGE_DECODE_FAILED") from exc
    if type(width) is not int or type(height) is not int or width < 1 or height < 1 or not isinstance(image_format, str) or not image_format:
        raise VideoIndexError("FRAME_IMAGE_METADATA_INVALID")
    return {"frame_order": order, "source_frame_reference": source_reference, "source_frame_path": source_reference,
            "resolved_frame_path": str(resolved), "frame_sha256": _sha(resolved), "width": width, "height": height,
            "image_format": image_format, "timestamp_seconds": None, "timestamp_provenance": None}


def load_timebase_policy(path: Path) -> tuple[dict[str, Any], str]:
    policy = _canonical_object(path, code="TIMEBASE_POLICY")
    if set(policy) != {"format", "policy_version", "allowed_sources", "default_status"} or policy["format"] != TIMEBASE_POLICY_FORMAT or policy["policy_version"] != TIMEBASE_POLICY_VERSION or policy["default_status"] != "UNRESOLVED_TIMEBASE" or not isinstance(policy["allowed_sources"], list) or set(policy["allowed_sources"]) != ALLOWED_SOURCE_TYPES or len(policy["allowed_sources"]) != len(ALLOWED_SOURCE_TYPES):
        raise VideoIndexError("TIMEBASE_POLICY_SCHEMA_INVALID")
    return policy, _sha(path)


def _timestamps_from_manifest(path: Path, identity: dict[str, Any], frames: list[dict[str, Any]]) -> tuple[list[float], str]:
    rows = _strict_rows(path, code="PUBLIC_TIMESTAMP_MANIFEST")
    fields = {"source_record_index", "sample_id", "public_record_sha256", "question_sha256", "frame_order", "source_frame_reference", "timestamp_seconds", "timestamp_unit"}
    if len(rows) != len(frames):
        raise VideoIndexError("TIMEBASE_MANIFEST_FRAME_COUNT_MISMATCH")
    timestamps = []
    for order, (row, frame) in enumerate(zip(rows, frames)):
        if set(row) != fields or row["timestamp_unit"] != "seconds":
            raise VideoIndexError("TIMEBASE_MANIFEST_SCHEMA_INVALID")
        if any(row[key] != identity[key] for key in IDENTITY_FIELDS) or row["frame_order"] != order or row["source_frame_reference"] != frame["source_frame_reference"]:
            raise VideoIndexError("TIMEBASE_MANIFEST_BINDING_MISMATCH")
        timestamp = _finite(row["timestamp_seconds"], "TIMEBASE_TIMESTAMP_NONFINITE")
        if timestamp < 0:
            raise VideoIndexError("TIMEBASE_TIMESTAMP_NEGATIVE")
        timestamps.append(timestamp)
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise VideoIndexError("TIMEBASE_TIMESTAMP_NONMONOTONIC")
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        raise VideoIndexError("TIMEBASE_DURATION_NOT_POSITIVE")
    return timestamps, _sha(path)


def _load_selection(path: Path, requirement_manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    selection = _canonical_object(path, code="SELECTION_MANIFEST")
    required = {"format", "selection_status", "source_selector_sha256", "selection_basis", "identity_manifest_sha256", "gt_used", "items", "selection_content_sha256"}
    if set(selection) != required or selection["selection_status"] != "FROZEN_PRE_VIDEO_ACCESS" or selection["gt_used"] is not False:
        raise VideoIndexError("SELECTION_MANIFEST_SCHEMA_INVALID")
    content = {key: value for key, value in selection.items() if key != "selection_content_sha256"}
    if selection["selection_content_sha256"] != stable_hash(content):
        raise VideoIndexError("SELECTION_MANIFEST_CONTENT_HASH_MISMATCH")
    digest = _sha(path)
    if requirement_manifest["selection_manifest_sha256"] != digest:
        raise VideoIndexError("SELECTION_REQUIREMENT_BINDING_MISMATCH")
    if not isinstance(selection["items"], list) or not selection["items"]:
        raise VideoIndexError("SELECTION_MANIFEST_ITEMS_INVALID")
    for item in selection["items"]:
        if not isinstance(item, dict) or set(item) != IDENTITY_FIELDS:
            raise VideoIndexError("SELECTION_MANIFEST_ITEM_SCHEMA_INVALID")
    return selection, digest


def _load_requirements(requirement_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        validate_requirement_freeze_artifacts(requirement_dir)
    except RequirementFreezeError as exc:
        raise VideoIndexError("REQUIREMENT_FREEZE_INVALID") from exc
    manifest = _canonical_object(requirement_dir / "v2_requirement_manifest.json", code="REQUIREMENT_MANIFEST")
    try:
        specs = _strict_rows(requirement_dir / "v2_requirement_specs.jsonl", code="REQUIREMENT_SPECS", allow_empty=True)
    except VideoIndexError:
        raise
    return manifest, list(specs)


def _requirement_identity(spec: dict[str, Any]) -> tuple[Any, ...]:
    try:
        identity = spec["provenance"]["public_source_identity"]
        return tuple(identity[key] for key in ("source_record_index", "sample_id", "public_record_sha256", "question_sha256"))
    except (KeyError, TypeError) as exc:
        raise VideoIndexError("REQUIREMENT_PUBLIC_IDENTITY_MISSING") from exc


def _write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise VideoIndexError("IMMUTABLE_OUTPUT_EXISTS")
    path.write_bytes(payload)


def _audit_imports() -> dict[str, Any]:
    import ast
    root = Path(__file__).parent
    sources = [root / "video_index.py"]
    forbidden = ("relive.backends", "relive.runner", "relive.verification", "relive.certificate", "relive.evaluation", "relive.cache", "relive.phase4a")
    imported = []
    for node in ast.walk(ast.parse((root / "video_index.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import): imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imported.append(node.module)
    issues = [name for name in forbidden if any(item == name or item.startswith(name + ".") for item in imported)]
    return {"status": "PASS" if not issues else "FAIL", "files_checked": len(sources), "issues": issues,
            "limitation": "Static import audit cannot independently prove source-value access behavior."}


def freeze_video_index(*, requirement_dir: Path, selection_manifest_path: Path, source_json: Path, frame_root: Path,
                       source_prefix: str, timebase_policy_path: Path, output_dir: Path,
                       public_timestamp_manifest_path: Path | None = None, public_timestamp_provenance_path: Path | None = None) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise VideoIndexError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    imports = _audit_imports()
    if imports["status"] != "PASS":
        raise VideoIndexError("VIDEO_INDEX_STATIC_ISOLATION_AUDIT_FAILED")
    requirement_manifest, specs = _load_requirements(requirement_dir)
    selection, selection_sha = _load_selection(selection_manifest_path, requirement_manifest)
    _, policy_sha = load_timebase_policy(timebase_policy_path)
    mapper = FrameRootMapper.create(source_prefix, frame_root)
    selected = _load_selected_source_records(source_json, selection)
    spec_by_identity = {_requirement_identity(spec): spec for spec in specs}
    if len(spec_by_identity) != len(specs):
        raise VideoIndexError("REQUIREMENT_IDENTITY_DUPLICATE")
    projection_rows, index_rows, unresolved_rows = [], [], []
    for item in selection["items"]:
        identity_key = tuple(item[key] for key in ("source_record_index", "sample_id", "public_record_sha256", "question_sha256"))
        identity, references = selected[identity_key]
        frames = [_frame_row(order, reference, mapper) for order, reference in enumerate(references)]
        projection = {"sample_id": identity["sample_id"], "source_record_index": identity["source_record_index"],
                      "public_record_sha256": identity["public_record_sha256"], "question_sha256": identity["question_sha256"],
                      "frame_count": len(frames), "logical_frame_order_preserved": True, "frames": frames}
        projection["public_media_projection_sha256"] = stable_hash(projection)
        projection_rows.append(projection)
        spec = spec_by_identity.get(identity_key)
        if spec is None:
            raise VideoIndexError("SELECTION_REQUIREMENT_IDENTITY_MISMATCH")
        if public_timestamp_manifest_path is None and public_timestamp_provenance_path is None:
            unresolved_rows.append({"requirement_id": spec["requirement_id"], "sample_id": identity["sample_id"], "source_record_index": identity["source_record_index"], "public_record_sha256": identity["public_record_sha256"], "question_sha256": identity["question_sha256"], "timebase_status": "UNRESOLVED_TIMEBASE", "reason_code": "NO_VALIDATED_PUBLIC_SECONDS_SOURCE", "public_media_projection_sha256": projection["public_media_projection_sha256"]})
            continue
        if public_timestamp_manifest_path is None or public_timestamp_provenance_path is None:
            unresolved_rows.append({"requirement_id": spec["requirement_id"], "sample_id": identity["sample_id"], "source_record_index": identity["source_record_index"], "public_record_sha256": identity["public_record_sha256"], "question_sha256": identity["question_sha256"], "timebase_status": "UNRESOLVED_TIMEBASE", "reason_code": "TIMESTAMP_PROVENANCE_PAIR_REQUIRED", "public_media_projection_sha256": projection["public_media_projection_sha256"]})
            continue
        try:
            timestamps, timestamp_provenance, timestamp_manifest_sha, timestamp_provenance_sha = validate_timestamp_pair(timestamp_manifest_path=public_timestamp_manifest_path, provenance_path=public_timestamp_provenance_path, identity=identity, frames=frames)
        except TimestampProvenanceError as exc:
            unresolved_rows.append({"requirement_id": spec["requirement_id"], "sample_id": identity["sample_id"], "source_record_index": identity["source_record_index"], "public_record_sha256": identity["public_record_sha256"], "question_sha256": identity["question_sha256"], "timebase_status": "UNRESOLVED_TIMEBASE", "reason_code": str(exc), "public_media_projection_sha256": projection["public_media_projection_sha256"]})
            continue
        if timestamp_provenance.get("public_media_projection_sha256") != projection["public_media_projection_sha256"]:
            unresolved_rows.append({"requirement_id": spec["requirement_id"], "sample_id": identity["sample_id"], "source_record_index": identity["source_record_index"], "public_record_sha256": identity["public_record_sha256"], "question_sha256": identity["question_sha256"], "timebase_status": "UNRESOLVED_TIMEBASE", "reason_code": "TIMESTAMP_PROVENANCE_MEDIA_BINDING_MISMATCH", "public_media_projection_sha256": projection["public_media_projection_sha256"]})
            continue
        indexed_frames = []
        duplicate_timestamps = []
        for frame, timestamp in zip(frames, timestamps):
            indexed = dict(frame)
            indexed["timestamp_seconds"] = timestamp
            indexed["timestamp_provenance"] = {"adapter_version": timestamp_provenance["adapter_version"], "timestamp_source": timestamp_provenance["source_type"], "timestamp_unit": "seconds", "source_sha256": timestamp_manifest_sha, "provenance_sha256": timestamp_provenance_sha, "time_origin": timestamp_provenance["time_origin"]}
            indexed_frames.append(indexed)
        duplicate_timestamps = [order for order, (left, right) in enumerate(zip(timestamps, timestamps[1:]), start=1) if left == right]
        dataset_native = timestamp_provenance["source_type"] == "VERSIONED_DATASET_TIMEBASE_ADAPTER"
        index = {"requirement_id": spec["requirement_id"], "sample_id": identity["sample_id"], "source_record_index": identity["source_record_index"],
                 "public_record_sha256": identity["public_record_sha256"], "question_sha256": identity["question_sha256"],
                 "requirement_manifest_sha256": _sha(requirement_dir / "v2_requirement_manifest.json"),
                 "selection_manifest_sha256": selection_sha, "public_media_projection_sha256": projection["public_media_projection_sha256"],
                 "timebase_status": "RESOLVED_DATASET_NATIVE" if dataset_native else "RESOLVED", "timebase_adapter_version": timestamp_provenance["adapter_version"],
                 "timestamp_source": timestamp_provenance["source_type"], "timestamp_unit": "seconds", "timestamp_provenance_sha256": timestamp_provenance_sha, "time_origin": timestamp_provenance["time_origin"], "timestamp_origin": timestamp_provenance.get("timestamp_origin", "CLIP_START" if dataset_native else "CLIP_LOCAL_ZERO"),
                 "frame_count": len(indexed_frames), "duplicate_timestamp_frame_orders": duplicate_timestamps, "frames": indexed_frames}
        index["video_index_id"] = "video_index_" + stable_hash(index)[:24]
        index_rows.append(index)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in _ARTIFACTS}
    _write(paths["v2_public_media_projection.jsonl"], _jsonl(projection_rows))
    _write(paths["v2_video_index.jsonl"], _jsonl(index_rows))
    _write(paths["v2_video_index_unresolved.jsonl"], _jsonl(unresolved_rows))
    index_status = ("RESOLVED_DATASET_NATIVE" if index_rows and not unresolved_rows and all(row["timebase_status"] == "RESOLVED_DATASET_NATIVE" for row in index_rows) else "RESOLVED") if index_rows and not unresolved_rows else ("UNRESOLVED_TIMEBASE" if not index_rows else "PARTIAL_UNRESOLVED_TIMEBASE")
    manifest = {"format": VIDEO_INDEX_FORMAT, "status": "PASS", "index_status": index_status,
                "ready_for_hypothesis_generation": index_status == "RESOLVED_DATASET_NATIVE", "requirement_manifest_sha256": _sha(requirement_dir / "v2_requirement_manifest.json"),
                "selection_manifest_sha256": selection_sha, "timebase_policy_sha256": policy_sha,
                "public_timestamp_manifest_sha256": _sha(public_timestamp_manifest_path) if public_timestamp_manifest_path else None, "public_timestamp_provenance_sha256": _sha(public_timestamp_provenance_path) if public_timestamp_provenance_path else None,
                "public_media_projection_count": len(projection_rows), "video_index_count": len(index_rows), "unresolved_count": len(unresolved_rows),
                "public_media_projection_sha256": _sha(paths["v2_public_media_projection.jsonl"]), "video_index_sha256": _sha(paths["v2_video_index.jsonl"]),
                "video_index_unresolved_sha256": _sha(paths["v2_video_index_unresolved.jsonl"]), "source_container_opened": True,
                "assistant_or_gt_values_accessed": False, "gt_used": False, "frames_read": sum(row["frame_count"] for row in projection_rows),
                "videos_read": 0, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "claim_graph_created": False,
                "hypothesis_count": 0, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    manifest_path = output_dir / "v2_video_index_manifest.json"
    _write(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    audit = {"format": VIDEO_INDEX_FORMAT, "status": "PASS", "index_status": index_status,
             "source_container_opened": True, "assistant_or_gt_values_accessed": False, "imports_audited": imports,
             "allowed_inputs_opened": ["frozen RequirementSpec artifacts", "frozen selection manifest", "public source container", "selected public frame bytes", "versioned timebase policy"],
             "forbidden_inputs_not_opened": ["assistant answer", "reference answer", "temporal GT", "bbox", "mask", "ROI", "struc_info", "RC_info", "evaluation", "model output", "certificate"],
             "gt_used": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "claim_graph_created": False,
             "hypothesis_count": 0, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE",
             "manifest_sha256": _sha(manifest_path)}
    audit["audit_content_sha256"] = stable_hash(audit)
    _write(output_dir / "v2_video_index_audit.json", (canonical_json(audit) + "\n").encode("utf-8"))
    return audit


def validate_video_index_artifacts(output_dir: Path, *, materialize_frames: bool = False) -> dict[str, Any]:
    paths = {name: output_dir / name for name in (*_ARTIFACTS, "v2_video_index_manifest.json", "v2_video_index_audit.json")}
    if not all(path.is_file() for path in paths.values()):
        raise VideoIndexError("VIDEO_INDEX_ARTIFACT_MISSING")
    manifest = _canonical_object(paths["v2_video_index_manifest.json"], code="VIDEO_INDEX_MANIFEST")
    audit = _canonical_object(paths["v2_video_index_audit.json"], code="VIDEO_INDEX_AUDIT")
    required = {"format", "status", "index_status", "ready_for_hypothesis_generation", "requirement_manifest_sha256", "selection_manifest_sha256", "timebase_policy_sha256", "public_timestamp_manifest_sha256", "public_timestamp_provenance_sha256", "public_media_projection_count", "video_index_count", "unresolved_count", "public_media_projection_sha256", "video_index_sha256", "video_index_unresolved_sha256", "source_container_opened", "assistant_or_gt_values_accessed", "gt_used", "frames_read", "videos_read", "model_calls_made", "backend_loaded", "cache_opened", "claim_graph_created", "hypothesis_count", "certificate_created", "new_verified_count", "certificate_status", "manifest_content_sha256"}
    if set(manifest) != required or manifest["format"] != VIDEO_INDEX_FORMAT:
        raise VideoIndexError("VIDEO_INDEX_MANIFEST_SCHEMA_INVALID")
    if manifest["manifest_content_sha256"] != stable_hash({key: value for key, value in manifest.items() if key != "manifest_content_sha256"}):
        raise VideoIndexError("VIDEO_INDEX_BINDING_MISMATCH")
    hashes = {"public_media_projection_sha256": "v2_public_media_projection.jsonl", "video_index_sha256": "v2_video_index.jsonl", "video_index_unresolved_sha256": "v2_video_index_unresolved.jsonl"}
    if any(not isinstance(manifest[key], str) or not _HEX.fullmatch(manifest[key]) or manifest[key] != _sha(paths[name]) for key, name in hashes.items()):
        raise VideoIndexError("VIDEO_INDEX_ARTIFACT_BYTES_CHANGED")
    projection = _strict_rows(paths["v2_public_media_projection.jsonl"], code="MEDIA_PROJECTION", allow_empty=True)
    indexes = _strict_rows(paths["v2_video_index.jsonl"], code="VIDEO_INDEX", allow_empty=True)
    unresolved = _strict_rows(paths["v2_video_index_unresolved.jsonl"], code="VIDEO_INDEX_UNRESOLVED", allow_empty=True)
    if (len(projection), len(indexes), len(unresolved)) != (manifest["public_media_projection_count"], manifest["video_index_count"], manifest["unresolved_count"]):
        raise VideoIndexError("VIDEO_INDEX_COUNT_MISMATCH")
    if manifest["index_status"] == "UNRESOLVED_TIMEBASE" and indexes:
        raise VideoIndexError("VIDEO_INDEX_TIMEBASE_UNRESOLVED")
    if manifest["index_status"] in {"RESOLVED", "RESOLVED_DATASET_NATIVE"} and (not indexes or unresolved):
        raise VideoIndexError("VIDEO_INDEX_TIMEBASE_INVALID")
    if manifest["index_status"] not in {"RESOLVED", "RESOLVED_DATASET_NATIVE", "UNRESOLVED_TIMEBASE", "PARTIAL_UNRESOLVED_TIMEBASE"} or manifest["ready_for_hypothesis_generation"] is not (manifest["index_status"] == "RESOLVED_DATASET_NATIVE"):
        raise VideoIndexError("VIDEO_INDEX_MANIFEST_SCHEMA_INVALID")
    for name in ("public_timestamp_manifest_sha256", "public_timestamp_provenance_sha256"):
        if manifest[name] is not None and (not isinstance(manifest[name], str) or not _HEX.fullmatch(manifest[name])):
            raise VideoIndexError("VIDEO_INDEX_MANIFEST_SCHEMA_INVALID")
    if (manifest["public_timestamp_manifest_sha256"] is None) != (manifest["public_timestamp_provenance_sha256"] is None):
        raise VideoIndexError("VIDEO_INDEX_TIMESTAMP_PROVENANCE_PAIR_REQUIRED")
    projection_by_identity = {}
    projection_fields = {"sample_id", "source_record_index", "public_record_sha256", "question_sha256", "frame_count", "logical_frame_order_preserved", "frames", "public_media_projection_sha256"}
    frame_fields = {"frame_order", "source_frame_reference", "source_frame_path", "resolved_frame_path", "frame_sha256", "width", "height", "image_format", "timestamp_seconds", "timestamp_provenance"}
    for row in projection:
        if set(row) != projection_fields or row["logical_frame_order_preserved"] is not True or not isinstance(row["frames"], list) or len(row["frames"]) != row["frame_count"]:
            raise VideoIndexError("VIDEO_INDEX_PROJECTION_SCHEMA_INVALID")
        payload = {key: value for key, value in row.items() if key != "public_media_projection_sha256"}
        if row["public_media_projection_sha256"] != stable_hash(payload):
            raise VideoIndexError("VIDEO_INDEX_BINDING_MISMATCH")
        for order, frame in enumerate(row["frames"]):
            if not isinstance(frame, dict) or set(frame) != frame_fields or frame["frame_order"] != order or frame["timestamp_seconds"] is not None or frame["timestamp_provenance"] is not None:
                raise VideoIndexError("VIDEO_INDEX_PROJECTION_FRAME_SCHEMA_INVALID")
            if not isinstance(frame["resolved_frame_path"], str) or not _HEX.fullmatch(frame["frame_sha256"]) or type(frame["width"]) is not int or type(frame["height"]) is not int or frame["width"] < 1 or frame["height"] < 1:
                raise VideoIndexError("VIDEO_INDEX_PROJECTION_FRAME_INVALID")
        identity = tuple(row[key] for key in ("source_record_index", "sample_id", "public_record_sha256", "question_sha256"))
        if identity in projection_by_identity:
            raise VideoIndexError("VIDEO_INDEX_PROJECTION_DUPLICATE_IDENTITY")
        projection_by_identity[identity] = row
    index_fields = {"video_index_id", "requirement_id", "sample_id", "source_record_index", "public_record_sha256", "question_sha256", "requirement_manifest_sha256", "selection_manifest_sha256", "public_media_projection_sha256", "timebase_status", "timebase_adapter_version", "timestamp_source", "timestamp_unit", "timestamp_provenance_sha256", "time_origin", "timestamp_origin", "frame_count", "duplicate_timestamp_frame_orders", "frames"}
    for row in indexes:
        if set(row) != index_fields or row["timebase_status"] not in {"RESOLVED", "RESOLVED_DATASET_NATIVE"} or not isinstance(row["timebase_adapter_version"], str) or not row["timebase_adapter_version"] or row["timestamp_source"] not in ALLOWED_SOURCE_TYPES or row["timestamp_unit"] != "seconds" or row["time_origin"] not in {"CLIP_LOCAL_ZERO", "SOURCE_VIDEO_ABSOLUTE"} or row["timestamp_origin"] not in {"CLIP_START", "CLIP_LOCAL_ZERO", "SOURCE_VIDEO_ABSOLUTE"} or not _HEX.fullmatch(row["timestamp_provenance_sha256"]) or not isinstance(row["frames"], list) or len(row["frames"]) != row["frame_count"]:
            raise VideoIndexError("VIDEO_INDEX_SCHEMA_INVALID")
        identity = tuple(row[key] for key in ("source_record_index", "sample_id", "public_record_sha256", "question_sha256"))
        projected = projection_by_identity.get(identity)
        if projected is None or row["public_media_projection_sha256"] != projected["public_media_projection_sha256"]:
            raise VideoIndexError("VIDEO_INDEX_BINDING_MISMATCH")
        index_payload = {key: value for key, value in row.items() if key != "video_index_id"}
        if row["video_index_id"] != "video_index_" + stable_hash(index_payload)[:24]:
            raise VideoIndexError("VIDEO_INDEX_BINDING_MISMATCH")
        for order, frame in enumerate(row["frames"]):
            if not isinstance(frame, dict) or set(frame) != frame_fields or frame["frame_order"] != order or _finite(frame["timestamp_seconds"], "VIDEO_INDEX_TIMESTAMP_NONFINITE") < 0:
                raise VideoIndexError("VIDEO_INDEX_FRAME_SCHEMA_INVALID")
            provenance = frame["timestamp_provenance"]
            if not isinstance(provenance, dict) or set(provenance) != {"adapter_version", "timestamp_source", "timestamp_unit", "source_sha256", "provenance_sha256", "time_origin"} or provenance["timestamp_source"] not in ALLOWED_SOURCE_TYPES or provenance["timestamp_unit"] != "seconds" or provenance["time_origin"] not in {"CLIP_LOCAL_ZERO", "SOURCE_VIDEO_ABSOLUTE"} or not _HEX.fullmatch(provenance["source_sha256"]) or not _HEX.fullmatch(provenance["provenance_sha256"]):
                raise VideoIndexError("VIDEO_INDEX_TIMESTAMP_PROVENANCE_INVALID")
        times = [frame["timestamp_seconds"] for frame in row["frames"]]
        if any(right < left for left, right in zip(times, times[1:])) or len(times) < 2 or times[-1] <= times[0]:
            raise VideoIndexError("VIDEO_INDEX_TIMEBASE_INVALID")
    unresolved_fields = {"requirement_id", "sample_id", "source_record_index", "public_record_sha256", "question_sha256", "timebase_status", "reason_code", "public_media_projection_sha256"}
    for row in unresolved:
        if set(row) != unresolved_fields or row["timebase_status"] != "UNRESOLVED_TIMEBASE":
            raise VideoIndexError("VIDEO_INDEX_UNRESOLVED_SCHEMA_INVALID")
        identity = tuple(row[key] for key in ("source_record_index", "sample_id", "public_record_sha256", "question_sha256"))
        if identity not in projection_by_identity or row["public_media_projection_sha256"] != projection_by_identity[identity]["public_media_projection_sha256"]:
            raise VideoIndexError("VIDEO_INDEX_BINDING_MISMATCH")
    if audit.get("audit_content_sha256") != stable_hash({key: value for key, value in audit.items() if key != "audit_content_sha256"}) or audit.get("manifest_sha256") != _sha(paths["v2_video_index_manifest.json"]) or audit.get("imports_audited", {}).get("status") != "PASS":
        raise VideoIndexError("VIDEO_INDEX_BINDING_MISMATCH")
    if materialize_frames:
        for index in indexes:
            for frame in index["frames"]:
                path = Path(frame["resolved_frame_path"])
                try:
                    actual = _sha(path)
                except OSError as exc:
                    raise VideoIndexError("VIDEO_INDEX_FRAME_BYTES_CHANGED") from exc
                if actual != frame["frame_sha256"]:
                    raise VideoIndexError("VIDEO_INDEX_FRAME_BYTES_CHANGED")
    return {"status": "PASS", "index_status": manifest["index_status"], "video_index_count": manifest["video_index_count"],
            "ready_for_hypothesis_generation": manifest["ready_for_hypothesis_generation"], "materialized_frames_validated": materialize_frames,
            "gt_used": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
            "hypothesis_count": 0, "certificate_created": False, "new_verified_count": 0,
            "certificate_status": "NOT_APPLICABLE"}
