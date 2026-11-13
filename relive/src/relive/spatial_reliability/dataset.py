from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .common import SpatialReliabilityError, canonical_bytes, runtime_gt_audit, stable_hash, strict_json_load, write_once
from .protocol import PROTOCOL_VERSION, protocol_manifest


DATASET_FORMAT = "relive-spatial-rc-dataset-v1"
NORMALIZER_VERSION = "relive-caption-to-claim-normalizer-v1"
SCOPE_ROUTER_VERSION = "relive-spatial-claim-scope-router-v1"
RUNTIME_FIELDS = frozenset({
    "format", "sample_id", "source_record_index", "source_record_sha256", "video_id",
    "dataset_name", "frame_order", "frame_path", "frame_sha256", "claim_id", "claim_text",
    "claim_scope", "split", "normalizer_version", "scope_router_version",
})
SCOPES = frozenset({"LOCAL_ATOMIC", "LOCAL_RELATIONAL", "MULTI_SUPPORT_POSSIBLE",
                    "GLOBAL_DISTRIBUTED", "TEMPORAL", "UNRESOLVED_SCOPE"})


def _caption(row: dict[str, Any]) -> str:
    turns = row.get("conversations")
    if not isinstance(turns, list):
        raise SpatialReliabilityError("RC_CONVERSATIONS_INVALID")
    values = [turn.get("value") for turn in turns if isinstance(turn, dict) and turn.get("from") != "human"]
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise SpatialReliabilityError("RC_CAPTION_NOT_UNIQUE")
    return " ".join(values[0].split())


def _video_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("video_id"), str) and metadata["video_id"].strip():
        return metadata["video_id"].strip()
    identifier = row.get("id")
    if isinstance(identifier, str) and len(identifier.split("$")) >= 2:
        return identifier.split("$")[1]
    raise SpatialReliabilityError("RC_VIDEO_ID_UNRESOLVED")


def _rc(row: dict[str, Any]) -> tuple[str, tuple[float, float, float, float]]:
    info = row.get("RC_info")
    if not isinstance(info, dict) or set(info) != {"start_frame", "start_frame_bbox"}:
        raise SpatialReliabilityError("RC_INFO_SCHEMA_INVALID")
    frame, box = info["start_frame"], info["start_frame_bbox"]
    if not isinstance(frame, str) or not frame or not isinstance(box, list) or len(box) != 4:
        raise SpatialReliabilityError("RC_INFO_SCHEMA_INVALID")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
        raise SpatialReliabilityError("RC_BBOX_INVALID")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not 0 <= x1 < x2 or not 0 <= y1 < y2:
        raise SpatialReliabilityError("RC_BBOX_INVALID")
    return frame, (x1, y1, x2, y2)


def rc_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    frame, box = _rc(row)
    return (row.get("dataset_name"), _video_id(row), frame, box)


def _first_clause(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+|;", text.strip(), maxsplit=1)[0].strip()


def route_claim_scope(text: str) -> tuple[str, str]:
    folded = " ".join(text.casefold().split())
    if not folded:
        return "UNRESOLVED_SCOPE", "EMPTY_CLAIM"
    temporal = (r"\bthroughout\b", r"\bacross (?:the )?(?:sequence|frames|video)\b", r"\bfrom .+ to .+\b",
                r"\bduring\b", r"\bcontinu(?:es|ing|ously)\b", r"\bthen\b", r"\bmoves?\b")
    if any(re.search(pattern, folded) for pattern in temporal):
        return "TEMPORAL", "REQUIRES_CROSS_FRAME_OR_MOTION_EVIDENCE"
    if re.search(r"\b(?:at least one|one or more|any|some)\b", folded):
        return "MULTI_SUPPORT_POSSIBLE", "EXISTENTIAL_OR_INTERCHANGEABLE_SUPPORT"
    if re.search(r"\b(?:overall|entire|scene|procedure|surgical field|viewpoint)\b", folded):
        return "GLOBAL_DISTRIBUTED", "GLOBAL_SCENE_ASSERTION"
    relational = (r"\b(?:adjacent|next to|between|inside|outside|against|contact|touch|hold|grasp|retract|manipulat|insert)\w*\b",
                  r"\b(?:left|right|above|below) of\b")
    if any(re.search(pattern, folded) for pattern in relational):
        return "LOCAL_RELATIONAL", "MULTI_ENTITY_OR_RELATIONAL_CLAIM"
    if re.search(r"\b(?:visible|identified|present|open|closed|curved|metallic|tissue|needle|instrument|tool|hand|organ|lesion|incision)\b", folded):
        return "LOCAL_ATOMIC", "SINGLE_LOCAL_OBSERVABLE_PROPOSITION"
    return "UNRESOLVED_SCOPE", "SCOPE_NOT_DETERMINABLE"


def normalize_caption(caption: str) -> dict[str, Any]:
    """Conservative delete/normalize-only transform; bbox is not an input."""
    if not isinstance(caption, str) or not caption.strip():
        raise SpatialReliabilityError("CAPTION_MUST_BE_NONEMPTY")
    original = " ".join(caption.split())
    clause = _first_clause(original)
    clause = re.sub(r"^(?:the )?region (?:contains|shows|depicts)\s+", "", clause, flags=re.I)
    clause = re.sub(r"^(?:in the (?:specified|given) region,?\s*)", "", clause, flags=re.I)
    clause = clause.strip(" ,.;:")
    scope, reason = route_claim_scope(clause)
    if scope != "LOCAL_ATOMIC":
        return {"status": "EXCLUDED", "claim_text": None, "scope": scope, "reason": reason,
                "normalizer_version": NORMALIZER_VERSION}
    if not clause or clause.casefold() not in original.casefold():
        return {"status": "EXCLUDED", "claim_text": None, "scope": "UNRESOLVED_SCOPE",
                "reason": "NORMALIZATION_NOT_DELETION_ONLY", "normalizer_version": NORMALIZER_VERSION}
    claim = clause[0].upper() + clause[1:]
    if claim[-1] not in ".!?":
        claim += "."
    return {"status": "ACCEPTED", "claim_text": claim, "scope": scope,
            "reason": "DELETION_ONLY_LOCAL_ATOMIC_CLAUSE", "normalizer_version": NORMALIZER_VERSION}


def parse_proposal(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError):
        return {"status": "PARSE_FAILURE", "bbox_normalized_0_1000": None, "reason": "INVALID_JSON"}
    if not isinstance(value, dict) or set(value) != {"status", "bbox_normalized_0_1000", "reason"}:
        return {"status": "PARSE_FAILURE", "bbox_normalized_0_1000": None, "reason": "SCHEMA_VIOLATION"}
    status, box, reason = value["status"], value["bbox_normalized_0_1000"], value["reason"]
    if status not in {"PROPOSED", "UNRESOLVED"} or not isinstance(reason, str):
        return {"status": "PARSE_FAILURE", "bbox_normalized_0_1000": None, "reason": "SCHEMA_VIOLATION"}
    if status == "UNRESOLVED":
        return value if box is None else {"status": "PARSE_FAILURE", "bbox_normalized_0_1000": None, "reason": "UNRESOLVED_REQUIRES_NULL_BBOX"}
    if (not isinstance(box, list) or len(box) != 4 or any(type(item) is not int for item in box)
            or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)):
        return {"status": "PARSE_FAILURE", "bbox_normalized_0_1000": None, "reason": "INVALID_BBOX"}
    return value


def _map_frame(source: str, source_prefix: str | None, frame_root: str | None) -> str:
    if source_prefix is None and frame_root is None:
        return source
    if not source_prefix or not frame_root:
        raise SpatialReliabilityError("SOURCE_PREFIX_AND_FRAME_ROOT_REQUIRED_TOGETHER")
    source_path, prefix = PurePosixPath(source), PurePosixPath(source_prefix)
    try:
        relative = source_path.relative_to(prefix)
    except ValueError as exc:
        raise SpatialReliabilityError("FRAME_OUTSIDE_SOURCE_PREFIX") from exc
    root = Path(frame_root).expanduser().resolve(strict=True)
    candidate = root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise SpatialReliabilityError("FRAME_MAPPING_INVALID")
    return str(candidate)


def prepare_rc_dataset(*, source_json: str | Path, output_dir: str | Path,
                       source_prefix: str | None = None, frame_root: str | None = None,
                       dev_count: int = 20) -> dict[str, Any]:
    rows = strict_json_load(source_json)
    if not isinstance(rows, list) or not rows:
        raise SpatialReliabilityError("RC_SOURCE_MUST_BE_NONEMPTY_ARRAY")
    candidates = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not str(row.get("qa_type", "")).startswith("region_caption"):
            continue
        frame, box = _rc(row)
        videos = row.get("video")
        if not isinstance(videos, list) or frame not in videos:
            raise SpatialReliabilityError("RC_START_FRAME_NOT_IN_VIDEO")
        record_sha = stable_hash(row)
        candidates.append({"index": index, "row": row, "identity": rc_identity(row), "record_sha": record_sha,
                           "caption": _caption(row), "frame": frame, "box": box, "frame_order": videos.index(frame)})
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        groups[item["identity"]].append(item)
    retained, duplicate_rows = [], []
    for identity, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        ordered = sorted(group, key=lambda item: (item["record_sha"], item["index"]))
        retained.append(ordered[0])
        if len(ordered) > 1:
            duplicate_rows.append({
                "identity_sha256": stable_hash(identity), "source_record_indexes": [item["index"] for item in ordered],
                "retained_source_record_index": ordered[0]["index"], "retained_source_record_sha256": ordered[0]["record_sha"],
                "discarded_source_record_indexes": [item["index"] for item in ordered[1:]],
            })
    normalized, exclusions = [], []
    for item in retained:
        decision = normalize_caption(item["caption"])
        entry = {"source_record_index": item["index"], "source_record_sha256": item["record_sha"], **decision}
        (normalized if decision["status"] == "ACCEPTED" else exclusions).append((item, entry))
    ordered_eligible = sorted(normalized, key=lambda pair: stable_hash({"identity": pair[0]["identity"], "record": pair[0]["record_sha"]}))
    runtime, oracle, split_rows = [], [], []
    for rank, (item, decision) in enumerate(ordered_eligible):
        row = item["row"]
        sample_id = "relive-spatial-" + stable_hash(item["identity"])[:24]
        split = "development" if rank < min(dev_count, len(ordered_eligible)) else "frozen_evaluation"
        mapped = _map_frame(item["frame"], source_prefix, frame_root)
        frame_sha = hashlib.sha256(Path(mapped).read_bytes()).hexdigest() if Path(mapped).is_file() else None
        claim_id = "claim-" + stable_hash({"sample_id": sample_id, "text": decision["claim_text"]})[:24]
        runtime_row = {
            "format": DATASET_FORMAT, "sample_id": sample_id, "source_record_index": item["index"],
            "source_record_sha256": item["record_sha"], "video_id": _video_id(row),
            "dataset_name": row.get("dataset_name"), "frame_order": item["frame_order"], "frame_path": mapped,
            "frame_sha256": frame_sha, "claim_id": claim_id, "claim_text": decision["claim_text"],
            "claim_scope": "LOCAL_ATOMIC", "split": split, "normalizer_version": NORMALIZER_VERSION,
            "scope_router_version": SCOPE_ROUTER_VERSION,
        }
        if set(runtime_row) != RUNTIME_FIELDS:
            raise AssertionError("runtime schema drift")
        runtime.append(runtime_row)
        oracle.append({
            "format": DATASET_FORMAT + "-oracle", "sample_id": sample_id,
            "source_record_index": item["index"], "source_record_sha256": item["record_sha"],
            "original_region_caption": item["caption"], "gt_bbox": list(item["box"]),
            "source_metadata": {"dataset_name": row.get("dataset_name"), "video_id": _video_id(row),
                                "start_frame": item["frame"], "qa_type": row.get("qa_type")},
        })
        split_rows.append({"sample_id": sample_id, "split": split,
                           "hash": stable_hash({"identity": item["identity"], "record": item["record_sha"]})})
    gt_audit = runtime_gt_audit(runtime)
    if gt_audit["status"] != "PASS":
        raise SpatialReliabilityError("RUNTIME_GT_ISOLATION_FAILED")
    output = Path(output_dir)
    dedup = {"format": DATASET_FORMAT + "-dedup", "raw_rc_count": len(candidates),
             "unique_spatial_regions": len(groups), "duplicate_group_count": len(duplicate_rows),
             "duplicate_groups": duplicate_rows,
             "tie_breaking_rule": "minimum canonical source-record SHA-256, then minimum source_record_index"}
    scope_report = {"format": DATASET_FORMAT + "-claim-scope", "router_version": SCOPE_ROUTER_VERSION,
                    "counts": dict(sorted(Counter(entry[1]["scope"] for entry in [*normalized, *exclusions]).items())),
                    "excluded": [entry for _, entry in exclusions], "eligible_count": len(runtime)}
    summary = {"format": DATASET_FORMAT, "protocol_version": PROTOCOL_VERSION, "source_sha256": hashlib.sha256(Path(source_json).read_bytes()).hexdigest(),
               "raw_rc_count": len(candidates), "deduplicated_count": len(retained), "eligible_count": len(runtime),
               "excluded_count": len(exclusions), "development_count": sum(row["split"] == "development" for row in runtime),
               "frozen_evaluation_count": sum(row["split"] == "frozen_evaluation" for row in runtime),
               "model_calls_made": 0, "cache_writes": 0, "certificate_created": False, "new_verified_count": 0}
    artifacts = {
        "protocol.json": protocol_manifest(), "runtime_manifest.jsonl": runtime,
        "oracle_manifest.jsonl": oracle, "dedup_report.json": dedup,
        "claim_scope_report.json": scope_report, "split_manifest.json": {"format": DATASET_FORMAT + "-split", "rows": split_rows},
        "dataset_summary.json": summary, "gt_isolation_audit.json": gt_audit,
    }
    for name, value in artifacts.items():
        write_once(output / name, value, jsonl=name.endswith(".jsonl"))
    return {**summary, "status": "PASS", "output_dir": str(output)}
