"""Immutable pre-model temporal search planning for ReliVE-v2 TAL.

This stage consumes only frozen Stage 1 and Stage 2 artifacts.  It defines
future claim contracts and schedules windows; it never creates a claim,
opens media, scores a window, or answers a temporal question.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

from relive.storage.artifacts import canonical_json, stable_hash
from .requirement_freeze import RequirementFreezeError, validate_requirement_freeze_artifacts
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl
from .video_index import VideoIndexError, validate_video_index_artifacts


SEARCH_FORMAT = "relive-v2-tal-temporal-search-freeze-v1"
SCHEMA_FORMAT = "relive-v2-tal-claim-graph-schema-v1"
HYPOTHESIS_CONTRACT_FORMAT = "relive-v2-tal-hypothesis-contract-v1"
POLICY_FORMAT = "relive-v2-tal-temporal-pyramid-policy-v1"
POLICY_VERSION = "relive-v2-tal-temporal-pyramid-policy-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACTS = (
    "v2_claim_graph_schema.json", "v2_hypothesis_contract.json",
    "v2_temporal_pyramid_policy.json", "v2_temporal_search_plan.jsonl",
    "v2_temporal_search_manifest.json", "v2_temporal_search_freeze_audit.json",
)
_FORBIDDEN = ("answer", "reference_answer", "assistant_answer", "temporal_gt", "ground_truth", "evaluation",
              "bbox", "mask", "roi", "model", "score", "certificate", "verdict", "iou")


class TemporalSearchPlanError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise TemporalSearchPlanError("IMMUTABLE_OUTPUT_EXISTS")
    path.write_bytes(payload)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _strict_object(path: Path, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise TemporalSearchPlanError(f"{code}_INVALID") from exc
    if not isinstance(value, dict):
        raise TemporalSearchPlanError(f"{code}_OBJECT_REQUIRED")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise TemporalSearchPlanError(f"{code}_NONCANONICAL_BYTES")
    return value


def _strict_rows(path: Path, code: str) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_bytes()
        rows = strict_jsonl(path, error_code=code)
    except (OSError, TALSelectionError) as exc:
        raise TemporalSearchPlanError(f"{code}_INVALID") from exc
    if raw != _jsonl(list(rows)):
        raise TemporalSearchPlanError(f"{code}_NONCANONICAL_BYTES")
    return rows


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TemporalSearchPlanError(code)
    return float(value)


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(any(word in str(key).casefold() for word in _FORBIDDEN) or _contains_forbidden(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    return False


def _policy(path: Path) -> tuple[dict[str, Any], str]:
    policy = _strict_object(path, "TEMPORAL_POLICY")
    required = {"format", "policy_version", "window_scales_seconds", "window_overlap_fraction",
                "include_full_clip_window", "timestamp_domain"}
    if set(policy) != required or policy["format"] != POLICY_FORMAT or policy["policy_version"] != POLICY_VERSION:
        raise TemporalSearchPlanError("TEMPORAL_POLICY_SCHEMA_INVALID")
    scales = policy["window_scales_seconds"]
    if (scales != [8, 16, 32]
            or policy["window_overlap_fraction"] != 0.5 or policy["include_full_clip_window"] is not True
            or policy["timestamp_domain"] != "CLIP_LOCAL"):
        raise TemporalSearchPlanError("TEMPORAL_POLICY_VALUE_INVALID")
    return policy, _sha(path)


def claim_graph_schema() -> dict[str, Any]:
    """The closed contract for future nodes; this stage emits no node instance."""
    return {
        "format": SCHEMA_FORMAT, "schema_version": "relive-v2-claim-graph-schema-v1",
        "requirement_root_binding_fields": ["requirement_id", "requirement_spec_sha256", "target_event", "task",
                                            "answer_schema", "temporal_requirement", "spatial_requirement", "required_evidence"],
        "hypothesis_claim": {"required": ["hypothesis_id", "requirement_id", "parent_claim_id", "claim_role", "target_event", "polarity", "candidate_interval", "generation_source", "status", "supporting_window_ids", "provenance"],
                             "status": "CANDIDATE_UNVERIFIED", "positive_polarity": "POSITIVE",
                             "no_visible_event_polarity": "NO_VISIBLE_EVENT", "candidate_timestamp_domain": "CLIP_LOCAL"},
        "observation_claim": {"required": ["claim_id", "parent_claim_id", "claim_role", "subject", "predicate", "object", "polarity", "temporal_quantifier", "frame_scope", "observability", "evidence_geometry", "required_components", "generation_reason", "provenance", "status"],
                              "claim_role": "OBSERVATION", "status": "CANDIDATE_UNVERIFIED",
                              "immutable_parent_required": True, "new_claim_id_required": True},
        "prohibited_fields": ["VERIFIED", "SUPPORTED", "final_answer", "gt_overlap", "iou", "certificate_status"],
    }


def hypothesis_contract() -> dict[str, Any]:
    return {"format": HYPOTHESIS_CONTRACT_FORMAT, "contract_version": "relive-v2-tal-hypothesis-contract-v1",
            "max_positive_hypotheses": 5, "include_no_visible_event_hypothesis": True,
            "max_total_hypotheses": 6, "candidate_timestamp_domain": "CLIP_LOCAL",
            "candidate_status": "CANDIDATE_UNVERIFIED", "hypothesis_rewrite_allowed": False,
            "gt_ranking_allowed": False, "certificate_creation_allowed": False,
            "instances_created_in_stage_3a": 0}


def validate_hypothesis_claim(value: Mapping[str, Any]) -> None:
    """Validate a future hypothesis without treating it as evidence."""
    required = {"hypothesis_id", "requirement_id", "parent_claim_id", "claim_role", "target_event", "polarity",
                "candidate_interval", "generation_source", "status", "supporting_window_ids", "provenance"}
    optional = {"retrieval_rank", "coarse_support_margin"}
    if not isinstance(value, Mapping) or not required.issubset(value) or not set(value).issubset(required | optional) or _contains_forbidden(value):
        raise TemporalSearchPlanError("HYPOTHESIS_SCHEMA_INVALID")
    if value["claim_role"] != "TARGET_HYPOTHESIS" or value["status"] != "CANDIDATE_UNVERIFIED" or value["parent_claim_id"] is not None:
        raise TemporalSearchPlanError("HYPOTHESIS_STATUS_INVALID")
    if not all(isinstance(value[key], str) and value[key] for key in ("hypothesis_id", "requirement_id", "target_event", "generation_source")) or not isinstance(value["supporting_window_ids"], list) or not isinstance(value["provenance"], dict):
        raise TemporalSearchPlanError("HYPOTHESIS_VALUE_INVALID")
    if "retrieval_rank" in value and value["retrieval_rank"] is not None and (type(value["retrieval_rank"]) is not int or value["retrieval_rank"] < 1):
        raise TemporalSearchPlanError("HYPOTHESIS_RETRIEVAL_RANK_INVALID")
    if "coarse_support_margin" in value and value["coarse_support_margin"] is not None:
        _finite(value["coarse_support_margin"], "HYPOTHESIS_SUPPORT_MARGIN_INVALID")
    if value["polarity"] == "NO_VISIBLE_EVENT":
        if value["candidate_interval"] is not None:
            raise TemporalSearchPlanError("NULL_HYPOTHESIS_INTERVAL_FORBIDDEN")
        return
    interval = value["candidate_interval"]
    if value["polarity"] != "POSITIVE" or not isinstance(interval, dict) or set(interval) != {"start_seconds", "end_seconds", "timestamp_domain"} or interval["timestamp_domain"] != "CLIP_LOCAL":
        raise TemporalSearchPlanError("HYPOTHESIS_INTERVAL_INVALID")
    if _finite(interval["start_seconds"], "HYPOTHESIS_INTERVAL_INVALID") > _finite(interval["end_seconds"], "HYPOTHESIS_INTERVAL_INVALID"):
        raise TemporalSearchPlanError("HYPOTHESIS_INTERVAL_INVALID")


def validate_observation_claim(value: Mapping[str, Any], *, immutable_parent_ids: frozenset[str] = frozenset()) -> None:
    required = {"claim_id", "parent_claim_id", "claim_role", "subject", "predicate", "object", "polarity", "temporal_quantifier", "frame_scope", "observability", "evidence_geometry", "required_components", "generation_reason", "provenance", "status"}
    if not isinstance(value, Mapping) or set(value) != required or _contains_forbidden(value):
        raise TemporalSearchPlanError("OBSERVATION_SCHEMA_INVALID")
    if value["claim_role"] != "OBSERVATION" or value["status"] != "CANDIDATE_UNVERIFIED" or not isinstance(value["parent_claim_id"], str) or value["parent_claim_id"] not in immutable_parent_ids or value["claim_id"] == value["parent_claim_id"]:
        raise TemporalSearchPlanError("OBSERVATION_PARENT_IMMUTABILITY_INVALID")
    if not all(isinstance(value[key], str) and value[key] for key in ("claim_id", "subject", "predicate", "polarity", "temporal_quantifier", "frame_scope", "observability", "evidence_geometry", "generation_reason")) or not isinstance(value["required_components"], list) or not isinstance(value["provenance"], dict):
        raise TemporalSearchPlanError("OBSERVATION_VALUE_INVALID")


def _window_id(payload: dict[str, Any]) -> str:
    return "temporal_window_" + stable_hash(payload)[:24]


def _unique_views(frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    by_visual: dict[tuple[Any, ...], str] = {}
    mapping, unique = [], []
    for frame in frames:
        key = (frame["frame_sha256"], frame["source_frame_reference"], frame["timestamp_seconds"])
        unique_id = by_visual.setdefault(key, "unique_visual_frame_" + stable_hash({"frame_sha256": key[0], "source_frame_reference": key[1], "timestamp_seconds": key[2]})[:24])
        mapping.append({"logical_frame_order": frame["frame_order"], "unique_visual_frame_id": unique_id})
        if unique_id not in {item["unique_visual_frame_id"] for item in unique}:
            unique.append({"unique_visual_frame_id": unique_id, "frame_sha256": frame["frame_sha256"], "source_frame_reference": frame["source_frame_reference"], "timestamp_seconds": frame["timestamp_seconds"]})
    return mapping, unique, len(frames) - len(unique)


def _windows(*, requirement_id: str, frames: list[dict[str, Any]], video_index_manifest_sha256: str, policy_sha256: str, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    last = _finite(frames[-1]["timestamp_seconds"], "VIDEO_INDEX_TIME_INVALID")
    logical_to_unique, unique, _ = _unique_views(frames)
    unique_by_order = {item["logical_frame_order"]: item["unique_visual_frame_id"] for item in logical_to_unique}
    output: list[dict[str, Any]] = []
    for scale_value in policy["window_scales_seconds"]:
        scale = float(scale_value)
        starts = [0.0]
        if last > scale:
            stride = scale * 0.5
            starts = []
            current = 0.0
            while current + scale < last:
                starts.append(current)
                current += stride
            starts.append(max(0.0, last - scale))
        dedup = []
        for start in starts:
            if not any(abs(start - prior) < 1e-12 for prior in dedup):
                dedup.append(start)
        for start in sorted(dedup):
            end = min(last, start + scale)
            members = [frame for frame in frames if start <= frame["timestamp_seconds"] <= end]
            if not members:
                raise TemporalSearchPlanError("TEMPORAL_SEARCH_PLAN_INCOMPLETE")
            row = {"scale_seconds": scale_value, "start_seconds": start, "end_seconds": end,
                   "window_role": "PYRAMID", "included_unique_frame_ids": list(dict.fromkeys(unique_by_order[item["frame_order"]] for item in members)),
                   "included_logical_frame_orders": [item["frame_order"] for item in members], "requirement_id": requirement_id,
                   "video_index_manifest_sha256": video_index_manifest_sha256, "policy_sha256": policy_sha256,
                   "status": "PLANNED_UNSCORED"}
            row["window_id"] = _window_id(row)
            output.append(row)
    full = {"scale_seconds": last, "start_seconds": 0.0, "end_seconds": last, "window_role": "FULL_CLIP",
            "included_unique_frame_ids": [item["unique_visual_frame_id"] for item in unique],
            "included_logical_frame_orders": [item["frame_order"] for item in frames], "requirement_id": requirement_id,
            "video_index_manifest_sha256": video_index_manifest_sha256, "policy_sha256": policy_sha256,
            "status": "PLANNED_UNSCORED"}
    full["window_id"] = _window_id(full)
    output.append(full)
    return output


def _audit_imports() -> dict[str, Any]:
    forbidden = ("relive.backends", "relive.runner", "relive.cache", "relive.certificate", "relive.verification", "relive.evaluation", "relive.data.medvidu", "relive.phase4a")
    source = Path(__file__)
    imported: list[str] = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"), filename=str(source))):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    issues = [item for item in forbidden if any(name == item or name.startswith(item + ".") for name in imported)]
    return {"status": "PASS" if not issues else "FAIL", "files_checked": 1, "issues": issues,
            "limitation": "Static import audit cannot independently prove source-value access behavior."}


def _load_inputs(*, requirement_dir: Path, selection_manifest_path: Path, video_index_dir: Path,
                 timestamp_manifest_path: Path, timestamp_provenance_path: Path, policy_path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, Any], tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any], str]:
    try:
        validate_requirement_freeze_artifacts(requirement_dir)
        indexed = validate_video_index_artifacts(video_index_dir, materialize_frames=False)
    except (RequirementFreezeError, VideoIndexError) as exc:
        raise TemporalSearchPlanError("FROZEN_STAGE_INPUT_INVALID") from exc
    if indexed["index_status"] != "RESOLVED_DATASET_NATIVE_CLIP_LOCAL" or indexed["ready_for_hypothesis_generation"] is not True:
        raise TemporalSearchPlanError("VIDEO_INDEX_NOT_READY_FOR_COARSE_HYPOTHESIS_GENERATION")
    req_manifest = _strict_object(requirement_dir / "v2_requirement_manifest.json", "REQUIREMENT_MANIFEST")
    specs = _strict_rows(requirement_dir / "v2_requirement_specs.jsonl", "REQUIREMENT_SPECS")
    selection = _strict_object(selection_manifest_path, "SELECTION_MANIFEST")
    index_manifest = _strict_object(video_index_dir / "v2_video_index_manifest.json", "VIDEO_INDEX_MANIFEST")
    indexes = _strict_rows(video_index_dir / "v2_video_index.jsonl", "VIDEO_INDEX")
    timestamps = _strict_rows(timestamp_manifest_path, "TIMESTAMP_MANIFEST")
    provenance = _strict_object(timestamp_provenance_path, "TIMESTAMP_PROVENANCE")
    policy, policy_sha = _policy(policy_path)
    if _sha(selection_manifest_path) != index_manifest.get("selection_manifest_sha256") or _sha(requirement_dir / "v2_requirement_manifest.json") != index_manifest.get("requirement_manifest_sha256"):
        raise TemporalSearchPlanError("STAGE_INPUT_BINDING_MISMATCH")
    if _sha(timestamp_manifest_path) != index_manifest.get("public_timestamp_manifest_sha256") or _sha(timestamp_provenance_path) != index_manifest.get("public_timestamp_provenance_sha256") or provenance.get("timestamp_manifest_sha256") != _sha(timestamp_manifest_path):
        raise TemporalSearchPlanError("TIMESTAMP_INPUT_BINDING_MISMATCH")
    # RequirementSpec legitimately contains the closed ``answer_schema`` field
    # and timestamp provenance records zero-model audit fields.  Their own
    # frozen-artifact validators establish GT isolation, so do not apply the
    # future-claim output field deny-list to these upstream contracts.
    if len(specs) != len(indexes) or not specs:
        raise TemporalSearchPlanError("STAGE_INPUT_SCHEMA_OR_ISOLATION_INVALID")
    return req_manifest, specs, selection, indexes, provenance, policy, policy_sha


def freeze_temporal_search_plan(*, requirement_dir: Path, selection_manifest_path: Path, video_index_dir: Path,
                                timestamp_manifest_path: Path, timestamp_provenance_path: Path,
                                policy_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise TemporalSearchPlanError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    imports = _audit_imports()
    if imports["status"] != "PASS":
        raise TemporalSearchPlanError("STATIC_ISOLATION_AUDIT_FAILED")
    req_manifest, specs, selection, indexes, provenance, policy, policy_sha = _load_inputs(
        requirement_dir=requirement_dir, selection_manifest_path=selection_manifest_path, video_index_dir=video_index_dir,
        timestamp_manifest_path=timestamp_manifest_path, timestamp_provenance_path=timestamp_provenance_path, policy_path=policy_path)
    index_manifest_path = video_index_dir / "v2_video_index_manifest.json"
    index_manifest_sha = _sha(index_manifest_path)
    by_identity = {tuple(item[key] for key in ("source_record_index", "sample_id", "public_record_sha256", "question_sha256")): item for item in indexes}
    plan: list[dict[str, Any]] = []
    logical_map: list[dict[str, Any]] = []
    unique_total = 0
    duplicate_total = 0
    for spec in specs:
        source_identity = spec.get("provenance", {}).get("public_source_identity")
        if not isinstance(source_identity, dict):
            raise TemporalSearchPlanError("REQUIREMENT_SOURCE_IDENTITY_MISSING")
        identity = (source_identity.get("source_record_index"), source_identity.get("sample_id"),
                    source_identity.get("public_record_sha256"), spec["question_sha256"])
        index = by_identity.get(identity)
        if index is None or index["requirement_id"] != spec["requirement_id"]:
            raise TemporalSearchPlanError("REQUIREMENT_VIDEO_INDEX_BINDING_MISMATCH")
        frames = index["frames"]
        generated = _windows(requirement_id=spec["requirement_id"], frames=frames, video_index_manifest_sha256=index_manifest_sha, policy_sha256=policy_sha, policy=policy)
        mapping, unique, duplicates = _unique_views(frames)
        logical_map.extend(mapping)
        unique_total += len(unique); duplicate_total += duplicates
        plan.extend(generated)
    all_logical = {item["logical_frame_order"] for item in logical_map}
    covered_logical = {item for row in plan for item in row["included_logical_frame_orders"]}
    all_unique = {item["unique_visual_frame_id"] for item in logical_map}
    covered_unique = {item for row in plan for item in row["included_unique_frame_ids"]}
    if not all_logical or all_logical != covered_logical or all_unique != covered_unique:
        raise TemporalSearchPlanError("TEMPORAL_SEARCH_PLAN_INCOMPLETE")
    schema = claim_graph_schema(); contract = hypothesis_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "v2_claim_graph_schema.json", (canonical_json(schema) + "\n").encode("utf-8"))
    _write(output_dir / "v2_hypothesis_contract.json", (canonical_json(contract) + "\n").encode("utf-8"))
    _write(output_dir / "v2_temporal_pyramid_policy.json", (canonical_json(policy) + "\n").encode("utf-8"))
    _write(output_dir / "v2_temporal_search_plan.jsonl", _jsonl(plan))
    manifest = {"format": SEARCH_FORMAT, "status": "PASS", "search_plan_status": "FROZEN_PRE_MODEL", "timestamp_domain": "CLIP_LOCAL",
                "requirement_manifest_sha256": _sha(requirement_dir / "v2_requirement_manifest.json"), "requirement_specs_sha256": req_manifest["requirement_specs_sha256"],
                "selection_manifest_sha256": _sha(selection_manifest_path), "video_index_manifest_sha256": index_manifest_sha,
                "timestamp_manifest_sha256": _sha(timestamp_manifest_path), "timestamp_provenance_sha256": _sha(timestamp_provenance_path),
                "claim_graph_schema_sha256": _sha(output_dir / "v2_claim_graph_schema.json"), "hypothesis_contract_sha256": _sha(output_dir / "v2_hypothesis_contract.json"),
                "temporal_policy_sha256": _sha(output_dir / "v2_temporal_pyramid_policy.json"), "search_plan_sha256": _sha(output_dir / "v2_temporal_search_plan.jsonl"),
                "requirement_count": len(specs), "search_window_count": len(plan), "logical_frame_count": len(logical_map),
                "unique_visual_frame_count": unique_total, "duplicate_logical_frame_count": duplicate_total,
                "logical_to_unique_frame_map": logical_map, "unique_visual_frame_coverage": 1.0, "logical_frame_coverage": 1.0,
                "uncovered_unique_frame_count": 0, "uncovered_logical_frame_count": 0, "hypothesis_count": 0, "observation_claim_count": 0,
                "claim_graph_created": False, "ready_for_coarse_hypothesis_generation": True, "assistant_or_gt_values_accessed": False, "gt_used": False,
                "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False,
                "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    manifest_path = output_dir / "v2_temporal_search_manifest.json"
    _write(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    audit = {"format": SEARCH_FORMAT, "status": "PASS", "search_plan_status": "FROZEN_PRE_MODEL",
             "allowed_inputs_opened": ["frozen RequirementSpec artifacts", "frozen selection manifest", "Stage 2 immutable VideoIndex", "Stage 2 timestamp manifest", "Stage 2 timebase provenance", "versioned Stage 3A temporal planning policy"],
             "forbidden_inputs_not_opened": ["assistant answer", "reference answer", "conversations[1]", "struc_info", "temporal GT", "evaluation", "bbox", "mask", "ROI", "model output", "certificate"],
             "imports_audited": imports, "manifest_sha256": _sha(manifest_path), "hypothesis_count": 0, "observation_claim_count": 0,
             "claim_graph_created": False, "assistant_or_gt_values_accessed": False, "gt_used": False, "model_calls_made": 0,
             "backend_loaded": False, "cache_opened": False, "certificate_created": False, "new_verified_count": 0,
             "certificate_status": "NOT_APPLICABLE"}
    audit["audit_content_sha256"] = stable_hash(audit)
    _write(output_dir / "v2_temporal_search_freeze_audit.json", (canonical_json(audit) + "\n").encode("utf-8"))
    return {"status": "PASS", "search_plan_status": "FROZEN_PRE_MODEL", "timestamp_domain": "CLIP_LOCAL", "search_window_count": len(plan),
            "logical_frame_count": len(logical_map), "unique_visual_frame_count": unique_total, "duplicate_logical_frame_count": duplicate_total,
            "unique_visual_frame_coverage": 1.0, "logical_frame_coverage": 1.0, "hypothesis_count": 0, "observation_claim_count": 0,
            "claim_graph_created": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
            "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE",
            "ready_for_coarse_hypothesis_generation": True}


def validate_temporal_search_plan_artifacts(output_dir: Path) -> dict[str, Any]:
    paths = {name: output_dir / name for name in _ARTIFACTS}
    if not all(path.is_file() for path in paths.values()):
        raise TemporalSearchPlanError("SEARCH_ARTIFACT_MISSING")
    schema = _strict_object(paths["v2_claim_graph_schema.json"], "CLAIM_SCHEMA")
    contract = _strict_object(paths["v2_hypothesis_contract.json"], "HYPOTHESIS_CONTRACT")
    policy = _strict_object(paths["v2_temporal_pyramid_policy.json"], "FROZEN_TEMPORAL_POLICY")
    manifest = _strict_object(paths["v2_temporal_search_manifest.json"], "SEARCH_MANIFEST")
    audit = _strict_object(paths["v2_temporal_search_freeze_audit.json"], "SEARCH_AUDIT")
    plan = _strict_rows(paths["v2_temporal_search_plan.jsonl"], "SEARCH_PLAN")
    if schema.get("format") != SCHEMA_FORMAT or contract.get("format") != HYPOTHESIS_CONTRACT_FORMAT or policy.get("format") != POLICY_FORMAT:
        raise TemporalSearchPlanError("SEARCH_SCHEMA_INVALID")
    bindings = {"claim_graph_schema_sha256": "v2_claim_graph_schema.json", "hypothesis_contract_sha256": "v2_hypothesis_contract.json", "temporal_policy_sha256": "v2_temporal_pyramid_policy.json", "search_plan_sha256": "v2_temporal_search_plan.jsonl"}
    if any(manifest.get(field) != _sha(paths[name]) for field, name in bindings.items()) or manifest.get("manifest_content_sha256") != stable_hash({key: value for key, value in manifest.items() if key != "manifest_content_sha256"}):
        raise TemporalSearchPlanError("SEARCH_ARTIFACT_BYTES_CHANGED")
    if manifest.get("search_window_count") != len(plan) or manifest.get("hypothesis_count") != 0 or manifest.get("observation_claim_count") != 0 or manifest.get("claim_graph_created") is not False or manifest.get("ready_for_coarse_hypothesis_generation") is not True:
        raise TemporalSearchPlanError("SEARCH_MANIFEST_SCHEMA_INVALID")
    if audit.get("audit_content_sha256") != stable_hash({key: value for key, value in audit.items() if key != "audit_content_sha256"}) or audit.get("manifest_sha256") != _sha(paths["v2_temporal_search_manifest.json"]) or audit.get("imports_audited", {}).get("status") != "PASS":
        raise TemporalSearchPlanError("SEARCH_AUDIT_BINDING_MISMATCH")
    if any(_contains_forbidden(row) for row in plan):
        raise TemporalSearchPlanError("SEARCH_PLAN_FORBIDDEN_FIELD")
    for row in plan:
        required = {"window_id", "scale_seconds", "start_seconds", "end_seconds", "window_role", "included_unique_frame_ids", "included_logical_frame_orders", "requirement_id", "video_index_manifest_sha256", "policy_sha256", "status"}
        if set(row) != required or row["status"] != "PLANNED_UNSCORED" or row["window_role"] not in {"PYRAMID", "FULL_CLIP"} or _finite(row["start_seconds"], "SEARCH_WINDOW_INVALID") > _finite(row["end_seconds"], "SEARCH_WINDOW_INVALID") or not row["included_unique_frame_ids"] or not row["included_logical_frame_orders"] or row["policy_sha256"] != manifest["temporal_policy_sha256"] or row["video_index_manifest_sha256"] != manifest["video_index_manifest_sha256"]:
            raise TemporalSearchPlanError("SEARCH_WINDOW_INVALID")
        payload = {key: value for key, value in row.items() if key != "window_id"}
        if row["window_id"] != _window_id(payload):
            raise TemporalSearchPlanError("SEARCH_WINDOW_ID_INVALID")
    return {"status": "PASS", "search_plan_status": "FROZEN_PRE_MODEL", "search_window_count": len(plan),
            "logical_frame_count": manifest["logical_frame_count"], "unique_visual_frame_count": manifest["unique_visual_frame_count"],
            "logical_frame_coverage": manifest["logical_frame_coverage"], "unique_visual_frame_coverage": manifest["unique_visual_frame_coverage"],
            "hypothesis_count": 0, "observation_claim_count": 0, "claim_graph_created": False, "gt_used": False,
            "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False,
            "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "ready_for_coarse_hypothesis_generation": True}
