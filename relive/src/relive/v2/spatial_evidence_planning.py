"""ReliVE-v2 Stage 3F: metadata-only typed spatial evidence planning.

This stage converts frozen temporal evidence chains into *ungrounded* spatial
contracts.  It intentionally creates no ROI, coordinate, mask, point, or image
request, and it never calls a model or certificate code.
"""
from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .observation_retrieval import ObservationRetrievalError, validate as validate_stage3d
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl
from .temporal_evidence_composition import TemporalCompositionError, validate as validate_stage3e

FORMAT = "relive-v2-typed-spatial-evidence-planning-v1"
PLAN_STATUS = "PLANNED_UNGROUNDED"
FORBIDDEN_SPATIAL_KEYS = frozenset({"bbox", "box", "point", "mask", "masklet", "roi", "support_tube"})

# (geometry type, required component roles, contextual requirements, propagation)
GEOMETRY: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str]] = {
    "PRECONDITION_ALIGNMENT": (
        "RELATIONAL_COMPOSITE",
        ("BASE_PLATE", "TARGET_SKIN_AREA", "ALIGNMENT_INTERFACE"),
        (),
        "STATIC_WINDOW",
    ),
    "ACTION_CORE_PRESSING": (
        "DYNAMIC_SUPPORT_TUBE",
        ("OPERATOR_HAND", "BASE_PLATE", "HAND_BASE_INTERFACE"),
        ("TARGET_SKIN_AREA", "BASE_SKIN_INTERFACE"),
        "FULL_PHYSICAL_WINDOW",
    ),
    "POSTCONDITION_ATTACHMENT": (
        "DYNAMIC_SUPPORT_TUBE",
        ("BASE_PLATE", "TARGET_SKIN_AREA", "ATTACHMENT_INTERFACE"),
        ("BASE_REMAINS_ALIGNED_OR_ATTACHED_AFTER_INTERACTION", "OPERATOR_HAND_NOT_REQUIRED_INSIDE_TARGET_REGION"),
        "FULL_PHYSICAL_WINDOW",
    ),
}


class SpatialPlanError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _obj(path: Path, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise SpatialPlanError(f"{code}_INVALID") from exc
    if not isinstance(value, dict) or raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise SpatialPlanError(f"{code}_NONCANONICAL")
    return value


def _rows(path: Path, code: str) -> list[dict[str, Any]]:
    try:
        value = list(strict_jsonl(path, error_code=code))
        raw = path.read_bytes()
    except (OSError, TALSelectionError) as exc:
        raise SpatialPlanError(f"{code}_INVALID") from exc
    if raw != _jsonl(value):
        raise SpatialPlanError(f"{code}_NONCANONICAL")
    return value


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise SpatialPlanError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical_json(value) + "\n").encode("utf-8") if isinstance(value, dict) else _jsonl(value)
    path.write_bytes(raw)


def _anchor_schedule(window: dict[str, Any], frame_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    orders = window.get("included_logical_frame_orders")
    if not isinstance(orders, list) or not orders:
        raise SpatialPlanError("PHYSICAL_WINDOW_FRAME_MEMBERSHIP_INVALID")
    aliases: dict[str, list[int]] = {}
    for frame in frame_map.values():
        key = stable_hash({"frame_sha256": frame["frame_sha256"], "source_frame_reference": frame["source_frame_reference"], "timestamp_seconds": frame["timestamp_seconds"]})
        aliases.setdefault(key, []).append(frame["frame_order"])
    visible: dict[str, dict[str, Any]] = {}
    for order in orders:
        frame = frame_map.get(order)
        if frame is None:
            raise SpatialPlanError("PHYSICAL_WINDOW_FRAME_MEMBERSHIP_INVALID")
        key = stable_hash({"frame_sha256": frame["frame_sha256"], "source_frame_reference": frame["source_frame_reference"], "timestamp_seconds": frame["timestamp_seconds"]})
        visible.setdefault(key, frame)
    ranked = sorted(visible.items(), key=lambda item: (Decimal(str(item[1]["timestamp_seconds"])), item[0]))
    midpoint = (Decimal(str(window["start_seconds"])) + Decimal(str(window["end_seconds"]))) / 2
    selected: list[tuple[str, dict[str, Any], str]] = []
    for label, choices in (
        ("MIDPOINT_NEAREST", sorted(ranked, key=lambda item: (abs(Decimal(str(item[1]["timestamp_seconds"])) - midpoint), Decimal(str(item[1]["timestamp_seconds"])), item[0]))),
        ("FIRST", ranked),
        ("LAST", list(reversed(ranked))),
    ):
        key, frame = choices[0]
        if key not in {item[0] for item in selected}:
            selected.append((key, frame, label))
    return [
        {
            "anchor_selector": label,
            "unique_visual_frame_id": "unique_visual_frame_" + key[:24],
            "logical_aliases": sorted(aliases[key]),
            "source_frame_reference": frame["source_frame_reference"],
            "timestamp_seconds": frame["timestamp_seconds"],
        }
        for key, frame, label in selected
    ]


def _assert_no_geometry_payload(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN_SPATIAL_KEYS.intersection(value):
            raise SpatialPlanError("GEOMETRY_PAYLOAD_FORBIDDEN")
        for child in value.values():
            _assert_no_geometry_payload(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_geometry_payload(child)


def _input_hashes(stage3c_dir: Path, stage3d_dir: Path, stage3e_dir: Path, video_index_dir: Path) -> dict[str, str]:
    return {
        "stage3c_manifest_sha256": _sha(stage3c_dir / "v2_stage3c_manifest.json"),
        "stage3d_call_plan_sha256": _sha(stage3d_dir / "v2_stage3d_call_plan.json"),
        "stage3d_run_summary_sha256": _sha(stage3d_dir / "v2_stage3d_run_summary.json"),
        "stage3e_manifest_sha256": _sha(stage3e_dir / "v2_stage3e_manifest.json"),
        "video_index_manifest_sha256": _sha(video_index_dir / "v2_video_index_manifest.json"),
    }


def prepare(*, stage3c_dir: Path, stage3d_dir: Path, stage3e_dir: Path, video_index_dir: Path,
            policy_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SpatialPlanError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    try:
        stage3d = validate_stage3d(stage3d_dir)
        stage3e = validate_stage3e(stage3e_dir)
    except (ObservationRetrievalError, TemporalCompositionError) as exc:
        raise SpatialPlanError("FROZEN_UPSTREAM_ARTIFACT_INVALID") from exc
    if stage3d.get("status") != "PASS" or stage3e.get("status") != "PASS" or not stage3e.get("ready_for_typed_spatial_evidence_planning"):
        raise SpatialPlanError("UPSTREAM_STATUS_NOT_READY")
    policy = _obj(policy_path, "POLICY")
    if policy.get("format") != "relive-v2-typed-spatial-evidence-policy-v1" or policy.get("max_anchor_candidates") != 3:
        raise SpatialPlanError("POLICY_INVALID")
    policy_sha = _sha(policy_path)
    geometry_sha = stable_hash(GEOMETRY)
    upstream = _input_hashes(stage3c_dir, stage3d_dir, stage3e_dir, video_index_dir)

    claims = {item["claim_id"]: item for item in _rows(stage3c_dir / "v2_stage3c_observation_claims.jsonl", "OBSERVATION_CLAIMS")}
    windows = {item["acquisition_window_id"]: item for item in _rows(stage3c_dir / "v2_stage3c_physical_acquisition_windows.jsonl", "PHYSICAL_WINDOWS")}
    index_rows = _rows(video_index_dir / "v2_video_index.jsonl", "VIDEO_INDEX")
    if len(index_rows) != 1 or not isinstance(index_rows[0].get("frames"), list):
        raise SpatialPlanError("VIDEO_INDEX_SCHEMA_INVALID")
    frame_map = {item["frame_order"]: item for item in index_rows[0]["frames"]}
    chains = {item["chain_id"]: item for item in _rows(stage3e_dir / "v2_stage3e_compatible_chains.jsonl", "COMPATIBLE_CHAINS")}
    chain_bindings = _rows(stage3e_dir / "v2_stage3e_hypothesis_chain_bindings.jsonl", "HYPOTHESIS_CHAIN_BINDINGS")

    plans: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    tasks_by_key: dict[str, dict[str, Any]] = {}
    slot_specs = (
        ("PRE", "pre_observation_claim_id", "pre_evidence_window_id"),
        ("ACTION", "action_observation_claim_id", "action_evidence_window_id"),
        ("POST", "post_observation_claim_id", "post_evidence_window_id"),
    )
    for chain_binding in sorted(chain_bindings, key=lambda row: (row["parent_hypothesis_id"], row["chain_id"])):
        chain = chains.get(chain_binding["chain_id"])
        if chain is None:
            raise SpatialPlanError("UNKNOWN_CHAIN_REFERENCE")
        for slot, claim_key, window_key in slot_specs:
            claim = claims.get(chain.get(claim_key))
            window = windows.get(chain.get(window_key))
            if claim is None or window is None:
                raise SpatialPlanError("UNKNOWN_REFERENCE_ID")
            role = claim.get("template_id")
            if role not in GEOMETRY:
                unresolved.append({"parent_hypothesis_id": chain_binding["parent_hypothesis_id"], "chain_id": chain["chain_id"], "chain_slot": slot, "observation_claim_id": claim["claim_id"], "reason_code": "UNSUPPORTED_OBSERVATION_ROLE", "status": "UNRESOLVED"})
                continue
            geometry_type, required_roles, contextual, propagation = GEOMETRY[role]
            anchors = _anchor_schedule(window, frame_map)
            semantics_hash = stable_hash({"observation_claim_id": claim["claim_id"], "observation_role": role, "surface": claim.get("surface")})
            task_key = stable_hash({"observation_semantics_hash": semantics_hash, "observation_role": role, "physical_window_id": window["acquisition_window_id"], "geometry_contract_sha256": geometry_sha, "anchor_policy_sha256": policy_sha, "required_component_roles": required_roles})
            task = tasks_by_key.setdefault(task_key, {
                "spatial_grounding_task_id": "spatial_grounding_task_" + task_key[:24],
                "spatial_grounding_task_key": task_key,
                "observation_semantics_hash": semantics_hash,
                "observation_role": role,
                "physical_window_id": window["acquisition_window_id"],
                "geometry_type": geometry_type,
                "required_component_roles": list(required_roles),
                "contextual_requirements": list(contextual),
                "anchor_policy_sha256": policy_sha,
                "geometry_contract_sha256": geometry_sha,
                "anchor_candidates": anchors,
                "propagation_scope": propagation,
                "future_intervention_target": "COMPOSITE_EVIDENCE_UNION",
                "matched_control_required": True,
                "component_roles_preserved": True,
                "status": PLAN_STATUS,
            })
            plan_key = stable_hash({"observation_claim_id": claim["claim_id"], "physical_window_id": window["acquisition_window_id"], "spatial_grounding_task_id": task["spatial_grounding_task_id"], "chain_id": chain["chain_id"], "slot": slot})
            plan = {
                "spatial_plan_id": "spatial_plan_" + plan_key[:24],
                "observation_claim_id": claim["claim_id"],
                "observation_role": role,
                "physical_window_id": window["acquisition_window_id"],
                "geometry_type": geometry_type,
                "required_component_roles": list(required_roles),
                "contextual_requirements": list(contextual),
                "anchor_candidates": anchors,
                "propagation_scope": propagation,
                "future_intervention_target": "COMPOSITE_EVIDENCE_UNION",
                "matched_control_required": True,
                "component_roles_preserved": True,
                "spatial_grounding_task_id": task["spatial_grounding_task_id"],
                "status": PLAN_STATUS,
            }
            plans.append(plan)
            bindings.append({
                "parent_hypothesis_id": chain_binding["parent_hypothesis_id"], "chain_id": chain["chain_id"],
                "physical_chain_id": chain_binding["physical_chain_id"], "chain_slot": slot,
                "observation_claim_id": claim["claim_id"], "physical_window_id": window["acquisition_window_id"],
                "spatial_plan_id": plan["spatial_plan_id"], "spatial_grounding_task_id": task["spatial_grounding_task_id"],
                "status": PLAN_STATUS,
            })

    plans.sort(key=lambda row: row["spatial_plan_id"])
    bindings.sort(key=lambda row: (row["parent_hypothesis_id"], row["chain_id"], row["chain_slot"]))
    tasks = sorted(tasks_by_key.values(), key=lambda row: row["spatial_grounding_task_id"])
    artifacts: dict[str, Any] = {
        "v2_tal_spatial_evidence_schema.json": {"format": FORMAT, "plan_status": PLAN_STATUS, "geometry_contract_sha256": geometry_sha, "forbidden_geometry_payload_keys": sorted(FORBIDDEN_SPATIAL_KEYS)},
        "v2_tal_spatial_geometry_contract.json": {key: {"geometry_type": item[0], "required_component_roles": list(item[1]), "contextual_requirements": list(item[2]), "propagation_scope": item[3]} for key, item in GEOMETRY.items()},
        "v2_tal_spatial_anchor_policy.json": policy,
        "v2_tal_spatial_evidence_plans.jsonl": plans,
        "v2_tal_spatial_grounding_tasks.jsonl": tasks,
        "v2_tal_spatial_plan_bindings.jsonl": bindings,
        "v2_tal_spatial_plan_unresolved.jsonl": unresolved,
    }
    output_dir.mkdir(parents=True)
    for name, value in artifacts.items():
        _assert_no_geometry_payload(value)
        _write(output_dir / name, value)
    hashes = {name: _sha(output_dir / name) for name in artifacts}
    manifest = {
        "format": FORMAT, "status": "PASS" if tasks else "UNRESOLVED", "stage_status": "SPATIAL_EVIDENCE_PLANS_FROZEN_UNGROUNDED",
        "upstream_sha256": upstream, "policy_sha256": policy_sha, "geometry_contract_sha256": geometry_sha, "artifact_sha256": hashes,
        "input_hypothesis_chain_binding_count": len(chain_bindings), "spatial_plan_count": len(plans), "deduplicated_grounding_task_count": len(tasks),
        "binding_count": len(bindings), "tasks_saved_by_deduplication": len(plans) - len(tasks), "unresolved_count": len(unresolved),
        "frames_read": 0, "frame_bytes_opened": False, "videos_read": 0, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
        "gt_used": False, "assistant_or_gt_values_accessed": False, "boxes_created": 0, "points_created": 0, "masks_created": 0, "masklets_created": 0,
        "support_tubes_created": 0, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE",
        "ready_for_stage3g_spatial_grounding": bool(tasks),
    }
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    _write(output_dir / "v2_tal_spatial_plan_manifest.json", manifest)
    audit = {
        "format": FORMAT, "status": "PASS", "allowed_inputs_opened": ["frozen RequirementSpec provenance through Stage 3D", "Stage 3C observation metadata", "Stage 3E chain metadata", "VideoIndex metadata", "versioned spatial planning policy"],
        "forbidden_inputs_not_opened": ["frame pixels", "video bytes", "assistant answer", "reference answer", "temporal GT", "bbox", "mask", "ROI", "model output", "certificate"],
        "artifact_sha256": {**hashes, "v2_tal_spatial_plan_manifest.json": _sha(output_dir / "v2_tal_spatial_plan_manifest.json")},
        "frames_read": 0, "frame_bytes_opened": False, "videos_read": 0, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
        "gt_used": False, "assistant_or_gt_values_accessed": False, "boxes_created": 0, "points_created": 0, "masks_created": 0, "masklets_created": 0,
        "support_tubes_created": 0, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE",
    }
    audit["audit_content_sha256"] = stable_hash(audit)
    _write(output_dir / "v2_tal_spatial_plan_audit.json", audit)
    return manifest


def validate(output_dir: Path) -> dict[str, Any]:
    manifest = _obj(output_dir / "v2_tal_spatial_plan_manifest.json", "MANIFEST")
    if manifest.get("format") != FORMAT or manifest.get("manifest_content_sha256") != stable_hash({key: value for key, value in manifest.items() if key != "manifest_content_sha256"}):
        raise SpatialPlanError("MANIFEST_TAMPERED")
    for name, digest in manifest.get("artifact_sha256", {}).items():
        if _sha(output_dir / name) != digest:
            raise SpatialPlanError("ARTIFACT_TAMPERED")
    audit = _obj(output_dir / "v2_tal_spatial_plan_audit.json", "AUDIT")
    if audit.get("audit_content_sha256") != stable_hash({key: value for key, value in audit.items() if key != "audit_content_sha256"}):
        raise SpatialPlanError("AUDIT_TAMPERED")
    plans = _rows(output_dir / "v2_tal_spatial_evidence_plans.jsonl", "PLANS")
    tasks = _rows(output_dir / "v2_tal_spatial_grounding_tasks.jsonl", "TASKS")
    if len(plans) != manifest.get("spatial_plan_count") or len(tasks) != manifest.get("deduplicated_grounding_task_count"):
        raise SpatialPlanError("COUNT_BINDING_INVALID")
    for row in plans + tasks:
        _assert_no_geometry_payload(row)
        if row.get("status") != PLAN_STATUS or len(row.get("anchor_candidates", [])) > 3:
            raise SpatialPlanError("SPATIAL_PLAN_SCHEMA_INVALID")
    return {"status": manifest["status"], "stage_status": manifest["stage_status"], "spatial_plan_count": len(plans), "deduplicated_grounding_task_count": len(tasks), "binding_count": manifest["binding_count"], "frames_read": 0, "frame_bytes_opened": False, "videos_read": 0, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "gt_used": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "ready_for_stage3g_spatial_grounding": manifest["ready_for_stage3g_spatial_grounding"]}
