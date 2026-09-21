"""Immutable, diagnostic-only integration of Stage 3G human overlay reviews.

This module consumes a frozen grounding JSONL and a human-authored, per-role
overlay review JSONL.  It never edits the model result, calls a backend, opens
frames, creates a certificate, or asserts that an observation claim is true.
"""
from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from relive.interventions import generate_control_regions
from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl


FORMAT = "relive-v2-stage3g-human-overlay-adjudication-v1"
REVIEW_FORMAT = "relive-v2-stage3g-human-overlay-review-v1"
REVIEW_FIELDS = frozenset({"anchor_candidate_id", "role", "model_visibility", "human_overlay_label", "reason_code"})
VISIBILITIES = frozenset({"VISIBLE", "NOT_VISIBLE", "AMBIGUOUS"})
VISIBLE_LABELS = frozenset({"ACCEPTED", "BOX_TOO_BROAD", "BOX_TOO_TIGHT", "WRONG_OBJECT", "INTERFACE_NOT_LOCALIZED", "SHOULD_BE_NOT_VISIBLE", "UNRESOLVED"})
NONVISIBLE_LABELS = frozenset({"NOT_VISIBLE_CONFIRMED", "UNRESOLVED"})
AMBIGUOUS_LABELS = frozenset({"AMBIGUOUS_CONFIRMED", "UNRESOLVED"})
ALL_LABELS = VISIBLE_LABELS | NONVISIBLE_LABELS | AMBIGUOUS_LABELS
TEMPORAL_CONTEXTUAL_ROLES = frozenset({"BASE_REMAINS_ALIGNED_OR_ATTACHED_AFTER_INTERACTION", "OPERATOR_HAND_NOT_REQUIRED_INSIDE_TARGET_REGION"})
RELATIONAL_OBSERVATION_ROLES = frozenset({"PRECONDITION_ALIGNMENT", "POSTCONDITION_ATTACHMENT"})


class HumanOverlayReviewError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _strict_rows(path: Path, code: str) -> list[dict[str, Any]]:
    try:
        rows = list(strict_jsonl(path, error_code=code))
    except (OSError, TALSelectionError) as exc:
        raise HumanOverlayReviewError(f"{code}_INVALID") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise HumanOverlayReviewError(f"{code}_ROWS_MUST_BE_OBJECTS")
    return rows


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise HumanOverlayReviewError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical_json(value) + "\n").encode("utf-8") if isinstance(value, dict) else _jsonl(value)
    path.write_bytes(raw)


def _review_label_allowed(visibility: str, label: str) -> bool:
    return ((visibility == "VISIBLE" and label in VISIBLE_LABELS)
            or (visibility == "NOT_VISIBLE" and label in NONVISIBLE_LABELS)
            or (visibility == "AMBIGUOUS" and label in AMBIGUOUS_LABELS))


def _review_index(raw_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]], *, expected_anchor_count: int) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    anchors = [row.get("anchor_candidate_id") for row in raw_rows]
    if any(not isinstance(anchor, str) or not anchor for anchor in anchors):
        raise HumanOverlayReviewError("RAW_ANCHOR_ID_INVALID")
    if len(set(anchors)) != expected_anchor_count or len(raw_rows) != expected_anchor_count:
        raise HumanOverlayReviewError("RAW_ANCHOR_COUNT_MUST_EQUAL_EXPECTED")
    expected: dict[tuple[str, str], str] = {}
    for row in raw_rows:
        components = row.get("components")
        if not isinstance(components, list) or not components:
            raise HumanOverlayReviewError("RAW_COMPONENTS_INVALID")
        declared_roles = row.get("required_component_roles", []) + row.get("contextual_requirements", [])
        if not isinstance(row.get("required_component_roles"), list) or not isinstance(row.get("contextual_requirements"), list):
            raise HumanOverlayReviewError("RAW_ROLE_CONTRACT_INVALID")
        seen_roles: set[str] = set()
        for component in components:
            if not isinstance(component, dict):
                raise HumanOverlayReviewError("RAW_COMPONENT_INVALID")
            role, visibility = component.get("role"), component.get("visibility")
            key = (row["anchor_candidate_id"], role)
            bbox = component.get("bbox_normalized_xyxy")
            if not isinstance(role, str) or visibility not in VISIBILITIES or key in expected or role in seen_roles:
                raise HumanOverlayReviewError("RAW_COMPONENT_BINDING_INVALID")
            seen_roles.add(role)
            if visibility == "VISIBLE":
                if (not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in bbox)
                        or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
                    raise HumanOverlayReviewError("RAW_VISIBLE_BBOX_INVALID")
            elif bbox is not None:
                raise HumanOverlayReviewError("RAW_NONVISIBLE_BBOX_INVALID")
            expected[key] = visibility
        if set(declared_roles) != seen_roles:
            raise HumanOverlayReviewError("RAW_COMPONENT_ROLE_OMISSION_OR_EXTRA")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for item in review_rows:
        if set(item) != REVIEW_FIELDS:
            errors.append("REVIEW_SCHEMA_INVALID")
            continue
        anchor, role = item.get("anchor_candidate_id"), item.get("role")
        vis, label, reason = item.get("model_visibility"), item.get("human_overlay_label"), item.get("reason_code")
        key = (anchor, role)
        if not isinstance(anchor, str) or not isinstance(role, str) or key not in expected:
            errors.append("REVIEW_UNKNOWN_ANCHOR_OR_ROLE")
        elif key in index:
            errors.append("REVIEW_DUPLICATE_COMPONENT")
        elif vis != expected[key]:
            errors.append("REVIEW_MODEL_VISIBILITY_MISMATCH")
        elif not isinstance(label, str) or not label or label not in ALL_LABELS:
            errors.append("REVIEW_LABEL_EMPTY_OR_ILLEGAL")
        elif not _review_label_allowed(vis, label):
            errors.append("REVIEW_VISIBILITY_LABEL_INCOMPATIBLE")
        elif not isinstance(reason, str):
            errors.append("REVIEW_REASON_CODE_INVALID")
        else:
            index[key] = item
    missing = sorted(set(expected) - set(index))
    if missing:
        errors.append("REVIEW_COMPONENT_MISSING")
    if errors:
        raise HumanOverlayReviewError(";".join(sorted(set(errors))))

    duplicate_keys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        for component in row["components"]:
            signature = stable_hash({"frame_sha256": row.get("frame_sha256"), "role": component["role"], "bbox_normalized_xyxy": component.get("bbox_normalized_xyxy")})
            duplicate_keys[signature].append({"anchor_candidate_id": row["anchor_candidate_id"], "role": component["role"], "human_overlay_label": index[(row["anchor_candidate_id"], component["role"])]["human_overlay_label"]})
    warnings = []
    for signature, rows in sorted(duplicate_keys.items()):
        labels = sorted({row["human_overlay_label"] for row in rows})
        if len(rows) > 1 and len(labels) > 1:
            warnings.append({"warning_code": "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW", "grounding_signature": signature, "labels": labels, "reviews": sorted(rows, key=lambda x: (x["anchor_candidate_id"], x["role"]))})
    return index, warnings


def _adjudicate_component(component: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    label = review["human_overlay_label"]
    visibility = component["visibility"]
    status, vis, bbox, usable, routing = {
        "ACCEPTED": ("VALID_VISIBLE_LOCALIZATION", visibility, component.get("bbox_normalized_xyxy"), True, "SPATIAL_COMPONENT_VALID"),
        "BOX_TOO_BROAD": ("INVALID_GEOMETRY", visibility, None, False, "RELOCALIZE_REQUIRED"),
        "BOX_TOO_TIGHT": ("INVALID_GEOMETRY", visibility, None, False, "RELOCALIZE_REQUIRED"),
        "WRONG_OBJECT": ("INVALID_OBJECT", visibility, None, False, "RELOCALIZE_REQUIRED"),
        "INTERFACE_NOT_LOCALIZED": ("INVALID_INTERFACE", visibility, None, False, "RELATIONAL_RELOCALIZATION_REQUIRED"),
        "SHOULD_BE_NOT_VISIBLE": ("VISIBILITY_FALSE_POSITIVE", "NOT_VISIBLE", None, False, "REACQUIRE_REQUIRED"),
        "NOT_VISIBLE_CONFIRMED": ("VALID_NONVISIBLE", "NOT_VISIBLE", None, False, "NOT_VISIBLE_CONFIRMED"),
        "AMBIGUOUS_CONFIRMED": ("VALID_AMBIGUOUS", "AMBIGUOUS", None, False, "AMBIGUOUS_CONFIRMED"),
        "UNRESOLVED": ("UNRESOLVED", visibility, None, False, "ABSTAIN"),
    }[label]
    return {**component, "human_overlay_label": label, "human_reason_code": review["reason_code"],
            "component_review_status": status, "adjudicated_visibility": vis,
            "adjudicated_bbox": bbox, "bbox_usable_for_intervention": usable,
            "routing_reason": routing}


def _anchor_status(row: dict[str, Any]) -> str:
    by_role = {x["role"]: x for x in row["adjudicated_components"]}
    required = [by_role[role] for role in row["required_component_roles"]]
    labels = {x["human_overlay_label"] for x in required}
    if labels == {"ACCEPTED"}:
        return "SPATIAL_COMPONENTS_VALID"
    if any(x["human_overlay_label"] == "INTERFACE_NOT_LOCALIZED" and "INTERFACE" in x["role"] for x in required):
        return "RELATIONAL_RELOCALIZATION_REQUIRED"
    if labels & {"BOX_TOO_BROAD", "BOX_TOO_TIGHT", "WRONG_OBJECT"}:
        return "RELOCALIZE_REQUIRED"
    if labels & {"SHOULD_BE_NOT_VISIBLE", "NOT_VISIBLE_CONFIRMED", "AMBIGUOUS_CONFIRMED"}:
        return "REACQUIRE_REQUIRED"
    return "UNRESOLVED_REQUIRED_COMPONENT"


def _route_anchor(row: dict[str, Any]) -> tuple[str, list[str]]:
    status = row["spatial_anchor_status"]
    components = {x["role"]: x for x in row["adjudicated_components"]}
    temporal_unresolved = [role for role in TEMPORAL_CONTEXTUAL_ROLES if role in components and components[role]["human_overlay_label"] == "UNRESOLVED"]
    schema_flags = ["APPARENT_2D_ONLY"] if (row["observation_role"] in RELATIONAL_OBSERVATION_ROLES or any("INTERFACE" in role for role in row["required_component_roles"])) else []
    if temporal_unresolved:
        return "TEMPORAL_REACQUIRE", schema_flags
    if row["observation_role"] in RELATIONAL_OBSERVATION_ROLES:
        return "RELATIONAL_COMPOSITE_REQUIRED", schema_flags
    if status == "SPATIAL_COMPONENTS_VALID":
        return "ELIGIBLE_SPATIAL_ANCHOR", schema_flags
    return ({"RELOCALIZE_REQUIRED": "RELOCALIZE_REQUIRED", "RELATIONAL_RELOCALIZATION_REQUIRED": "RELATIONAL_RELOCALIZATION_REQUIRED", "REACQUIRE_REQUIRED": "REACQUIRE_REQUIRED", "UNRESOLVED_REQUIRED_COMPONENT": "ABSTAIN"}[status], schema_flags)


def _intervention_components(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for component in row["adjudicated_components"]:
        if component["role"] not in row["required_component_roles"] or not component["bbox_usable_for_intervention"]:
            continue
        bbox = component["adjudicated_bbox"]
        control = generate_control_regions(bbox, 1)
        candidates.append({"intervention_component_id": "reviewed_component_" + stable_hash({"anchor": row["anchor_candidate_id"], "role": component["role"]})[:24],
                           "role": component["role"], "target_roi_normalized_0_1_xyxy": bbox,
                           "frame_path": row["image_path"], "frame_sha256": row["frame_sha256"],
                           "required_variants": ["ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"],
                           "matched_control_geometry": control,
                           "runner_input_status": "READY_FOR_REGISTERED_OPERATOR" if control["available"] else "CONTROL_UNAVAILABLE"})
    return candidates


def _observation_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["observation_claim_id"]].append(row)
    decisions = []
    for observation_id, items in sorted(grouped.items()):
        def rank(row: dict[str, Any]) -> tuple[Any, ...]:
            required = [x for x in row["adjudicated_components"] if x["role"] in row["required_component_roles"]]
            contextual = [x for x in row["adjudicated_components"] if x["role"] in row["contextual_requirements"]]
            warnings = sum(x["human_overlay_label"] != "ACCEPTED" for x in contextual)
            return (0 if row["claim_evidence_route"] == "ELIGIBLE_SPATIAL_ANCHOR" else 1,
                    -sum(x["human_overlay_label"] == "ACCEPTED" for x in required),
                    -sum(x["human_overlay_label"] == "ACCEPTED" for x in contextual), warnings,
                    row["original_anchor_order"], row["anchor_candidate_id"])
        selected = sorted(items, key=rank)[0]
        decision = {"observation_claim_id": observation_id, "observation_claim": selected["observation_claim"],
                    "observation_role": selected["observation_role"], "selected_anchor_candidate_id": selected["anchor_candidate_id"],
                    "route": selected["claim_evidence_route"], "schema_flags": selected["schema_flags"],
                    "candidate_anchor_count": len(items), "certificate_status": "NOT_APPLICABLE",
                    "new_verified_count": 0, "diagnostic_only": True}
        decisions.append(decision)
    return decisions


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["role", "model_visibility", "human_overlay_label", "count"]
    if path.exists():
        raise HumanOverlayReviewError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def adjudicate(*, raw_anchor_manifest: Path, review_jsonl: Path, output_dir: Path, expected_anchor_count: int = 75) -> dict[str, Any]:
    """Validate reviews and write immutable diagnostic-only adjudication artifacts."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise HumanOverlayReviewError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    raw_rows = _strict_rows(raw_anchor_manifest, "RAW_GROUNDING")
    review_rows = _strict_rows(review_jsonl, "HUMAN_REVIEW")
    review_by_key, warnings = _review_index(raw_rows, review_rows, expected_anchor_count=expected_anchor_count)

    adjudicated = []
    for order, raw in enumerate(raw_rows):
        components = [_adjudicate_component(component, review_by_key[(raw["anchor_candidate_id"], component["role"])]) for component in raw["components"]]
        row = {"anchor_candidate_id": raw["anchor_candidate_id"], "spatial_grounding_task_id": raw["spatial_grounding_task_id"],
               "observation_claim_id": raw["observation_claim_id"], "observation_claim": raw["observation_claim"],
               "observation_role": raw["observation_role"], "geometry_type": raw["geometry_type"],
               "required_component_roles": raw["required_component_roles"], "contextual_requirements": raw["contextual_requirements"],
               "image_path": raw["image_path"], "frame_sha256": raw["frame_sha256"], "timestamp_seconds": raw["timestamp_seconds"],
               "source_frame_reference": raw["source_frame_reference"], "original_anchor_order": order,
               "raw_grounding_row_sha256": stable_hash(raw), "adjudicated_components": components,
               "diagnostic_only": True, "certificate_status": "NOT_APPLICABLE", "new_verified_count": 0, "gt_used": False}
        row["spatial_anchor_status"] = _anchor_status(row)
        row["claim_evidence_route"], row["schema_flags"] = _route_anchor(row)
        row["intervention_components"] = _intervention_components(row) if row["claim_evidence_route"] == "ELIGIBLE_SPATIAL_ANCHOR" else []
        adjudicated.append(row)
    adjudicated.sort(key=lambda row: (row["original_anchor_order"], row["anchor_candidate_id"]))
    decisions = _observation_decisions(adjudicated)
    chosen = {row["selected_anchor_candidate_id"] for row in decisions if row["route"] == "ELIGIBLE_SPATIAL_ANCHOR"}
    eligible = [row for row in adjudicated if row["anchor_candidate_id"] in chosen and row["claim_evidence_route"] == "ELIGIBLE_SPATIAL_ANCHOR" and row["intervention_components"]]

    label_counts = Counter(item["human_overlay_label"] for item in review_rows)
    role_label = Counter((item["role"], item["model_visibility"], item["human_overlay_label"]) for item in review_rows)
    observation_role_by_anchor = {row["anchor_candidate_id"]: row["observation_role"] for row in raw_rows}
    observation_role_label = Counter((observation_role_by_anchor[item["anchor_candidate_id"]], item["human_overlay_label"]) for item in review_rows)
    csv_rows = [{"role": role, "model_visibility": vis, "human_overlay_label": label, "count": count}
                for (role, vis, label), count in sorted(role_label.items())]
    spatial_counts = Counter(row["spatial_anchor_status"] for row in adjudicated)
    route_counts = Counter(row["route"] for row in decisions)
    representatives: dict[str, list[str]] = {}
    for label in sorted(label_counts):
        representatives[label] = sorted(item["anchor_candidate_id"] for item in review_rows if item["human_overlay_label"] == label)[:3]
    issue_rows = list(warnings)
    for row in adjudicated:
        for component in row["adjudicated_components"]:
            if component["human_overlay_label"] != "ACCEPTED":
                issue_rows.append({"warning_code": "HUMAN_REVIEW_NONACCEPTED", "anchor_candidate_id": row["anchor_candidate_id"], "observation_claim_id": row["observation_claim_id"], "role": component["role"], "model_visibility": component["visibility"], "human_overlay_label": component["human_overlay_label"], "routing_reason": component["routing_reason"]})
    issue_rows.sort(key=lambda row: canonical_json(row))
    rng = random.Random(1729)
    queue_by_key: dict[str, dict[str, Any]] = {canonical_json(row): {**row, "queue_reason": "WARNING_OR_NONACCEPTED"} for row in issue_rows}
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        strata[(row["role"], row["human_overlay_label"])].append(row)
    for key, choices in sorted(strata.items()):
        item = rng.choice(sorted(choices, key=lambda x: (x["anchor_candidate_id"], x["role"])))
        queue = {**item, "queue_reason": "FIXED_SEED_ROLE_LABEL_STRATUM", "stratum": list(key), "fixed_seed": 1729}
        queue_by_key.setdefault(canonical_json(queue), queue)
    queue = sorted(queue_by_key.values(), key=canonical_json)

    report = {"format": FORMAT, "review_contract_format": REVIEW_FORMAT, "status": "PASS", "raw_anchor_manifest": str(raw_anchor_manifest), "raw_anchor_manifest_sha256": _sha(raw_anchor_manifest),
              "review_jsonl": str(review_jsonl), "review_jsonl_sha256": _sha(review_jsonl), "raw_anchor_count": len(raw_rows), "review_component_count": len(review_rows),
              "label_distribution": dict(sorted(label_counts.items())), "label_ratios": {key: value / len(review_rows) for key, value in sorted(label_counts.items())},
              "observation_role_label_distribution": {f"{role}|{label}": count for (role, label), count in sorted(observation_role_label.items())},
              "spatial_anchor_status_counts": dict(sorted(spatial_counts.items())), "observation_route_counts": dict(sorted(route_counts.items())),
              "required_complete_pass_rate": spatial_counts["SPATIAL_COMPONENTS_VALID"] / len(adjudicated),
              "observation_count": len(decisions), "observations_with_eligible_anchor": sum(row["route"] == "ELIGIBLE_SPATIAL_ANCHOR" for row in decisions),
              "eligible_anchor_count": len(eligible), "visibility_false_positive_count": label_counts["SHOULD_BE_NOT_VISIBLE"],
              "wrong_object_count": label_counts["WRONG_OBJECT"], "interface_not_localized_count": label_counts["INTERFACE_NOT_LOCALIZED"],
              "box_geometry_error_count": label_counts["BOX_TOO_BROAD"] + label_counts["BOX_TOO_TIGHT"], "unresolved_role_distribution": dict(sorted(Counter(row["role"] for row in review_rows if row["human_overlay_label"] == "UNRESOLVED").items())),
              "warning_count": len(warnings), "warning_examples": warnings[:5], "second_pass_queue_count": len(queue),
              "representative_anchor_ids_by_label": representatives,
              "raw_grounding_unchanged": True, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False,
              "certificate_status": "NOT_APPLICABLE", "new_verified_count": 0, "gt_used": False, "diagnostic_only": True}
    markdown = "\n".join(["# Stage 3G human overlay adjudication", "", "This is a localization audit only. It does not establish claim truth, create a certificate, or create VERIFIED evidence.", "", f"- Raw anchors: {len(raw_rows)}", f"- Reviewed components: {len(review_rows)}", f"- Eligible observation anchors: {len(eligible)}", f"- Warnings: {len(warnings)}", "", "## Label distribution", "", *[f"- `{key}`: {value}" for key, value in sorted(label_counts.items())], "", "## Observation routes", "", *[f"- `{key}`: {value}" for key, value in sorted(route_counts.items())], ""])
    output_dir.mkdir(parents=True)
    _write(output_dir / "adjudicated_anchor_manifest.jsonl", adjudicated)
    _write(output_dir / "observation_anchor_decisions.jsonl", decisions)
    _write(output_dir / "eligible_anchor_manifest.jsonl", eligible)
    _write(output_dir / "schema_issue_candidates.jsonl", issue_rows)
    _write(output_dir / "second_pass_review_queue.jsonl", queue)
    _csv(output_dir / "role_label_summary.csv", csv_rows)
    report["artifact_sha256"] = {name: _sha(output_dir / name) for name in ("adjudicated_anchor_manifest.jsonl", "observation_anchor_decisions.jsonl", "eligible_anchor_manifest.jsonl", "schema_issue_candidates.jsonl", "second_pass_review_queue.jsonl", "role_label_summary.csv")}
    report["report_content_sha256"] = stable_hash(report)
    _write(output_dir / "human_review_validation_report.json", report)
    (output_dir / "human_review_validation_report.md").write_text(markdown, encoding="utf-8")
    return report
