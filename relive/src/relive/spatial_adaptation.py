"""Phase 3 GT-free, within-candidate spatial evidence adaptation.

This module owns only the reason-specific R0 -> R1 transition.  It consumes
immutable Phase 2 and 2.5 artifacts, never generates temporal candidates, and
reuses :meth:`SampleRunner.spatial_with_proposal` plus the existing certificate
builder for R1 admission.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any

from .claims import PROMPT_VERSIONS, stable_id
from .json_protocol import strict_json
from .spatial import normalized_1000_to_region, parse_proposal
from .types import EvidenceCandidate, ExecutionStatus, SpatialProposal

PHASE3_FORMAT = "relive-phase3-spatial-adaptation-v1"
MAX_SPATIAL_REFINEMENT_ROUNDS = 1


class SpatialAdaptationError(ValueError):
    pass


def _status(references: dict[str, Any], name: str) -> str | None:
    value = references.get(name) if isinstance(references, dict) else None
    return value.get("semantic_status") if isinstance(value, dict) else None


def route_refinement(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Map only diagnosed failure states to a versioned R1 proposal route."""
    reasons = set(diagnosis.get("original_failure_reasons", []))
    mismatch = diagnosis.get("control_count_audit", {}).get("classification")
    root = diagnosis.get("root_cause_class")
    effect = diagnosis.get("intervention_no_effect_audit", {}).get("classification")
    if "CONTROL_RESULT_COUNT_MISMATCH" in reasons:
        if mismatch == "B_LEGITIMATE_PROTOCOL_UNAVAILABLE":
            return {"allowed": True, "routing_reason": "CONTROL_GEOMETRY_UNAVAILABLE",
                    "action": "SPATIAL_REFINE_OVERBROAD_ROI",
                    "stage": "spatial_refine_control_geometry",
                    "prompt_version": PROMPT_VERSIONS["spatial_refine_control_geometry"]}
        return {"allowed": False, "routing_reason": "UNRESOLVED_CONTROL_RESULT_COUNT_MISMATCH",
                "action": "SPATIAL_UNRESOLVED", "stage": None, "prompt_version": None}
    if "ORIGINAL_INSUFFICIENT" in reasons or root == "TEMPORAL_OR_SEMANTIC_INSUFFICIENT":
        return {"allowed": False, "routing_reason": "ORIGINAL_INSUFFICIENT",
                "action": "TEMPORAL_REACQUIRE", "stage": None, "prompt_version": None}
    if root == "TECHNICAL_ERROR" or "TECHNICAL_FAILURE" in reasons:
        return {"allowed": False, "routing_reason": "TECHNICAL_DIAGNOSTIC",
                "action": "SPATIAL_UNRESOLVED", "stage": None, "prompt_version": None}
    if root == "SPATIAL_PROPOSAL_OR_ROI_FAILURE" and effect in {
            "CLIPPED_TO_ZERO_AREA", "FULL_FRAME_OR_NO_COMPLEMENT_ROI", "UNIFORM_REGION"}:
        return {"allowed": True, "routing_reason": "OVERBROAD_OR_DEGENERATE_ROI",
                "action": "SPATIAL_REFINE_VISUALLY_GROUNDED_ROI",
                "stage": "spatial_refine_intervention_effect",
                "prompt_version": PROMPT_VERSIONS["spatial_refine_intervention_effect"]}
    if "DEPENDENCE_UNRESOLVED" in reasons:
        return {"allowed": True, "routing_reason": "DEPENDENCE_UNRESOLVED",
                "action": "SPATIAL_REFINE_RESIDUAL_SUPPORT",
                "stage": "spatial_refine_dependence",
                "prompt_version": PROMPT_VERSIONS["spatial_refine_dependence"]}
    if reasons & {"KEEP_SUPPORT_LOST", "CONTROL_SUPPORT_LOST", "NONSPECIFIC_INTERVENTION_RESPONSE"}:
        return {"allowed": True, "routing_reason": "SUFFICIENCY_UNRESOLVED",
                "action": "SPATIAL_REFINE_CLAIM_SUFFICIENCY",
                "stage": "spatial_refine_sufficiency",
                "prompt_version": PROMPT_VERSIONS["spatial_refine_sufficiency"]}
    return {"allowed": False, "routing_reason": "UNRESOLVED_FAILURE_MODE",
            "action": "SPATIAL_UNRESOLVED", "stage": None, "prompt_version": None}


class SpatialEvidenceAdaptationController:
    """Bounded, reason-specific R0 -> R1 spatial proposal controller.

    The controller has no certificate authority and does not touch temporal
    acquisition.  It only selects a frozen prompt route and parses one new
    proposal; callers must send that proposal through the formal protocol.
    """

    def __init__(self, max_rounds: int = MAX_SPATIAL_REFINEMENT_ROUNDS):
        if type(max_rounds) is not int or max_rounds != MAX_SPATIAL_REFINEMENT_ROUNDS:
            raise SpatialAdaptationError("PHASE3_V1_REQUIRES_EXACTLY_ONE_SPATIAL_REFINEMENT_ROUND")
        self.max_rounds = max_rounds

    def route(self, diagnosis: dict[str, Any], current_round: int) -> dict[str, Any]:
        if type(current_round) is not int or current_round < 0:
            raise SpatialAdaptationError("INVALID_SPATIAL_REFINEMENT_ROUND")
        if current_round >= self.max_rounds:
            return {"allowed": False, "routing_reason": "MAX_SPATIAL_REFINEMENT_ROUNDS",
                    "action": "SPATIAL_UNRESOLVED", "stage": None, "prompt_version": None}
        return route_refinement(diagnosis)

    def parse(self, raw: str, candidate: EvidenceCandidate, claim, *, parent: SpatialProposal,
              route: dict[str, Any], raw_response_ref: str | None) -> SpatialProposal:
        if route.get("allowed") is not True:
            raise SpatialAdaptationError("REFINEMENT_ROUTE_FORBIDS_PROPOSAL")
        proposal = parse_refined_proposal(raw, candidate, claim, parent=parent, route=route,
                                          raw_response_ref=raw_response_ref)
        if proposal.parser_status == ExecutionStatus.OK and proposal.support_region == parent.support_region:
            return replace(proposal, parser_status=ExecutionStatus.PARSE_ERROR,
                           provenance={**proposal.provenance, "failure_reason": "REFINEMENT_BBOX_UNCHANGED"})
        return proposal


def parse_refined_proposal(raw: str, candidate: EvidenceCandidate, claim, *, parent: SpatialProposal,
                           route: dict[str, Any], raw_response_ref: str | None) -> SpatialProposal:
    """Require the closed integer 0..1000 R1 schema and bind parent provenance."""
    try:
        data = strict_json(raw)
        if not isinstance(data, dict) or set(data) != {"support_region", "coordinate_system"}:
            raise ValueError("EXPECTED_REFINEMENT_SUPPORT_REGION_AND_COORDINATE_SYSTEM")
        if data.get("coordinate_system") != "normalized_0_1000_xyxy":
            raise ValueError("REFINEMENT_REQUIRES_NORMALIZED_0_1000_XYXY")
        box = data.get("support_region")
        if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
            raise ValueError("REFINEMENT_COORDINATES_MUST_BE_FOUR_INTEGERS")
        normalized_1000_to_region(box)
    except (ValueError, TypeError) as exc:
        failed = parse_proposal("{}", candidate, claim, raw_response_ref=raw_response_ref)
        return replace(failed, proposal_id=stable_id("refinement", [candidate.candidate_id, claim.claim_id,
                       parent.proposal_id, route.get("prompt_version"), raw]),
                       provenance={**failed.provenance, "phase3_format": PHASE3_FORMAT,
                                   "parent_proposal_id": parent.proposal_id,
                                   "refinement_prompt_version": route.get("prompt_version"),
                                   "refinement_action": route.get("action"),
                                   "failure_reason": str(exc)})
    parsed = parse_proposal(raw, candidate, claim, raw_response_ref=raw_response_ref)
    return replace(parsed, proposal_id=stable_id("refinement", [candidate.candidate_id, claim.claim_id,
                   parent.proposal_id, route["prompt_version"], 1, raw]),
                   provenance={**parsed.provenance, "phase3_format": PHASE3_FORMAT,
                               "parent_proposal_id": parent.proposal_id,
                               "refinement_prompt_version": route["prompt_version"],
                               "refinement_action": route["action"], "spatial_round": 1})


def geometry_change(previous: list[float] | tuple[float, ...], current: list[float] | tuple[float, ...]) -> dict[str, float]:
    px1, py1, px2, py2 = (float(value) for value in previous)
    cx1, cy1, cx2, cy2 = (float(value) for value in current)
    previous_area, current_area = (px2 - px1) * (py2 - py1), (cx2 - cx1) * (cy2 - cy1)
    inter = max(0.0, min(px2, cx2) - max(px1, cx1)) * max(0.0, min(py2, cy2) - max(py1, cy1))
    union = previous_area + current_area - inter
    return {"delta_area_fraction": current_area - previous_area,
            "center_shift_x": ((cx1 + cx2) - (px1 + px2)) / 2,
            "center_shift_y": ((cy1 + cy2) - (py1 + py2)) / 2,
            "intersection_area_fraction": inter,
            "iou": inter / union if union else 0.0}


def candidate_from_manifest(row: dict[str, Any], sample) -> EvidenceCandidate:
    """Reconstruct only a frozen candidate; no acquisition function is called."""
    prohibited = {"reference_answer", "assistant_answer", "temporal_gt", "bbox", "mask", "struc_info", "rc_info", "gt_iou", "true_support", "spurious_support", "evaluation_artifact"}
    # Reject declared annotation fields only.  A substring test would wrongly
    # reject legitimate fixed-pool fields such as ``window_length``.
    if any(str(key).lower() in prohibited or str(key).lower().startswith("gt_")
           or str(key).lower().endswith("_gt") for key in row):
        raise SpatialAdaptationError("PHASE3_CANDIDATE_MANIFEST_CONTAINS_GT_SHAPED_FIELD")
    if row.get("sample_id") != sample.sample_id or not isinstance(row.get("candidate_id"), str):
        raise SpatialAdaptationError("PHASE3_CANDIDATE_MANIFEST_BINDING_MISMATCH")
    ids = row.get("frame_ids")
    lookup = {frame.frame_id: frame for frame in sample.frames}
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(frame_id not in lookup for frame_id in ids):
        raise SpatialAdaptationError("PHASE3_FROZEN_CANDIDATE_FRAMES_INVALID")
    paths = row.get("source_frame_paths")
    if paths is not None:
        expected_paths = [lookup[frame_id].path for frame_id in ids]
        if not isinstance(paths, list) or paths != expected_paths:
            raise SpatialAdaptationError("PHASE3_FROZEN_CANDIDATE_SOURCE_PATH_MISMATCH")
    references = row.get("source_frame_references")
    expected_references = [lookup[frame_id].source_reference for frame_id in ids]
    if references is not None and any(value is not None for value in expected_references):
        if not isinstance(references, list) or references != expected_references:
            raise SpatialAdaptationError("PHASE3_FROZEN_CANDIDATE_SOURCE_REFERENCE_MISMATCH")
    return EvidenceCandidate(row["candidate_id"], sample.sample_id, tuple(ids),
                             tuple(lookup[frame_id].timestamp for frame_id in ids), row.get("candidate_rank"),
                             row.get("rank_source", "chronological"), provenance={
                                 "phase3_frozen_candidate_manifest_hash": row.get("manifest_hash"),
                                 "source_frame_references": row.get("source_frame_references", []),
                                 "source_selector": "phase2_frozen_candidate_manifest"})


def historical_pattern(certificate: dict[str, Any]) -> dict[str, Any]:
    spatial = certificate.get("checks", {}).get("spatial", {})
    spatial = spatial if isinstance(spatial, dict) else {}
    refs = spatial.get("references", {}) if isinstance(spatial.get("references"), dict) else {}
    proposal = spatial.get("proposal", {}) if isinstance(spatial.get("proposal"), dict) else {}
    controls = refs.get("controls", []) if isinstance(refs.get("controls"), list) else []
    return {"proposal_id": proposal.get("proposal_id"), "bbox": proposal.get("support_region"),
            "bbox_area_fraction": spatial.get("support_area_fraction"),
            "original_status": certificate.get("checks", {}).get("semantic", {}).get("status"),
            "keep_status": _status(refs, "keep"), "drop_status": _status(refs, "drop"),
            "control_status": [item.get("semantic_status") for item in controls if isinstance(item, dict)],
            "control_available": spatial.get("control_available"),
            "certificate_status": certificate.get("final_status"),
            "failure_reasons": list(certificate.get("failure_reasons", []))}
