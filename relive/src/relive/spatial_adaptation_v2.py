"""Phase 3-v2 refiner contract: canonical coordinates and explicit unresolved state."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from typing import Any

from .claims import PROMPT_VERSIONS, stable_id
from .json_protocol import strict_json
from .spatial import normalized_1000_to_region, parse_proposal, validate_region
from .types import EvidenceCandidate, SpatialProposal
from .spatial_adaptation import MAX_SPATIAL_REFINEMENT_ROUNDS, SpatialAdaptationError, route_refinement

PHASE3_V2_FORMAT = "relive-phase3-spatial-adaptation-v2"


def canonicalize_region_1000(region) -> list[int]:
    """Convert an internal 0–1 rectangle to the closed integer refiner contract."""
    x1, y1, x2, y2 = validate_region(region)
    result = [math.floor(x1 * 1000), math.floor(y1 * 1000),
              math.ceil(x2 * 1000), math.ceil(y2 * 1000)]
    normalized_1000_to_region(result)
    return result


def route_refinement_v2(diagnosis: dict[str, Any]) -> dict[str, Any]:
    route = dict(route_refinement(diagnosis))
    if route["allowed"]:
        route["stage"] = {
            "spatial_refine_dependence": "spatial_refine_dependence_v2",
            "spatial_refine_sufficiency": "spatial_refine_sufficiency_v2",
            "spatial_refine_control_geometry": "spatial_refine_control_geometry_v2",
            "spatial_refine_intervention_effect": "spatial_refine_intervention_effect_v2",
        }[route["stage"]]
        route["prompt_version"] = PROMPT_VERSIONS[route["stage"]]
    return route


def _failure(code: str, *, raw: str | None, raw_response_ref: str | None,
             previous_bbox_1000: list[int], reason: str | None = None) -> dict[str, Any]:
    return {"outcome": "PARSE_FAILURE", "failure_code": code, "raw_model_response": raw,
            "raw_response_ref": raw_response_ref, "previous_bbox_normalized_0_1000": previous_bbox_1000,
            "model_reason": reason, "proposal": None, "bbox_normalized_0_1000": None}


def parse_refinement_v2(raw: str, candidate: EvidenceCandidate, claim, *, parent: SpatialProposal,
                        route: dict[str, Any], raw_response_ref: str | None) -> dict[str, Any]:
    """Parse the closed v2 contract without inventing a fallback rectangle."""
    previous = canonicalize_region_1000(parent.support_region)
    try:
        data = strict_json(raw)
    except (ValueError, TypeError) as exc:
        return _failure("REFINEMENT_JSON_PARSE_FAILURE", raw=raw, raw_response_ref=raw_response_ref,
                        previous_bbox_1000=previous, reason=str(exc))
    if not isinstance(data, dict) or set(data) != {"status", "bbox_normalized_0_1000", "reason"}:
        return _failure("REFINEMENT_SCHEMA_FAILURE", raw=raw, raw_response_ref=raw_response_ref,
                        previous_bbox_1000=previous)
    status, bbox, reason = data["status"], data["bbox_normalized_0_1000"], data["reason"]
    if status not in {"PROPOSED", "UNRESOLVED"} or not isinstance(reason, str) or len(reason) > 2000:
        return _failure("REFINEMENT_SCHEMA_FAILURE", raw=raw, raw_response_ref=raw_response_ref,
                        previous_bbox_1000=previous)
    if status == "UNRESOLVED":
        if bbox is not None:
            return _failure("REFINEMENT_UNRESOLVED_REQUIRES_NULL_BBOX", raw=raw, raw_response_ref=raw_response_ref,
                            previous_bbox_1000=previous, reason=reason)
        return {"outcome": "UNRESOLVED", "failure_code": None, "raw_model_response": raw,
                "raw_response_ref": raw_response_ref, "model_reason": reason,
                "previous_bbox_normalized_0_1000": previous, "bbox_normalized_0_1000": None, "proposal": None}
    if not isinstance(bbox, list) or len(bbox) != 4 or any(type(value) is not int for value in bbox):
        return _failure("REFINEMENT_COORDINATE_VIOLATION", raw=raw, raw_response_ref=raw_response_ref,
                        previous_bbox_1000=previous, reason=reason)
    try:
        region = normalized_1000_to_region(bbox)
    except (TypeError, ValueError) as exc:
        return _failure("REFINEMENT_COORDINATE_VIOLATION", raw=raw, raw_response_ref=raw_response_ref,
                        previous_bbox_1000=previous, reason=f"{reason}: {exc}")
    if bbox == previous:
        return {"outcome": "REFINEMENT_NO_OP", "failure_code": "REFINEMENT_NO_OP", "raw_model_response": raw,
                "raw_response_ref": raw_response_ref, "model_reason": reason,
                "previous_bbox_normalized_0_1000": previous, "bbox_normalized_0_1000": bbox, "proposal": None}
    proposal = parse_proposal(json.dumps({"support_region": bbox,
                                          "coordinate_system": "normalized_0_1000_xyxy"}), candidate, claim,
                              raw_response_ref=raw_response_ref)
    proposal = replace(proposal, proposal_id=stable_id("refinement-v2", [candidate.candidate_id, claim.claim_id,
                       parent.proposal_id, route["prompt_version"], 1, bbox]),
                       provenance={**proposal.provenance, "phase3_format": PHASE3_V2_FORMAT,
                                   "parent_proposal_id": parent.proposal_id,
                                   "previous_bbox_normalized_0_1_xyxy": list(parent.support_region),
                                   "previous_bbox_normalized_0_1000": previous,
                                   "refinement_prompt_version": route["prompt_version"],
                                   "refinement_action": route["action"], "spatial_round": 1,
                                   "model_reason": reason})
    return {"outcome": "PROPOSED", "failure_code": None, "raw_model_response": raw,
            "raw_response_ref": raw_response_ref, "model_reason": reason,
            "previous_bbox_normalized_0_1000": previous, "bbox_normalized_0_1000": bbox,
            "proposal": proposal}


class SpatialEvidenceAdaptationControllerV2:
    """One-round v2 controller that returns a proposal or explicit unresolved."""

    def __init__(self, max_rounds: int = MAX_SPATIAL_REFINEMENT_ROUNDS):
        if type(max_rounds) is not int or max_rounds != 1:
            raise SpatialAdaptationError("PHASE3_V2_REQUIRES_EXACTLY_ONE_SPATIAL_REFINEMENT_ROUND")
        self.max_rounds = max_rounds

    def route(self, diagnosis: dict[str, Any], current_round: int = 0) -> dict[str, Any]:
        if type(current_round) is not int or current_round < 0:
            raise SpatialAdaptationError("INVALID_SPATIAL_REFINEMENT_ROUND")
        if current_round >= self.max_rounds:
            return {"allowed": False, "routing_reason": "MAX_SPATIAL_REFINEMENT_ROUNDS",
                    "action": "SPATIAL_UNRESOLVED", "stage": None, "prompt_version": None}
        return route_refinement_v2(diagnosis)

    def parse(self, raw: str, candidate: EvidenceCandidate, claim, *, parent: SpatialProposal,
              route: dict[str, Any], raw_response_ref: str | None) -> dict[str, Any]:
        if route.get("allowed") is not True:
            raise SpatialAdaptationError("REFINEMENT_ROUTE_FORBIDS_PROPOSAL")
        return parse_refinement_v2(raw, candidate, claim, parent=parent, route=route,
                                   raw_response_ref=raw_response_ref)
