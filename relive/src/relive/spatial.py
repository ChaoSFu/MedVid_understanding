"""Claim-conditioned support rectangles, independently typed from target boxes."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from .types import Claim, EvidenceCandidate, ExecutionStatus, SpatialProposal
from .json_protocol import strict_json

COORDINATE_SYSTEM = "normalized_0_1_xyxy"
COORDINATE_SYSTEM_1000 = "normalized_0_1000_xyxy"
COORDINATE_MAPPING_VERSION = "relive-floor-start-ceil-end-v1"


def validate_region(region: Sequence[float]) -> tuple[float, float, float, float]:
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("REGION_MUST_HAVE_FOUR_COORDINATES")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in region):
        raise ValueError("REGION_COORDINATES_MUST_BE_FINITE_NUMBERS")
    x1, y1, x2, y2 = (float(v) for v in region)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("REGION_OUT_OF_RANGE_OR_INVERTED")
    return x1, y1, x2, y2


def normalized_1000_to_region(region: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate an explicit 0–1000 normalized box and convert it to 0–1."""
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("REGION_MUST_HAVE_FOUR_COORDINATES")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in region):
        raise ValueError("REGION_COORDINATES_MUST_BE_FINITE_NUMBERS")
    x1, y1, x2, y2 = (float(v) for v in region)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError("REGION_1000_OUT_OF_RANGE_OR_INVERTED")
    return validate_region((x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000))


def normalized_to_pixel_bbox(region: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Half-open rectangle mapping. Invalid coordinates are rejected, never clipped.

    Uses the H4 floor-start/ceil-end convention on internal 0–1 coordinates.
    """
    box = validate_region(region)
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("IMAGE_DIMENSIONS_MUST_BE_POSITIVE_INTEGERS")
    return math.floor(box[0] * width), math.floor(box[1] * height), math.ceil(box[2] * width), math.ceil(box[3] * height)


def parse_proposal(raw: str, candidate: EvidenceCandidate, claim: Claim,
                   raw_response_ref: str | None = None) -> SpatialProposal:
    payload = {"candidate": candidate.candidate_id, "claim": claim.claim_id, "raw": raw}
    proposal_id = "proposal-" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]
    provenance: dict[str, Any] = {
        "raw_response_ref": raw_response_ref,
        "parser_version": "relive-spatial-parser-v2",
        "coordinate_mapping_version": COORDINATE_MAPPING_VERSION,
        "limitation": "One static support rectangle is applied to all supplied frames.",
    }
    mapping = {"mode": "static_across_candidate", "frame_ids": list(candidate.frame_ids),
               "timestamps": list(candidate.timestamps), "timestamp_unit": "seconds_if_supplied"}
    try:
        parsed = strict_json(raw)
        if not isinstance(parsed, dict) or "support_region" not in parsed:
            raise ValueError("EXPECTED_OBJECT_WITH_SUPPORT_REGION")
        if set(parsed) - {"support_region", "target_bbox", "coordinate_system"}:
            raise ValueError("UNEXPECTED_PROPOSAL_FIELDS")
        source_system = parsed.get("coordinate_system", COORDINATE_SYSTEM)
        if source_system not in {COORDINATE_SYSTEM, COORDINATE_SYSTEM_1000}:
            raise ValueError("UNSUPPORTED_COORDINATE_SYSTEM")
        if source_system == COORDINATE_SYSTEM_1000:
            region = normalized_1000_to_region(parsed["support_region"])
            target = (normalized_1000_to_region(parsed["target_bbox"])
                      if parsed.get("target_bbox") is not None else None)
            conversion = "divide_by_1000"
        else:
            region = validate_region(parsed["support_region"])
            target = validate_region(parsed["target_bbox"]) if parsed.get("target_bbox") is not None else None
            conversion = "identity_normalized_0_1"
        area = (region[2] - region[0]) * (region[3] - region[1])
        provenance.update({"support_area_fraction": area, "large_region": area >= 0.8,
                           "full_frame_region": region == (0.0, 0.0, 1.0, 1.0),
                           "original_coordinates": parsed["support_region"],
                           "source_coordinate_system": source_system,
                           "coordinate_conversion": conversion})
    except (ValueError, TypeError) as exc:
        provenance["failure_reason"] = str(exc)
        return SpatialProposal(proposal_id, candidate.candidate_id, claim.claim_id, None,
                               frame_mapping=mapping, parser_status=ExecutionStatus.PARSE_ERROR,
                               provenance=provenance)
    return SpatialProposal(proposal_id, candidate.candidate_id, claim.claim_id, region,
                           target_bbox=target, frame_mapping=mapping, provenance=provenance)
