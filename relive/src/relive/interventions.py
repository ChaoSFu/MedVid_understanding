"""Deterministic image interventions with byte-exact unchanged-region audits.

The GaussianBlur + hard-mask Image.composite implementation is extracted from
evidence_stability/scripts/16_generate_h4_spatial_interventions.py:make_roi_image
(H4 h4_spatial_v1), retaining its historical file unchanged. The extraction avoids
that script's GT-dependent imports; configurable radius, strict normalized boxes,
non-overlapping matched controls and RGB/RGBA/L pixel audits are ReliVE-v1 changes.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from PIL import Image, ImageChops, ImageFilter

from .spatial import COORDINATE_MAPPING_VERSION, normalized_to_pixel_bbox, validate_region
from .types import ExecutionStatus, SemanticStatus, VerificationResult, to_dict

INTERVENTION_VERSION = "relive-h4-pure-gaussian-hard-mask-v1"
CONTROL_VERSION = "relive-matched-corners-edges-v1"
VARIANTS = ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL")


def _intersection_area(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def generate_control_regions(region: Sequence[float], count: int) -> dict[str, Any]:
    """Predeclared corner/edge/adjacent placements, disjoint from target and each other.

    All controls have the target's normalized dimensions. Too few valid positions
    yields CONTROL_UNAVAILABLE; partial positions remain in the audit. Positions
    never depend on image content, model responses, or hidden annotations.
    """
    box = validate_region(region)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("CONTROL_COUNT_MUST_BE_NONNEGATIVE_INTEGER")
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    placements = [(0.0, 0.0), (1.0 - width, 0.0), (0.0, 1.0 - height),
                  (1.0 - width, 1.0 - height), ((1.0 - width) / 2, 0.0),
                  ((1.0 - width) / 2, 1.0 - height), (0.0, (1.0 - height) / 2),
                  (1.0 - width, (1.0 - height) / 2),
                  (x1 - width, y1), (x2, y1), (x1, y1 - height), (x1, y2)]
    selected = []
    for x, y in placements:
        if len(selected) >= count:
            break
        proposal = (x, y, x + width, y + height)
        if not (0 <= x < x + width <= 1 and 0 <= y < y + height <= 1):
            continue
        if _intersection_area(box, proposal) > 0 or any(_intersection_area(old, proposal) > 0 for old in selected):
            continue
        if proposal not in selected:
            selected.append(proposal)
    available = len(selected) == count
    return {"status": "CONTROL_AVAILABLE" if available else "CONTROL_UNAVAILABLE",
            "available": available, "regions": selected, "requested_count": count,
            "valid_count": len(selected), "version": CONTROL_VERSION,
            "geometry_rule": "fixed_corners_then_edge_centers_then_target_adjacent; target/control pairwise disjoint",
            "matching": "identical_normalized_width_and_height",
            "limitation": "Controls may contain other evidence; rasterization can change pixel area by one row/column."}


def _pixel_stats(original: Image.Image, altered: Image.Image, box) -> dict[str, Any]:
    mask = Image.new("L", original.size, 0)
    mask.paste(255, box=box)
    changed_channels = ImageChops.difference(original, altered)
    # RGBA getbbox can ignore RGB differences at zero alpha; compare every band.
    changed = Image.new("L", original.size, 0)
    for band in changed_channels.split():
        changed = ImageChops.lighter(changed, band)
    inside = ImageChops.multiply(changed, mask)
    outside = ImageChops.multiply(changed, ImageChops.invert(mask))
    return {"inside_unchanged": inside.getbbox() is None,
            "outside_unchanged": outside.getbbox() is None,
            "inside_max_channel_delta": inside.getextrema()[1],
            "outside_max_channel_delta": outside.getextrema()[1],
            "any_pixel_changed": changed.getbbox() is not None}


def apply_intervention(image: Image.Image, region: Sequence[float], variant: str,
                       blur_radius: float) -> tuple[Image.Image, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"UNSUPPORTED_INTERVENTION: {variant}")
    if isinstance(blur_radius, bool) or not isinstance(blur_radius, (int, float)) or not math.isfinite(blur_radius) or blur_radius <= 0:
        raise ValueError("BLUR_RADIUS_MUST_BE_FINITE_POSITIVE")
    if image.mode not in {"RGB", "RGBA", "L"}:
        raise ValueError("INTERVENTION_IMAGE_MODE_MUST_BE_RGB_RGBA_OR_L")
    box = normalized_to_pixel_bbox(region, *image.size)
    if variant == "ORIGINAL":
        altered = image.copy()
    else:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        mask = Image.new("L", image.size, 0)
        mask.paste(255, box=box)
        altered = Image.composite(image, blurred, mask) if variant == "KEEP_TARGET" else Image.composite(blurred, image, mask)
    stats = _pixel_stats(image, altered, box)
    expected = "both" if variant == "ORIGINAL" else "inside" if variant == "KEEP_TARGET" else "outside"
    passed = ((stats["inside_unchanged"] and stats["outside_unchanged"]) if expected == "both" else stats[f"{expected}_unchanged"])
    audit = {"version": INTERVENTION_VERSION, "variant": variant, "region": list(region),
             "pixel_bbox": list(box), "pixel_bbox_convention": "half_open_xyxy",
             "mapping_version": COORDINATE_MAPPING_VERSION, "blur_radius": float(blur_radius),
             "original_size": list(image.size), "output_size": list(altered.size),
             "resolution_preserved": altered.size == image.size,
             "unchanged_region": expected, "pixel_audit_pass": bool(passed), **stats}
    if not passed or altered.size != image.size:
        raise RuntimeError("INTERVENTION_PIXEL_AUDIT_FAILED")
    return altered, audit


def check_spatial(original: VerificationResult, keep: VerificationResult | None,
                  drop: VerificationResult | None, controls: Sequence[VerificationResult | None],
                  control_available: bool, require_controls: bool = True,
                  expected_control_count: int | None = None) -> dict[str, Any]:
    """Evaluate the predeclared protocol, preserving technical/semantic separation.

    Callers running matched controls should supply the frozen control count so a
    partial set of successful results cannot stand in for all planned controls.
    """
    if expected_control_count is not None and (isinstance(expected_control_count, bool)
            or not isinstance(expected_control_count, int) or expected_control_count < 0):
        raise ValueError("EXPECTED_CONTROL_COUNT_MUST_BE_NONNEGATIVE_INTEGER")
    checks = {"original": original, "keep": keep, "drop": drop}
    reasons = []
    semantic = lambda result: result.semantic_status if result and result.execution_status == ExecutionStatus.OK else None
    if semantic(original) != SemanticStatus.SUPPORTED:
        reasons.append("ORIGINAL_NOT_SUPPORTED")
    if any(result is None or result.execution_status != ExecutionStatus.OK for result in checks.values()):
        reasons.append("TECHNICAL_FAILURE")
    if semantic(keep) in {SemanticStatus.INSUFFICIENT, SemanticStatus.CONTRADICTED}:
        reasons.append("KEEP_SUPPORT_LOST")
    if semantic(drop) == SemanticStatus.SUPPORTED:
        reasons.append("DEPENDENCE_UNRESOLVED")
    intervention_results = [keep, drop] + (list(controls) if require_controls else [])
    if any(semantic(result) == SemanticStatus.CONTRADICTED for result in intervention_results):
        reasons.append("INTERVENTION_CONTRADICTION")
    if require_controls:
        if not control_available or not controls:
            reasons.append("CONTROL_UNAVAILABLE")
        if expected_control_count is not None and len(controls) != expected_control_count:
            reasons.append("CONTROL_RESULT_COUNT_MISMATCH")
        if any(result is None or result.execution_status != ExecutionStatus.OK for result in controls):
            reasons.append("TECHNICAL_FAILURE")
        lost_controls = any(semantic(result) in {SemanticStatus.INSUFFICIENT, SemanticStatus.CONTRADICTED} for result in controls)
        if lost_controls:
            reasons.append("CONTROL_SUPPORT_LOST")
            if semantic(drop) in {SemanticStatus.INSUFFICIENT, SemanticStatus.CONTRADICTED}:
                reasons.append("NONSPECIFIC_INTERVENTION_RESPONSE")
    passed = (semantic(original) == SemanticStatus.SUPPORTED and semantic(keep) == SemanticStatus.SUPPORTED
              and semantic(drop) == SemanticStatus.INSUFFICIENT
              and (not require_controls or (control_available and bool(controls)
                   and (expected_control_count is None or len(controls) == expected_control_count)
                   and all(semantic(result) == SemanticStatus.SUPPORTED for result in controls))))
    reasons = list(dict.fromkeys(reasons))
    if passed:
        status = "REGION_SPECIFICITY_PASS" if require_controls else "KEEP_DROP_PASS"
    else:
        priority = ["TECHNICAL_FAILURE", "ORIGINAL_NOT_SUPPORTED", "CONTROL_UNAVAILABLE", "CONTROL_RESULT_COUNT_MISMATCH",
                    "INTERVENTION_CONTRADICTION", "NONSPECIFIC_INTERVENTION_RESPONSE", "KEEP_SUPPORT_LOST",
                    "DEPENDENCE_UNRESOLVED", "CONTROL_SUPPORT_LOST"]
        status = next((reason for reason in priority if reason in reasons), "SPATIAL_UNRESOLVED")
    return {"pass": passed, "status": status, "reasons": reasons, "require_controls": require_controls,
            "control_available": control_available, "expected_control_count": expected_control_count,
            "observed_control_count": len(controls),
            "aggregation": "all_predeclared_valid_controls_supported" if require_controls else "keep_drop_without_controls",
            "references": {**{key: to_dict(value) for key, value in checks.items()}, "controls": to_dict(controls)},
            "interpretation": "Spatial intervention response under this protocol; not a causal proof."}
