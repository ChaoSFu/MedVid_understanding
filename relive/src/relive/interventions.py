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
"""Version of the historical Gaussian implementation retained for compatibility."""

# The protocol is separate from certificate policy.  It identifies exactly how
# pixels were transformed, without changing what semantic_spatial requires.
INTERVENTION_PROTOCOL_VERSION = "relive-spatial-intervention-protocol-v2"
GAUSSIAN_OPERATOR = "gaussian_blur"
OPAQUE_GRAY_OPERATOR = "opaque_gray"
OPAQUE_GRAY_VERSION = "relive-opaque-gray-hard-mask-v1"
CONTROL_VERSION = "relive-matched-corners-edges-v1"
VARIANTS = ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL")


def resolve_intervention_spec(spec: dict[str, Any] | None, blur_radius: float) -> dict[str, Any]:
    """Validate and canonicalize a closed core spatial-operator declaration.

    ``blur_radius`` remains the compatibility input for old Gaussian configs.
    The normalized result always contains an explicit operator, immutable
    implementation version, and closed parameters, so it can be bound to cache
    identity and certificate provenance.
    """
    if isinstance(blur_radius, bool) or not isinstance(blur_radius, (int, float)) or not math.isfinite(blur_radius) or blur_radius <= 0:
        raise ValueError("BLUR_RADIUS_MUST_BE_FINITE_POSITIVE")
    if spec is None:
        return {"protocol_version": INTERVENTION_PROTOCOL_VERSION, "operator": GAUSSIAN_OPERATOR,
                "operator_version": INTERVENTION_VERSION, "parameters": {"blur_radius": float(blur_radius)}}
    if not isinstance(spec, dict):
        raise ValueError("SPATIAL_INTERVENTION_REQUIRES_OPERATOR_VERSION_AND_PARAMETERS")
    # validate_config stores the canonical result, so a second validation pass
    # must accept its protocol marker while still rejecting any other field.
    if set(spec) == {"protocol_version", "operator", "operator_version", "parameters"}:
        if spec.get("protocol_version") != INTERVENTION_PROTOCOL_VERSION:
            raise ValueError("UNSUPPORTED_SPATIAL_INTERVENTION_PROTOCOL_VERSION")
        spec = {key: spec[key] for key in ("operator", "operator_version", "parameters")}
    if set(spec) != {"operator", "operator_version", "parameters"}:
        raise ValueError("SPATIAL_INTERVENTION_REQUIRES_OPERATOR_VERSION_AND_PARAMETERS")
    operator, version, parameters = spec.get("operator"), spec.get("operator_version"), spec.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("SPATIAL_INTERVENTION_PARAMETERS_MUST_BE_OBJECT")
    if operator == GAUSSIAN_OPERATOR:
        if version != INTERVENTION_VERSION or set(parameters) != {"blur_radius"}:
            raise ValueError("UNSUPPORTED_GAUSSIAN_INTERVENTION_SPEC")
        radius = parameters["blur_radius"]
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
            raise ValueError("BLUR_RADIUS_MUST_BE_FINITE_POSITIVE")
        if float(radius) != float(blur_radius):
            raise ValueError("SPATIAL_INTERVENTION_BLUR_RADIUS_MISMATCH")
        return {"protocol_version": INTERVENTION_PROTOCOL_VERSION, "operator": operator,
                "operator_version": version, "parameters": {"blur_radius": float(radius)}}
    if operator == OPAQUE_GRAY_OPERATOR:
        if version != OPAQUE_GRAY_VERSION or set(parameters) != {"fill_rgb"} or parameters.get("fill_rgb") != [127, 127, 127]:
            raise ValueError("UNSUPPORTED_OPAQUE_GRAY_INTERVENTION_SPEC")
        return {"protocol_version": INTERVENTION_PROTOCOL_VERSION, "operator": operator,
                "operator_version": version, "parameters": {"fill_rgb": [127, 127, 127]}}
    raise ValueError("UNSUPPORTED_SPATIAL_INTERVENTION_OPERATOR")


def _intersection_area(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _control_placements(box: Sequence[float]) -> list[tuple[float, float]]:
    """Return the closed, ordered placement list used by the core protocol."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return [(0.0, 0.0), (1.0 - width, 0.0), (0.0, 1.0 - height),
            (1.0 - width, 1.0 - height), ((1.0 - width) / 2, 0.0),
            ((1.0 - width) / 2, 1.0 - height), (0.0, (1.0 - height) / 2),
            (1.0 - width, (1.0 - height) / 2),
            (x1 - width, y1), (x2, y1), (x1, y1 - height), (x1, y2)]


def audit_control_geometry(region: Sequence[float], count: int) -> dict[str, Any]:
    """Explain the deterministic control-placement result without changing it.

    This is a diagnostic view of the existing fixed corner/edge/adjacent
    protocol.  It neither selects a different control nor considers image
    content, model output, or annotations.  ``generate_control_regions`` uses
    this function so its existing placement behaviour remains the one source
    of truth.
    """
    box = validate_region(region)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("CONTROL_COUNT_MUST_BE_NONNEGATIVE_INTEGER")
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    selected: list[tuple[float, float, float, float]] = []
    candidates = []
    for placement_index, (x, y) in enumerate(_control_placements(box)):
        proposal = (x, y, x + width, y + height)
        if len(selected) >= count:
            state = "NOT_EVALUATED_AFTER_REQUEST_SATISFIED"
        elif not (0 <= x < x + width <= 1 and 0 <= y < y + height <= 1):
            state = "OUT_OF_BOUNDS"
        elif _intersection_area(box, proposal) > 0:
            state = "OVERLAPS_TARGET"
        elif any(_intersection_area(old, proposal) > 0 for old in selected):
            state = "OVERLAPS_PREVIOUS_CONTROL"
        elif proposal in selected:
            state = "DUPLICATE_PLACEMENT"
        else:
            selected.append(proposal)
            state = "SELECTED"
        candidates.append({"placement_index": placement_index, "region": list(proposal), "state": state})
    available = len(selected) == count
    return {"status": "CONTROL_AVAILABLE" if available else "CONTROL_UNAVAILABLE",
            "available": available, "regions": [list(item) for item in selected],
            "requested_count": count, "valid_count": len(selected),
            "target_region": list(box), "target_area_fraction": width * height,
            "candidate_placements": candidates,
            "geometry_rule": "fixed_corners_then_edge_centers_then_target_adjacent; target/control pairwise disjoint",
            "matching": "identical_normalized_width_and_height",
            "version": CONTROL_VERSION}


def generate_control_regions(region: Sequence[float], count: int) -> dict[str, Any]:
    """Predeclared corner/edge/adjacent placements, disjoint from target and each other.

    All controls have the target's normalized dimensions. Too few valid positions
    yields CONTROL_UNAVAILABLE; partial positions remain in the audit. Positions
    never depend on image content, model responses, or hidden annotations.
    """
    diagnostic = audit_control_geometry(region, count)
    return {"status": diagnostic["status"], "available": diagnostic["available"],
            "regions": diagnostic["regions"], "requested_count": diagnostic["requested_count"],
            "valid_count": diagnostic["valid_count"], "version": CONTROL_VERSION,
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
    inside_unchanged = inside.getbbox() is None
    outside_unchanged = outside.getbbox() is None
    return {"inside_unchanged": inside_unchanged,
            "outside_unchanged": outside_unchanged,
            "inside_changed": not inside_unchanged,
            "outside_changed": not outside_unchanged,
            "inside_max_channel_delta": inside.getextrema()[1],
            "outside_max_channel_delta": outside.getextrema()[1],
            "any_pixel_changed": changed.getbbox() is not None}


def _audit(image: Image.Image, altered: Image.Image, box, region: Sequence[float], variant: str,
           operator_spec: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    stats = _pixel_stats(image, altered, box)
    unchanged_region = "both" if variant == "ORIGINAL" else "inside" if variant == "KEEP_TARGET" else "outside"
    changed_region = None if variant == "ORIGINAL" else "outside" if variant == "KEEP_TARGET" else "inside"
    unchanged_region_pass = ((stats["inside_unchanged"] and stats["outside_unchanged"])
                             if unchanged_region == "both" else stats[f"{unchanged_region}_unchanged"])
    expected_region_changed = (False if changed_region is None else stats[f"{changed_region}_changed"])
    passed = unchanged_region_pass and (changed_region is None or expected_region_changed)
    if not unchanged_region_pass or altered.size != image.size:
        audit_status = "INTERVENTION_PIXEL_AUDIT_FAILED"
    elif changed_region is not None and not expected_region_changed:
        audit_status = "INTERVENTION_NO_EFFECT"
    else:
        audit_status = "PASS"
    return {"version": operator_spec["operator_version"], "intervention_protocol": operator_spec,
            "variant": variant, "region": list(region), "pixel_bbox": list(box),
            "pixel_bbox_convention": "half_open_xyxy", "mapping_version": COORDINATE_MAPPING_VERSION,
            "original_size": list(image.size), "output_size": list(altered.size),
            "resolution_preserved": altered.size == image.size, "unchanged_region": unchanged_region,
            "expected_changed_region": changed_region, "unchanged_region_pass": bool(unchanged_region_pass),
            "expected_region_changed": bool(expected_region_changed), "pixel_audit_pass": bool(passed),
            "audit_status": audit_status, **(extra or {}), **stats}


def apply_intervention(image: Image.Image, region: Sequence[float], variant: str,
                       blur_radius: float) -> tuple[Image.Image, dict[str, Any]]:
    """Historical Gaussian operation, preserved byte-for-byte for old callers."""
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
    operator_spec = resolve_intervention_spec(None, float(blur_radius))
    audit = _audit(image, altered, box, region, variant, operator_spec, {"blur_radius": float(blur_radius)})
    return altered, audit


def apply_spatial_intervention(image: Image.Image, region: Sequence[float], variant: str,
                               operator_spec: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Apply a registered, version-bound core spatial operator.

    Opaque gray is intentionally closed to RGB 127.  It is a VLM sensitivity
    intervention, not a medical reconstruction or causal proof.
    """
    canonical = resolve_intervention_spec(operator_spec, float(operator_spec.get("parameters", {}).get("blur_radius", 1.0)))
    if canonical["operator"] == GAUSSIAN_OPERATOR:
        return apply_intervention(image, region, variant, canonical["parameters"]["blur_radius"])
    if variant not in VARIANTS:
        raise ValueError(f"UNSUPPORTED_INTERVENTION: {variant}")
    if image.mode not in {"RGB", "RGBA", "L"}:
        raise ValueError("INTERVENTION_IMAGE_MODE_MUST_BE_RGB_RGBA_OR_L")
    source = image.convert("RGB")
    box = normalized_to_pixel_bbox(region, *source.size)
    fill = tuple(canonical["parameters"]["fill_rgb"])
    if variant == "ORIGINAL":
        altered = source.copy()
    elif variant == "KEEP_TARGET":
        altered = Image.new("RGB", source.size, fill)
        altered.paste(source.crop(box), box)
    else:
        altered = source.copy()
        altered.paste(fill, box)
    return altered, _audit(source, altered, box, region, variant, canonical, {"fill_rgb": list(fill)})


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
