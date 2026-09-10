"""Frozen, calibration-only image interventions.

These operators are not part of the ReliVE core verification protocol. They
exist solely so a human-reviewed calibration manifest can bind its image
operator before a calibration-only model run.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from PIL import Image, ImageChops

from .interventions import INTERVENTION_VERSION, apply_intervention
from .spatial import COORDINATE_MAPPING_VERSION, normalized_to_pixel_bbox


OPAQUE_GRAY_VERSION = "relive-calibration-opaque-gray-v1"


def calibration_intervention_spec(manifest: dict[str, Any], blur_radius: float) -> dict[str, Any]:
    """Return a closed, cache-bindable intervention specification.

    Old frozen calibration manifests retain their reviewed Gaussian behavior.
    Opaque gray is accepted only as an explicit, pre-inference calibration
    choice; arbitrary fills and operators are intentionally rejected.
    """
    declared = manifest.get("calibration_intervention")
    if declared is None:
        if isinstance(blur_radius, bool) or not isinstance(blur_radius, (int, float)) or not math.isfinite(blur_radius) or blur_radius <= 0:
            raise ValueError("CALIBRATION_BLUR_RADIUS_MUST_BE_FINITE_POSITIVE")
        return {"operator": "gaussian_blur", "operator_version": INTERVENTION_VERSION,
                "blur_radius": float(blur_radius)}
    if not isinstance(declared, dict):
        raise ValueError("CALIBRATION_INTERVENTION_MUST_BE_OBJECT")
    if set(declared) != {"operator", "operator_version", "fill_rgb", "selection_basis"}:
        raise ValueError("CALIBRATION_INTERVENTION_FIELDS_INVALID")
    if (declared.get("operator") != "opaque_gray" or declared.get("operator_version") != OPAQUE_GRAY_VERSION
            or declared.get("fill_rgb") != [127, 127, 127]
            or declared.get("selection_basis") != "human_visual_review_pre_inference"):
        raise ValueError("CALIBRATION_INTERVENTION_NOT_FROZEN_OPAQUE_GRAY")
    return {"operator": "opaque_gray", "operator_version": OPAQUE_GRAY_VERSION,
            "fill_rgb": [127, 127, 127]}


def _pixel_stats(original: Image.Image, altered: Image.Image, box: tuple[int, int, int, int]) -> dict[str, Any]:
    mask = Image.new("L", original.size, 0)
    mask.paste(255, box=box)
    changed_channels = ImageChops.difference(original, altered)
    changed = Image.new("L", original.size, 0)
    for band in changed_channels.split():
        changed = ImageChops.lighter(changed, band)
    inside = ImageChops.multiply(changed, mask)
    outside = ImageChops.multiply(changed, ImageChops.invert(mask))
    inside_unchanged = inside.getbbox() is None
    outside_unchanged = outside.getbbox() is None
    return {"inside_unchanged": inside_unchanged, "outside_unchanged": outside_unchanged,
            "inside_changed": not inside_unchanged, "outside_changed": not outside_unchanged,
            "inside_max_channel_delta": inside.getextrema()[1], "outside_max_channel_delta": outside.getextrema()[1],
            "any_pixel_changed": changed.getbbox() is not None}


def _opaque_gray(image: Image.Image, region: Sequence[float], variant: str,
                 spec: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    if variant not in {"ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"}:
        raise ValueError("CALIBRATION_INTERVENTION_VARIANT_INVALID")
    if image.mode not in {"RGB", "RGBA", "L"}:
        raise ValueError("CALIBRATION_INTERVENTION_IMAGE_MODE_INVALID")
    source = image.convert("RGB")
    box = normalized_to_pixel_bbox(region, *source.size)
    if variant == "ORIGINAL":
        altered = source.copy()
    elif variant == "KEEP_TARGET":
        altered = Image.new("RGB", source.size, tuple(spec["fill_rgb"]))
        altered.paste(source.crop(box), box)
    else:
        altered = source.copy()
        altered.paste(tuple(spec["fill_rgb"]), box)
    stats = _pixel_stats(source, altered, box)
    unchanged_region = "both" if variant == "ORIGINAL" else "inside" if variant == "KEEP_TARGET" else "outside"
    changed_region = None if variant == "ORIGINAL" else "outside" if variant == "KEEP_TARGET" else "inside"
    unchanged_region_pass = ((stats["inside_unchanged"] and stats["outside_unchanged"])
                             if unchanged_region == "both" else stats[f"{unchanged_region}_unchanged"])
    expected_region_changed = False if changed_region is None else stats[f"{changed_region}_changed"]
    passed = unchanged_region_pass and (changed_region is None or expected_region_changed)
    audit_status = ("INTERVENTION_PIXEL_AUDIT_FAILED" if not unchanged_region_pass or altered.size != source.size
                    else "INTERVENTION_NO_EFFECT" if changed_region is not None and not expected_region_changed else "PASS")
    return altered, {"version": spec["operator_version"], "variant": variant, "region": list(region),
                     "pixel_bbox": list(box), "pixel_bbox_convention": "half_open_xyxy",
                     "mapping_version": COORDINATE_MAPPING_VERSION, "fill_rgb": spec["fill_rgb"],
                     "original_size": list(source.size), "output_size": list(altered.size),
                     "resolution_preserved": altered.size == source.size, "unchanged_region": unchanged_region,
                     "expected_changed_region": changed_region, "unchanged_region_pass": bool(unchanged_region_pass),
                     "expected_region_changed": bool(expected_region_changed), "pixel_audit_pass": bool(passed),
                     "audit_status": audit_status, **stats}


def apply_calibration_intervention(image: Image.Image, region: Sequence[float], variant: str,
                                   spec: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Apply a frozen calibration operator without changing core intervention behavior."""
    if spec["operator"] == "gaussian_blur":
        return apply_intervention(image, region, variant, float(spec["blur_radius"]))
    if spec["operator"] == "opaque_gray":
        return _opaque_gray(image, region, variant, spec)
    raise ValueError("CALIBRATION_INTERVENTION_OPERATOR_INVALID")
