"""Frozen calibration declarations mapped onto the core intervention registry.

This module retains the old calibration-manifest spelling for replaying frozen
inputs. It owns no image transform: calibration and the formal runner call the
same implementation in :mod:`relive.interventions`.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from PIL import Image

from .interventions import (
    INTERVENTION_VERSION,
    OPAQUE_GRAY_OPERATOR,
    OPAQUE_GRAY_VERSION as CORE_OPAQUE_GRAY_VERSION,
    apply_spatial_intervention,
    resolve_intervention_spec,
)

# Historical frozen-manifest identifier. It is accepted only by this adapter;
# formal core configuration uses CORE_OPAQUE_GRAY_VERSION.
OPAQUE_GRAY_VERSION = "relive-calibration-opaque-gray-v1"


def calibration_intervention_spec(manifest: dict[str, Any], blur_radius: float) -> dict[str, Any]:
    """Return the frozen calibration declaration used in request provenance."""
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


def _core_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("operator") == "gaussian_blur":
        radius = spec.get("blur_radius")
        return resolve_intervention_spec({"operator": "gaussian_blur",
                                          "operator_version": INTERVENTION_VERSION,
                                          "parameters": {"blur_radius": radius}}, float(radius))
    if (spec.get("operator") == "opaque_gray" and spec.get("operator_version") == OPAQUE_GRAY_VERSION
            and spec.get("fill_rgb") == [127, 127, 127]):
        return resolve_intervention_spec({"operator": OPAQUE_GRAY_OPERATOR,
                                          "operator_version": CORE_OPAQUE_GRAY_VERSION,
                                          "parameters": {"fill_rgb": [127, 127, 127]}}, 1.0)
    raise ValueError("CALIBRATION_INTERVENTION_OPERATOR_INVALID")


def apply_calibration_intervention(image: Image.Image, region: Sequence[float], variant: str,
                                   spec: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Apply the registered core operator while retaining calibration provenance."""
    altered, audit = apply_spatial_intervention(image, region, variant, _core_spec(spec))
    audit["calibration_intervention"] = dict(spec)
    return altered, audit
