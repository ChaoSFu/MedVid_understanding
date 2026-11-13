from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageChops

from relive.interventions import apply_spatial_intervention
from relive.spatial import normalized_to_pixel_bbox, validate_region


def image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def _mask(size: tuple[int, int], regions: Sequence[Sequence[float]]) -> tuple[Image.Image, list[list[int]]]:
    if not 1 <= len(regions) <= 2:
        raise ValueError("EVIDENCE_SET_MUST_HAVE_ONE_OR_TWO_REGIONS")
    mask = Image.new("L", size, 0)
    boxes = []
    for region in regions:
        box = normalized_to_pixel_bbox(validate_region(region), *size)
        mask.paste(255, box)
        boxes.append(list(box))
    return mask, boxes


def apply_union_intervention(image: Image.Image, regions: Sequence[Sequence[float]], variant: str,
                             operator: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    if variant not in {"KEEP_UNION", "DROP_UNION"}:
        raise ValueError("UNSUPPORTED_UNION_VARIANT")
    source = image.convert("RGB")
    mask, boxes = _mask(source.size, regions)
    fill_rgb = operator.get("parameters", {}).get("fill_rgb")
    if operator.get("operator") != "opaque_gray" or fill_rgb != [127, 127, 127]:
        raise ValueError("UNION_REQUIRES_FROZEN_OPAQUE_GRAY_OPERATOR")
    gray = Image.new("RGB", source.size, tuple(fill_rgb))
    altered = Image.composite(source, gray, mask) if variant == "KEEP_UNION" else Image.composite(gray, source, mask)
    changed = ImageChops.difference(source, altered)
    changed_mask = Image.new("L", source.size, 0)
    for band in changed.split():
        changed_mask = ImageChops.lighter(changed_mask, band)
    expected_unchanged_mask = mask if variant == "KEEP_UNION" else ImageChops.invert(mask)
    expected_changed_mask = ImageChops.invert(mask) if variant == "KEEP_UNION" else mask
    unchanged_diff = ImageChops.multiply(changed_mask, expected_unchanged_mask)
    changed_diff = ImageChops.multiply(changed_mask, expected_changed_mask)
    passed = altered.size == source.size and unchanged_diff.getbbox() is None and changed_diff.getbbox() is not None
    return altered, {
        "variant": variant, "regions": [list(validate_region(region)) for region in regions],
        "pixel_boxes": boxes, "region_count": len(regions), "operator": operator,
        "resolution_preserved": altered.size == source.size, "unchanged_region_pass": unchanged_diff.getbbox() is None,
        "expected_region_changed": changed_diff.getbbox() is not None, "pixel_audit_pass": passed,
        "audit_status": "PASS" if passed else "INTERVENTION_PIXEL_AUDIT_FAILED",
        "input_pixel_sha256": image_sha256(source), "output_pixel_sha256": image_sha256(altered),
    }


def keep_residual(drop_primary: Image.Image, residual_region: Sequence[float],
                  operator: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    altered, audit = apply_spatial_intervention(drop_primary.convert("RGB"), residual_region, "KEEP_TARGET", operator)
    return altered, {**audit, "variant": "KEEP_RESIDUAL", "primary_region_remains_masked": True,
                     "input_pixel_sha256": image_sha256(drop_primary), "output_pixel_sha256": image_sha256(altered)}


def regions_equal(a: Sequence[float], b: Sequence[float]) -> bool:
    return tuple(validate_region(a)) == tuple(validate_region(b))


def save_intervention(path: str | Path, image: Image.Image) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with Image.open(target) as prior:
            same = prior.convert("RGB").size == image.convert("RGB").size and ImageChops.difference(prior.convert("RGB"), image.convert("RGB")).getbbox() is None
        if not same:
            raise ValueError("IMMUTABLE_INTERVENTION_OUTPUT_EXISTS")
    else:
        image.save(target, format="PNG")
    return {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "size": list(image.size)}
