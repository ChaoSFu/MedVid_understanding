#!/usr/bin/env python3
"""Create no-model visual intervention sweeps for a frozen calibration manifest.

This diagnostic is intentionally outside the core runner and fixed-ROI
calibration runner. It reads only a frozen calibration manifest and its public
frame images, then writes visual alternatives for human review. It never loads
a model, opens a cache, creates a runtime, or changes certification policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from relive.interventions import apply_intervention
from relive.spatial import normalized_to_pixel_bbox


class SweepError(RuntimeError):
    pass


DEFAULT_RADII = (4.0, 8.0, 16.0, 32.0)
OPAQUE_OPERATORS = ("opaque_mean", "opaque_gray")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SweepError(f"unreadable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SweepError("JSON object required")
    return payload


def _box(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise SweepError(f"{name} must be a normalized xyxy list")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise SweepError(f"{name} coordinates must be finite numbers")
    x1, y1, x2, y2 = (float(v) for v in value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise SweepError(f"{name} must be in bounds with positive area")
    return x1, y1, x2, y2


def _area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _controls(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("format") != "relive-calibration-manifest-frozen-v1":
        raise SweepError("requires a frozen calibration manifest")
    if manifest.get("calibration_only") is not True or manifest.get("selection_status") != "FROZEN_PRE_INFERENCE":
        raise SweepError("manifest is not frozen calibration-only input")
    supplied = manifest.get("frozen_manifest_sha256")
    bound = dict(manifest)
    bound.pop("frozen_manifest_sha256", None)
    if supplied != _canonical_hash(bound):
        raise SweepError("frozen manifest SHA-256 does not bind its content")
    selected, ancillary = manifest.get("selected_positive_control"), manifest.get("ancillary_controls")
    if not isinstance(selected, dict) or not isinstance(ancillary, list):
        raise SweepError("manifest must contain calibration controls")
    rows = [selected, *ancillary]
    if len(rows) != 3:
        raise SweepError("requires exactly three frozen controls")
    ids = set()
    for row in rows:
        identifier = row.get("candidate_id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise SweepError("unique control IDs required")
        ids.add(identifier)
        paths, frame_ids, orders = row.get("frame_paths"), row.get("frame_ids"), row.get("frame_orders")
        if not (isinstance(paths, list) and isinstance(frame_ids, list) and isinstance(orders, list)
                and len(paths) == len(frame_ids) == len(orders) == 1):
            raise SweepError("sweep requires one frozen public frame per control")
        if not all(isinstance(value, str) and value for value in (*paths, *frame_ids)):
            raise SweepError("invalid frozen frame identity")
        if type(orders[0]) is not int or orders[0] < 0 or not Path(paths[0]).is_file():
            raise SweepError("missing frozen public frame")
        expected_frame_hash = _canonical_hash([{"frame_id": frame_ids[0], "path": paths[0], "order": orders[0]}])
        if row.get("frame_sha256") != expected_frame_hash:
            raise SweepError("frozen frame hash mismatch")
        target = _box(row.get("diagnostic_only_bbox_normalized_xyxy"), "target ROI")
        matched = _box(row.get("matched_control_bbox_normalized_xyxy"), "matched control ROI")
        if not 0.05 <= _area(target) <= 0.30 or abs(_area(target) - _area(matched)) > 1e-12 or _intersection(target, matched) != 0:
            raise SweepError("invalid frozen target/control geometry")
    return rows


def _opaque(image: Image.Image, region: tuple[float, float, float, float], variant: str, operator: str) -> tuple[Image.Image, dict[str, Any]]:
    if variant not in {"KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"}:
        raise SweepError(f"unsupported opaque variant: {variant}")
    source = image.convert("RGB")
    box = normalized_to_pixel_bbox(region, *source.size)
    crop = source.crop(box)
    if operator == "opaque_mean":
        values = list(crop.getdata())
        fill = tuple(round(sum(pixel[channel] for pixel in values) / len(values)) for channel in range(3))
    elif operator == "opaque_gray":
        fill = (127, 127, 127)
    else:
        raise SweepError(f"unknown opaque operator: {operator}")
    if variant == "KEEP_TARGET":
        altered = Image.new("RGB", source.size, fill)
        altered.paste(source.crop(box), box)
    else:
        altered = source.copy()
        altered.paste(fill, box)
    return altered, {"operator": operator, "fill_rgb": list(fill), "pixel_bbox": list(box),
                     "pixel_bbox_convention": "half_open_xyxy", "region": list(region)}


def _diff_stats(original: Image.Image, altered: Image.Image, box: tuple[int, int, int, int]) -> dict[str, Any]:
    width, height = original.size
    before, after = original.convert("RGB").tobytes(), altered.convert("RGB").tobytes()
    total_pixels = width * height
    changed_total = changed_inside = changed_outside = 0
    abs_total = abs_inside = abs_outside = 0
    channel_inside = channel_outside = 0
    x1, y1, x2, y2 = box
    for pixel_index in range(total_pixels):
        base = pixel_index * 3
        delta = abs(before[base] - after[base]) + abs(before[base + 1] - after[base + 1]) + abs(before[base + 2] - after[base + 2])
        is_changed = delta != 0
        x, y = pixel_index % width, pixel_index // width
        inside = x1 <= x < x2 and y1 <= y < y2
        abs_total += delta
        if is_changed:
            changed_total += 1
        if inside:
            abs_inside += delta
            channel_inside += 3
            if is_changed:
                changed_inside += 1
        else:
            abs_outside += delta
            channel_outside += 3
            if is_changed:
                changed_outside += 1
    roi_pixels = (x2 - x1) * (y2 - y1)
    return {
        "changed_pixel_count": changed_total,
        "changed_pixel_fraction": changed_total / total_pixels,
        "mean_absolute_pixel_difference": abs_total / (total_pixels * 3),
        "roi_changed_pixel_count": changed_inside,
        "roi_changed_pixel_fraction": changed_inside / roi_pixels,
        "roi_mean_absolute_pixel_difference": abs_inside / channel_inside,
        "outside_changed_pixel_count": changed_outside,
        "outside_changed_pixel_fraction": changed_outside / (total_pixels - roi_pixels),
        "outside_mean_absolute_pixel_difference": abs_outside / channel_outside,
    }


def _effect_class(stats: dict[str, Any], box: tuple[int, int, int, int]) -> str:
    if box[0] == box[2] or box[1] == box[3]:
        return "CLIPPED_TO_ZERO_AREA"
    if stats["changed_pixel_count"] == 0:
        return "UNIFORM_REGION"
    if stats["roi_changed_pixel_count"] == 0:
        return "ARTIFACT_NOT_APPLIED"
    return "EFFECTIVE_PIXEL_CHANGE"


def _save(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return str(path)


def _overlay(image: Image.Image, box: tuple[int, int, int, int], label: str, color: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=color, width=max(2, min(image.size) // 160))
    draw.rectangle((0, 0, max(130, len(label) * 7 + 10), 18), fill="black")
    draw.text((4, 3), label, font=ImageFont.load_default(), fill=color)
    return canvas


def _sheet(images: Iterable[tuple[str, Image.Image]], path: Path) -> str:
    images = list(images)
    width, height = images[0][1].size
    label_height = 24
    canvas = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(images):
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 4, 5), label, fill="black", font=font)
    return _save(canvas, path)


def _operator_specs(radii: tuple[float, ...]) -> list[tuple[str, str, float | None]]:
    return [(f"gaussian_r{radius:g}", "gaussian", radius) for radius in radii] + [(name, "opaque", None) for name in OPAQUE_OPERATORS]


def _alter(image: Image.Image, region: tuple[float, float, float, float], variant: str,
           family: str, radius: float | None, name: str) -> tuple[Image.Image, dict[str, Any]]:
    if family == "gaussian":
        altered, audit = apply_intervention(image, region, variant, float(radius))
        return altered.convert("RGB"), {"operator": name, "family": family, **audit}
    altered, audit = _opaque(image, region, variant, name)
    return altered, {"family": family, **audit}


def main() -> int:
    parser = argparse.ArgumentParser(description="No-model intervention/operator visual sweep for frozen calibration controls")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blur-radii", default=",".join(str(int(radius)) for radius in DEFAULT_RADII), help="comma-separated positive Gaussian radii")
    args = parser.parse_args()
    try:
        radii = tuple(float(part) for part in args.blur_radii.split(",") if part.strip())
        if not radii or any(not math.isfinite(radius) or radius <= 0 for radius in radii) or len(set(radii)) != len(radii):
            raise SweepError("blur radii must be unique finite positive numbers")
        manifest_path, output = Path(args.manifest).resolve(), Path(args.output_dir).resolve()
        if output.exists():
            raise SweepError(f"refusing to overwrite output directory: {output}")
        manifest = _json(manifest_path)
        controls = _controls(manifest)
        output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest_path, output / "frozen_manifest.input.json")
        report_controls = []
        for control in controls:
            identifier = control["candidate_id"]
            source_path = Path(control["frame_paths"][0])
            with Image.open(source_path) as opened:
                original = opened.convert("RGB")
            target = _box(control["diagnostic_only_bbox_normalized_xyxy"], "target ROI")
            matched = _box(control["matched_control_bbox_normalized_xyxy"], "matched control ROI")
            target_box = normalized_to_pixel_bbox(target, *original.size)
            matched_box = normalized_to_pixel_bbox(matched, *original.size)
            original_path = _save(original, output / "images" / identifier / "original.png")
            rows = []
            for name, family, radius in _operator_specs(radii):
                variants: dict[str, Image.Image] = {"ORIGINAL": original}
                variant_records = []
                for variant, region, pixel_box in (("KEEP_TARGET", target, target_box),
                                                   ("DROP_TARGET", target, target_box),
                                                   ("DROP_MATCHED_CONTROL", matched, matched_box)):
                    altered, audit = _alter(original, region, variant, family, radius, name)
                    stats = _diff_stats(original, altered, pixel_box)
                    record = {"variant": variant, "region": list(region), "pixel_bbox": list(pixel_box),
                              "effect_class": _effect_class(stats, pixel_box), "audit": audit, "pixel_metrics": stats}
                    record["output_path"] = _save(altered, output / "images" / identifier / f"{name}__{variant.lower()}.png")
                    variants[variant] = altered
                    variant_records.append(record)
                sheet = _sheet([
                    ("ORIGINAL", original),
                    ("TARGET ROI", _overlay(original, target_box, "TARGET ROI", "red")),
                    ("KEEP TARGET", variants["KEEP_TARGET"]),
                    ("DROP TARGET", variants["DROP_TARGET"]),
                    ("DROP MATCHED CONTROL", variants["DROP_MATCHED_CONTROL"]),
                ], output / "contact_sheets" / f"{identifier}__{name}.png")
                rows.append({"operator": name, "family": family, "blur_radius": radius,
                             "contact_sheet": sheet, "variants": variant_records})
            report_controls.append({
                "candidate_id": identifier, "kind": control["kind"], "claim": control["atomic_claim_en"],
                "frame_id": control["frame_ids"][0], "frame_order": control["frame_orders"][0],
                "source_frame_path": str(source_path), "source_frame_sha256": _file_hash(source_path),
                "original_output_path": original_path, "original_size": list(original.size),
                "target_roi_normalized_xyxy": list(target), "matched_control_roi_normalized_xyxy": list(matched),
                "target_pixel_bbox": list(target_box), "matched_control_pixel_bbox": list(matched_box),
                "target_area_fraction": _area(target), "operator_sweeps": rows,
            })
        report = {
            "status": "PASS", "phase": "ReliVE calibration intervention/operator visual sweep",
            "model_calls_made": 0, "cache_mutated": False, "calibration_only": True,
            "not_benchmark_evidence": True, "formal_certificate_created": False,
            "manifest_path": str(manifest_path), "manifest_sha256": manifest["frozen_manifest_sha256"],
            "allowed_inputs": ["frozen_human_authored_public_frame_calibration_manifest", "public_frame_images"],
            "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts", "model_weights", "inference_cache"],
            "operators": [{"name": name, "family": family, "blur_radius": radius}
                          for name, family, radius in _operator_specs(radii)],
            "controls": report_controls,
            "human_review_required": "Select one operator family/setting before creating a new manifest; do not choose from model verdicts.",
        }
        (output / "intervention_sweep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("status", "model_calls_made", "cache_mutated", "manifest_sha256", "operators", "human_review_required")}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, SweepError) as exc:
        print(f"ReliVE calibration intervention sweep error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
