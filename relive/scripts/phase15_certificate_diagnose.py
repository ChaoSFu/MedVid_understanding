#!/usr/bin/env python3
"""Read-only Phase 1.5 diagnosis for one completed ReliVE run.

This utility deliberately does not import a model backend or the evaluation
package.  It reads only a public runtime JSONL, completed run artifacts, raw
inference records already present in a caller-supplied cache, and existing
intervention images.  It never invokes inference and writes only to a new
diagnostic directory outside the run directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


FAILURE_CATEGORIES = (
    "RETRIEVAL_MISS", "ORIGINAL_MODEL_INSUFFICIENT", "CLAIM_TOO_BROAD",
    "MULTIPLE_INDEPENDENT_SUPPORTS", "CLAIM_NOT_SPATIALLY_LOCALIZABLE",
    "INVALID_OR_OVERSIZED_PROPOSAL", "INEFFECTIVE_INTERVENTION",
    "CONTROL_CONSTRUCTION_FAILURE", "MODEL_INTERVENTION_INSENSITIVITY",
    "ADAPTATION_MISMATCH", "TECHNICAL_ERROR",
)


class DiagnosticError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON artifact must be an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DiagnosticError(f"unreadable runtime: {path}") from exc
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiagnosticError(f"invalid runtime JSONL line {number}") from exc
        if not isinstance(value, dict):
            raise DiagnosticError(f"runtime line {number} must be an object")
        records.append(value)
    if not records:
        raise DiagnosticError("runtime has no records")
    return records


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _raw_text(reference: str | None, cache_root: Path) -> dict[str, Any] | None:
    if not isinstance(reference, str) or not reference:
        return None
    path = Path(reference).expanduser()
    if not _inside(path, cache_root):
        return {"status": "REF_OUTSIDE_CACHE", "raw_response_ref": reference}
    row = _json(path)
    return {
        "status": "OK",
        "raw_response_ref": str(path.resolve()),
        "raw_text": row.get("raw_text"),
        "execution_status": row.get("execution_status"),
        "failure_reason": row.get("failure_reason"),
    }


def _status(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict) or result.get("execution_status") != "OK":
        return None
    value = result.get("semantic_status")
    return value if value in {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"} else None


def _result_summary(result: dict[str, Any] | None, cache_root: Path,
                    candidate_frame_ids: list[str]) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    references = result.get("frame_references") if isinstance(result.get("frame_references"), list) else []
    indexes = [candidate_frame_ids.index(frame_id) for frame_id in references if frame_id in candidate_frame_ids]
    return {
        "execution_status": result.get("execution_status"),
        "semantic_status": _status(result),
        "failure_reason": result.get("failure_reason"),
        "frame_references": references,
        "candidate_frame_indexes": indexes,
        "raw": _raw_text(result.get("raw_response_ref"), cache_root),
        "input_references": result.get("input_references", {}),
    }


def _pixel_bbox(region: list[float] | tuple[float, ...] | None, size: tuple[int, int]) -> list[int] | None:
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in region):
        return None
    x1, y1, x2, y2 = (float(value) for value in region)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    width, height = size
    return [math.floor(x1 * width), math.floor(y1 * height), math.ceil(x2 * width), math.ceil(y2 * height)]


def _image_array(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.int16), rgb.size


def _metrics(original_path: str, altered_path: str, pixel_bbox: list[int] | None,
             variant: str) -> dict[str, Any]:
    try:
        original, size = _image_array(original_path)
        altered, altered_size = _image_array(altered_path)
    except (OSError, ValueError) as exc:
        return {"status": "ARTIFACT_NOT_APPLIED", "reason": type(exc).__name__, "variant": variant}
    if size != altered_size or original.shape != altered.shape:
        return {"status": "ARTIFACT_NOT_APPLIED", "reason": "IMAGE_SIZE_MISMATCH", "variant": variant}
    if pixel_bbox is None:
        return {"status": "EMPTY_OR_INVALID_ROI", "variant": variant, "image_size": list(size)}
    x1, y1, x2, y2 = pixel_bbox
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return {"status": "CLIPPED_TO_ZERO_AREA", "variant": variant, "image_size": list(size),
                "pixel_bbox": pixel_bbox}
    delta = np.abs(original - altered)
    changed = np.any(delta != 0, axis=2)
    mask = np.zeros(changed.shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    inside, outside = delta[mask], delta[~mask]
    changed_inside, changed_outside = changed[mask], changed[~mask]
    roi_original = original[y1:y2, x1:x2]
    return {
        "status": "OK", "variant": variant, "image_size": list(size), "pixel_bbox": pixel_bbox,
        "changed_pixel_count": int(changed.sum()),
        "changed_pixel_fraction": float(changed.mean()),
        "mean_absolute_pixel_difference": float(delta.mean()),
        "roi_mean_absolute_pixel_difference": float(inside.mean()) if inside.size else None,
        "outside_roi_mean_absolute_pixel_difference": float(outside.mean()) if outside.size else None,
        "roi_changed_pixel_count": int(changed_inside.sum()),
        "outside_roi_changed_pixel_count": int(changed_outside.sum()),
        "roi_changed_pixel_fraction": float(changed_inside.mean()) if changed_inside.size else None,
        "outside_roi_changed_pixel_fraction": float(changed_outside.mean()) if changed_outside.size else None,
        "roi_original_channel_variance": float(roi_original.var()) if roi_original.size else None,
        "drop_only_expected_region_changed": bool(changed_outside.sum() == 0) if variant == "DROP_TARGET" else None,
        "keep_roi_preserved": bool(changed_inside.sum() == 0) if variant == "KEEP_TARGET" else None,
        "keep_outside_changed": bool(changed_outside.sum() > 0) if variant == "KEEP_TARGET" else None,
    }


def _no_effect_subtype(metric: dict[str, Any], audit: dict[str, Any] | None,
                       expected_bbox: list[int] | None) -> str | None:
    if not isinstance(metric, dict):
        return None
    if metric.get("status") == "ARTIFACT_NOT_APPLIED":
        return "ARTIFACT_NOT_APPLIED"
    if metric.get("status") == "EMPTY_OR_INVALID_ROI":
        return "EMPTY_OR_INVALID_ROI"
    if metric.get("status") == "CLIPPED_TO_ZERO_AREA":
        return "CLIPPED_TO_ZERO_AREA"
    if not isinstance(audit, dict) or audit.get("audit_status") != "INTERVENTION_NO_EFFECT":
        return None
    observed = audit.get("pixel_bbox")
    if expected_bbox is not None and observed != expected_bbox:
        return "WRONG_COORDINATE_MAPPING"
    if metric.get("roi_original_channel_variance") == 0:
        return "UNIFORM_REGION"
    if metric.get("changed_pixel_count") == 0:
        return "BLUR_TOO_WEAK"
    return "OTHER"


def _aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in metrics if row.get("status") == "OK"]
    if not valid:
        return {"frame_count": len(metrics), "valid_frame_count": 0}
    keys = ("changed_pixel_count", "changed_pixel_fraction", "mean_absolute_pixel_difference",
            "roi_mean_absolute_pixel_difference", "outside_roi_mean_absolute_pixel_difference",
            "roi_changed_pixel_count", "outside_roi_changed_pixel_count")
    output = {"frame_count": len(metrics), "valid_frame_count": len(valid)}
    for key in keys:
        values = [row[key] for row in valid if isinstance(row.get(key), (int, float))]
        if values:
            output[f"mean_{key}"] = float(sum(values) / len(values))
            output[f"sum_{key}"] = float(sum(values))
    return output


def _proposal_summary(proposal: dict[str, Any] | None, cache_root: Path,
                      image_size: tuple[int, int] | None) -> dict[str, Any] | None:
    if not isinstance(proposal, dict):
        return None
    region = proposal.get("support_region")
    valid = isinstance(region, list) and len(region) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in region)
    area = None
    out_of_bounds = True
    empty = True
    if valid:
        x1, y1, x2, y2 = (float(value) for value in region)
        out_of_bounds = not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)
        empty = x2 <= x1 or y2 <= y1
        if not out_of_bounds and not empty:
            area = (x2 - x1) * (y2 - y1)
    provenance = proposal.get("provenance") if isinstance(proposal.get("provenance"), dict) else {}
    return {
        "parser_status": proposal.get("parser_status"),
        "raw_output": _raw_text(provenance.get("raw_response_ref"), cache_root),
        "support_bbox_normalized": region,
        "target_bbox_normalized": proposal.get("target_bbox"),
        "support_bbox_pixels": _pixel_bbox(region, image_size) if image_size else None,
        "roi_area_fraction": area,
        "out_of_bounds": out_of_bounds,
        "empty": empty,
        "large": bool(area is not None and area >= 0.8),
        "near_full_frame": bool(area is not None and area >= 0.95),
        "provenance": provenance,
    }


def _pick_frame_indexes(count: int) -> list[int]:
    if count <= 0:
        return []
    values = [0, count // 2, count - 1]
    return list(dict.fromkeys(values))


def _panel(canvas: Image.Image, draw: ImageDraw.ImageDraw, image_path: str | None,
           box: list[int] | None, x: int, y: int, width: int, height: int, caption: str) -> None:
    draw.rectangle((x, y, x + width, y + height), outline="black", width=1)
    if not image_path:
        draw.text((x + 5, y + 5), caption + "\nnot available", fill="black")
        return
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if box is not None:
                overlay = ImageDraw.Draw(image)
                overlay.rectangle(tuple(box), outline="red", width=max(2, image.width // 300))
            image.thumbnail((width - 4, height - 28))
            canvas.paste(image, (x + (width - image.width) // 2, y + 2))
    except (OSError, ValueError):
        draw.text((x + 5, y + 5), caption + "\nunreadable artifact", fill="red")
        return
    draw.text((x + 3, y + height - 24), caption[:95], fill="black")


def _contact_sheet(destination: Path, candidate: dict[str, Any], source_paths: list[str],
                   keep_paths: list[str], drop_paths: list[str], control_paths: list[str],
                   pixel_box: list[int] | None, area: float | None, verdicts: dict[str, Any]) -> None:
    indexes = _pick_frame_indexes(len(source_paths))
    cols, panel_w, panel_h, header_h = max(1, len(indexes)), 320, 250, 70
    rows = [("ORIGINAL", source_paths), ("ROI_OVERLAY", source_paths), ("KEEP_TARGET", keep_paths),
            ("DROP_TARGET", drop_paths), ("DROP_MATCHED_CONTROL", control_paths)]
    canvas = Image.new("RGB", (cols * panel_w, header_h + len(rows) * panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    title = (f"{candidate['sample_id']} | {candidate['candidate_id']} | rank={candidate['candidate_rank']} | "
             f"ROI={pixel_box} area={area} | original={verdicts.get('original')} "
             f"keep={verdicts.get('keep')} drop={verdicts.get('drop')}")
    draw.text((5, 5), title[:260], fill="black")
    for col, index in enumerate(indexes):
        frame_id = candidate["candidate_frame_ids"][index] if index < len(candidate["candidate_frame_ids"]) else str(index)
        for row, (label, paths) in enumerate(rows):
            path = paths[index] if index < len(paths) else None
            overlay = pixel_box if label == "ROI_OVERLAY" else None
            caption = f"{label} | {frame_id}"
            _panel(canvas, draw, path, overlay, col * panel_w, header_h + row * panel_h,
                   panel_w, panel_h, caption)
    canvas.save(destination, format="PNG")


def _classification(claim: str, certificate: dict[str, Any], spatial: dict[str, Any] | None,
                    intervention_subtypes: list[str], sample_coverage: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    reasons = certificate.get("failure_reasons", []) if isinstance(certificate.get("failure_reasons"), list) else []
    lowered = claim.lower()
    if "ORIGINAL_INSUFFICIENT" in reasons:
        categories.append("ORIGINAL_MODEL_INSUFFICIENT")
        if sample_coverage.get("coverage_fraction", 1.0) < 1.0:
            categories.append("RETRIEVAL_MISS")
    if "DEPENDENCE_UNRESOLVED" in reasons:
        categories.append("MODEL_INTERVENTION_INSENSITIVITY")
        if "at least one" in lowered or "at least" in lowered:
            categories.extend(["CLAIM_TOO_BROAD", "MULTIPLE_INDEPENDENT_SUPPORTS"])
    if "CONTROL_UNAVAILABLE" in reasons or "CONTROL_RESULT_COUNT_MISMATCH" in reasons:
        categories.append("CONTROL_CONSTRUCTION_FAILURE")
        if spatial and spatial.get("roi_area_fraction") is not None and spatial["roi_area_fraction"] >= 0.25:
            categories.append("INVALID_OR_OVERSIZED_PROPOSAL")
    if "INTERVENTION_NO_EFFECT" in reasons or intervention_subtypes:
        categories.append("INEFFECTIVE_INTERVENTION")
    if "MAX_SPATIAL_PROPOSALS" in reasons:
        categories.append("ADAPTATION_MISMATCH")
    if any(item in {"ARTIFACT_NOT_APPLIED", "WRONG_COORDINATE_MAPPING", "EMPTY_OR_INVALID_ROI", "CLIPPED_TO_ZERO_AREA"}
           for item in intervention_subtypes):
        categories.append("TECHNICAL_ERROR")
    if ("open surgical procedure" in lowered or "egocentric viewpoint" in lowered):
        categories.append("CLAIM_NOT_SPATIALLY_LOCALIZABLE")
    return list(dict.fromkeys(categories))


def _policy_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies: dict[str, list[str]] = {"semantic_only": [], "semantic_keep_drop": [], "semantic_spatial": []}
    original_supported = keep_pass = drop_pass = control_pass = 0
    for row in rows:
        original, keep, drop = row["verdicts"].get("original"), row["verdicts"].get("keep"), row["verdicts"].get("drop")
        original_supported += original == "SUPPORTED"
        keep_pass += keep == "SUPPORTED"
        drop_pass += drop == "INSUFFICIENT"
        controls = row["controls"]
        control_ok = bool(controls.get("available")) and controls.get("expected_count") == controls.get("observed_count") and all(
            status == "SUPPORTED" for status in controls.get("verdicts", []))
        control_pass += control_ok
        policies["semantic_only"].append("VERIFIED" if original == "SUPPORTED" else "REJECTED" if original == "CONTRADICTED" else "UNCERTAIN")
        policies["semantic_keep_drop"].append(
            "REJECTED" if original == "CONTRADICTED" else "VERIFIED" if original == "SUPPORTED" and keep == "SUPPORTED" and drop == "INSUFFICIENT" else "UNCERTAIN")
        policies["semantic_spatial"].append(row["certificate_status"])
    output = {}
    for name, statuses in policies.items():
        output[name] = {
            "interpretation": "diagnostic upper bound only; not final reliable evidence" if name == "semantic_only" else "offline diagnostic recomputation from existing artifacts only",
            "original_supported_count": original_supported,
            "keep_pass_count": keep_pass,
            "drop_necessity_pass_count": drop_pass,
            "control_discriminability_pass_count": control_pass,
            "certificate_distribution": dict(Counter(statuses)),
            "candidate_count": len(statuses),
        }
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# ReliVE Phase 1.5 certificate failure diagnosis", "", "## Scope", "",
             "Read-only diagnostic. No model invocation, GT import, cache mutation, or run-artifact mutation occurred.", "",
             "## Summary", "",
             f"- Run: `{report['run_directory']}`",
             f"- Runtime: `{report['runtime_path']}`",
             f"- Candidates diagnosed: {report['summary']['candidate_count']}",
             f"- Existing raw cache records read: {report['summary']['raw_cache_records_read']}",
             "", "## Policy decomposition", "",
             "| Policy | Original supported | KEEP pass | DROP necessity pass | Control discriminability | VERIFIED | REJECTED | UNCERTAIN |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, row in report["policy_decomposition"].items():
        dist = row["certificate_distribution"]
        lines.append(f"| {policy} | {row['original_supported_count']} | {row['keep_pass_count']} | {row['drop_necessity_pass_count']} | {row['control_discriminability_pass_count']} | {dist.get('VERIFIED', 0)} | {dist.get('REJECTED', 0)} | {dist.get('UNCERTAIN', 0)} |")
    lines.extend(["", "`semantic_only` is a diagnostic upper bound only and is not a reliability result.", "",
                  "## Candidate summary", "", "| Sample | Candidate rank | Original | Certificate | Failure categories |", "|---|---:|---|---|---|"])
    for row in report["candidates"]:
        lines.append(f"| {row['sample_id']} | {row['candidate_rank']} | {row['verdicts'].get('original')} | {row['certificate_status']} | {', '.join(row['failure_categories'])} |")
    lines.extend(["", "## Retrieval coverage", ""])
    for row in report["retrieval_coverage"]:
        lines.append(f"- `{row['sample_id']}`: accessed {row['accessed_frame_count']}/{row['total_runtime_frames']} public frames ({row['coverage_percent']:.1f}%), orders {row['earliest_accessed_order']}–{row['latest_accessed_order']}; {row['event_access_assessment']}")
    lines.extend(["", "## Next-stage recommendation", "", *[f"- {item}" for item in report["next_stage_recommendations"]], ""])
    return "\n".join(lines)


def diagnose(run_dir: Path, runtime_path: Path, cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir, runtime_path, cache_dir, output_dir = (path.expanduser().resolve() for path in (run_dir, runtime_path, cache_dir, output_dir))
    if not run_dir.is_dir() or not runtime_path.is_file() or not cache_dir.is_dir():
        raise DiagnosticError("run directory, runtime path, and cache directory must exist")
    if _inside(output_dir, run_dir):
        raise DiagnosticError("diagnostic output must be outside the original run directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DiagnosticError("diagnostic output directory already exists and is nonempty")
    manifest = _json(run_dir / "manifest" / "run.json")
    runtime = {row.get("sample_id"): row for row in _jsonl(runtime_path)}
    samples = [_json(path) for path in sorted((run_dir / "samples").glob("*.json"))]
    if not samples:
        raise DiagnosticError("run has no completed sample results")
    completed_ids = {row.get("sample_id") for row in samples}
    if not completed_ids.issubset(runtime):
        raise DiagnosticError("completed samples are not a subset of the supplied runtime")
    pixel_events = {}
    for path in (run_dir / "events" / "pixel_audits").glob("*.json"):
        event = _json(path)
        key = (event.get("candidate_id"), event.get("variant"), event.get("control_index"), event.get("frame_id"))
        pixel_events[key] = event
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=False)
    candidate_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    raw_read = 0
    for sample in samples:
        public = runtime[sample["sample_id"]]
        public_frames = public.get("frames", [])
        frame_lookup = {frame["frame_id"]: frame for frame in public_frames}
        all_orders = {frame["frame_id"]: frame["order"] for frame in public_frames}
        candidates = {candidate["candidate_id"]: candidate for candidate in sample.get("candidates", [])}
        accessed = sorted({all_orders[frame_id] for candidate in candidates.values() for frame_id in candidate.get("frame_ids", []) if frame_id in all_orders})
        total = len(public_frames)
        coverage = {
            "sample_id": sample["sample_id"], "total_runtime_frames": total,
            "accessed_frame_count": len(accessed), "accessed_frame_orders": accessed,
            "coverage_fraction": len(accessed) / total if total else 0.0,
            "coverage_percent": 100 * len(accessed) / total if total else 0.0,
            "earliest_accessed_order": min(accessed) if accessed else None,
            "latest_accessed_order": max(accessed) if accessed else None,
            "only_early_windows": bool(accessed and max(accessed) < total - 1),
            "event_access_assessment": "POSSIBLE_UNACCESSED_PUBLIC_FRAMES" if accessed and max(accessed) < total - 1 else "NO_UNACCESSED_RUNTIME_POSITION",
        }
        coverage_rows.append(coverage)
        claim = public.get("target_claim", {})
        claim_text = claim.get("text") if isinstance(claim, dict) else None
        proposals = {(proposal.get("candidate_id"), proposal.get("claim_id")): proposal for proposal in sample.get("spatial_proposals", [])}
        for certificate in sample.get("certificates", []):
            candidate = candidates.get(certificate.get("candidate_id"))
            if not candidate:
                raise DiagnosticError("certificate references candidate absent from completed sample")
            candidate_ids = list(candidate.get("frame_ids", []))
            spatial_check = certificate.get("checks", {}).get("spatial", {})
            references = spatial_check.get("references", {}) if isinstance(spatial_check, dict) else {}
            original = references.get("original") if isinstance(references, dict) else None
            keep = references.get("keep") if isinstance(references, dict) else None
            drop = references.get("drop") if isinstance(references, dict) else None
            controls = references.get("controls", []) if isinstance(references, dict) and isinstance(references.get("controls"), list) else []
            original = original or certificate.get("checks", {}).get("semantic", {}).get("references")
            source_paths = original.get("input_references", {}).get("image_paths", []) if isinstance(original, dict) else []
            image_size = None
            if source_paths:
                try:
                    with Image.open(source_paths[0]) as image:
                        image_size = image.size
                except (OSError, ValueError):
                    pass
            proposal = spatial_check.get("proposal") if isinstance(spatial_check, dict) else None
            proposal = proposal if isinstance(proposal, dict) else proposals.get((candidate["candidate_id"], certificate.get("claim_id")))
            proposal_info = _proposal_summary(proposal, cache_dir, image_size)
            if proposal_info and proposal_info.get("raw_output", {}).get("status") == "OK":
                raw_read += 1
            pixel_box = proposal_info.get("support_bbox_pixels") if proposal_info else None
            verdict_rows = {
                "original": _result_summary(original, cache_dir, candidate_ids),
                "keep": _result_summary(keep, cache_dir, candidate_ids),
                "drop": _result_summary(drop, cache_dir, candidate_ids),
                "controls": [_result_summary(control, cache_dir, candidate_ids) for control in controls],
            }
            raw_read += sum(1 for result in [verdict_rows["original"], verdict_rows["keep"], verdict_rows["drop"], *verdict_rows["controls"]]
                            if result and result.get("raw", {}).get("status") == "OK")
            variant_paths = {
                "KEEP_TARGET": keep.get("input_references", {}).get("image_paths", []) if isinstance(keep, dict) else [],
                "DROP_TARGET": drop.get("input_references", {}).get("image_paths", []) if isinstance(drop, dict) else [],
                "DROP_MATCHED_CONTROL": controls[0].get("input_references", {}).get("image_paths", []) if controls else [],
            }
            interventions = {}
            subtypes = []
            for variant, paths in variant_paths.items():
                details = []
                for frame_id, original_path, altered_path in zip(candidate_ids, source_paths, paths):
                    event = pixel_events.get((candidate["candidate_id"], variant,
                                              0 if variant == "DROP_MATCHED_CONTROL" else None, frame_id))
                    metric = _metrics(original_path, altered_path, pixel_box, variant)
                    subtype = _no_effect_subtype(metric, event, pixel_box)
                    if subtype:
                        subtypes.append(subtype)
                    details.append({"frame_id": frame_id, "source_path": original_path,
                                    "artifact_path": altered_path, "existing_pixel_audit": event,
                                    "metrics": metric, "no_effect_subtype": subtype})
                interventions[variant] = {"frames": details, "aggregate": _aggregate([item["metrics"] for item in details])}
            control_geo = spatial_check.get("controls", {}) if isinstance(spatial_check, dict) else {}
            control_verdicts = [item.get("semantic_status") for item in verdict_rows["controls"] if item]
            controls_summary = {
                "available": control_geo.get("available"), "expected_count": spatial_check.get("expected_control_count"),
                "observed_count": spatial_check.get("observed_control_count"), "regions": control_geo.get("regions", []),
                "verdicts": control_verdicts, "equal_size_nonoverlap_in_bounds": bool(control_geo.get("available")),
            }
            row = {
                "sample_id": sample["sample_id"], "dataset_name": public.get("metadata", {}).get("dataset_name"),
                "source_qa_type": public.get("metadata", {}).get("source_qa_type"), "claim": claim_text,
                "claim_id": certificate.get("claim_id"), "candidate_id": candidate["candidate_id"],
                "candidate_rank": candidate.get("acquisition_rank"), "candidate_rank_source": candidate.get("rank_source"),
                "candidate_parent_id": candidate.get("parent_id"), "candidate_round_index": candidate.get("round_index"),
                "candidate_frame_ids": candidate_ids,
                "candidate_frame_orders": [all_orders.get(frame_id) for frame_id in candidate_ids],
                "candidate_relative_positions": [all_orders[frame_id] / max(1, total - 1) for frame_id in candidate_ids if frame_id in all_orders],
                "temporal_window": {"frame_order_start": min((all_orders.get(fid) for fid in candidate_ids), default=None),
                                    "frame_order_end": max((all_orders.get(fid) for fid in candidate_ids), default=None),
                                    "seconds_available": False, "timestamps": []},
                "original_semantic": verdict_rows["original"], "verdict_details": verdict_rows,
                "verdicts": {"original": _status(original), "keep": _status(keep), "drop": _status(drop),
                             "controls": control_verdicts},
                "spatial_proposal": proposal_info, "support_bbox_tube": {"static_across_candidate": True,
                    "frame_ids": candidate_ids, "frame_mapping": proposal.get("frame_mapping") if isinstance(proposal, dict) else None},
                "interventions": interventions, "controls": controls_summary,
                "certificate_status": certificate.get("final_status"), "failure_reasons": certificate.get("failure_reasons", []),
                "spatial_status": spatial_check.get("status") if isinstance(spatial_check, dict) else None,
                "adaptation_actions": [], "next_candidate_change": [],
                "sample_retrieval_coverage": coverage,
            }
            for event in sample.get("adaptation_events", []):
                if event.get("previous_candidate_id") == candidate["candidate_id"]:
                    action = {"action": event.get("action"), "reason": event.get("termination_reason") or event.get("parameters", {}).get("decision_reason"),
                              "next_candidate_id": event.get("next_candidate_id"), "parameters": event.get("parameters", {})}
                    row["adaptation_actions"].append(action)
                    next_candidate = candidates.get(event.get("next_candidate_id"))
                    if next_candidate:
                        row["next_candidate_change"].append({"next_candidate_id": next_candidate["candidate_id"],
                            "frame_orders": [all_orders.get(fid) for fid in next_candidate.get("frame_ids", [])],
                            "frame_ids": next_candidate.get("frame_ids", [])})
            row["failure_categories"] = _classification(claim_text or "", certificate, proposal_info, subtypes, coverage)
            sheet = contact_dir / f"{sample['sample_id'].replace('/', '_').replace(':', '_')}_{candidate['candidate_id']}.png"
            _contact_sheet(sheet, row, source_paths, variant_paths["KEEP_TARGET"], variant_paths["DROP_TARGET"],
                           variant_paths["DROP_MATCHED_CONTROL"], pixel_box,
                           proposal_info.get("roi_area_fraction") if proposal_info else None, row["verdicts"])
            row["contact_sheet"] = str(sheet)
            candidate_rows.append(row)
    decomposition = _policy_decomposition(candidate_rows)
    categories = Counter(category for row in candidate_rows for category in row["failure_categories"])
    report = {
        "diagnostic_version": "relive-phase15-certificate-diagnostic-v1",
        "read_only": True, "model_calls_made": 0,
        "allowed_inputs": ["public_runtime_record", "public_frame_images", "existing_raw_inference_cache", "existing_intervention_artifacts", "completed_run_artifacts"],
        "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"],
        "run_directory": str(run_dir), "runtime_path": str(runtime_path), "cache_directory": str(cache_dir),
        "run_manifest": {key: manifest.get(key) for key in ("git_commit", "runtime_sha256", "prompt_versions", "policy_version", "phase", "synthetic")},
        "retrieval_coverage": coverage_rows, "candidates": candidate_rows,
        "policy_decomposition": decomposition,
        "summary": {"sample_count": len(samples), "candidate_count": len(candidate_rows),
                    "failure_category_counts": dict(sorted(categories.items())), "raw_cache_records_read": raw_read,
                    "diagnostic_output_directory": str(output_dir)},
        "next_stage_recommendations": [
            "Do not change certificate admission rules, semantic schema, verifier prompt, confidence thresholds, cache keys, or the five claims based on this diagnostic.",
            "Before Phase 2, create one manually inspected public development/calibration positive control with a visibly localizable object or action, measurable intervention pixel change, and a geometrically feasible matched control.",
            "Treat semantic_only VERIFIED counts only as diagnostic upper bounds; they are not final reliable evidence.",
            "Use the contact sheets and per-frame pixel metrics to distinguish broad/global claims, multiple support regions, control geometry failure, and ineffective interventions before changing any proposal or acquisition component.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "diagnostic_report.md").write_text(_markdown(report), encoding="utf-8")
    with (output_dir / "summary_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "candidate_id", "candidate_rank", "original_verdict", "keep_verdict", "drop_verdict", "certificate_status", "spatial_status", "failure_categories"])
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({"sample_id": row["sample_id"], "candidate_id": row["candidate_id"], "candidate_rank": row["candidate_rank"],
                             "original_verdict": row["verdicts"]["original"], "keep_verdict": row["verdicts"]["keep"],
                             "drop_verdict": row["verdicts"]["drop"], "certificate_status": row["certificate_status"],
                             "spatial_status": row["spatial_status"], "failure_categories": ";".join(row["failure_categories"])})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only ReliVE Phase 1.5 certificate diagnostic")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        report = diagnose(Path(args.run_dir), Path(args.runtime), Path(args.cache_dir), Path(args.output_dir))
    except DiagnosticError as exc:
        print(f"Phase 1.5 diagnostic error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "PASS", "model_calls_made": 0,
                      "diagnostic_output_directory": report["summary"]["diagnostic_output_directory"],
                      "candidate_count": report["summary"]["candidate_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
