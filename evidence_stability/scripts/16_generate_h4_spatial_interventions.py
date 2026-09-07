#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.cache import stable_hash  # noqa: E402
from evidence_stability.phase_d2 import runtime_frame_path  # noqa: E402
from evidence_stability.spatial import (  # noqa: E402
    H4_PROTOCOL_VERSION,
    bbox_area_fraction,
    box_iou,
    control_bbox_norm,
    h4_gt_leakage_audit,
    normalized_to_pixel_bbox,
)
from evidence_stability.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


logger = logging.getLogger("h4_spatial_interventions")
INTERVENTION_TYPES = ("KEEP_ROI_V1", "DROP_ROI_V1", "KEEP_CONTROL_V1", "DROP_CONTROL_V1")
BLUR_VERSION = "gaussian_blur_sigma_0.05_min_hw_v1"


def setup_logging(log_path: str | Path | None) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def environment_snapshot() -> str:
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
    ]
    for module_name in ["PIL", "numpy"]:
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}: {getattr(module, '__version__', 'UNKNOWN')}")
        except Exception as exc:
            lines.append(f"{module_name}: UNAVAILABLE ({exc!r})")
    return "\n".join(lines) + "\n"


def resolve_frame_paths(row: dict[str, Any], frame_root: str | None, source_frame_prefix: str) -> list[str]:
    if frame_root and row.get("logical_frame_paths"):
        return [str(Path(frame_root) / Path(str(path))) for path in row["logical_frame_paths"]]
    return [
        runtime_frame_path(str(path), frame_root, source_frame_prefix) if frame_root else str(path)
        for path in row.get("frame_paths", [])
    ]


def gaussian_blur_radius(width: int, height: int) -> float:
    return 0.05 * min(width, height)


def make_roi_image(image: Any, pixel_box: list[int], intervention_type: str) -> Any:
    from PIL import Image, ImageFilter

    blurred = image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius(*image.size)))
    region = Image.new("L", image.size, 0)
    region.paste(255, box=tuple(pixel_box))
    if intervention_type.startswith("KEEP"):
        return Image.composite(image, blurred, region)
    return Image.composite(blurred, image, region)


def image_delta_stats(original: Any, altered: Any, pixel_box: list[int]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {"available": False, "reason": "NUMPY_UNAVAILABLE"}
    orig = np.asarray(original.convert("RGB"), dtype=np.int16)
    alt = np.asarray(altered.convert("RGB"), dtype=np.int16)
    diff = np.abs(orig - alt).mean(axis=2)
    x1, y1, x2, y2 = pixel_box
    inside = np.zeros(diff.shape, dtype=bool)
    inside[y1:y2, x1:x2] = True
    outside = ~inside
    out: dict[str, Any] = {"available": True}
    for name, mask in [("inside", inside), ("outside", outside)]:
        if mask.any():
            values = diff[mask]
            out[f"{name}_max_abs_mean_channel_delta"] = float(values.max())
            out[f"{name}_mean_abs_mean_channel_delta"] = float(values.mean())
        else:
            out[f"{name}_max_abs_mean_channel_delta"] = None
            out[f"{name}_mean_abs_mean_channel_delta"] = None
    return out


def write_visual_sheet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    thumbs = []
    labels = []
    if rows:
        source_paths = rows[0].get("source_frame_paths") or []
        if source_paths and Path(source_paths[0]).exists():
            img = Image.open(source_paths[0]).convert("RGB")
            if rows[0].get("predicted_bbox_norm"):
                draw = ImageDraw.Draw(img)
                box = normalized_to_pixel_bbox(rows[0]["predicted_bbox_norm"], img.size[0], img.size[1])
                draw.rectangle(box, outline=(255, 0, 0), width=4)
            img.thumbnail((240, 180))
            thumbs.append(img.copy())
            labels.append("ORIGINAL_BBOX")
    for row in rows:
        frame_paths = row.get("intervention_frame_paths") or []
        if not frame_paths or not Path(frame_paths[0]).exists():
            continue
        img = Image.open(frame_paths[0]).convert("RGB")
        img.thumbnail((240, 180))
        thumbs.append(img.copy())
        labels.append(row["intervention_type"])
    if not thumbs:
        return False
    width = 240 * len(thumbs)
    height = 210
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(thumbs):
        x = i * 240
        sheet.paste(img, (x, 24))
        draw.text((x + 6, 4), labels[i], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=90)
    return True


def strip_model_manifest(row: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "intervention_id",
        "candidate_id",
        "qa_id",
        "clip_id",
        "window_id",
        "dataset_name",
        "human_question",
        "support_prediction",
        "predicted_bbox_norm",
        "bbox_valid",
        "bbox_area_fraction",
        "control_bbox_norm",
        "control_valid",
        "intervention_type",
        "intervention_family",
        "intervention_frame_paths",
        "logical_frame_paths",
        "n_frames",
        "n_unique_frames",
        "blur_version",
        "blur_sigma_rule",
        "prompt_version",
        "prompt_hash",
    ]
    stripped = {key: row.get(key) for key in allowed if key in row}
    leakage = h4_gt_leakage_audit([stripped])
    if leakage["gt_leakage"]:
        raise RuntimeError(f"H4 spatial intervention model manifest leaked GT: {leakage}")
    return stripped


def generate_for_candidate(row: dict[str, Any], out_dir: Path, frame_root: str | None, source_frame_prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from PIL import Image

    source_paths = resolve_frame_paths(row, frame_root, source_frame_prefix)
    missing = [path for path in source_paths if not Path(path).exists()]
    if missing:
        logger.warning("candidate=%s missing_frames=%d", row["candidate_id"], len(missing))
        return [], [], [{"candidate_id": row["candidate_id"], "error_type": "MISSING_FRAMES", "missing_frame_paths": missing[:20]}]
    bbox = [int(v) for v in row["predicted_bbox_norm"]]
    control_bbox, control_valid, control_iou = control_bbox_norm(bbox)
    variants = [
        ("KEEP_ROI_V1", bbox, True),
        ("DROP_ROI_V1", bbox, True),
        ("KEEP_CONTROL_V1", control_bbox, control_valid),
        ("DROP_CONTROL_V1", control_bbox, control_valid),
    ]
    manifest_rows: list[dict[str, Any]] = []
    pixel_audit_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for intervention_type, intervention_bbox, generation_valid in variants:
        intervention_id = f"{row['candidate_id']}::{intervention_type}"
        candidate_dir = out_dir / "interventions" / stable_hash({"intervention_id": intervention_id})[:18] / intervention_type
        frame_out_paths: list[str] = []
        resolutions: list[list[int]] = []
        if generation_valid:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "candidate=%s intervention=%s frames=%d bbox=%s",
                row["candidate_id"],
                intervention_type,
                len(source_paths),
                intervention_bbox,
            )
            for idx, source_path in enumerate(source_paths):
                image = Image.open(source_path).convert("RGB")
                width, height = image.size
                pixel_box = normalized_to_pixel_bbox(intervention_bbox, width, height)
                altered = make_roi_image(image, pixel_box, intervention_type)
                output_path = candidate_dir / f"frame_{idx:04d}.png"
                altered.save(output_path, format="PNG")
                frame_out_paths.append(str(output_path))
                resolutions.append([width, height])
                if idx in {0, len(source_paths) // 2, len(source_paths) - 1}:
                    stats = image_delta_stats(image, altered, pixel_box)
                    keep = intervention_type.startswith("KEEP")
                    pixel_audit_rows.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "intervention_id": intervention_id,
                            "intervention_type": intervention_type,
                            "frame_position": idx,
                            "source_frame_path": source_path,
                            "intervention_frame_path": str(output_path),
                            "predicted_bbox_norm": bbox,
                            "intervention_bbox_norm": intervention_bbox,
                            "pixel_bbox": pixel_box,
                            "width": width,
                            "height": height,
                            "bbox_area_fraction": bbox_area_fraction(intervention_bbox),
                            "inside_should_preserve": keep,
                            "outside_should_preserve": not keep,
                            "delta_stats": stats,
                        }
                    )
        manifest_rows.append(
            {
                "protocol_version": H4_PROTOCOL_VERSION,
                "intervention_id": intervention_id,
                "candidate_id": row["candidate_id"],
                "qa_id": row["qa_id"],
                "clip_id": row["clip_id"],
                "window_id": row["window_id"],
                "dataset_name": row["dataset_name"],
                "human_question": row["human_question"],
                "support_prediction": row.get("support_prediction", "YES"),
                "predicted_bbox_norm": bbox,
                "bbox_valid": True,
                "bbox_area_fraction": row.get("bbox_area_fraction"),
                "control_bbox_norm": control_bbox,
                "control_valid": control_valid,
                "control_iou_with_predicted_bbox": control_iou,
                "intervention_type": intervention_type,
                "intervention_family": "H4_SPATIAL_ROI",
                "generation_valid": generation_valid,
                "generation_reason": "OK" if generation_valid else "CONTROL_REGION_INVALID",
                "source_frame_paths": source_paths,
                "logical_frame_paths": row.get("logical_frame_paths", []),
                "intervention_frame_paths": frame_out_paths,
                "n_frames": len(frame_out_paths) if generation_valid else 0,
                "n_unique_frames": len(set(row.get("logical_frame_paths") or source_paths)),
                "same_frame_count": generation_valid and len(frame_out_paths) == len(source_paths),
                "same_frame_order": True,
                "same_resolution": len({tuple(r) for r in resolutions}) <= 1 if resolutions else None,
                "blur_version": BLUR_VERSION,
                "blur_sigma_rule": "sigma = 0.05 * min(height, width); PIL.ImageFilter.GaussianBlur(radius=sigma)",
                "prompt_version": row.get("prompt_version"),
                "prompt_hash": row.get("prompt_hash"),
            }
        )
    return manifest_rows, pixel_audit_rows, errors


def summarize_pixel_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in rows:
        stats = row.get("delta_stats") or {}
        if not stats.get("available"):
            continue
        inside_max = stats.get("inside_max_abs_mean_channel_delta")
        outside_max = stats.get("outside_max_abs_mean_channel_delta")
        inside_mean = stats.get("inside_mean_abs_mean_channel_delta")
        outside_mean = stats.get("outside_mean_abs_mean_channel_delta")
        keep = row["intervention_type"].startswith("KEEP")
        if keep and inside_max not in (None, 0.0):
            failures.append({"intervention_id": row["intervention_id"], "frame_position": row["frame_position"], "reason": "KEEP_INSIDE_CHANGED"})
        if (not keep) and outside_max not in (None, 0.0):
            failures.append({"intervention_id": row["intervention_id"], "frame_position": row["frame_position"], "reason": "DROP_OUTSIDE_CHANGED"})
        if keep and outside_mean is not None and outside_mean <= 0.0:
            failures.append({"intervention_id": row["intervention_id"], "frame_position": row["frame_position"], "reason": "KEEP_OUTSIDE_UNCHANGED"})
        if (not keep) and inside_mean is not None and inside_mean <= 0.0:
            failures.append({"intervention_id": row["intervention_id"], "frame_position": row["frame_position"], "reason": "DROP_INSIDE_UNCHANGED"})
    return {
        "n_pixel_audit_rows": len(rows),
        "n_failures": len(failures),
        "failures": failures[:50],
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# H4 Spatial Intervention Generation",
        "",
        "## Input",
        f"- spatial pointer candidates: {summary['input']['n_pointer_rows']}",
        f"- valid bbox candidates: {summary['input']['n_valid_bbox_candidates']}",
        "",
        "## Generated",
        f"- interventions: {summary['generation']['n_interventions']}",
        f"- generation valid: {summary['generation']['generation_valid']}",
        f"- generation invalid: {summary['generation']['generation_invalid']}",
        f"- type counts: {summary['generation']['intervention_type_counts']}",
        "",
        "## Audit",
        f"- GT leakage: {summary['gt_leakage_audit']['gt_leakage']}",
        f"- pixel audit failures: {summary['pixel_audit']['n_failures']}",
        f"- missing frame errors: {summary['errors']['n_errors']}",
        f"- visual sheets written: {summary['visual_audit']['n_written']}",
        "",
        "No model inference, 50-QA discovery, formal statistics, or independent confirmation was run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate H4 preflight KEEP/DROP ROI interventions and engineering audits.")
    p.add_argument("--spatial_pointer_predictions", default="outputs/stg_pilot/phase_g/h4_spatial_v1/predictions/spatial_pointer_predictions.jsonl")
    p.add_argument("--supporting_candidates", default="outputs/stg_pilot/phase_g/h4_spatial_v1/manifest/h4_supporting_candidates_gt_free.jsonl")
    p.add_argument("--output_dir", default="outputs/stg_pilot/phase_g/h4_spatial_v1")
    p.add_argument("--frame_root", default=None)
    p.add_argument(
        "--source_frame_prefix",
        default="/root/data,/mnt/hdd3/huihui/hh_datas/MedVidU/valdata,/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
    )
    p.add_argument("--visual_limit", type=int, default=10)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["manifest", "audit", "summary", "visualizations/preflight", "provenance", "interventions"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_path or out_dir / "audit" / "h4_spatial_intervention_generation.log")
    supporting_by_candidate = {}
    if Path(args.supporting_candidates).exists():
        supporting_by_candidate = {
            row["candidate_id"]: row for row in read_jsonl(args.supporting_candidates)
            if row.get("candidate_id")
        }
    pointer_rows = [
        {**supporting_by_candidate.get(row.get("candidate_id"), {}), **row}
        for row in read_jsonl(args.spatial_pointer_predictions)
        if row.get("support_prediction") == "YES" and row.get("bbox_valid") and row.get("predicted_bbox_norm")
    ]
    logger.info(
        "loaded pointer_rows=%d valid_bbox_candidates=%d output_dir=%s",
        len(read_jsonl(args.spatial_pointer_predictions)),
        len(pointer_rows),
        out_dir,
    )

    interventions: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, row in enumerate(pointer_rows, start=1):
        logger.info("processing candidate %d/%d id=%s", i, len(pointer_rows), row["candidate_id"])
        generated, pixel_audit, row_errors = generate_for_candidate(row, out_dir, args.frame_root, args.source_frame_prefix)
        interventions.extend(generated)
        pixel_rows.extend(pixel_audit)
        errors.extend(row_errors)
        logger.info(
            "candidate %d/%d done generated=%d pixel_audit=%d errors=%d",
            i,
            len(pointer_rows),
            len(generated),
            len(pixel_audit),
            len(row_errors),
        )

    ids = [row["intervention_id"] for row in interventions]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate H4 spatial intervention IDs: {duplicates[:20]}")
    model_manifest = [strip_model_manifest(row) for row in interventions if row.get("generation_valid")]
    leakage = h4_gt_leakage_audit(model_manifest)
    if leakage["gt_leakage"]:
        raise RuntimeError(f"H4 intervention model manifest leakage detected: {leakage}")

    visual_rows = []
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in interventions:
        if row.get("generation_valid"):
            by_candidate.setdefault(row["candidate_id"], []).append(row)
    for idx, (candidate_id, rows) in enumerate(sorted(by_candidate.items())):
        if idx >= args.visual_limit:
            break
        sheet_path = out_dir / "visualizations" / "preflight" / f"{stable_hash({'candidate_id': candidate_id})[:16]}.jpg"
        written = write_visual_sheet(sheet_path, rows)
        visual_rows.append({"candidate_id": candidate_id, "visualization_path": str(sheet_path), "written": written})

    pixel_summary = summarize_pixel_audit(pixel_rows)
    generation_audit = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "n_interventions": len(interventions),
        "generation_valid": sum(bool(row.get("generation_valid")) for row in interventions),
        "generation_invalid": sum(not bool(row.get("generation_valid")) for row in interventions),
        "intervention_type_counts": dict(Counter(row["intervention_type"] for row in interventions)),
        "n_duplicate_intervention_ids": len(duplicates),
        "same_frame_count_all_valid": all(row.get("same_frame_count") for row in interventions if row.get("generation_valid")),
        "same_frame_order_all_valid": all(row.get("same_frame_order") for row in interventions if row.get("generation_valid")),
        "blur_version": BLUR_VERSION,
    }
    repo = {
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
    }
    summary = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "input": {
            "spatial_pointer_predictions": args.spatial_pointer_predictions,
            "n_pointer_rows": len(read_jsonl(args.spatial_pointer_predictions)),
            "n_valid_bbox_candidates": len(pointer_rows),
        },
        "generation": generation_audit,
        "pixel_audit": pixel_summary,
        "gt_leakage_audit": leakage,
        "errors": {"n_errors": len(errors), "examples": errors[:20]},
        "visual_audit": {"requested": args.visual_limit, "n_written": sum(row["written"] for row in visual_rows), "cases": visual_rows},
        "repo": repo,
        "model_inference_executed": False,
        "discovery_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
    }

    write_jsonl(out_dir / "manifest" / "h4_spatial_interventions.jsonl", interventions)
    write_jsonl(out_dir / "manifest" / "h4_spatial_intervention_manifest_gt_free.jsonl", model_manifest)
    write_json(out_dir / "audit" / "intervention_generation_audit.json", generation_audit)
    write_json(out_dir / "audit" / "spatial_pixel_audit.json", {"summary": pixel_summary, "rows": pixel_rows})
    write_json(out_dir / "audit" / "gt_leakage_audit_interventions.json", leakage)
    write_json(out_dir / "audit" / "visual_audit.json", summary["visual_audit"])
    write_json(out_dir / "predictions" / "intervention_generation_errors.json", errors)
    write_json(out_dir / "summary" / "h4_spatial_intervention_generation_summary.json", summary)
    write_summary_md(out_dir / "summary" / "h4_spatial_intervention_generation_summary.md", summary)
    (out_dir / "provenance" / "environment_snapshot_intervention_generation.txt").write_text(environment_snapshot(), encoding="utf-8")
    (out_dir / "provenance" / "git_status_intervention_generation.txt").write_text(repo["git_status_short"] + "\n", encoding="utf-8")
    logger.info(
        "done interventions=%d valid=%d invalid=%d pixel_failures=%d errors=%d",
        generation_audit["n_interventions"],
        generation_audit["generation_valid"],
        generation_audit["generation_invalid"],
        pixel_summary["n_failures"],
        len(errors),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
