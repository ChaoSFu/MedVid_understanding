#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.spatial import (  # noqa: E402
    H4_PROTOCOL_VERSION,
    H4_SPATIAL_LABEL_PROTOCOL,
    H4_SPATIAL_TRUE_IOU_THRESHOLD,
    H4_SPURIOUS_IOU_THRESHOLD,
    box_iou,
    normalized_to_pixel_bbox,
)
from evidence_stability.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else None


def clean_median(values: list[Any]) -> float | None:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v)))
    return float(median(clean)) if clean else None


def index_latest(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            out[str(value)] = row
    return out


def duplicate_ids(rows: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(str(row.get(key)) for row in rows)
    return sorted(value for value, count in counts.items() if count > 1)


def image_dimensions(pointer: dict[str, Any]) -> tuple[int, int] | None:
    metadata = pointer.get("processor_metadata") or {}
    sizes = metadata.get("image_sizes") or []
    if sizes and isinstance(sizes[0], list) and len(sizes[0]) == 2:
        return int(sizes[0][0]), int(sizes[0][1])
    try:
        from PIL import Image
    except ImportError:
        return None
    for path in pointer.get("frame_paths") or []:
        if Path(path).exists():
            with Image.open(path) as img:
                return int(img.size[0]), int(img.size[1])
    return None


def timestamp_in_window(timestamp: float, local_timestamps: list[Any]) -> bool:
    times = [float(t) for t in local_timestamps if isinstance(t, (int, float))]
    if not times:
        return False
    eps = 1e-6
    return min(times) - eps <= timestamp <= max(times) + eps


def spatial_match(pointer: dict[str, Any], window_gt: dict[str, Any]) -> dict[str, Any]:
    if not pointer.get("bbox_valid") or not pointer.get("predicted_bbox_norm"):
        return {"spatial_match_score": None, "spatial_match_reason": "INVALID_PREDICTED_BBOX", "n_spatial_gt_boxes_evaluated": 0}
    dims = image_dimensions(pointer)
    if dims is None:
        return {"spatial_match_score": None, "spatial_match_reason": "MISSING_IMAGE_DIMENSIONS", "n_spatial_gt_boxes_evaluated": 0}
    width, height = dims
    predicted_pixel = normalized_to_pixel_bbox(pointer["predicted_bbox_norm"], width, height)
    bbox_dict = window_gt.get("stg_bbox_dict") or {}
    local_times = window_gt.get("local_timestamps") or []
    ious = []
    evaluated = []
    for key, gt_box in sorted(bbox_dict.items(), key=lambda item: float(item[0])):
        timestamp = float(key)
        if not timestamp_in_window(timestamp, local_times):
            continue
        if not (isinstance(gt_box, list) and len(gt_box) == 4):
            continue
        iou = box_iou(predicted_pixel, gt_box)
        ious.append(iou)
        evaluated.append({"timestamp": timestamp, "gt_bbox": gt_box, "predicted_pixel_bbox": predicted_pixel, "iou": iou})
    if not ious:
        return {
            "spatial_match_score": None,
            "spatial_match_reason": "NO_GT_TIMESTAMP_IN_WINDOW",
            "n_spatial_gt_boxes_evaluated": 0,
            "predicted_pixel_bbox": predicted_pixel,
            "image_width": width,
            "image_height": height,
        }
    score = float(mean(ious))
    return {
        "spatial_match_score": score,
        "spatial_match_reason": "OK",
        "n_spatial_gt_boxes_evaluated": len(ious),
        "per_timestamp_iou": evaluated,
        "predicted_pixel_bbox": predicted_pixel,
        "image_width": width,
        "image_height": height,
    }


def classify_spatial_candidate(match: dict[str, Any], window_gt: dict[str, Any]) -> str:
    if not window_gt.get("h4_temporally_eligible"):
        return "NOT_TEMPORALLY_ELIGIBLE"
    score = match.get("spatial_match_score")
    if score is None:
        return "NO_SPATIAL_GT_EVALUATED"
    if float(score) >= H4_SPATIAL_TRUE_IOU_THRESHOLD:
        return "TRUE_SPATIAL_SUPPORT"
    if float(score) == H4_SPURIOUS_IOU_THRESHOLD:
        return "SPURIOUS_SPATIAL_SUPPORT"
    return "WEAK_SPATIAL_SUPPORT"


def intervention_outcomes(candidate_id: str, predictions_by_candidate: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = predictions_by_candidate.get(candidate_id, [])
    by_type = {row["intervention_type"]: row for row in rows}

    def yes_value(intervention_type: str) -> int | None:
        row = by_type.get(intervention_type)
        if not row or row.get("parsed_prediction") not in {"YES", "NO"}:
            return None
        return 1 if row["parsed_prediction"] == "YES" else 0

    keep_yes = yes_value("KEEP_ROI_V1")
    drop_yes = yes_value("DROP_ROI_V1")
    keep_control_yes = yes_value("KEEP_CONTROL_V1")
    drop_control_yes = yes_value("DROP_CONTROL_V1")
    suff = keep_yes
    nec = None if drop_yes is None else 1 - drop_yes
    c_s = None if suff is None or nec is None else (suff + nec) / 2
    return {
        "keep_yes": keep_yes,
        "drop_yes": drop_yes,
        "suff": suff,
        "nec": nec,
        "C_S": c_s,
        "keep_control_yes": keep_control_yes,
        "drop_control_yes": drop_control_yes,
        "n_intervention_predictions": len(rows),
    }


def candidate_rows(
    pointer_rows: list[dict[str, Any]],
    window_gt_rows: list[dict[str, Any]],
    intervention_prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt_by_candidate = index_latest(window_gt_rows, "candidate_id")
    predictions_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in intervention_prediction_rows:
        predictions_by_candidate[str(row["candidate_id"])].append(row)

    rows = []
    for pointer in pointer_rows:
        if pointer.get("support_prediction") != "YES" or not pointer.get("bbox_valid"):
            continue
        candidate_id = str(pointer["candidate_id"])
        window_gt = gt_by_candidate.get(candidate_id)
        if not window_gt:
            continue
        match = spatial_match(pointer, window_gt)
        row = {
            "candidate_id": candidate_id,
            "qa_id": pointer["qa_id"],
            "clip_id": pointer["clip_id"],
            "window_id": pointer["window_id"],
            "dataset_name": pointer["dataset_name"],
            "h4_temporally_eligible": bool(window_gt.get("h4_temporally_eligible")),
            "temporal_eligibility_rule": window_gt.get("temporal_eligibility_rule"),
            "n_gt_visible_in_window": window_gt.get("n_gt_visible_in_window"),
            "n_gt_visible_total": window_gt.get("n_gt_visible_total"),
            "evidence_density": window_gt.get("evidence_density"),
            "gt_evidence_recall": window_gt.get("gt_evidence_recall"),
            "bbox_valid": bool(pointer.get("bbox_valid")),
            "predicted_bbox_norm": pointer.get("predicted_bbox_norm"),
            "bbox_area_fraction": pointer.get("bbox_area_fraction"),
            **match,
            "candidate_spatial_type": classify_spatial_candidate(match, window_gt),
            **intervention_outcomes(candidate_id, predictions_by_candidate),
        }
        rows.append(row)
    audit = {
        "n_pointer_rows": len(pointer_rows),
        "n_candidates_joined": len(rows),
        "n_missing_window_gt": sum(1 for row in pointer_rows if row.get("candidate_id") not in gt_by_candidate),
        "candidate_spatial_type_counts": dict(Counter(row["candidate_spatial_type"] for row in rows)),
        "n_temporally_eligible": sum(bool(row["h4_temporally_eligible"]) for row in rows),
        "n_with_C_S": sum(row["C_S"] is not None for row in rows),
    }
    return rows, audit


def qa_rows(candidate_rows_: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows_:
        if row["candidate_spatial_type"] in {"TRUE_SPATIAL_SUPPORT", "SPURIOUS_SPATIAL_SUPPORT"}:
            groups[row["qa_id"]].append(row)
    out = []
    for qa_id, rows in sorted(groups.items()):
        true_rows = [row for row in rows if row["candidate_spatial_type"] == "TRUE_SPATIAL_SUPPORT" and row["C_S"] is not None]
        spur_rows = [row for row in rows if row["candidate_spatial_type"] == "SPURIOUS_SPATIAL_SUPPORT" and row["C_S"] is not None]
        if not true_rows or not spur_rows:
            continue
        mean_true_c = clean_mean([row["C_S"] for row in true_rows])
        mean_spur_c = clean_mean([row["C_S"] for row in spur_rows])
        mean_true_suff = clean_mean([row["suff"] for row in true_rows])
        mean_spur_suff = clean_mean([row["suff"] for row in spur_rows])
        mean_true_nec = clean_mean([row["nec"] for row in true_rows])
        mean_spur_nec = clean_mean([row["nec"] for row in spur_rows])
        out.append(
            {
                "qa_id": qa_id,
                "dataset_name": rows[0]["dataset_name"],
                "n_true_spatial_candidates": len(true_rows),
                "n_spurious_spatial_candidates": len(spur_rows),
                "mean_true_C_S": mean_true_c,
                "mean_spurious_C_S": mean_spur_c,
                "delta_C": None if mean_true_c is None or mean_spur_c is None else mean_true_c - mean_spur_c,
                "mean_true_suff": mean_true_suff,
                "mean_spurious_suff": mean_spur_suff,
                "delta_suff": None if mean_true_suff is None or mean_spur_suff is None else mean_true_suff - mean_spur_suff,
                "mean_true_nec": mean_true_nec,
                "mean_spurious_nec": mean_spur_nec,
                "delta_nec": None if mean_true_nec is None or mean_spur_nec is None else mean_true_nec - mean_spur_nec,
            }
        )
    return out


def sign_counts(values: list[Any]) -> dict[str, int]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return {
        "positive": sum(v > 0 for v in clean),
        "zero": sum(v == 0 for v in clean),
        "negative": sum(v < 0 for v in clean),
    }


def persistence_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in ["TRUE_SPATIAL_SUPPORT", "SPURIOUS_SPATIAL_SUPPORT", "WEAK_SPATIAL_SUPPORT", "NOT_TEMPORALLY_ELIGIBLE"]:
        sub = [row for row in rows if row["candidate_spatial_type"] == label]
        out[label] = {
            "n": len(sub),
            "suff_mean": clean_mean([row["suff"] for row in sub]),
            "nec_mean": clean_mean([row["nec"] for row in sub]),
            "C_S_mean": clean_mean([row["C_S"] for row in sub]),
            "keep_yes": sum(row["keep_yes"] == 1 for row in sub),
            "keep_no": sum(row["keep_yes"] == 0 for row in sub),
            "drop_yes": sum(row["drop_yes"] == 1 for row in sub),
            "drop_no": sum(row["drop_yes"] == 0 for row in sub),
        }
    return out


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    primary = summary["primary_preflight_qa"]
    lines = [
        "# H4 Spatial Preflight Join Report",
        "",
        "## Scope",
        f"- protocol version: {summary['protocol_version']}",
        f"- spatial label protocol: {summary['spatial_label_protocol']}",
        f"- candidate rows: {summary['candidate_join_audit']['n_candidates_joined']}",
        f"- intervention predictions: {summary['intervention_audit']['n_intervention_predictions']}",
        f"- discovery executed: {summary['discovery_executed']}",
        "",
        "## Candidate Labels",
        f"- counts: {summary['candidate_join_audit']['candidate_spatial_type_counts']}",
        f"- temporally eligible: {summary['candidate_join_audit']['n_temporally_eligible']}",
        f"- candidates with C_S: {summary['candidate_join_audit']['n_with_C_S']}",
        "",
        "## Primary QA Preflight",
        f"- paired QA: {primary['n_paired_qa']}",
        f"- delta_C positive: {primary['delta_C_positive']}",
        f"- delta_C zero: {primary['delta_C_zero']}",
        f"- delta_C negative: {primary['delta_C_negative']}",
        f"- mean delta_C: {primary['mean_delta_C']}",
        f"- median delta_C: {primary['median_delta_C']}",
        f"- mean delta_suff: {primary['mean_delta_suff']}",
        f"- mean delta_nec: {primary['mean_delta_nec']}",
        "",
        "This is a 12-QA engineering preflight join. No 50-QA discovery, formal statistics, or independent confirmation was run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc H4 preflight GT join and spatial causality summary.")
    p.add_argument("--window_gt", default="outputs/stg_pilot/phase_g/h4_spatial_v1/joined/h4_preflight_windows_with_temporal_gt.jsonl")
    p.add_argument("--spatial_pointer_predictions", default="outputs/stg_pilot/phase_g/h4_spatial_v1/predictions/spatial_pointer_predictions.jsonl")
    p.add_argument("--intervention_predictions", default="outputs/stg_pilot/phase_g/h4_spatial_v1/predictions/intervention_predictions.jsonl")
    p.add_argument("--output_dir", default="outputs/stg_pilot/phase_g/h4_spatial_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["joined", "candidate", "qa", "summary", "audit", "provenance"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    window_gt_rows = read_jsonl(args.window_gt)
    pointer_rows = read_jsonl(args.spatial_pointer_predictions)
    intervention_rows = read_jsonl(args.intervention_predictions)
    candidate_rows_, candidate_audit = candidate_rows(pointer_rows, window_gt_rows, intervention_rows)
    qa_rows_ = qa_rows(candidate_rows_)
    delta_counts = sign_counts([row["delta_C"] for row in qa_rows_])
    primary = {
        "n_paired_qa": len(qa_rows_),
        "delta_C_positive": delta_counts["positive"],
        "delta_C_zero": delta_counts["zero"],
        "delta_C_negative": delta_counts["negative"],
        "mean_delta_C": clean_mean([row["delta_C"] for row in qa_rows_]),
        "median_delta_C": clean_median([row["delta_C"] for row in qa_rows_]),
        "mean_delta_suff": clean_mean([row["delta_suff"] for row in qa_rows_]),
        "median_delta_suff": clean_median([row["delta_suff"] for row in qa_rows_]),
        "mean_delta_nec": clean_mean([row["delta_nec"] for row in qa_rows_]),
        "median_delta_nec": clean_median([row["delta_nec"] for row in qa_rows_]),
    }
    intervention_audit = {
        "n_intervention_predictions": len(intervention_rows),
        "duplicate_intervention_prediction_ids": duplicate_ids(intervention_rows, "intervention_id"),
        "prediction_counts": dict(Counter(row.get("parsed_prediction") for row in intervention_rows)),
        "by_intervention_type": {
            key: dict(Counter(row.get("parsed_prediction") for row in rows))
            for key, rows in sorted(
                {
                    intervention_type: [
                        row for row in intervention_rows
                        if row.get("intervention_type") == intervention_type
                    ]
                    for intervention_type in {row.get("intervention_type") for row in intervention_rows}
                }.items()
            )
        },
    }
    summary = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "spatial_label_protocol": H4_SPATIAL_LABEL_PROTOCOL,
        "window_gt": args.window_gt,
        "spatial_pointer_predictions": args.spatial_pointer_predictions,
        "intervention_predictions": args.intervention_predictions,
        "candidate_join_audit": candidate_audit,
        "intervention_audit": intervention_audit,
        "primary_preflight_qa": primary,
        "persistence_diagnostic": persistence_diagnostic(candidate_rows_),
        "repo": {
            "git_commit": git_output(["rev-parse", "HEAD"]),
            "git_status_short": git_output(["status", "--short"]),
        },
        "model_inference_executed": False,
        "discovery_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
    }
    write_jsonl(out_dir / "joined" / "h4_candidates_with_gt.jsonl", candidate_rows_)
    write_csv(out_dir / "candidate" / "candidate_spatial_causality.csv", candidate_rows_)
    write_csv(out_dir / "qa" / "h4_primary_paired_qa.csv", qa_rows_)
    write_json(out_dir / "summary" / "h4_spatial_persistence_diagnostic.json", summary["persistence_diagnostic"])
    write_json(out_dir / "summary" / "h4_preflight_join_summary.json", summary)
    write_summary_md(out_dir / "summary" / "h4_preflight_join_summary.md", summary)
    write_json(out_dir / "audit" / "join_audit.json", {"candidate_join_audit": candidate_audit, "intervention_audit": intervention_audit})
    (out_dir / "provenance" / "git_status_h4_preflight_join.txt").write_text(summary["repo"]["git_status_short"] + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
