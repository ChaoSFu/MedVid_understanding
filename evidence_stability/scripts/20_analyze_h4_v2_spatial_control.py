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

from evidence_stability.spatial import H4_PROTOCOL_VERSION  # noqa: E402
from evidence_stability.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


H4_V2_PROTOCOL_VERSION = "h4_spatial_v2_exploratory"
ALIGNED_THRESHOLD = 0.5


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def binary_value(value: Any) -> int | None:
    if value in {0, 1}:
        return int(value)
    return None


def clean_mean(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if isinstance(value, (int, float)) and not math.isnan(float(value))]
    return float(mean(valid)) if valid else None


def clean_median(values: list[Any]) -> float | None:
    valid = sorted(float(value) for value in values if isinstance(value, (int, float)) and not math.isnan(float(value)))
    return float(median(valid)) if valid else None


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: csv_value(row.get(field)) for field in fields} for row in rows)


def h4_v2_label(row: dict[str, Any]) -> str:
    if not row.get("h4_temporally_eligible"):
        return "EXCLUDED_NOT_TEMPORALLY_ELIGIBLE"
    score = row.get("spatial_match_score")
    if not isinstance(score, (int, float)):
        return "EXCLUDED_NO_SPATIAL_SCORE"
    score = float(score)
    if score >= ALIGNED_THRESHOLD:
        return "ALIGNED_SPATIAL_SUPPORT"
    if score > 0.0:
        return "MISALIGNED_SPATIAL_SUPPORT"
    return "ZERO_OVERLAP_SPATIAL_SUPPORT"


def causal_scores(row: dict[str, Any]) -> dict[str, Any]:
    keep_roi = binary_value(row.get("keep_yes"))
    drop_roi = binary_value(row.get("drop_yes"))
    keep_control = binary_value(row.get("keep_control_yes"))
    drop_control = binary_value(row.get("drop_control_yes"))
    roi_sufficiency = keep_roi
    roi_necessity = None if drop_roi is None else 1 - drop_roi
    control_sufficiency = keep_control
    control_necessity = None if drop_control is None else 1 - drop_control
    c_roi = None if roi_sufficiency is None or roi_necessity is None else (roi_sufficiency + roi_necessity) / 2
    c_control = None if control_sufficiency is None or control_necessity is None else (control_sufficiency + control_necessity) / 2
    return {
        "roi_sufficiency": roi_sufficiency,
        "roi_necessity": roi_necessity,
        "C_ROI": c_roi,
        "control_sufficiency": control_sufficiency,
        "control_necessity": control_necessity,
        "C_CONTROL": c_control,
        "delta_sufficiency_roi_minus_control": None if roi_sufficiency is None or control_sufficiency is None else roi_sufficiency - control_sufficiency,
        "delta_necessity_roi_minus_control": None if roi_necessity is None or control_necessity is None else roi_necessity - control_necessity,
        "roi_control_advantage": None if c_roi is None or c_control is None else c_roi - c_control,
    }


def candidate_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in source_rows:
        scores = causal_scores(source)
        c_roi = scores["C_ROI"]
        h4_v1_c = source.get("C_S")
        if c_roi is not None and isinstance(h4_v1_c, (int, float)) and not math.isclose(c_roi, float(h4_v1_c), abs_tol=1e-9):
            raise RuntimeError(f"H4 v1 C_S mismatch for {source.get('candidate_id')}: computed={c_roi} stored={h4_v1_c}")
        rows.append(
            {
                "candidate_id": source.get("candidate_id"),
                "qa_id": source.get("qa_id"),
                "clip_id": source.get("clip_id"),
                "window_id": source.get("window_id"),
                "dataset_name": source.get("dataset_name"),
                "h4_v1_candidate_spatial_type": source.get("candidate_spatial_type"),
                "h4_temporally_eligible": bool(source.get("h4_temporally_eligible")),
                "spatial_match_score": source.get("spatial_match_score"),
                "h4_v2_spatial_label": h4_v2_label(source),
                "keep_yes": source.get("keep_yes"),
                "drop_yes": source.get("drop_yes"),
                "keep_control_yes": source.get("keep_control_yes"),
                "drop_control_yes": source.get("drop_control_yes"),
                **scores,
                "control_complete": scores["C_CONTROL"] is not None,
                "v2_analysis_eligible": (
                    h4_v2_label(source) in {"ALIGNED_SPATIAL_SUPPORT", "MISALIGNED_SPATIAL_SUPPORT"}
                    and scores["roi_control_advantage"] is not None
                ),
            }
        )
    return rows


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_candidates": len(rows),
        "n_with_complete_control": sum(bool(row["control_complete"]) for row in rows),
        "n_v2_analysis_eligible": sum(bool(row["v2_analysis_eligible"]) for row in rows),
        "mean_iou": clean_mean([row["spatial_match_score"] for row in rows]),
        "mean_C_ROI": clean_mean([row["C_ROI"] for row in rows]),
        "mean_C_CONTROL": clean_mean([row["C_CONTROL"] for row in rows]),
        "mean_roi_control_advantage": clean_mean([row["roi_control_advantage"] for row in rows]),
        "median_roi_control_advantage": clean_median([row["roi_control_advantage"] for row in rows]),
        "mean_delta_sufficiency": clean_mean([row["delta_sufficiency_roi_minus_control"] for row in rows]),
        "mean_delta_necessity": clean_mean([row["delta_necessity_roi_minus_control"] for row in rows]),
    }


def qa_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["v2_analysis_eligible"]:
            groups[str(row["qa_id"])].append(row)
    paired = []
    for qa_id, group in sorted(groups.items()):
        aligned = [row for row in group if row["h4_v2_spatial_label"] == "ALIGNED_SPATIAL_SUPPORT"]
        misaligned = [row for row in group if row["h4_v2_spatial_label"] == "MISALIGNED_SPATIAL_SUPPORT"]
        if not aligned or not misaligned:
            continue
        aligned_advantage = clean_mean([row["roi_control_advantage"] for row in aligned])
        misaligned_advantage = clean_mean([row["roi_control_advantage"] for row in misaligned])
        paired.append(
            {
                "qa_id": qa_id,
                "dataset_name": group[0]["dataset_name"],
                "n_aligned": len(aligned),
                "n_misaligned": len(misaligned),
                "mean_aligned_roi_control_advantage": aligned_advantage,
                "mean_misaligned_roi_control_advantage": misaligned_advantage,
                "delta_roi_control_advantage": None if aligned_advantage is None or misaligned_advantage is None else aligned_advantage - misaligned_advantage,
            }
        )
    return paired


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    groups = summary["spatial_groups"]
    pairing = summary["within_qa_pairing"]
    lines = [
        "# H4 v2 Exploratory Spatial-Control Analysis",
        "",
        "This is a new exploratory post-hoc analysis of existing H4 v1 discovery outputs.",
        "It does not alter H4 v1 labels, run new model inference, or provide independent confirmation.",
        "",
        "## Frozen v2 Labels",
        f"- ALIGNED_SPATIAL_SUPPORT: temporally eligible and mean IoU >= {ALIGNED_THRESHOLD}",
        "- MISALIGNED_SPATIAL_SUPPORT: temporally eligible and 0 < mean IoU < 0.5",
        "- ZERO_OVERLAP_SPATIAL_SUPPORT: temporally eligible and mean IoU == 0",
        "",
        "## ROI Versus Control",
        "- C_ROI = (KEEP_ROI YES + DROP_ROI NO) / 2",
        "- C_CONTROL = (KEEP_CONTROL YES + DROP_CONTROL NO) / 2",
        "- ROI-control advantage = C_ROI - C_CONTROL",
        "",
        "## Spatial Groups",
    ]
    for label in ["ALIGNED_SPATIAL_SUPPORT", "MISALIGNED_SPATIAL_SUPPORT", "ZERO_OVERLAP_SPATIAL_SUPPORT"]:
        values = groups.get(label, {})
        lines.extend(
            [
                f"- {label}: n={values.get('n_candidates', 0)}, complete_control={values.get('n_with_complete_control', 0)}, eligible={values.get('n_v2_analysis_eligible', 0)}, mean_advantage={values.get('mean_roi_control_advantage')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Within-QA Exploratory Pairing",
            f"- paired QA with both ALIGNED and MISALIGNED candidates: {pairing['n_paired_qa']}",
            f"- mean paired delta ROI-control advantage: {pairing['mean_delta_roi_control_advantage']}",
            "",
            "No formal statistics were calculated.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H4 v2 exploratory ROI-versus-control analysis from existing joined candidates.")
    parser.add_argument("--joined_candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.joined_candidates)
    if not source_rows:
        raise RuntimeError("No H4 joined candidate rows found.")
    ids = [str(row.get("candidate_id")) for row in source_rows]
    duplicates = [candidate_id for candidate_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate H4 joined candidate IDs: {duplicates[:10]}")

    rows = candidate_rows(source_rows)
    pair_rows = qa_pair_rows(rows)
    labels = Counter(row["h4_v2_spatial_label"] for row in rows)
    groups = {
        label: group_summary([row for row in rows if row["h4_v2_spatial_label"] == label])
        for label in ["ALIGNED_SPATIAL_SUPPORT", "MISALIGNED_SPATIAL_SUPPORT", "ZERO_OVERLAP_SPATIAL_SUPPORT"]
    }
    paired_deltas = [row["delta_roi_control_advantage"] for row in pair_rows]
    summary = {
        "protocol_version": H4_V2_PROTOCOL_VERSION,
        "source_protocol_version": H4_PROTOCOL_VERSION,
        "study_stage": "exploratory_post_hoc_existing_discovery",
        "joined_candidates": args.joined_candidates,
        "model_inference_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
        "n_source_candidates": len(source_rows),
        "n_duplicate_candidate_ids": len(duplicates),
        "label_counts_all_joined_candidates": dict(labels),
        "spatial_groups": groups,
        "within_qa_pairing": {
            "n_paired_qa": len(pair_rows),
            "delta_positive": sum(value > 0 for value in paired_deltas if isinstance(value, (int, float))),
            "delta_zero": sum(value == 0 for value in paired_deltas if isinstance(value, (int, float))),
            "delta_negative": sum(value < 0 for value in paired_deltas if isinstance(value, (int, float))),
            "mean_delta_roi_control_advantage": clean_mean(paired_deltas),
            "median_delta_roi_control_advantage": clean_median(paired_deltas),
        },
        "repo": {
            "git_commit": git_output(["rev-parse", "HEAD"]),
            "git_status_short": git_output(["status", "--short"]),
        },
    }
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "joined" / "h4_v2_exploratory_candidates.jsonl", rows)
    write_csv(out_dir / "candidate" / "h4_v2_exploratory_spatial_control.csv", rows)
    write_csv(out_dir / "qa" / "h4_v2_exploratory_paired_qa.csv", pair_rows)
    write_json(out_dir / "summary" / "h4_v2_exploratory_spatial_control_summary.json", summary)
    write_summary_md(out_dir / "summary" / "h4_v2_exploratory_spatial_control_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
