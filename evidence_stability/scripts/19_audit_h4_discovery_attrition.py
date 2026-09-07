#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.spatial import H4_PROTOCOL_VERSION  # noqa: E402
from evidence_stability.utils import read_jsonl, write_json  # noqa: E402


SPATIAL_TYPES = (
    "TRUE_SPATIAL_SUPPORT",
    "SPURIOUS_SPATIAL_SUPPORT",
    "WEAK_SPATIAL_SUPPORT",
    "NOT_TEMPORALLY_ELIGIBLE",
    "NO_SPATIAL_GT_EVALUATED",
)


def latest_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: csv_value(row.get(field)) for field in fields} for row in rows)


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def outcome(row: dict[str, Any] | None) -> str:
    if row is None:
        return "MISSING"
    value = row.get("parsed_prediction")
    return str(value) if value in {"YES", "NO", "INVALID"} else "INVALID"


def iou_bin(score: Any) -> str:
    if not isinstance(score, (int, float)):
        return "MISSING"
    score = float(score)
    if score == 0.0:
        return "0"
    if score < 0.5:
        return "(0,0.5)"
    return ">=0.5"


def add_count(bucket: dict[str, Any], key: str, value: int = 1) -> None:
    bucket[key] = int(bucket.get(key, 0)) + value


def build_audit(
    manifest_rows: list[dict[str, Any]],
    window_gt_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    pointer_rows: list[dict[str, Any]],
    joined_candidate_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    intervention_prediction_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_by_id = latest_by_key(manifest_rows, "candidate_id")
    window_by_id = latest_by_key(window_gt_rows, "candidate_id")
    support_by_id = latest_by_key(support_rows, "candidate_id")
    pointer_by_id = latest_by_key(pointer_rows, "candidate_id")
    joined_by_id = latest_by_key(joined_candidate_rows, "candidate_id")

    if set(window_by_id) - set(manifest_by_id):
        raise RuntimeError("Window-GT file contains candidates absent from the GT-free model manifest.")

    by_dataset: dict[str, dict[str, Any]] = defaultdict(dict)
    by_qa: dict[str, dict[str, Any]] = defaultdict(dict)
    all_counts: dict[str, Any] = {}
    support_counts = Counter()
    support_eligible_counts = Counter()
    pointer_counts = Counter()
    pointer_eligible_counts = Counter()
    spatial_counts = Counter()
    spatial_eligible_counts = Counter()
    iou_counts = Counter()

    def record_level(bucket: dict[str, Any], prefix: str, *, eligible: bool, support: str, pointer: dict[str, Any] | None, joined: dict[str, Any] | None) -> None:
        add_count(bucket, f"{prefix}_windows")
        if eligible:
            add_count(bucket, f"{prefix}_temporally_eligible")
        if support == "YES":
            add_count(bucket, f"{prefix}_support_yes")
            if eligible:
                add_count(bucket, f"{prefix}_eligible_support_yes")
        if pointer is not None:
            pointer_status = "VALID" if pointer.get("bbox_valid") else "INVALID"
            add_count(bucket, f"{prefix}_pointer_{pointer_status.lower()}")
            if eligible:
                add_count(bucket, f"{prefix}_eligible_pointer_{pointer_status.lower()}")
        if joined is not None:
            spatial_type = str(joined.get("candidate_spatial_type", "MISSING"))
            add_count(bucket, f"{prefix}_{spatial_type}")

    for candidate_id, manifest in sorted(manifest_by_id.items()):
        dataset = str(manifest.get("dataset_name", "UNKNOWN"))
        qa_id = str(manifest.get("qa_id", "UNKNOWN"))
        gt = window_by_id.get(candidate_id, {})
        eligible = bool(gt.get("h4_temporally_eligible"))
        support = outcome(support_by_id.get(candidate_id))
        pointer = pointer_by_id.get(candidate_id)
        joined = joined_by_id.get(candidate_id)

        add_count(all_counts, "n_model_windows")
        if eligible:
            add_count(all_counts, "n_temporally_eligible_windows")
        support_counts[support] += 1
        if eligible:
            support_eligible_counts[support] += 1
        if pointer is not None:
            pointer_status = "VALID_BBOX" if pointer.get("bbox_valid") else "INVALID_BBOX"
            pointer_counts[pointer_status] += 1
            if eligible:
                pointer_eligible_counts[pointer_status] += 1
        if joined is not None:
            spatial_type = str(joined.get("candidate_spatial_type", "MISSING"))
            spatial_counts[spatial_type] += 1
            if eligible:
                spatial_eligible_counts[spatial_type] += 1
                iou_counts[iou_bin(joined.get("spatial_match_score"))] += 1

        record_level(by_dataset[dataset], "n", eligible=eligible, support=support, pointer=pointer, joined=joined)
        record_level(by_qa[qa_id], "n", eligible=eligible, support=support, pointer=pointer, joined=joined)

    dataset_rows = []
    for dataset, values in sorted(by_dataset.items()):
        dataset_rows.append({"dataset_name": dataset, **values})

    qa_rows = []
    for qa_id, values in sorted(by_qa.items()):
        true_count = int(values.get("n_TRUE_SPATIAL_SUPPORT", 0))
        spur_count = int(values.get("n_SPURIOUS_SPATIAL_SUPPORT", 0))
        first = next(row for row in manifest_by_id.values() if str(row.get("qa_id")) == qa_id)
        qa_rows.append(
            {
                "qa_id": qa_id,
                "dataset_name": first.get("dataset_name"),
                **values,
                "n_true_spatial_support": true_count,
                "n_spurious_spatial_support": spur_count,
                "is_primary_paired_qa": true_count > 0 and spur_count > 0,
            }
        )

    intervention_type_counts = Counter(str(row.get("intervention_type", "MISSING")) for row in intervention_rows)
    valid_intervention_type_counts = Counter(
        str(row.get("intervention_type", "MISSING"))
        for row in intervention_rows
        if row.get("generation_valid")
    )
    intervention_prediction_by_type: dict[str, Counter] = defaultdict(Counter)
    for row in intervention_prediction_rows:
        intervention_prediction_by_type[str(row.get("intervention_type", "MISSING"))][outcome(row)] += 1

    summary = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "study_stage": "discovery",
        "analysis_type": "post_hoc_attrition_diagnostic_no_model_inference",
        "model_inference_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
        "window_funnel": {
            **all_counts,
            "temporal_eligibility_rate": all_counts.get("n_temporally_eligible_windows", 0) / max(1, all_counts.get("n_model_windows", 0)),
            "support_prediction_counts_all_windows": dict(support_counts),
            "support_prediction_counts_temporally_eligible_windows": dict(support_eligible_counts),
            "pointer_counts_all_windows": dict(pointer_counts),
            "pointer_counts_temporally_eligible_windows": dict(pointer_eligible_counts),
        },
        "spatial_labels": {
            "counts_all_joined_candidates": dict(spatial_counts),
            "counts_temporally_eligible_joined_candidates": dict(spatial_eligible_counts),
            "iou_bins_temporally_eligible_joined_candidates": dict(iou_counts),
            "n_joined_candidates": len(joined_by_id),
            "n_pointer_candidates_without_post_hoc_join": sum(1 for candidate_id in pointer_by_id if candidate_id not in joined_by_id),
        },
        "qa_pairing": {
            "n_qa": len(by_qa),
            "n_primary_paired_qa": sum(bool(row["is_primary_paired_qa"]) for row in qa_rows),
            "n_qa_with_true_spatial_support": sum(row["n_true_spatial_support"] > 0 for row in qa_rows),
            "n_qa_with_spurious_spatial_support": sum(row["n_spurious_spatial_support"] > 0 for row in qa_rows),
        },
        "interventions": {
            "n_intervention_rows": len(intervention_rows),
            "n_generation_valid": sum(bool(row.get("generation_valid")) for row in intervention_rows),
            "n_generation_invalid": sum(not bool(row.get("generation_valid")) for row in intervention_rows),
            "generation_by_type": dict(intervention_type_counts),
            "generation_valid_by_type": dict(valid_intervention_type_counts),
            "prediction_by_type": {key: dict(value) for key, value in sorted(intervention_prediction_by_type.items())},
            "n_intervention_predictions": len(intervention_prediction_rows),
        },
    }
    return summary, dataset_rows, qa_rows


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    funnel = summary["window_funnel"]
    labels = summary["spatial_labels"]
    pairing = summary["qa_pairing"]
    lines = [
        "# H4 Discovery Attrition Diagnostic",
        "",
        "This audit is post-hoc and reads existing outputs only; it did not run model inference.",
        "",
        "## Window Funnel",
        f"- model windows: {funnel['n_model_windows']}",
        f"- temporally eligible windows: {funnel['n_temporally_eligible_windows']}",
        f"- temporal eligibility rate: {funnel['temporal_eligibility_rate']:.4f}",
        f"- support predictions, all windows: {funnel['support_prediction_counts_all_windows']}",
        f"- support predictions, temporally eligible: {funnel['support_prediction_counts_temporally_eligible_windows']}",
        f"- pointer outcomes, all windows: {funnel['pointer_counts_all_windows']}",
        f"- pointer outcomes, temporally eligible: {funnel['pointer_counts_temporally_eligible_windows']}",
        "",
        "## Spatial Labels",
        f"- all joined candidates: {labels['counts_all_joined_candidates']}",
        f"- temporally eligible joined candidates: {labels['counts_temporally_eligible_joined_candidates']}",
        f"- temporal-eligible IoU bins: {labels['iou_bins_temporally_eligible_joined_candidates']}",
        "",
        "## Pairing",
        f"- QA with TRUE spatial support: {pairing['n_qa_with_true_spatial_support']}",
        f"- QA with SPURIOUS spatial support: {pairing['n_qa_with_spurious_spatial_support']}",
        f"- primary paired QA: {pairing['n_primary_paired_qa']}",
        "",
        "No spatial threshold was changed, and no formal statistic was calculated.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit H4 50-QA discovery attrition from existing outputs only.")
    p.add_argument("--model_manifest", required=True)
    p.add_argument("--window_gt", required=True)
    p.add_argument("--support_predictions", required=True)
    p.add_argument("--spatial_pointer_predictions", required=True)
    p.add_argument("--joined_candidates", required=True)
    p.add_argument("--interventions", required=True)
    p.add_argument("--intervention_predictions", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.model_manifest)
    window_gt_rows = read_jsonl(args.window_gt)
    support_rows = read_jsonl(args.support_predictions)
    pointer_rows = read_jsonl(args.spatial_pointer_predictions)
    joined_candidate_rows = read_jsonl(args.joined_candidates)
    intervention_rows = read_jsonl(args.interventions)
    intervention_prediction_rows = read_jsonl(args.intervention_predictions)
    if not manifest_rows:
        raise RuntimeError("The discovery model manifest is empty.")

    summary, dataset_rows, qa_rows = build_audit(
        manifest_rows,
        window_gt_rows,
        support_rows,
        pointer_rows,
        joined_candidate_rows,
        intervention_rows,
        intervention_prediction_rows,
    )
    summary["inputs"] = vars(args)
    summary["repo"] = {
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
    }

    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary" / "h4_discovery_attrition_diagnostic.json", summary)
    write_summary_md(out_dir / "summary" / "h4_discovery_attrition_diagnostic.md", summary)
    write_csv(out_dir / "audit" / "h4_discovery_attrition_by_dataset.csv", dataset_rows)
    write_csv(out_dir / "audit" / "h4_discovery_attrition_by_qa.csv", qa_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
