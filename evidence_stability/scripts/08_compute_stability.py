#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.stability import (  # noqa: E402
    ANALYSIS_PROTOCOL_VERSION,
    CORE_CANDIDATE_LABELS,
    FAMILY_TYPES,
    STABILITY_DEFINITION_VERSION,
    analysis_protocol_snapshot,
    candidate_stability_rows,
    clean_mean,
    clean_median,
    is_nan,
    missingness_rows,
    phase_e_summary,
    qa_stability_rows,
    retention_matrix_rows,
    stability_by_group_rows,
)
from evidence_stability.utils import read_jsonl, write_json  # noqa: E402


READY_NO_REAL_DATA_MESSAGE = (
    "Full Phase D.2b intervention predictions are not available. "
    "Analysis implementation is ready; no real H2 computation was performed."
)


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_jsonl_safe(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def quantile(values: list[float], q: float) -> float:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[int(position)]
    return clean[lower] * (upper - position) + clean[upper] * (position - lower)


def stddev(values: list[float]) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
    if len(clean) < 2:
        return float("nan")
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / (len(clean) - 1))


def score_distribution_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ["S_shift", "S_resample", "S_context", "S_overall_macro", "S_overall_micro"]
    rows = []
    for candidate_type in CORE_CANDIDATE_LABELS:
        sub = [row for row in candidate_rows if row.get("candidate_type") == candidate_type]
        for field in fields:
            values = [row.get(field) for row in sub]
            clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
            rows.append(
                {
                    "candidate_type": candidate_type,
                    "score": field,
                    "n": len(clean),
                    "mean": clean_mean(clean),
                    "std": stddev(clean),
                    "median": clean_median(clean),
                    "Q1": quantile(clean, 0.25),
                    "Q3": quantile(clean, 0.75),
                    "min": min(clean) if clean else float("nan"),
                    "max": max(clean) if clean else float("nan"),
                }
            )
    return rows


def selected_candidate_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "candidate_id",
        "qa_id",
        "dataset_name",
        "target_action",
        "target_field",
        "candidate_type",
        "S_shift",
        "S_resample",
        "S_context",
        "S_overall_macro",
        "S_overall_micro",
        "n_valid_shift",
        "n_valid_resample",
        "n_valid_context",
        "n_valid_total",
        "n_yes_shift",
        "n_yes_resample",
        "n_yes_context",
        "n_yes_total",
    ]
    return [{field: row.get(field) for field in fields} for row in rows]


def paired_qa_ids_from_candidates(candidate_rows: list[dict[str, Any]]) -> set[str]:
    by_qa: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        if row.get("candidate_type") in CORE_CANDIDATE_LABELS:
            by_qa[str(row["qa_id"])].add(str(row["candidate_type"]))
    return {qa_id for qa_id, labels in by_qa.items() if set(CORE_CANDIDATE_LABELS) <= labels}


def load_frozen_paired_qa_ids(path: str | None) -> set[str] | None:
    if not path or not Path(path).exists():
        return None
    return {str(row["qa_id"]) for row in read_jsonl(path)}


def assert_frozen_paired_consistency(candidate_rows: list[dict[str, Any]], paired_qa_path: str | None) -> set[str]:
    recomputed = paired_qa_ids_from_candidates(candidate_rows)
    frozen = load_frozen_paired_qa_ids(paired_qa_path)
    if frozen is not None and recomputed != frozen:
        raise RuntimeError(
            "Recomputed paired QA set differs from frozen Phase C paired QA list: "
            f"missing_from_recomputed={sorted(frozen - recomputed)[:10]} "
            f"new_in_recomputed={sorted(recomputed - frozen)[:10]}"
        )
    return recomputed


def qa_descriptive_rows(qa_rows: list[dict[str, Any]], paired_ids: set[str], target_field: str | None = None) -> list[dict[str, Any]]:
    selected = [
        row for row in qa_rows
        if row["qa_id"] in paired_ids and (target_field is None or row.get("target_field") == target_field)
    ]
    fields = [
        "qa_id",
        "dataset_name",
        "target_field",
        "target_action_or_phase",
        "n_true_candidates",
        "n_spurious_candidates",
        "mean_true_macro",
        "mean_spurious_macro",
        "delta_macro",
        "mean_true_micro",
        "mean_spurious_micro",
        "delta_micro",
        "true_shift",
        "spurious_shift",
        "delta_shift",
        "true_resample",
        "spurious_resample",
        "delta_resample",
        "true_context",
        "spurious_context",
        "delta_context",
    ]
    return [{field: row.get(field) for field in fields} for row in selected]


def sign_counts(values: list[Any]) -> dict[str, int]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
    return {
        "positive": sum(v > 0 for v in clean),
        "zero": sum(v == 0 for v in clean),
        "negative": sum(v < 0 for v in clean),
    }


def cohort_summary(rows: list[dict[str, Any]], cohort_name: str, role: str) -> dict[str, Any]:
    delta_macro = [row.get("delta_macro") for row in rows]
    summary = {
        "cohort": cohort_name,
        "role": role,
        "n_paired_QA": len(rows),
        "datasets": dict(Counter(row.get("dataset_name") for row in rows)),
        "n_delta_macro_positive": sign_counts(delta_macro)["positive"],
        "n_delta_macro_zero": sign_counts(delta_macro)["zero"],
        "n_delta_macro_negative": sign_counts(delta_macro)["negative"],
        "mean_delta_macro": clean_mean(delta_macro),
        "median_delta_macro": clean_median(delta_macro),
        "min_delta_macro": min((float(v) for v in delta_macro if isinstance(v, (int, float)) and not is_nan(v)), default=float("nan")),
        "max_delta_macro": max((float(v) for v in delta_macro if isinstance(v, (int, float)) and not is_nan(v)), default=float("nan")),
        "mean_TRUE_stability": clean_mean([row.get("mean_true_macro") for row in rows]),
        "mean_SPURIOUS_stability": clean_mean([row.get("mean_spurious_macro") for row in rows]),
    }
    for family in ["shift", "resample", "context"]:
        deltas = [row.get(f"delta_{family}") for row in rows]
        counts = sign_counts(deltas)
        summary[f"{family}_positive"] = counts["positive"]
        summary[f"{family}_zero"] = counts["zero"]
        summary[f"{family}_negative"] = counts["negative"]
        summary[f"{family}_mean_delta"] = clean_mean(deltas)
        summary[f"{family}_median_delta"] = clean_median(deltas)
    return summary


def write_h2_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {summary['cohort']} Summary",
        "",
        f"- role: {summary['role']}",
        f"- paired QA: {summary['n_paired_QA']}",
        f"- datasets: {summary['datasets']}",
        f"- delta_macro positive: {summary['n_delta_macro_positive']}",
        f"- delta_macro zero: {summary['n_delta_macro_zero']}",
        f"- delta_macro negative: {summary['n_delta_macro_negative']}",
        f"- mean delta_macro: {summary['mean_delta_macro']}",
        f"- median delta_macro: {summary['median_delta_macro']}",
        f"- SHIFT median delta: {summary['shift_median_delta']}",
        f"- RESAMPLE median delta: {summary['resample_median_delta']}",
        f"- CONTEXT median delta: {summary['context_median_delta']}",
        "",
        "Descriptive QA-level paired aggregate only. No inferential statistics were run.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def yes_stickiness_rows(joined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict_rows = [
        row for row in joined_rows
        if row.get("generation_valid")
        and row.get("strict_valid")
        and row.get("candidate_type") in CORE_CANDIDATE_LABELS
        and row.get("intervention_prediction") in {"YES", "NO"}
    ]
    specs = {
        "candidate_type": lambda row: row.get("candidate_type"),
        "intervention_type": lambda row: row.get("intervention_type"),
        "intervention_family": lambda row: row.get("intervention_family"),
        "target_field": lambda row: row.get("target_field"),
        "dataset": lambda row: row.get("dataset_name"),
        "candidate_type_x_intervention_type": lambda row: f"{row.get('candidate_type')}::{row.get('intervention_type')}",
    }
    out = []
    for dimension, getter in specs.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in strict_rows:
            groups[str(getter(row))].append(row)
        for group, rows in sorted(groups.items()):
            yes = sum(row.get("intervention_prediction") == "YES" for row in rows)
            no = sum(row.get("intervention_prediction") == "NO" for row in rows)
            out.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n_strict_valid": len(rows),
                    "YES": yes,
                    "NO": no,
                    "yes_retention_rate": yes / len(rows) if rows else float("nan"),
                }
            )
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def protocol_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "stability_version": STABILITY_DEFINITION_VERSION,
        "candidate_label_source": args.joined_results,
        "intervention_manifest_path": args.intervention_manifest,
        "prediction_source_path": args.prediction_source,
        "strict_validity_definition_source": args.intervention_manifest,
        "primary_cohort_definition": "action-only paired QA from frozen Phase C paired QA set",
        "primary_unit": "QA",
        "primary_target_field": "action",
        "primary_score": "S_overall_macro",
        "secondary_score": "S_overall_micro",
        "intervention_families": FAMILY_TYPES,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "threshold_selection": None,
        "learned_weights": None,
        "inferential_statistics_run": False,
    }


def write_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    if not summary.get("joined_intervention_results_available", True):
        lines = [
            "# Phase E Stability Summary",
            "",
            READY_NO_REAL_DATA_MESSAGE,
            "",
            "No real H2 stability computation was performed.",
        ]
    else:
        protocol = summary.get("primary_h2_analysis", {})
        lines = [
            "# Phase E Stability Summary",
            "",
            "## Protocol",
            f"- stability definition: {summary.get('stability_definition_version')}",
            f"- analysis protocol: {summary.get('analysis_protocol_version')}",
            f"- validity mode: {summary.get('validity_mode')}",
            f"- primary target type: {protocol.get('target_type')}",
            f"- primary unit: {protocol.get('primary_unit')}",
            f"- preferred score: {protocol.get('preferred_summary_score')}",
            "",
            "## Counts",
            f"- candidates: {summary.get('n_candidates')}",
            f"- TRUE_SUPPORT candidates: {summary.get('n_true_candidates')}",
            f"- SPURIOUS_SUPPORT candidates: {summary.get('n_spurious_candidates')}",
            f"- QA: {summary.get('n_qa')}",
            f"- action QA: {summary.get('n_action_qa')}",
            f"- phase QA: {summary.get('n_phase_qa')}",
            f"- paired macro usable QA: {summary.get('paired_macro_usable_qa')}",
            f"- paired shift usable QA: {summary.get('paired_shift_analysis_usable_qa')}",
            f"- paired resample usable QA: {summary.get('paired_resample_analysis_usable_qa')}",
            f"- paired context usable QA: {summary.get('paired_context_analysis_usable_qa')}",
            "",
            "Candidate-window analysis is secondary because windows within a QA are correlated. "
            "No inferential statistics were run by this script.",
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_threshold(value: str | None) -> float | None:
    return None if value is None else float(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase E post-hoc intervention stability computation.")
    p.add_argument("--joined_results", default="outputs/tal_pilot/phase_e/h2_stability_v1/d3/joined_intervention_results.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_e/h2_stability_v1")
    p.add_argument("--validity_mode", choices=["strict", "any_gt"], default="strict")
    p.add_argument("--paired_qa", default=None)
    p.add_argument("--intervention_manifest", default="outputs/tal_pilot/phase_d/intervention_pilot/interventions.jsonl")
    p.add_argument("--prediction_source", default="outputs/tal_pilot/phase_d/qwen3_vl_8b_intervention_full/intervention_probe_results.jsonl")
    p.add_argument("--stable_hallucination_threshold", default=None)
    p.add_argument("--fragile_true_threshold", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stable_hallucination_threshold is not None or args.fragile_true_threshold is not None:
        raise RuntimeError("Threshold selection is frozen off for Phase E descriptive analysis.")
    out_dir = Path(args.output_dir)
    candidate_dir = out_dir / "candidate"
    qa_dir = out_dir / "qa"
    summary_dir = out_dir / "summary"
    provenance_dir = out_dir / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "phase_e_summary.json"
    summary_md_path = summary_dir / "phase_e_summary.md"
    protocol = analysis_protocol_snapshot(args.validity_mode)
    write_json(provenance_dir / "analysis_protocol_snapshot.json", protocol)

    if not Path(args.joined_results).exists():
        summary = {
            "joined_intervention_results_available": False,
            "message": READY_NO_REAL_DATA_MESSAGE,
            "joined_results": args.joined_results,
            "real_h2_analysis_executed": False,
            **protocol,
        }
        write_json(summary_path, summary)
        write_summary_md(summary_md_path, summary)
        print(READY_NO_REAL_DATA_MESSAGE)
        return

    joined_rows = read_jsonl(args.joined_results)
    candidate_rows = candidate_stability_rows(joined_rows, validity_mode=args.validity_mode)
    candidate_ids = [row["candidate_id"] for row in candidate_rows]
    duplicate_candidate_ids = [value for value, count in Counter(candidate_ids).items() if count > 1]
    if duplicate_candidate_ids:
        raise RuntimeError(f"Duplicate candidate_id in candidate stability rows: {duplicate_candidate_ids[:5]}")
    for row in candidate_rows:
        for field in ["S_shift", "S_resample", "S_context", "S_overall_macro", "S_overall_micro"]:
            value = row.get(field)
            if isinstance(value, (int, float)) and not is_nan(value) and not (0 <= value <= 1):
                raise RuntimeError(f"Score out of range for {row['candidate_id']} {field}: {value}")
    qa_rows = qa_stability_rows(candidate_rows)
    paired_ids = assert_frozen_paired_consistency(candidate_rows, args.paired_qa)
    primary_rows = qa_descriptive_rows(qa_rows, paired_ids, "action")
    all_paired_rows = qa_descriptive_rows(qa_rows, paired_ids, None)
    phase_only_rows = qa_descriptive_rows(qa_rows, paired_ids, "phase")
    missing_rows = missingness_rows(joined_rows, candidate_rows, validity_mode=args.validity_mode)
    dataset_rows = stability_by_group_rows(candidate_rows, qa_rows, "dataset_name")
    target_rows = stability_by_group_rows(candidate_rows, qa_rows, "target_field")
    gt_duration_rows = stability_by_group_rows(candidate_rows, qa_rows, "gt_duration_bin")
    matrix_rows = retention_matrix_rows(candidate_rows)
    distribution_rows = score_distribution_rows(candidate_rows)
    spurious_high = sorted(
        [row for row in candidate_rows if row.get("candidate_type") == "SPURIOUS_SUPPORT"],
        key=lambda row: (-1 if is_nan(row["S_overall_macro"]) else -row["S_overall_macro"], row["candidate_id"]),
    )
    true_low = sorted(
        [row for row in candidate_rows if row.get("candidate_type") == "TRUE_SUPPORT"],
        key=lambda row: (float("inf") if is_nan(row["S_overall_macro"]) else row["S_overall_macro"], row["candidate_id"]),
    )
    primary_summary = cohort_summary(primary_rows, "h2_primary_action", "PRIMARY")
    all_paired_summary = cohort_summary(all_paired_rows, "h2_all_paired", "SECONDARY")
    phase_only_summary = cohort_summary(phase_only_rows, "h2_phase_only", "EXPLORATORY / GENERALIZATION")
    stickiness_rows = yes_stickiness_rows(joined_rows)
    stickiness_summary = {
        "diagnostic": "YES retention among strict-valid interventions only",
        "rows": stickiness_rows,
    }
    summary = {
        "joined_intervention_results_available": True,
        "joined_results": args.joined_results,
        "real_h2_analysis_executed": False,
        "inferential_statistics_run": False,
        "threshold_selection_run": False,
        "learned_weights": None,
        "primary_cohort": "action-only paired QA",
        "secondary_cohorts": ["all paired QA", "phase-only paired QA"],
        "primary_rows": len(primary_rows),
        "all_paired_rows": len(all_paired_rows),
        "phase_only_rows": len(phase_only_rows),
        "candidate_macro_nan": sum(is_nan(row["S_overall_macro"]) for row in candidate_rows),
        "candidate_micro_nan": sum(is_nan(row["S_overall_micro"]) for row in candidate_rows),
        "primary_action_summary": primary_summary,
        "all_paired_summary": all_paired_summary,
        "phase_only_summary": phase_only_summary,
        **phase_e_summary(candidate_rows, qa_rows, args.validity_mode),
    }
    summary.update(
        {
            "phase_c_model_identity": next((row.get("model_revision") for row in joined_rows if row.get("model_revision")), None),
            "phase_d2_model_identity": next((row.get("model_identity_hash") for row in joined_rows if row.get("model_identity_hash")), None),
            "prompt_versions": sorted({row.get("prompt_version") for row in joined_rows if row.get("prompt_version")}),
            "prompt_hashes": sorted({row.get("prompt_hash") for row in joined_rows if row.get("prompt_hash")}),
            "protocol_snapshot": protocol,
        }
    )

    write_jsonl_safe(candidate_dir / "candidate_stability.jsonl", candidate_rows)
    write_csv(candidate_dir / "candidate_stability.csv", candidate_rows)
    write_csv(candidate_dir / "candidate_stability_distribution.csv", distribution_rows)
    write_csv(candidate_dir / "spurious_high_stability.csv", selected_candidate_columns(spurious_high))
    write_csv(candidate_dir / "true_low_stability.csv", selected_candidate_columns(true_low))
    write_csv(qa_dir / "h2_primary_action_qa.csv", primary_rows)
    write_csv(qa_dir / "h2_all_paired_qa.csv", all_paired_rows)
    write_csv(qa_dir / "h2_phase_only_qa.csv", phase_only_rows)
    write_csv(out_dir / "candidate_stability.csv", candidate_rows)
    write_csv(out_dir / "qa_stability.csv", qa_rows)
    write_csv(out_dir / "stability_missingness.csv", missing_rows)
    write_csv(out_dir / "stability_by_dataset.csv", dataset_rows)
    write_csv(out_dir / "stability_by_target_type.csv", target_rows)
    write_csv(out_dir / "stability_by_gt_duration.csv", gt_duration_rows)
    write_csv(out_dir / "intervention_retention_matrix.csv", matrix_rows)
    write_json(summary_dir / "h2_primary_action_summary.json", json_safe(primary_summary))
    write_json(summary_dir / "h2_all_paired_summary.json", json_safe(all_paired_summary))
    write_json(summary_dir / "h2_phase_only_summary.json", json_safe(phase_only_summary))
    write_h2_summary_md(summary_dir / "h2_primary_action_summary.md", primary_summary)
    write_h2_summary_md(summary_dir / "h2_all_paired_summary.md", all_paired_summary)
    write_h2_summary_md(summary_dir / "h2_phase_only_summary.md", phase_only_summary)
    write_json(summary_dir / "yes_stickiness_diagnostic.json", json_safe(stickiness_summary))
    write_csv(summary_dir / "yes_stickiness_diagnostic.csv", stickiness_rows)
    write_json(provenance_dir / "h2_protocol_provenance.json", json_safe(protocol_provenance(args)))
    write_json(summary_path, json_safe(summary))
    write_summary_md(summary_md_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
