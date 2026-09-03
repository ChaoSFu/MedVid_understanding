#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.stability import (  # noqa: E402
    analysis_protocol_snapshot,
    candidate_stability_rows,
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
    p.add_argument("--joined_results", default="outputs/tal_pilot/phase_d/analysis/intervention_labeled_results.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_e")
    p.add_argument("--validity_mode", choices=["strict", "any_gt"], default="strict")
    p.add_argument("--stable_hallucination_threshold", default=None)
    p.add_argument("--fragile_true_threshold", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase_e_summary.json"
    summary_md_path = out_dir / "phase_e_summary.md"
    protocol = analysis_protocol_snapshot(args.validity_mode)
    write_json(out_dir / "analysis_protocol_snapshot.json", protocol)

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
    candidate_rows = candidate_stability_rows(
        joined_rows,
        validity_mode=args.validity_mode,
        stable_hallucination_threshold=parse_threshold(args.stable_hallucination_threshold),
        fragile_true_threshold=parse_threshold(args.fragile_true_threshold),
    )
    qa_rows = qa_stability_rows(candidate_rows)
    missing_rows = missingness_rows(joined_rows, candidate_rows, validity_mode=args.validity_mode)
    dataset_rows = stability_by_group_rows(candidate_rows, qa_rows, "dataset_name")
    target_rows = stability_by_group_rows(candidate_rows, qa_rows, "target_field")
    gt_duration_rows = stability_by_group_rows(candidate_rows, qa_rows, "gt_duration_bin")
    matrix_rows = retention_matrix_rows(candidate_rows)
    summary = {
        "joined_intervention_results_available": True,
        "joined_results": args.joined_results,
        "real_h2_analysis_executed": False,
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

    write_csv(out_dir / "candidate_stability.csv", candidate_rows)
    write_csv(out_dir / "qa_stability.csv", qa_rows)
    write_csv(out_dir / "stability_missingness.csv", missing_rows)
    write_csv(out_dir / "stability_by_dataset.csv", dataset_rows)
    write_csv(out_dir / "stability_by_target_type.csv", target_rows)
    write_csv(out_dir / "stability_by_gt_duration.csv", gt_duration_rows)
    write_csv(out_dir / "intervention_retention_matrix.csv", matrix_rows)
    write_json(summary_path, summary)
    write_summary_md(summary_md_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
