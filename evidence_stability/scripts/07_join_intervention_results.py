#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.join import d2_final_prediction_audit, join_intervention_results  # noqa: E402
from evidence_stability.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


READY_NO_REAL_DATA_MESSAGE = (
    "Full Phase D.2b intervention predictions are not available. "
    "Analysis implementation is ready; no real H2 computation was performed."
)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    if not summary.get("full_phase_d2b_predictions_available", True):
        lines = [
            "# Phase D.3 Join Summary",
            "",
            READY_NO_REAL_DATA_MESSAGE,
            "",
            "No intervention join, stability computation, or H2 analysis was performed.",
        ]
    else:
        lines = [
            "# Phase D.3 Join Summary",
            "",
            "## Join",
            f"- manifest total: {summary.get('n_manifest_total')}",
            f"- generation-valid manifest: {summary.get('n_generation_valid_manifest')}",
            f"- generation-invalid manifest: {summary.get('n_generation_invalid_manifest')}",
            f"- probe results: {summary.get('n_probe_results')}",
            f"- matched: {summary.get('n_matched')}",
            f"- missing generation-valid predictions: {summary.get('n_missing_generation_valid_predictions')}",
            f"- extra predictions: {summary.get('n_extra_predictions')}",
            f"- historical error records: {summary.get('historical_error_records')}",
            f"- historical errors resolved by final result: {summary.get('historical_errors_resolved_by_final_result')}",
            f"- unresolved errors: {summary.get('unresolved_errors')}",
            "",
            "## Audit",
            f"- duplicate manifest IDs: {summary.get('n_duplicate_intervention_ids_manifest')}",
            f"- duplicate probe IDs: {summary.get('n_duplicate_intervention_ids_probe')}",
            f"- TRUE_SUPPORT origin interventions: {summary.get('n_true_origin_interventions')}",
            f"- SPURIOUS_SUPPORT origin interventions: {summary.get('n_spurious_origin_interventions')}",
            "",
            "D.3 is join/audit only. No stability interpretation was performed.",
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_d2_final_audit_md(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Phase D.2b Final Audit",
        "",
        "Phase D.2b full run completed successfully."
        if audit.get("d2b_full_pass")
        else "Phase D.2b final audit did not pass.",
        "",
        f"- total interventions: {audit.get('n_total_interventions')}",
        f"- generation-valid interventions: {audit.get('n_generation_valid')}",
        f"- generation-invalid interventions: {audit.get('n_generation_invalid')}",
        f"- final predictions: {audit.get('n_final_predictions')}",
        f"- missing generation-valid predictions: {audit.get('missing_generation_valid_predictions')}",
        f"- extra predictions: {audit.get('extra_predictions')}",
        f"- duplicate prediction intervention IDs: {audit.get('duplicate_prediction_intervention_ids')}",
        f"- prediction counts: {audit.get('prediction_counts')}",
        f"- historical error records: {audit.get('historical_error_records')}",
        f"- historical errors resolved by final result: {audit.get('historical_errors_resolved_by_final_result')}",
        f"- unresolved errors: {audit.get('unresolved_errors')}",
        "",
        "errors.jsonl is historical engineering provenance only; it is not used as the authoritative prediction source.",
        "No Phase D.3 or Phase E result is included in this D2 final audit.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_d2_final_audit(
    audit: dict[str, Any],
    *,
    expected_total_interventions: int,
    expected_generation_valid: int,
    expected_yes: int,
    expected_no: int,
) -> None:
    failures = []
    if expected_total_interventions >= 0 and audit["n_total_interventions"] != expected_total_interventions:
        failures.append("n_total_interventions")
    if expected_generation_valid >= 0 and audit["n_generation_valid"] != expected_generation_valid:
        failures.append("n_generation_valid")
    if audit["n_final_predictions"] != audit["n_generation_valid"]:
        failures.append("n_final_predictions")
    if audit["missing_generation_valid_predictions"] != 0:
        failures.append("missing_generation_valid_predictions")
    if audit["extra_predictions"] != 0:
        failures.append("extra_predictions")
    if audit["predictions_for_generation_invalid"] != 0:
        failures.append("predictions_for_generation_invalid")
    if audit["duplicate_prediction_intervention_ids"] != 0:
        failures.append("duplicate_prediction_intervention_ids")
    if not audit["prediction_values_only_yes_no"]:
        failures.append("prediction_values_only_yes_no")
    counts = audit["prediction_counts"]
    if expected_yes >= 0 and counts.get("YES", 0) != expected_yes:
        failures.append("YES")
    if expected_no >= 0 and counts.get("NO", 0) != expected_no:
        failures.append("NO")
    if audit["unresolved_errors"] != 0:
        failures.append("unresolved_errors")
    if failures:
        raise RuntimeError(f"Phase D.2 final preflight failed: {failures}; audit={audit}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D.3 post-hoc intervention prediction join.")
    p.add_argument("--manifest", default="outputs/tal_pilot/phase_d/intervention_pilot/interventions.jsonl")
    p.add_argument("--probe_results", default="outputs/tal_pilot/phase_d/qwen3_vl_8b_intervention_full/intervention_probe_results.jsonl")
    p.add_argument("--errors", default="outputs/tal_pilot/phase_d/qwen3_vl_8b_intervention_full/errors.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_e/h2_stability_v1/d3")
    p.add_argument("--joined_output", default=None)
    p.add_argument("--summary_output", default=None)
    p.add_argument("--d2_final_audit_output", default=None)
    p.add_argument("--expected_total_interventions", type=int, default=-1)
    p.add_argument("--expected_generation_valid", type=int, default=-1)
    p.add_argument("--expected_yes", type=int, default=-1)
    p.add_argument("--expected_no", type=int, default=-1)
    p.add_argument("--allow_incomplete", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joined_output = args.joined_output or str(out_dir / "joined_intervention_results.jsonl")
    summary_output = args.summary_output or str(out_dir / "phase_d3_join_audit.json")
    summary_md_output = str(Path(summary_output).with_suffix(".md"))
    d2_audit_output = args.d2_final_audit_output or str(out_dir / "phase_d2_final_audit.json")
    d2_audit_md_output = str(Path(d2_audit_output).with_suffix(".md"))

    if not Path(args.probe_results).exists():
        summary = {
            "full_phase_d2b_predictions_available": False,
            "message": READY_NO_REAL_DATA_MESSAGE,
            "manifest": args.manifest,
            "probe_results": args.probe_results,
            "errors": args.errors,
            "real_h2_analysis_executed": False,
        }
        write_json(summary_output, summary)
        write_summary_md(summary_md_output, summary)
        print(READY_NO_REAL_DATA_MESSAGE)
        return

    manifest_rows = read_jsonl(args.manifest)
    probe_rows = read_jsonl(args.probe_results)
    error_rows = read_jsonl(args.errors) if Path(args.errors).exists() else []
    d2_audit = d2_final_prediction_audit(manifest_rows, probe_rows, error_rows)
    d2_audit["d2b_full_pass"] = (
        d2_audit["n_final_predictions"] == d2_audit["n_generation_valid"]
        and d2_audit["missing_generation_valid_predictions"] == 0
        and d2_audit["extra_predictions"] == 0
        and d2_audit["predictions_for_generation_invalid"] == 0
        and d2_audit["duplicate_prediction_intervention_ids"] == 0
        and d2_audit["prediction_values_only_yes_no"]
        and d2_audit["unresolved_errors"] == 0
    )
    write_json(d2_audit_output, d2_audit)
    write_d2_final_audit_md(d2_audit_md_output, d2_audit)
    assert_d2_final_audit(
        d2_audit,
        expected_total_interventions=args.expected_total_interventions,
        expected_generation_valid=args.expected_generation_valid,
        expected_yes=args.expected_yes,
        expected_no=args.expected_no,
    )
    expected = args.expected_generation_valid if args.expected_generation_valid >= 0 else None
    joined, summary = join_intervention_results(
        manifest_rows,
        probe_rows,
        error_rows=error_rows,
        expected_generation_valid=expected,
        require_complete=not args.allow_incomplete,
    )
    summary = {
        "full_phase_d2b_predictions_available": True,
        "manifest": args.manifest,
        "probe_results": args.probe_results,
        "errors": args.errors,
        "joined_output": joined_output,
        "real_h2_analysis_executed": False,
        **summary,
    }
    write_jsonl(joined_output, joined)
    write_json(summary_output, summary)
    write_summary_md(summary_md_output, summary)
    write_csv(out_dir / "phase_d3_join_by_intervention_type.csv", summary["by_intervention_type"])
    write_csv(out_dir / "phase_d3_join_by_candidate_type_and_intervention_type.csv", summary["counts_by_candidate_type_and_intervention_type"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
