#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.h2_statistics import exact_wilcoxon_signed_rank  # noqa: E402
from evidence_stability.utils import read_json, write_json  # noqa: E402


TEST_NAME = "exact_two_sided_wilcoxon_signed_rank"
ALPHA = 0.05


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def read_primary_qa_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Primary QA file is empty: {path}")
    required = {"qa_id", "target_field", "delta_macro"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Primary QA file is missing columns: {sorted(missing)}")
    qa_ids = [row["qa_id"] for row in rows]
    duplicates = sorted({qa_id for qa_id in qa_ids if qa_ids.count(qa_id) > 1})
    if duplicates:
        raise RuntimeError(f"Primary QA file contains duplicate qa_id values: {duplicates[:5]}")
    non_action = [row["qa_id"] for row in rows if row["target_field"] != "action"]
    if non_action:
        raise RuntimeError(f"Primary QA file contains non-action rows: {non_action[:5]}")
    for row in rows:
        try:
            row["delta_macro_float"] = float(row["delta_macro"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid delta_macro for QA {row['qa_id']}: {row['delta_macro']!r}") from exc
    return rows


def validate_phase_e_summary(path: Path, n_rows: int) -> dict[str, Any]:
    summary = read_json(path)
    failures = []
    if not summary.get("joined_intervention_results_available"):
        failures.append("joined_intervention_results_available")
    if not summary.get("real_h2_analysis_executed"):
        failures.append("real_h2_analysis_executed")
    if summary.get("primary_cohort") != "action-only paired QA":
        failures.append("primary_cohort")
    if summary.get("primary_rows") != n_rows:
        failures.append("primary_rows")
    protocol = summary.get("primary_h2_analysis", {})
    if protocol.get("preferred_summary_score") != "S_overall_macro":
        failures.append("preferred_summary_score")
    if summary.get("inferential_statistics_run"):
        failures.append("inferential_statistics_run_already_true")
    if failures:
        raise RuntimeError(f"Phase E summary is incompatible with the frozen H2 primary test: {failures}")
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    test = result["primary_test"]
    lines = [
        "# H2 Primary Inferential Test",
        "",
        "## Frozen Test",
        "- unit: QA",
        "- cohort: action-only paired QA",
        "- score: `S_overall_macro`",
        "- contrast: `TRUE_SUPPORT - SPURIOUS_SUPPORT`",
        "- test: exact two-sided Wilcoxon signed-rank (Wilcox zero handling; average ranks for ties)",
        "",
        "## Result",
        f"- paired QA: {test['n_pairs']}",
        f"- zero deltas excluded from rank test: {test['n_zero_deltas']}",
        f"- nonzero ranked pairs: {test['n_nonzero_deltas']}",
        f"- W+: {test['W_plus']}",
        f"- rank-biserial correlation: {test['rank_biserial_correlation']}",
        f"- mean delta: {test['mean_delta']}",
        f"- median delta: {test['median_delta']}",
        f"- exact two-sided p value: {test['p_value_two_sided_exact']}",
        f"- p < {ALPHA}: {test['reject_null_at_alpha_0_05']}",
        "",
        "This is the sole formal H2 test. All secondary cohorts and intervention-family summaries remain descriptive.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the frozen QA-level primary H2 signed-rank test.")
    p.add_argument("--primary_qa", required=True, help="Phase E qa/h2_primary_action_qa.csv")
    p.add_argument("--phase_e_summary", required=True, help="Phase E summary/phase_e_summary.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--expected_paired_qa", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    primary_qa_path = Path(args.primary_qa)
    summary_path = Path(args.phase_e_summary)
    rows = read_primary_qa_rows(primary_qa_path)
    if args.expected_paired_qa >= 0 and len(rows) != args.expected_paired_qa:
        raise RuntimeError(f"Expected {args.expected_paired_qa} paired QA, found {len(rows)}")
    phase_e_summary = validate_phase_e_summary(summary_path, len(rows))
    deltas = [float(row["delta_macro_float"]) for row in rows]
    test = exact_wilcoxon_signed_rank(deltas)
    test["test_name"] = TEST_NAME
    test["alternative"] = "two-sided"
    test["zero_method"] = "wilcox"
    test["tie_method"] = "average_rank"
    test["exact_distribution"] = "all sign assignments of nonzero paired ranks"
    test["alpha"] = ALPHA
    test["reject_null_at_alpha_0_05"] = test["p_value_two_sided_exact"] < ALPHA

    out_dir = Path(args.output_dir)
    audit_rows = []
    for row in rows:
        delta = float(row["delta_macro_float"])
        audit_rows.append(
            {
                "qa_id": row["qa_id"],
                "dataset_name": row.get("dataset_name"),
                "target_action_or_phase": row.get("target_action_or_phase"),
                "delta_S_macro": delta,
                "included_in_signed_rank": abs(delta) > 1e-12,
            }
        )
    result = {
        "analysis_protocol_version": phase_e_summary.get("analysis_protocol_version"),
        "stability_definition_version": phase_e_summary.get("stability_definition_version"),
        "formal_statistics_run": True,
        "independent_confirmation_run": False,
        "input": {
            "primary_qa": str(primary_qa_path),
            "primary_qa_sha256": sha256_file(primary_qa_path),
            "phase_e_summary": str(summary_path),
            "phase_e_summary_sha256": sha256_file(summary_path),
            "phase_c_model_identity": phase_e_summary.get("phase_c_model_identity"),
            "phase_d2_model_identity": phase_e_summary.get("phase_d2_model_identity"),
        },
        "primary_test": test,
        "secondary_tests_run": False,
        "git_commit": git_commit(),
    }
    write_csv(out_dir / "h2_primary_qa_delta_audit.csv", audit_rows)
    write_json(out_dir / "h2_primary_inferential_statistics.json", result)
    write_markdown(out_dir / "h2_primary_inferential_statistics.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
