#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.dynamic import (  # noqa: E402
    BINARY_PREDICTIONS,
    CORE_CANDIDATE_TYPES,
    INTERVENTION_TYPES,
    PROTOCOL_VERSION,
    candidate_dynamic_rows,
    clean_mean,
    clean_median,
    cohort_summary,
    final_records_by_intervention,
    is_nan,
    persistence_diagnostic_rows,
    protocol_provenance,
    qa_dynamic_rows,
)
from evidence_stability.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


READY_NO_REAL_DATA_MESSAGE = (
    "Full H3 dynamic predictions are not available. "
    "Analysis implementation is ready; no real H3 descriptive computation was performed."
)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, (dict, list)):
        return json.dumps(json_safe(value), ensure_ascii=False)
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def write_jsonl_safe(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def raw_result_is_gt_free(row: dict[str, Any]) -> bool:
    serialized = json.dumps(row, ensure_ascii=False).lower()
    forbidden = ["candidate_type", "true_support", "spurious_support", "gt_alignment", "evidence_density"]
    return not any(term in serialized for term in forbidden)


def join_h3_results(
    manifest: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    expected_ids = {row["intervention_id"] for row in manifest if row.get("generation_valid")}
    final_results, completion = final_records_by_intervention(result_rows, error_rows, expected_ids)
    manifest_ids = [row["intervention_id"] for row in manifest]
    duplicate_manifest = [value for value, count in Counter(manifest_ids).items() if count > 1]
    if duplicate_manifest:
        raise RuntimeError(f"Duplicate H3 manifest intervention IDs: {duplicate_manifest[:5]}")
    if any(not raw_result_is_gt_free(row) for row in result_rows):
        raise RuntimeError("GT-aware/candidate-type field leaked into H3 raw prediction rows.")

    joined = []
    for row in manifest:
        candidate = candidate_by_id.get(row["candidate_id"])
        if candidate is None:
            raise RuntimeError(f"Missing candidate reconstruction metadata: {row['candidate_id']}")
        result = final_results.get(row["intervention_id"])
        prediction = result.get("prediction") if result else "ERROR"
        joined.append(
            {
                "intervention_id": row["intervention_id"],
                "candidate_id": row["candidate_id"],
                "qa_id": row["qa_id"],
                "clip_id": row.get("clip_id"),
                "dataset_name": row.get("dataset_name"),
                "target_action": row.get("target_action"),
                "target_field": row.get("target_field"),
                "intervention_type": row.get("intervention_type"),
                "intervention_family": row.get("intervention_family"),
                "n_frames": row.get("n_frames"),
                "ordered_frame_hash": result.get("ordered_frame_hash") if result else None,
                "generation_valid": bool(row.get("generation_valid")),
                "generation_reason": row.get("generation_reason"),
                "prediction": prediction,
                "inference_valid": bool(result and result.get("inference_valid")),
                "raw_response": result.get("raw_response") if result else None,
                "cache_key": result.get("cache_key") if result else None,
                "model_identity_hash": result.get("model_identity_hash") if result else None,
                "prompt_hash": result.get("prompt_hash") if result else None,
                "decoding_config": result.get("decoding_config") if result else None,
                "candidate_type": candidate.get("candidate_type"),
                "original_prediction": candidate.get("original_prediction"),
            }
        )

    counts = Counter(row["candidate_type"] for row in candidates)
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "expected_generation_valid": len(expected_ids),
        "n_joined_rows": len(joined),
        "n_candidates": len(candidates),
        "candidate_type_counts": dict(counts),
        "prediction_counts": dict(Counter(row["prediction"] for row in joined)),
        "raw_predictions_gt_free": all(raw_result_is_gt_free(row) for row in result_rows),
        "completion": completion,
        "generation_invalid": sum(not bool(row.get("generation_valid")) for row in manifest),
    }
    return joined, audit


def assert_frozen_h3_counts(candidates: list[dict[str, Any]], qa_rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["candidate_type"] for row in candidates)
    if len(candidates) != 158 or counts["TRUE_SUPPORT"] != 64 or counts["SPURIOUS_SUPPORT"] != 94:
        raise RuntimeError(f"H3 frozen candidate counts changed: total={len(candidates)} counts={dict(counts)}")
    if Counter(row.get("original_prediction") for row in candidates) != Counter({"YES": 158}):
        raise RuntimeError("H3 original candidate predictions are not all frozen YES.")
    if len(qa_rows) != 16:
        raise RuntimeError(f"H3 all paired QA count changed: {len(qa_rows)}")
    if sum(row.get("target_field") == "action" for row in qa_rows) != 9:
        raise RuntimeError("H3 primary action QA count changed.")
    if sum(row.get("target_field") == "phase" for row in qa_rows) != 7:
        raise RuntimeError("H3 phase-only QA count changed.")


def selected_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "candidate_id",
        "qa_id",
        "dataset_name",
        "target_action",
        "target_field",
        "candidate_type",
        "original_prediction",
        "pred_full_shuffle",
        "pred_block_shuffle",
        "pred_freeze",
        "D_full_shuffle",
        "D_block_shuffle",
        "D_order",
        "D_freeze",
        "D_macro_exploratory",
        "n_dynamic_flips",
    ]
    return [{field: row.get(field) for field in fields} for row in rows]


def diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for candidate_type in CORE_CANDIDATE_TYPES:
        for intervention_type in INTERVENTION_TYPES:
            sub = [
                row for row in rows
                if row.get("candidate_type") == candidate_type
                and row.get("intervention_type") == intervention_type
                and row.get("prediction") in BINARY_PREDICTIONS
            ]
            yes = sum(row["prediction"] == "YES" for row in sub)
            no = sum(row["prediction"] == "NO" for row in sub)
            key = f"{candidate_type}::{intervention_type}"
            out[key] = {
                "n": len(sub),
                "YES": yes,
                "NO": no,
                "P_YES": yes / len(sub) if sub else None,
                "dynamic_sensitivity_rate": no / len(sub) if sub else None,
            }
    return out


def write_h3_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# H3 Phase F-D Discovery Summary",
        "",
        "DESCRIPTIVE RESULT.",
        "",
        f"- protocol version: {summary['protocol']['protocol_version']}",
        f"- study stage: {summary['protocol']['study_stage']}",
        f"- primary score: {summary['protocol']['primary_score']}",
        f"- training: {not summary['protocol']['no_training']}",
        f"- total candidates: {summary['candidate_cohort']['total']}",
        f"- TRUE: {summary['candidate_cohort']['TRUE_SUPPORT']}",
        f"- SPURIOUS: {summary['candidate_cohort']['SPURIOUS_SUPPORT']}",
        f"- primary action QA: {summary['candidate_cohort']['primary_action_qa']}",
        f"- all paired QA: {summary['candidate_cohort']['all_paired_qa']}",
        f"- phase-only QA: {summary['candidate_cohort']['phase_only_qa']}",
        f"- attempted interventions: {summary['interventions']['attempted']}",
        f"- generation valid: {summary['interventions']['generation_valid']}",
        f"- predictions: {summary['inference']['predictions']}",
        f"- YES: {summary['inference']['YES']}",
        f"- NO: {summary['inference']['NO']}",
        f"- INVALID: {summary['inference']['INVALID']}",
        f"- ERROR: {summary['inference']['ERROR']}",
        "",
        "## Primary H3 - Action QA",
        f"- paired QA: {summary['primary_h3_action_qa']['paired_QA']}",
        f"- delta_D_order positive: {summary['primary_h3_action_qa']['delta_D_order_positive']}",
        f"- delta_D_order zero: {summary['primary_h3_action_qa']['delta_D_order_zero']}",
        f"- delta_D_order negative: {summary['primary_h3_action_qa']['delta_D_order_negative']}",
        f"- mean delta_D_order: {summary['primary_h3_action_qa']['mean_delta_D_order']}",
        f"- median delta_D_order: {summary['primary_h3_action_qa']['median_delta_D_order']}",
        "",
        "Formal statistics run: false.",
        "",
        "Independent confirmation run: false.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cohort_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {summary['cohort']} H3 Dynamic Summary",
        "",
        "DESCRIPTIVE RESULT.",
        "",
        f"- role: {summary['role']}",
        f"- QA: {summary['n_QA']}",
        f"- delta_D_order positive: {summary['delta_D_order_positive']}",
        f"- delta_D_order zero: {summary['delta_D_order_zero']}",
        f"- delta_D_order negative: {summary['delta_D_order_negative']}",
        f"- mean delta_D_order: {summary['mean_delta_D_order']}",
        f"- median delta_D_order: {summary['median_delta_D_order']}",
        f"- full shuffle median delta: {summary['median_full_shuffle_delta']}",
        f"- block shuffle median delta: {summary['median_block_shuffle_delta']}",
        f"- freeze median delta: {summary['median_freeze_delta']}",
        "",
        "No formal statistics or independent confirmation were run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_join_audit_md(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# H3 Dynamic Join Audit",
        "",
        f"- expected generation-valid interventions: {audit.get('expected_generation_valid')}",
        f"- joined rows: {audit.get('n_joined_rows')}",
        f"- candidates: {audit.get('n_candidates')}",
        f"- candidate type counts: {audit.get('candidate_type_counts')}",
        f"- prediction counts: {audit.get('prediction_counts')}",
        f"- raw predictions GT-free: {audit.get('raw_predictions_gt_free')}",
        f"- generation invalid: {audit.get('generation_invalid')}",
        f"- completion: {audit.get('completion')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc H3 dynamic join and descriptive analysis.")
    p.add_argument("--manifest", default="outputs/tal_pilot/phase_f/h3_dynamic_v1/manifest/h3_model_intervention_manifest_gt_free.jsonl")
    p.add_argument("--candidate_reconstruction", default="outputs/tal_pilot/phase_f/h3_dynamic_v1/manifest/frozen_candidate_reconstruction.jsonl")
    p.add_argument("--paired_qa", default="outputs/tal_pilot/phase_c_server/qwen3_vl_8b_pilot50/paired_h2_qa.jsonl")
    p.add_argument("--probe_results", default="outputs/tal_pilot/phase_f/h3_dynamic_v1/predictions/full/h3_dynamic_probe_results.jsonl")
    p.add_argument("--errors", default="outputs/tal_pilot/phase_f/h3_dynamic_v1/predictions/full/h3_dynamic_errors.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_f/h3_dynamic_v1")
    p.add_argument("--allow_incomplete", action="store_true")
    p.add_argument("--model_specific_cohort", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["joined", "candidate", "qa", "summary", "audit", "provenance"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    if not Path(args.probe_results).exists():
        summary = {
            "full_h3_predictions_available": False,
            "message": READY_NO_REAL_DATA_MESSAGE,
            "real_h3_analysis_executed": False,
            "formal_statistics_run": False,
            "independent_confirmation_run": False,
        }
        write_json(out_dir / "summary" / "h3_dynamic_discovery_summary.json", summary)
        print(READY_NO_REAL_DATA_MESSAGE)
        return

    manifest = read_jsonl(args.manifest)
    candidates = read_jsonl(args.candidate_reconstruction)
    paired_qa = read_jsonl(args.paired_qa)
    if not args.model_specific_cohort:
        assert_frozen_h3_counts(candidates, paired_qa)
    result_rows = read_jsonl(args.probe_results)
    error_rows = read_jsonl(args.errors) if Path(args.errors).exists() else []
    joined, join_audit = join_h3_results(manifest, candidates, result_rows, error_rows)
    if not args.allow_incomplete:
        completion = join_audit["completion"]
        expected_interventions = len([row for row in manifest if row.get("generation_valid")])
        if (
            completion["n_final_success_results"] != expected_interventions
            or completion["unresolved_errors"]
            or completion["missing"]
            or completion["extra"]
        ):
            raise RuntimeError(f"H3 full prediction completion audit failed: {completion}")
        if join_audit["prediction_counts"].get("INVALID", 0) or join_audit["prediction_counts"].get("ERROR", 0):
            raise RuntimeError(f"H3 full predictions contain INVALID/ERROR: {join_audit['prediction_counts']}")

    candidate_rows = candidate_dynamic_rows(joined)
    qa_rows = qa_dynamic_rows(candidate_rows, paired_qa)
    primary_rows = [row for row in qa_rows if row.get("target_field") == "action"]
    all_paired_rows = list(qa_rows)
    phase_only_rows = [row for row in qa_rows if row.get("target_field") == "phase"]
    if not args.model_specific_cohort and (len(primary_rows) != 9 or len(all_paired_rows) != 16 or len(phase_only_rows) != 7):
        raise RuntimeError(
            f"H3 QA cohort counts changed: primary={len(primary_rows)} all={len(all_paired_rows)} phase={len(phase_only_rows)}"
        )
    primary_summary = cohort_summary(primary_rows, "h3_primary_action", "PRIMARY")
    all_summary = cohort_summary(all_paired_rows, "h3_all_paired", "SECONDARY")
    phase_summary = cohort_summary(phase_only_rows, "h3_phase_only", "EXPLORATORY")
    diagnostic_rows = persistence_diagnostic_rows(joined)
    diagnostic = {
        "definition": "YES persistence and dynamic sensitivity by frozen candidate type and intervention.",
        "summary": diagnostic_summary(joined),
        "rows": diagnostic_rows,
    }
    true_rows = [row for row in candidate_rows if row.get("candidate_type") == "TRUE_SUPPORT"]
    spurious_rows = [row for row in candidate_rows if row.get("candidate_type") == "SPURIOUS_SUPPORT"]
    true_high = sorted(true_rows, key=lambda row: (-(row["D_order"] if not is_nan(row["D_order"]) else -1), row["candidate_id"]))
    true_low = sorted(true_rows, key=lambda row: ((row["D_order"] if not is_nan(row["D_order"]) else 999), row["candidate_id"]))
    spurious_high = sorted(spurious_rows, key=lambda row: (-(row["D_order"] if not is_nan(row["D_order"]) else -1), row["candidate_id"]))
    spurious_low = sorted(spurious_rows, key=lambda row: ((row["D_order"] if not is_nan(row["D_order"]) else 999), row["candidate_id"]))
    predictions = Counter(row["prediction"] for row in joined)
    protocol = protocol_provenance()
    summary = {
        "protocol": protocol,
        "candidate_cohort": {
            "model_specific_cohort": args.model_specific_cohort,
            "total": len(candidates),
            "TRUE_SUPPORT": Counter(row["candidate_type"] for row in candidates)["TRUE_SUPPORT"],
            "SPURIOUS_SUPPORT": Counter(row["candidate_type"] for row in candidates)["SPURIOUS_SUPPORT"],
            "primary_action_qa": len(primary_rows),
            "all_paired_qa": len(all_paired_rows),
            "phase_only_qa": len(phase_only_rows),
        },
        "interventions": {
            "attempted": len(manifest),
            "generation_valid": sum(bool(row.get("generation_valid")) for row in manifest),
            "generation_invalid": sum(not bool(row.get("generation_valid")) for row in manifest),
            "FULL_count": sum(row.get("intervention_type") == "FULL_SHUFFLE_V1" for row in manifest),
            "BLOCK_count": sum(row.get("intervention_type") == "BLOCK_SHUFFLE_4X4_V1" for row in manifest),
            "FREEZE_count": sum(row.get("intervention_type") == "FREEZE_MID_V1" for row in manifest),
        },
        "inference": {
            "predictions": sum(predictions.values()),
            "YES": predictions.get("YES", 0),
            "NO": predictions.get("NO", 0),
            "INVALID": predictions.get("INVALID", 0),
            "ERROR": predictions.get("ERROR", 0),
            "missing": len(join_audit["completion"]["missing"]),
            "extra": len(join_audit["completion"]["extra"]),
            "duplicate": len(join_audit["completion"]["duplicate"]),
            "historical_errors": join_audit["completion"]["historical_errors"],
            "resolved": join_audit["completion"]["resolved_by_final_result"],
            "unresolved": join_audit["completion"]["unresolved_errors"],
        },
        "primary_h3_action_qa": {
            "paired_QA": len(primary_rows),
            **primary_summary,
        },
        "all_paired_summary": all_summary,
        "phase_only_summary": phase_summary,
        "diagnostic": diagnostic["summary"],
        "gt_leakage": not join_audit["raw_predictions_gt_free"],
        "model_config_mismatch": None,
        "prompt_mismatch": None,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
        "interpretation_guardrail": (
            "DESCRIPTIVE RESULT only. Do not state H3 confirmed/rejected without later scientific review "
            "and independent confirmation."
        ),
    }

    write_jsonl_safe(out_dir / "joined" / "joined_h3_dynamic_results.jsonl", joined)
    write_json(out_dir / "joined" / "h3_join_audit.json", json_safe(join_audit))
    write_join_audit_md(out_dir / "joined" / "h3_join_audit.md", json_safe(join_audit))
    write_jsonl_safe(out_dir / "candidate" / "candidate_dynamic_sensitivity.jsonl", candidate_rows)
    write_csv(out_dir / "candidate" / "candidate_dynamic_sensitivity.csv", candidate_rows)
    write_csv(out_dir / "candidate" / "true_high_dynamic_sensitivity.csv", selected_columns(true_high))
    write_csv(out_dir / "candidate" / "true_low_dynamic_sensitivity.csv", selected_columns(true_low))
    write_csv(out_dir / "candidate" / "spurious_high_dynamic_sensitivity.csv", selected_columns(spurious_high))
    write_csv(out_dir / "candidate" / "spurious_low_dynamic_sensitivity.csv", selected_columns(spurious_low))
    write_csv(out_dir / "qa" / "h3_primary_action_qa.csv", primary_rows)
    write_csv(out_dir / "qa" / "h3_all_paired_qa.csv", all_paired_rows)
    write_csv(out_dir / "qa" / "h3_phase_only_qa.csv", phase_only_rows)
    write_json(out_dir / "summary" / "h3_primary_action_summary.json", json_safe(primary_summary))
    write_json(out_dir / "summary" / "h3_all_paired_summary.json", json_safe(all_summary))
    write_json(out_dir / "summary" / "h3_phase_only_summary.json", json_safe(phase_summary))
    write_cohort_summary_md(out_dir / "summary" / "h3_primary_action_summary.md", json_safe(primary_summary))
    write_cohort_summary_md(out_dir / "summary" / "h3_all_paired_summary.md", json_safe(all_summary))
    write_cohort_summary_md(out_dir / "summary" / "h3_phase_only_summary.md", json_safe(phase_summary))
    write_json(out_dir / "summary" / "h3_dynamic_persistence_diagnostic.json", json_safe(diagnostic))
    write_csv(out_dir / "summary" / "h3_dynamic_persistence_diagnostic.csv", diagnostic_rows)
    write_json(out_dir / "summary" / "h3_dynamic_discovery_summary.json", json_safe(summary))
    write_h3_summary_md(out_dir / "summary" / "h3_dynamic_discovery_summary.md", json_safe(summary))
    write_json(out_dir / "provenance" / "h3_dynamic_protocol_provenance.json", protocol)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
