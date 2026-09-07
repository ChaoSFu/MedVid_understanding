#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from evidence_stability.dynamic import (  # noqa: E402
    BLOCK_ORDER,
    FREEZE_MID_IDX,
    GLOBAL_SEED,
    INTERVENTION_TYPES,
    PROTOCOL_VERSION,
    assert_dynamic_intervention_valid,
    assert_model_manifest_gt_free,
    deterministic_smoke_candidate_ids,
    generate_dynamic_interventions_for_candidate,
    multiset_preserved,
    protocol_provenance,
    reconstruct_h3_candidates,
    write_contact_sheet,
)
from evidence_stability.phase_d2 import remap_projection_frame_paths  # noqa: E402
from evidence_stability.utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402


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
    for module_name in ["torch", "transformers", "PIL"]:
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}: {getattr(module, '__version__', 'UNKNOWN')}")
        except Exception as exc:
            lines.append(f"{module_name}: UNAVAILABLE ({exc!r})")
    return "\n".join(lines) + "\n"


def strip_model_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    allowed = [
        "intervention_id",
        "candidate_id",
        "qa_id",
        "clip_id",
        "window_id",
        "dataset_name",
        "target_action",
        "target_field",
        "intervention_type",
        "intervention_family",
        "ordered_frame_paths",
        "n_frames",
        "n_unique_frames",
        "prompt_version",
        "generation_valid",
        "generation_reason",
        "gt_information_used_in_model_inference",
        "intervention_provenance",
    ]
    stripped = {key: row.get(key) for key in allowed}
    assert_model_manifest_gt_free(stripped)
    return stripped


def intervention_generation_audit(
    candidates: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_candidate = {row["candidate_id"]: row for row in candidates}
    failures = []
    for row in interventions:
        original = by_candidate[row["candidate_id"]]["ordered_frame_paths"]
        try:
            assert_dynamic_intervention_valid(row, original)
        except RuntimeError as exc:
            failures.append({"intervention_id": row["intervention_id"], "error": str(exc)})
    ids = [row["intervention_id"] for row in interventions]
    duplicate_ids = [value for value, count in Counter(ids).items() if count > 1]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "n_candidates": len(candidates),
        "n_interventions": len(interventions),
        "theoretical_count": len(candidates) * len(INTERVENTION_TYPES),
        "generation_valid": sum(bool(row.get("generation_valid")) for row in interventions),
        "generation_invalid": sum(not bool(row.get("generation_valid")) for row in interventions),
        "intervention_type_counts": dict(Counter(row["intervention_type"] for row in interventions)),
        "n_duplicate_intervention_ids": len(duplicate_ids),
        "duplicate_intervention_ids": duplicate_ids[:20],
        "n_validation_failures": len(failures),
        "validation_failures": failures[:20],
        "full_shuffle_valid": all(
            row["intervention_provenance"]["checks"].get("same_frame_multiset")
            and row["intervention_provenance"]["checks"].get("ordering_changed")
            for row in interventions
            if row["intervention_type"] == "FULL_SHUFFLE_V1" and row.get("generation_valid")
        ),
        "block_shuffle_block_order": BLOCK_ORDER,
        "block_shuffle_valid": all(
            row["intervention_provenance"]["checks"].get("same_frame_multiset")
            and row["intervention_provenance"]["checks"].get("within_block_order_preserved")
            for row in interventions
            if row["intervention_type"] == "BLOCK_SHUFFLE_4X4_V1" and row.get("generation_valid")
        ),
        "freeze_mid_idx": FREEZE_MID_IDX,
        "freeze_valid": all(
            row["intervention_provenance"]["checks"].get("freeze_mid_idx_is_7")
            and row["intervention_provenance"]["checks"].get("sixteen_repeated_logical_entries")
            for row in interventions
            if row["intervention_type"] == "FREEZE_MID_V1" and row.get("generation_valid")
        ),
    }


def gt_leakage_audit(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    leaked = []
    for row in manifest:
        try:
            assert_model_manifest_gt_free(row)
        except RuntimeError as exc:
            leaked.append({"intervention_id": row.get("intervention_id"), "error": str(exc)})
    serialized = "\n".join(json.dumps(row, ensure_ascii=False).lower() for row in manifest)
    terms = ["true_support", "spurious_support", "gt_alignment", "evidence_density", "gt_evidence_recall"]
    visible_terms = [term for term in terms if term in serialized]
    return {
        "gt_information_used_in_model_inference": False,
        "n_manifest_rows": len(manifest),
        "n_rows_with_forbidden_fields": len(leaked),
        "forbidden_field_examples": leaked[:20],
        "candidate_type_visible_to_model": "candidate_type" in serialized,
        "gt_terms_visible_to_model": visible_terms,
    }


def build_visualization_audit(
    candidates: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    output_dir: Path,
    frame_root: str | None,
    source_frame_prefix: str,
) -> dict[str, Any]:
    selected = deterministic_smoke_candidate_ids(candidates, n=5, seed=GLOBAL_SEED)
    by_candidate = {row["candidate_id"]: row for row in candidates}
    interventions_by_candidate: dict[str, dict[str, list[str]]] = {}
    for row in interventions:
        if row["candidate_id"] in selected and row.get("generation_valid"):
            interventions_by_candidate.setdefault(row["candidate_id"], {})[row["intervention_type"]] = list(row["ordered_frame_paths"])
    rows = []
    for candidate_id in selected:
        original = list(by_candidate[candidate_id]["ordered_frame_paths"])
        variants = interventions_by_candidate.get(candidate_id, {})
        if frame_root:
            projection = {"intervention_id": candidate_id, "ordered_frame_paths": original, "n_frames": len(original)}
            original_runtime = remap_projection_frame_paths(projection, frame_root, source_frame_prefix)["ordered_frame_paths"]
            runtime_variants = {}
            for key, paths in variants.items():
                proj = {"intervention_id": f"{candidate_id}::{key}", "ordered_frame_paths": paths, "n_frames": len(paths)}
                runtime_variants[key] = remap_projection_frame_paths(proj, frame_root, source_frame_prefix)["ordered_frame_paths"]
        else:
            original_runtime = original
            runtime_variants = variants
        sheet_path = output_dir / "audit" / "visualizations" / f"{stable_hash({'candidate_id': candidate_id})[:16]}.jpg"
        written = write_contact_sheet(original_runtime, runtime_variants, sheet_path)
        rows.append(
            {
                "candidate_id": candidate_id,
                "dataset_name": by_candidate[candidate_id]["dataset_name"],
                "visualization_path": str(sheet_path) if written else None,
                "written": written,
                "reason_if_not_written": None if written else "Pillow unavailable or frame files not present on this machine",
            }
        )
    return {"requested": 5, "cases": rows, "n_written": sum(row["written"] for row in rows)}


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase F-D H3 Dynamic Preparation",
        "",
        "## Repo",
        f"- git commit: {summary['repo']['git_commit']}",
        f"- working tree short: {summary['repo']['git_status_short']!r}",
        "",
        "## Candidate Reconstruction",
        f"- total: {summary['candidate_reconstruction']['n_total_candidates']}",
        f"- TRUE: {summary['candidate_reconstruction']['candidate_type_counts'].get('TRUE_SUPPORT', 0)}",
        f"- SPURIOUS: {summary['candidate_reconstruction']['candidate_type_counts'].get('SPURIOUS_SUPPORT', 0)}",
        f"- primary action QA: {summary['candidate_reconstruction']['n_primary_action_qa']}",
        f"- all paired QA: {summary['candidate_reconstruction']['n_all_paired_qa']}",
        f"- phase-only QA: {summary['candidate_reconstruction']['n_phase_only_qa']}",
        "",
        "## Intervention Generation",
        f"- interventions: {summary['intervention_generation']['n_interventions']}",
        f"- generation valid: {summary['intervention_generation']['generation_valid']}",
        f"- generation invalid: {summary['intervention_generation']['generation_invalid']}",
        f"- duplicate IDs: {summary['intervention_generation']['n_duplicate_intervention_ids']}",
        "",
        "No model inference, formal statistics, or independent confirmation was run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Phase F-D H3 dynamic intervention manifest.")
    p.add_argument("--phase_c_labeled", default="outputs/tal_pilot/phase_c_server/qwen3_vl_8b_pilot50/probe_labeled_results.jsonl")
    p.add_argument("--paired_qa", default="outputs/tal_pilot/phase_c_server/qwen3_vl_8b_pilot50/paired_h2_qa.jsonl")
    p.add_argument("--phase_c_summary", default="outputs/tal_pilot/phase_c_server/qwen3_vl_8b_pilot50/phase_c_pilot_probe_summary.json")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_f/h3_dynamic_v1")
    p.add_argument(
        "--model_specific_cohort",
        action="store_true",
        help="Allow H3 candidates to be reconstructed from a non-8B model's H2 pairing set.",
    )
    p.add_argument("--frame_root", default=None)
    p.add_argument(
        "--source_frame_prefix",
        default="/root/data,/mnt/hdd3/huihui/hh_datas/MedVidU/valdata,/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["manifest", "audit", "provenance"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    phase_c_rows = read_jsonl(args.phase_c_labeled)
    paired_qa_rows = read_jsonl(args.paired_qa)
    expected = {} if args.model_specific_cohort else {
        "expected_total": 158,
        "expected_true": 64,
        "expected_spurious": 94,
        "expected_primary_action_qa": 9,
        "expected_all_paired_qa": 16,
        "expected_phase_only_qa": 7,
    }
    if args.model_specific_cohort:
        expected = {
            "expected_total": None,
            "expected_true": None,
            "expected_spurious": None,
            "expected_primary_action_qa": None,
            "expected_all_paired_qa": None,
            "expected_phase_only_qa": None,
        }
    candidates, reconstruction_audit = reconstruct_h3_candidates(phase_c_rows, paired_qa_rows, **expected)
    if not candidates:
        raise RuntimeError("H3 requires at least one H2 TRUE/SPURIOUS paired candidate for this model.")
    interventions = [
        row
        for candidate in candidates
        for row in generate_dynamic_interventions_for_candidate(candidate)
    ]
    generation_audit = intervention_generation_audit(candidates, interventions)
    if generation_audit["n_duplicate_intervention_ids"] or generation_audit["n_validation_failures"]:
        raise RuntimeError(f"H3 intervention generation audit failed: {generation_audit}")
    model_manifest = [strip_model_manifest_row(row) for row in interventions]
    leakage = gt_leakage_audit(model_manifest)
    if leakage["n_rows_with_forbidden_fields"] or leakage["candidate_type_visible_to_model"] or leakage["gt_terms_visible_to_model"]:
        raise RuntimeError(f"H3 GT leakage audit failed: {leakage}")

    smoke_candidate_ids = deterministic_smoke_candidate_ids(candidates, n=12, seed=GLOBAL_SEED)
    smoke_artifact = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": GLOBAL_SEED,
        "n_candidates": len(smoke_candidate_ids),
        "candidate_ids": smoke_candidate_ids,
        "selection_rule": "GT-blind deterministic dataset coverage then candidate_id hash",
        "dataset_counts": dict(Counter(row["dataset_name"] for row in candidates if row["candidate_id"] in smoke_candidate_ids)),
    }
    visualization = build_visualization_audit(candidates, interventions, out_dir, args.frame_root, args.source_frame_prefix)
    phase_c_summary = read_json(args.phase_c_summary) if Path(args.phase_c_summary).exists() else {}
    provenance = protocol_provenance()
    provenance.update(
        {
            "phase_c_summary": args.phase_c_summary,
            "phase_c_model_fingerprint": phase_c_summary.get("model_fingerprint"),
            "phase_c_decoding_config": phase_c_summary.get("decoding_config"),
        }
    )
    repo = {
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
    }
    summary = {
        "phase": "Phase F-D H3 Dynamic Evidence Verification preparation",
        "protocol_version": PROTOCOL_VERSION,
        "study_stage": "h3_dynamic_discovery",
        "model_specific_cohort": args.model_specific_cohort,
        "candidate_reconstruction": reconstruction_audit,
        "intervention_generation": generation_audit,
        "gt_leakage_audit": leakage,
        "smoke": smoke_artifact,
        "visualization_audit": visualization,
        "repo": repo,
        "model_inference_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
    }

    write_jsonl(out_dir / "manifest" / "frozen_candidate_reconstruction.jsonl", candidates)
    write_jsonl(out_dir / "manifest" / "h3_model_intervention_manifest_gt_free.jsonl", model_manifest)
    write_json(out_dir / "manifest" / "smoke_candidate_ids.json", smoke_artifact)
    write_json(out_dir / "audit" / "candidate_reconstruction_audit.json", reconstruction_audit)
    write_json(out_dir / "audit" / "intervention_generation_audit.json", generation_audit)
    write_json(out_dir / "audit" / "gt_leakage_audit.json", leakage)
    write_json(out_dir / "audit" / "visualization_audit.json", visualization)
    write_json(out_dir / "provenance" / "h3_dynamic_protocol_provenance.json", provenance)
    write_json(out_dir / "provenance" / "model_fingerprint.json", phase_c_summary.get("model_fingerprint") or {})
    (out_dir / "provenance" / "environment_snapshot.txt").write_text(environment_snapshot(), encoding="utf-8")
    (out_dir / "provenance" / "git_commit.txt").write_text(repo["git_commit"] + "\n", encoding="utf-8")
    (out_dir / "provenance" / "git_diff.txt").write_text(git_output(["diff"]) + "\n", encoding="utf-8")
    write_json(out_dir / "summary_prepare.json", summary)
    write_md(out_dir / "summary_prepare.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
