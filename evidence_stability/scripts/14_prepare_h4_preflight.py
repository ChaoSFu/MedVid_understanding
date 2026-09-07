#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.cache import stable_hash  # noqa: E402
from evidence_stability.spatial import (  # noqa: E402
    H4_DISCOVERY_SEED,
    H4_PREFLIGHT_QA_COUNT,
    H4_PROTOCOL_VERSION,
    H4_WINDOW_SIZE,
    H4_WINDOW_STRIDE,
    audit_stg_schema,
    h4_generate_stg_windows,
    h4_gt_leakage_audit,
    h4_model_window_manifest_row,
    h4_protocol_freeze,
    h4_window_temporal_alignment,
    normalize_stg_sample,
    prompt_sha256,
    write_spatial_threshold_review_md,
    write_stg_schema_md,
    write_temporal_protocol_review_md,
)
from evidence_stability.utils import read_json, write_json, write_jsonl  # noqa: E402


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
    for module_name in ["torch", "transformers", "PIL", "numpy"]:
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}: {getattr(module, '__version__', 'UNKNOWN')}")
        except Exception as exc:
            lines.append(f"{module_name}: UNAVAILABLE ({exc!r})")
    return "\n".join(lines) + "\n"


def select_preflight_qa(samples: list[dict[str, Any]], n: int, seed: int, min_frames: int = H4_WINDOW_SIZE) -> list[dict[str, Any]]:
    pool = [s for s in samples if len(s.get("frame_paths") or []) >= min_frames]
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in pool:
        key = f"{sample['dataset_name']}::fps={sample['metadata_fps']}"
        by_stratum[key].append(sample)

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    shuffled_by_stratum = {}
    for stratum, rows in by_stratum.items():
        copied = list(rows)
        rng.shuffle(copied)
        shuffled_by_stratum[stratum] = copied

    while len(chosen) < n:
        progressed = False
        for stratum in sorted(shuffled_by_stratum):
            rows = shuffled_by_stratum[stratum]
            while rows and rows[0]["qa_id"] in seen:
                rows.pop(0)
            if not rows:
                continue
            sample = rows.pop(0)
            chosen.append(sample)
            seen.add(sample["qa_id"])
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break

    chosen.sort(key=lambda row: row["qa_id"])
    return chosen


def qa_manifest_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": sample["qa_id"],
        "clip_id": sample["clip_id"],
        "original_index": sample["original_index"],
        "dataset": sample["dataset_name"],
        "dataset_name": sample["dataset_name"],
        "metadata_fps": sample["metadata_fps"],
        "n_frames": len(sample["frame_paths"]),
        "n_unique_frames": len(set(sample["sampled_frame_indices"])),
        "clip_duration": sample["clip_duration"],
        "human_question": sample["human_question"],
        "selection_fields_gt_free": [
            "dataset_name",
            "metadata_fps",
            "n_frames",
            "n_unique_frames",
            "clip_duration",
            "qa_id",
        ],
    }


def write_protocol_freeze_md(path: Path, freeze: dict[str, Any]) -> None:
    lines = [
        "# H4 Protocol Freeze",
        "",
        f"- protocol version: {freeze['protocol_version']}",
        f"- temporal eligibility: {freeze['temporal_eligibility_rule']}",
        f"- temporal density threshold: {freeze['temporal_density_threshold']}",
        f"- temporal recall threshold: {freeze['temporal_recall_threshold']}",
        f"- TRUE_SPATIAL_SUPPORT: {freeze['true_spatial_support_rule']}",
        f"- SPURIOUS_SPATIAL_SUPPORT: {freeze['spurious_spatial_support_rule']}",
        f"- spatial label protocol: {freeze['spatial_label_protocol']}",
        f"- window size: {freeze['window_size']}",
        f"- stride: {freeze['window_stride']}",
        f"- drop last: {freeze['drop_last']}",
        "",
        "These choices were explicitly frozen before H4 preflight/discovery inference.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# H4 Spatial Preflight Preparation",
        "",
        "## Protocol",
        f"- protocol version: {summary['protocol_version']}",
        f"- temporal eligibility: {summary['protocol_freeze']['temporal_eligibility_rule']}",
        f"- TRUE spatial: {summary['protocol_freeze']['true_spatial_support_rule']}",
        f"- SPURIOUS spatial: {summary['protocol_freeze']['spurious_spatial_support_rule']}",
        "",
        "## STG",
        f"- total STG QA: {summary['n_stg_total']}",
        f"- normalized QA: {summary['n_normalized_qa']}",
        f"- normalization errors: {summary['n_normalization_errors']}",
        f"- datasets: {summary['dataset_counts']}",
        "",
        "## Preflight Cohort",
        f"- selected QA: {summary['preflight']['n_qa']}",
        f"- dataset counts: {summary['preflight']['dataset_counts']}",
        f"- cohort sha256: {summary['preflight']['cohort_sha256']}",
        "",
        "## Windows",
        f"- total windows: {summary['windows']['n_windows']}",
        f"- duplicate candidate IDs: {summary['windows']['n_duplicate_candidate_ids']}",
        f"- frame-count distribution: {summary['windows']['frame_count_distribution']}",
        f"- temporally eligible windows, post-hoc only: {summary['windows']['n_h4_temporally_eligible']}",
        "",
        "## GT-Free Manifest",
        f"- model manifest rows: {summary['gt_leakage_audit']['n_model_manifest_rows']}",
        f"- GT leakage: {summary['gt_leakage_audit']['gt_leakage']}",
        "",
        "No model inference, spatial pointing, KEEP/DROP intervention, formal statistics, or discovery run was executed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare GT-blind H4 STG preflight cohort and window manifests.")
    p.add_argument("--data_json", default="data_json/medvidu_filtered/trainval/stg.json")
    p.add_argument("--output_dir", default="outputs/stg_pilot/phase_g/h4_spatial_v1")
    p.add_argument("--frame_root", default=None)
    p.add_argument(
        "--source_frame_prefix",
        default="/root/data,/mnt/hdd3/huihui/hh_datas/MedVidU/valdata,/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
    )
    p.add_argument("--preflight_qa", type=int, default=H4_PREFLIGHT_QA_COUNT)
    p.add_argument("--seed", type=int, default=H4_DISCOVERY_SEED)
    p.add_argument("--window_size", type=int, default=H4_WINDOW_SIZE)
    p.add_argument("--stride", type=int, default=H4_WINDOW_STRIDE)
    p.add_argument("--verify_paths", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["audit", "summary", "provenance", "manifest", "predictions", "joined", "candidate", "qa", "visualizations/preflight", "visualizations/discovery"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    raw = read_json(args.data_json)
    if not isinstance(raw, list):
        raise TypeError("STG data must be a list")

    audit = audit_stg_schema(raw)
    freeze = h4_protocol_freeze()
    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for original_index, sample in enumerate(raw):
        row, error = normalize_stg_sample(
            sample,
            original_index,
            frame_root=args.frame_root,
            source_frame_prefix=args.source_frame_prefix,
            verify_paths=args.verify_paths,
        )
        if row is not None:
            normalized.append(row)
        if error is not None:
            errors.append(error)

    qa_ids = [row["qa_id"] for row in normalized]
    duplicate_qa_ids = [value for value, count in Counter(qa_ids).items() if count > 1]
    if duplicate_qa_ids:
        raise RuntimeError(f"Duplicate H4 qa_id values: {duplicate_qa_ids[:20]}")

    selected = select_preflight_qa(normalized, args.preflight_qa, args.seed, min_frames=args.window_size)
    selected_ids = {row["qa_id"] for row in selected}
    selected_manifest = [qa_manifest_row(row) for row in selected]

    windows: list[dict[str, Any]] = []
    joined_windows: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    sample_by_qa = {row["qa_id"]: row for row in selected}
    for sample in selected:
        for window in h4_generate_stg_windows(
            sample,
            window_size=args.window_size,
            stride=args.stride,
            drop_last=True,
        ):
            if window["qa_id"] not in selected_ids:
                continue
            windows.append(window)
            model_manifest.append(h4_model_window_manifest_row(sample, window))
            alignment = h4_window_temporal_alignment(sample, window)
            joined_windows.append(
                {
                    "qa_id": sample["qa_id"],
                    "clip_id": sample["clip_id"],
                    "candidate_id": window["candidate_id"],
                    "window_id": window["window_id"],
                    "dataset_name": sample["dataset_name"],
                    "human_question": sample["human_question"],
                    "frame_ids": list(window["frame_indices"]),
                    "local_timestamps": list(window["frame_times"]),
                    "raw_gt_spans": sample["raw_gt_spans"],
                    "processed_gt_spans": sample["processed_gt_spans"],
                    "stg_bbox_dict": sample["stg_bbox_dict"],
                    "n_gt_visible_total": sample["n_gt_visible_total"],
                    **alignment,
                }
            )

    candidate_ids = [row["candidate_id"] for row in model_manifest]
    duplicate_candidate_ids = [value for value, count in Counter(candidate_ids).items() if count > 1]
    if duplicate_candidate_ids:
        raise RuntimeError(f"Duplicate H4 candidate IDs: {duplicate_candidate_ids[:20]}")
    if any(row["n_frames"] != args.window_size for row in model_manifest):
        raise RuntimeError("H4 preflight model manifest contains non-16-frame window")

    leakage = h4_gt_leakage_audit(model_manifest)
    if leakage["gt_leakage"]:
        raise RuntimeError(f"H4 GT-free manifest leakage detected: {leakage}")

    frame_count_distribution = Counter(str(row["n_frames"]) for row in model_manifest)
    repo = {
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
    }
    prompt_fingerprint = {
        "support_prompt_version": freeze["support_prompt_version"],
        "support_prompt_sha256_first_example": (
            model_manifest[0]["support_prompt_hash"] if model_manifest else None
        ),
        "spatial_pointer_prompt_version": freeze["spatial_pointer_prompt_version"],
        "spatial_pointer_prompt_sha256_first_example": (
            model_manifest[0]["spatial_pointer_prompt_hash"] if model_manifest else None
        ),
        "structured_output": False,
        "spatial_pointer_parser": "strict_json_object_with_bbox_only",
    }
    cohort_payload = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "cohort": "h4_preflight_12_qa",
        "seed": args.seed,
        "selection_rule": "GT-blind deterministic stratification by dataset, then seeded shuffle over QA with >=16 input frames",
        "gt_blind_selection": True,
        "selection_strata": "dataset_name::metadata_fps",
        "qa": selected_manifest,
        "dataset_counts": dict(Counter(row["dataset_name"] for row in selected)),
        "dataset_fps_counts": dict(Counter(f"{row['dataset_name']}::fps={row['metadata_fps']}" for row in selected)),
    }
    summary = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "protocol_freeze": freeze,
        "data_json": args.data_json,
        "output_dir": str(out_dir),
        "frame_root": args.frame_root,
        "source_frame_prefix": args.source_frame_prefix,
        "n_stg_total": audit["n_stg"],
        "n_normalized_qa": len(normalized),
        "n_normalization_errors": len(errors),
        "dataset_counts": dict(Counter(row["dataset_name"] for row in normalized)),
        "preflight": {
            "n_qa": len(selected),
            "dataset_counts": dict(Counter(row["dataset_name"] for row in selected)),
            "cohort_sha256": stable_hash(cohort_payload),
            "qa_ids": [row["qa_id"] for row in selected],
        },
        "windows": {
            "n_windows": len(model_manifest),
            "n_duplicate_candidate_ids": len(duplicate_candidate_ids),
            "frame_count_distribution": dict(sorted(frame_count_distribution.items())),
            "n_h4_temporally_eligible": sum(bool(row["h4_temporally_eligible"]) for row in joined_windows),
        },
        "gt_leakage_audit": leakage,
        "prompt_fingerprint": prompt_fingerprint,
        "repo": repo,
        "model_inference_executed": False,
        "spatial_pointing_executed": False,
        "interventions_generated": False,
        "discovery_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
    }

    write_json(out_dir / "audit" / "medvidu_stg_schema.json", audit)
    write_stg_schema_md(str(out_dir / "audit" / "medvidu_stg_schema.md"), audit)
    write_stg_schema_md(str(out_dir / "provenance" / "medvidu_stg_schema.md"), audit)
    write_temporal_protocol_review_md(str(out_dir / "audit" / "temporal_eligibility_protocol_review.md"))
    write_spatial_threshold_review_md(str(out_dir / "audit" / "spatial_label_threshold_review.md"), audit)
    write_protocol_freeze_md(out_dir / "audit" / "h4_protocol_freeze.md", freeze)
    write_json(out_dir / "provenance" / "h4_protocol_freeze.json", freeze)
    write_json(out_dir / "provenance" / "prompt_fingerprint.json", prompt_fingerprint)
    write_json(out_dir / "provenance" / "run_config.json", vars(args))
    (out_dir / "provenance" / "environment_snapshot.txt").write_text(environment_snapshot(), encoding="utf-8")
    (out_dir / "provenance" / "git_status.txt").write_text(repo["git_status_short"] + "\n", encoding="utf-8")

    write_json(out_dir / "manifest" / "h4_preflight_12_qa.json", cohort_payload)
    write_jsonl(out_dir / "manifest" / "h4_preflight_windows_gt_free.jsonl", model_manifest)
    write_jsonl(out_dir / "manifest" / "h4_model_manifest_gt_free.jsonl", model_manifest)
    write_jsonl(out_dir / "joined" / "h4_preflight_windows_with_temporal_gt.jsonl", joined_windows)
    write_json(out_dir / "audit" / "gt_leakage_audit.json", leakage)
    write_json(out_dir / "audit" / "preflight_window_generation_audit.json", summary["windows"])
    write_json(out_dir / "audit" / "normalization_errors.json", errors)
    write_json(out_dir / "summary" / "h4_preflight_prepare_summary.json", summary)
    write_summary_md(out_dir / "summary" / "h4_preflight_prepare_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
