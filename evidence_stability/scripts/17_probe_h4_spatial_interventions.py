#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.cache import stable_hash  # noqa: E402
from evidence_stability.model_interface import build_model  # noqa: E402
from evidence_stability.phase_d2 import sanitize_processor_metadata  # noqa: E402
from evidence_stability.prompts import parse_yes_no  # noqa: E402
from evidence_stability.spatial import (  # noqa: E402
    H4_PROTOCOL_VERSION,
    H4_SUPPORT_PROMPT_VERSION,
    build_h4_support_prompt,
    h4_gt_leakage_audit,
    prompt_sha256,
)
from evidence_stability.utils import append_jsonl, read_jsonl, write_json  # noqa: E402


logger = logging.getLogger("h4_spatial_intervention_probe")


def setup_logging(log_path: str | Path | None) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


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


def set_model_max_new_tokens(model: Any, value: int) -> None:
    if hasattr(model, "max_new_tokens"):
        model.max_new_tokens = int(value)


def completed_cache_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["cache_key"]) for row in read_jsonl(path) if row.get("cache_key")}


def intervention_cache_key(
    row: dict[str, Any],
    prompt: str,
    model_fingerprint: dict[str, Any],
    decoding_config: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "protocol_version": H4_PROTOCOL_VERSION,
            "stage": "spatial_intervention_support_probe",
            "qa_id": row["qa_id"],
            "candidate_id": row["candidate_id"],
            "window_id": row["window_id"],
            "intervention_id": row["intervention_id"],
            "intervention_type": row["intervention_type"],
            "logical_frame_paths": list(row.get("logical_frame_paths") or []),
            "predicted_bbox_norm": row.get("predicted_bbox_norm"),
            "control_bbox_norm": row.get("control_bbox_norm"),
            "blur_version": row.get("blur_version"),
            "prompt_version": H4_SUPPORT_PROMPT_VERSION,
            "prompt_hash": prompt_sha256(prompt),
            "model_fingerprint": model_fingerprint,
            "decoding_config": decoding_config,
        }
    )


def latest_rows_by_id(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        intervention_id = str(row.get("intervention_id"))
        if intervention_id in expected_ids:
            out[intervention_id] = row
    return out


def processor_frame_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    represented_flags = []
    for row in rows[:24]:
        metadata = row.get("processor_metadata") or {}
        expected = int(row.get("n_frames") or 0)
        represented = None
        for key in ("image_grid_thw_rows", "video_grid_thw_rows"):
            if metadata.get(key) is not None:
                represented = int(metadata[key]) == expected
                break
        represented_flags.append(represented)
        cases.append(
            {
                "intervention_id": row.get("intervention_id"),
                "intervention_type": row.get("intervention_type"),
                "expected_frame_count": expected,
                "processor_represented_expected_frames": represented,
                "processor_metadata": metadata,
            }
        )
    return {
        "n_cases": len(cases),
        "cases": cases,
        "all_observed_cases_represent_expected_frames": (
            all(flag is True for flag in represented_flags if flag is not None)
            if represented_flags else None
        ),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# H4 Spatial Intervention Preflight Probe",
        "",
        "## Status",
        f"- protocol version: {summary['protocol_version']}",
        f"- model inference executed: {summary['model_inference_executed']}",
        f"- discovery executed: {summary['discovery_executed']}",
        "",
        "## Scope",
        f"- input interventions: {summary['n_input_interventions']}",
        f"- selected interventions: {summary['n_selected_interventions']}",
        f"- generation-valid only: {summary['generation_valid_only']}",
        "",
        "## Inference",
        f"- completed final records: {summary['n_completed_final_records']}",
        f"- new inference count: {summary['new_inference_count']}",
        f"- skipped cached count: {summary['skipped_cached_count']}",
        f"- remaining interventions: {summary['remaining_interventions']}",
        f"- YES: {summary['prediction_counts'].get('YES', 0)}",
        f"- NO: {summary['prediction_counts'].get('NO', 0)}",
        f"- INVALID: {summary['prediction_counts'].get('INVALID', 0)}",
        f"- ERROR: {summary['prediction_counts'].get('ERROR', 0)}",
        "",
        "## Breakdown",
        f"- by intervention type: {summary['prediction_by_intervention_type']}",
        "",
        "## Audit",
        f"- GT leakage: {summary['gt_leakage_audit']['gt_leakage']}",
        f"- missing frame errors: {summary['missing_frame_errors']}",
        f"- processor failures: {summary['processor_failures']}",
        f"- processor frame audit: {summary['processor_frame_audit'].get('all_observed_cases_represent_expected_frames')}",
        "",
        "This is still H4 preflight. No post-hoc GT join, C_S computation, 50-QA discovery, formal statistics, or independent confirmation was run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe H4 KEEP/DROP/CONTROL intervention frames with frozen support prompt.")
    p.add_argument("--intervention_manifest", default="outputs/stg_pilot/phase_g/h4_spatial_v1/manifest/h4_spatial_intervention_manifest_gt_free.jsonl")
    p.add_argument("--output_dir", default="outputs/stg_pilot/phase_g/h4_spatial_v1")
    p.add_argument("--probe_results", default=None)
    p.add_argument("--errors", default=None)
    p.add_argument("--summary_output", default=None)
    p.add_argument("--log_path", default=None)
    p.add_argument("--model_backend", choices=["dummy", "openai_compatible", "qwen3_vl"], default="dummy")
    p.add_argument("--model_name", default="dummy-video-vlm")
    p.add_argument("--model_revision", default="v1")
    p.add_argument("--model_path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--processor_min_pixels", type=int, default=None)
    p.add_argument("--processor_max_pixels", type=int, default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--limit_interventions", type=int, default=-1)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["predictions", "audit", "summary", "provenance"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    probe_results = Path(args.probe_results or out_dir / "predictions" / "intervention_predictions.jsonl")
    errors_path = Path(args.errors or out_dir / "predictions" / "intervention_errors.jsonl")
    summary_output = Path(args.summary_output or out_dir / "summary" / "h4_spatial_intervention_preflight_probe_summary.json")
    log_path = Path(args.log_path or out_dir / "predictions" / "h4_spatial_intervention_probe.log")
    probe_results.touch(exist_ok=True)
    errors_path.touch(exist_ok=True)
    setup_logging(log_path)

    rows = read_jsonl(args.intervention_manifest)
    if args.limit_interventions >= 0:
        rows = rows[: args.limit_interventions]
    if not rows:
        raise RuntimeError("No H4 spatial interventions found to probe.")
    leakage = h4_gt_leakage_audit(rows)
    if leakage["gt_leakage"]:
        raise RuntimeError(f"H4 intervention manifest contains GT leakage: {leakage}")
    if any(not row.get("intervention_frame_paths") for row in rows):
        raise RuntimeError("H4 intervention manifest contains rows without intervention_frame_paths")
    if args.model_backend == "qwen3_vl" and not args.model_path:
        raise ValueError("--model_path is required for --model_backend qwen3_vl")

    model = build_model(args)
    set_model_max_new_tokens(model, args.max_new_tokens)
    model_fingerprint = model.fingerprint()
    decoding_config = model.generation_config()
    completed = completed_cache_keys(probe_results)
    completed_at_start = len(completed)
    counts = Counter()
    new_count = 0
    missing_frame_errors = 0
    processor_failures = 0
    expected_ids = {str(row["intervention_id"]) for row in rows}
    if len(expected_ids) != len(rows):
        raise RuntimeError("Duplicate intervention_id in H4 intervention manifest.")

    logger.info("H4 intervention probing selected=%d completed_cache=%d", len(rows), completed_at_start)
    started = time.time()
    for i, row in enumerate(rows, start=1):
        prompt = build_h4_support_prompt(row["human_question"])
        frame_paths = [str(path) for path in row["intervention_frame_paths"]]
        cache_key = intervention_cache_key(row, prompt, model_fingerprint, decoding_config)
        if cache_key in completed:
            counts["skipped_cached"] += 1
            continue
        if args.dry_run:
            counts["dry_run"] += 1
            continue
        missing = [path for path in frame_paths if not Path(path).exists()]
        if missing:
            counts["ERROR"] += 1
            missing_frame_errors += 1
            append_jsonl(
                errors_path,
                {
                    "protocol_version": H4_PROTOCOL_VERSION,
                    "stage": "spatial_intervention_support_probe",
                    "qa_id": row["qa_id"],
                    "clip_id": row["clip_id"],
                    "candidate_id": row["candidate_id"],
                    "window_id": row["window_id"],
                    "intervention_id": row["intervention_id"],
                    "intervention_type": row["intervention_type"],
                    "error_type": "MISSING_FRAMES",
                    "missing_frame_paths": missing[:20],
                    "cache_key": cache_key,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            continue
        try:
            raw = model.infer(frame_paths, prompt)
            parsed = parse_yes_no(raw)
            processor_meta = sanitize_processor_metadata(
                getattr(model, "last_processor_metadata", {}) or {},
                expected_frame_count=len(frame_paths),
            )
            record = {
                "protocol_version": H4_PROTOCOL_VERSION,
                "stage": "spatial_intervention_support_probe",
                "qa_id": row["qa_id"],
                "clip_id": row["clip_id"],
                "candidate_id": row["candidate_id"],
                "window_id": row["window_id"],
                "intervention_id": row["intervention_id"],
                "intervention_type": row["intervention_type"],
                "intervention_family": row.get("intervention_family"),
                "dataset_name": row["dataset_name"],
                "model_name": model.model_name,
                "model_revision": model.model_revision,
                "model_identity_hash": model_fingerprint.get("model_identity_hash"),
                "prompt_version": H4_SUPPORT_PROMPT_VERSION,
                "prompt_hash": prompt_sha256(prompt),
                "decoding_config": decoding_config,
                "logical_frame_paths": list(row.get("logical_frame_paths") or []),
                "intervention_frame_paths": frame_paths,
                "predicted_bbox_norm": row.get("predicted_bbox_norm"),
                "bbox_area_fraction": row.get("bbox_area_fraction"),
                "control_bbox_norm": row.get("control_bbox_norm"),
                "control_valid": row.get("control_valid"),
                "blur_version": row.get("blur_version"),
                "n_frames": len(frame_paths),
                "n_unique_frames": row.get("n_unique_frames"),
                "raw_response": raw,
                "parsed_prediction": parsed,
                "cache_key": cache_key,
                "processor_metadata": processor_meta,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_jsonl(probe_results, record)
            completed.add(cache_key)
            counts[parsed] += 1
            new_count += 1
        except Exception as exc:
            counts["ERROR"] += 1
            processor_failures += 1
            append_jsonl(
                errors_path,
                {
                    "protocol_version": H4_PROTOCOL_VERSION,
                    "stage": "spatial_intervention_support_probe",
                    "qa_id": row["qa_id"],
                    "clip_id": row["clip_id"],
                    "candidate_id": row["candidate_id"],
                    "window_id": row["window_id"],
                    "intervention_id": row["intervention_id"],
                    "intervention_type": row["intervention_type"],
                    "error_type": exc.__class__.__name__,
                    "error": repr(exc),
                    "cache_key": cache_key,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        if i % 25 == 0:
            logger.info("progress=%d/%d YES=%d NO=%d INVALID=%d ERROR=%d", i, len(rows), counts["YES"], counts["NO"], counts["INVALID"], counts["ERROR"])

    result_by_id = latest_rows_by_id(probe_results, expected_ids)
    error_rows = [
        row for row in read_jsonl(errors_path)
        if str(row.get("intervention_id")) in expected_ids and str(row.get("intervention_id")) not in result_by_id
    ]
    final_counts = Counter(str(row.get("parsed_prediction", "INVALID")) for row in result_by_id.values())
    final_counts["ERROR"] += len(error_rows)
    prediction_by_type: dict[str, Counter] = defaultdict(Counter)
    rows_by_id = {str(row["intervention_id"]): row for row in rows}
    for intervention_id, result in result_by_id.items():
        prediction_by_type[str(rows_by_id[intervention_id]["intervention_type"])][str(result.get("parsed_prediction", "INVALID"))] += 1
    for row in error_rows:
        intervention_id = str(row.get("intervention_id"))
        if intervention_id in rows_by_id:
            prediction_by_type[str(rows_by_id[intervention_id]["intervention_type"])]["ERROR"] += 1
    processor_audit = processor_frame_audit(list(result_by_id.values()))
    summary = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "intervention_manifest": args.intervention_manifest,
        "output_dir": str(out_dir),
        "n_input_interventions": len(read_jsonl(args.intervention_manifest)),
        "n_selected_interventions": len(rows),
        "generation_valid_only": all(row.get("intervention_frame_paths") for row in rows),
        "model_backend": args.model_backend,
        "model_path": args.model_path,
        "model_fingerprint": model.fingerprint(),
        "generation_config": model.generation_config(),
        "n_completed_final_records": len(result_by_id) + len(error_rows),
        "new_inference_count": new_count,
        "skipped_cached_count": counts["skipped_cached"],
        "completed_cache_at_start": completed_at_start,
        "remaining_interventions": len(expected_ids) - len(result_by_id) - len(error_rows),
        "prediction_counts": dict(final_counts),
        "prediction_by_intervention_type": {key: dict(value) for key, value in sorted(prediction_by_type.items())},
        "gt_leakage_audit": leakage,
        "missing_frame_errors": missing_frame_errors,
        "processor_failures": processor_failures,
        "processor_frame_audit": processor_audit,
        "repo": {
            "git_commit": git_output(["rev-parse", "HEAD"]),
            "git_status_short": git_output(["status", "--short"]),
        },
        "model_inference_executed": not args.dry_run,
        "discovery_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(summary_output, summary)
    write_summary_md(summary_output.with_suffix(".md"), summary)
    write_json(out_dir / "audit" / "processor_frame_audit.json", processor_audit)
    write_json(out_dir / "audit" / "cache_restart_audit.json", {
        "completed_cache_at_start": completed_at_start,
        "new_inference_count": new_count,
        "skipped_cached_count": counts["skipped_cached"],
        "cache_resume_supported": True,
    })
    (out_dir / "provenance" / "environment_snapshot_h4_intervention_probe.txt").write_text(environment_snapshot(), encoding="utf-8")
    (out_dir / "provenance" / "git_status_h4_intervention_probe.txt").write_text(summary["repo"]["git_status_short"] + "\n", encoding="utf-8")
    logger.info("done completed=%d new=%d skipped=%d remaining=%d", summary["n_completed_final_records"], new_count, counts["skipped_cached"], summary["remaining_interventions"])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
