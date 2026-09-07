#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.cache import stable_hash  # noqa: E402
from evidence_stability.model_interface import build_model  # noqa: E402
from evidence_stability.phase_d2 import logical_relative_frame_path, runtime_frame_path  # noqa: E402
from evidence_stability.prompts import parse_yes_no  # noqa: E402
from evidence_stability.spatial import (  # noqa: E402
    H4_PROTOCOL_VERSION,
    H4_SUPPORT_PROMPT_VERSION,
    SPATIAL_POINTER_PROMPT_VERSION,
    bbox_area_fraction,
    build_h4_support_prompt,
    build_spatial_pointer_prompt,
    h4_gt_leakage_audit,
    parse_normalized_bbox_json,
    prompt_sha256,
)
from evidence_stability.utils import append_jsonl, read_jsonl, write_json, write_jsonl  # noqa: E402


logger = logging.getLogger("h4_spatial_preflight")


def setup_logging(log_path: str | Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
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
    return {row["cache_key"] for row in read_jsonl(path) if row.get("cache_key")}


def resolve_runtime_paths(
    row: dict[str, Any],
    frame_root: str | None,
    source_frame_prefix: str,
) -> tuple[list[str], list[str]]:
    frozen_paths = [str(path) for path in row["frame_paths"]]
    logical = [
        logical_relative_frame_path(path, frame_root, source_frame_prefix)
        for path in frozen_paths
    ]
    runtime = [
        runtime_frame_path(path, frame_root, source_frame_prefix)
        if frame_root else path
        for path in frozen_paths
    ]
    if len(runtime) != len(frozen_paths) or len(logical) != len(frozen_paths):
        raise RuntimeError(f"Frame path remap changed frame count for {row['candidate_id']}")
    return logical, runtime


def make_h4_cache_key(
    stage: str,
    row: dict[str, Any],
    logical_frame_paths: list[str],
    prompt: str,
    prompt_version: str,
    model_fingerprint: dict[str, Any],
    decoding_config: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "protocol_version": H4_PROTOCOL_VERSION,
            "stage": stage,
            "qa_id": row["qa_id"],
            "candidate_id": row["candidate_id"],
            "window_id": row["window_id"],
            "logical_frame_paths": list(logical_frame_paths),
            "prompt_version": prompt_version,
            "prompt_hash": prompt_sha256(prompt),
            "model_fingerprint": model_fingerprint,
            "decoding_config": decoding_config,
        }
    )


def latest_by_candidate(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("candidate_id"):
            out[row["candidate_id"]] = row
    return out


def bbox_diagnostic(pointer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    areas = [float(row["bbox_area_fraction"]) for row in pointer_rows if row.get("bbox_valid")]
    valid = sum(bool(row.get("bbox_valid")) for row in pointer_rows)
    invalid = len(pointer_rows) - valid
    boxes = [row["predicted_bbox_norm"] for row in pointer_rows if row.get("bbox_valid")]
    boundary_hits = Counter()
    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 == 0:
            boundary_hits["x1_eq_0"] += 1
        if y1 == 0:
            boundary_hits["y1_eq_0"] += 1
        if x2 == 1000:
            boundary_hits["x2_eq_1000"] += 1
        if y2 == 1000:
            boundary_hits["y2_eq_1000"] += 1
    sorted_areas = sorted(areas)
    median_area = sorted_areas[len(sorted_areas) // 2] if sorted_areas else None
    return {
        "n_bbox_requests": len(pointer_rows),
        "valid": valid,
        "invalid": invalid,
        "valid_rate": valid / max(1, len(pointer_rows)),
        "area_mean": sum(areas) / len(areas) if areas else None,
        "area_median": median_area,
        "p_area_gt_0_25": sum(a > 0.25 for a in areas) / max(1, len(areas)),
        "p_area_gt_0_5": sum(a > 0.5 for a in areas) / max(1, len(areas)),
        "p_area_gt_0_75": sum(a > 0.75 for a in areas) / max(1, len(areas)),
        "p_area_gt_0_9": sum(a > 0.9 for a in areas) / max(1, len(areas)),
        "boundary_hits": dict(boundary_hits),
        "whole_frame_like_rate": sum(a > 0.9 for a in areas) / max(1, len(areas)),
        "invalid_reason_counts": dict(Counter(row.get("bbox_invalid_reason") for row in pointer_rows if not row.get("bbox_valid"))),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    stage = summary["study_stage"]
    lines = [
        f"# H4 Spatial {stage.title()} Probe",
        "",
        "## Status",
        f"- protocol version: {summary['protocol_version']}",
        f"- model inference executed: {summary['model_inference_executed']}",
        f"- discovery executed: {summary['discovery_executed']}",
        "",
        "## Support",
        f"- windows: {summary['support']['n_windows']}",
        f"- YES: {summary['support']['counts'].get('YES', 0)}",
        f"- NO: {summary['support']['counts'].get('NO', 0)}",
        f"- INVALID: {summary['support']['counts'].get('INVALID', 0)}",
        f"- ERROR: {summary['support']['counts'].get('ERROR', 0)}",
        f"- new inference count: {summary['support']['new_inference_count']}",
        f"- skipped cached count: {summary['support']['skipped_cached_count']}",
        "",
        "## Spatial Pointer",
        f"- requests: {summary['spatial_pointer']['n_requests']}",
        f"- valid bbox: {summary['spatial_pointer']['bbox_diagnostic']['valid']}",
        f"- invalid bbox: {summary['spatial_pointer']['bbox_diagnostic']['invalid']}",
        f"- bbox area median: {summary['spatial_pointer']['bbox_diagnostic']['area_median']}",
        f"- bbox > 0.75: {summary['spatial_pointer']['bbox_diagnostic']['p_area_gt_0_75']}",
        f"- new inference count: {summary['spatial_pointer']['new_inference_count']}",
        f"- skipped cached count: {summary['spatial_pointer']['skipped_cached_count']}",
        "",
        "## Audit",
        f"- GT leakage: {summary['gt_leakage_audit']['gt_leakage']}",
        f"- missing frame errors: {summary['missing_frame_errors']}",
        f"- processor failures: {summary['processor_failures']}",
        "",
        "KEEP/DROP interventions, formal statistics, and independent confirmation were not run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run H4 support judgment and spatial pointer probing.")
    p.add_argument("--manifest", default="outputs/stg_pilot/phase_g/h4_spatial_v1/manifest/h4_model_manifest_gt_free.jsonl")
    p.add_argument("--output_dir", default="outputs/stg_pilot/phase_g/h4_spatial_v1")
    p.add_argument("--model_backend", choices=["dummy", "openai_compatible", "qwen3_vl"], default="dummy")
    p.add_argument("--model_name", default="dummy-video-vlm")
    p.add_argument("--model_revision", default="v1")
    p.add_argument("--model_path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--processor_min_pixels", type=int, default=None)
    p.add_argument("--processor_max_pixels", type=int, default=None)
    p.add_argument("--frame_root", default=None)
    p.add_argument(
        "--source_frame_prefix",
        default="/root/data,/mnt/hdd3/huihui/hh_datas/MedVidU/valdata,/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
    )
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--support_max_new_tokens", type=int, default=8)
    p.add_argument("--pointer_max_new_tokens", type=int, default=96)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--limit_windows", type=int, default=-1)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log_path", default=None)
    p.add_argument("--study_stage", choices=["preflight", "discovery"], default="preflight")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["predictions", "manifest", "audit", "summary", "provenance"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or out_dir / "predictions" / f"h4_spatial_{args.study_stage}_probe.log"
    setup_logging(log_path)

    rows = read_jsonl(args.manifest)
    if args.limit_windows >= 0:
        rows = rows[: args.limit_windows]
    leakage = h4_gt_leakage_audit(rows)
    if leakage["gt_leakage"]:
        raise RuntimeError(f"H4 model manifest contains GT leakage: {leakage}")

    support_path = out_dir / "predictions" / "support_predictions.jsonl"
    pointer_path = out_dir / "predictions" / "spatial_pointer_predictions.jsonl"
    errors_path = out_dir / "predictions" / "errors.jsonl"
    support_path.touch(exist_ok=True)
    pointer_path.touch(exist_ok=True)
    errors_path.touch(exist_ok=True)

    if args.model_backend == "qwen3_vl" and not args.model_path:
        raise ValueError("--model_path is required for --model_backend qwen3_vl")
    model = build_model(args)

    support_completed = completed_cache_keys(support_path)
    pointer_completed = completed_cache_keys(pointer_path)
    support_counts = Counter()
    pointer_counts = Counter()
    support_new = 0
    pointer_new = 0
    missing_frame_errors = 0
    processor_failures = 0
    started = time.time()

    logger.info("H4 %s support probing windows=%d completed_cache=%d", args.study_stage, len(rows), len(support_completed))
    for i, row in enumerate(rows, start=1):
        prompt = build_h4_support_prompt(row["human_question"])
        try:
            logical_paths, runtime_paths = resolve_runtime_paths(row, args.frame_root, args.source_frame_prefix)
        except Exception as exc:
            support_counts["ERROR"] += 1
            missing_frame_errors += 1
            append_jsonl(errors_path, {"stage": "support", "candidate_id": row.get("candidate_id"), "error_type": exc.__class__.__name__, "error": repr(exc)})
            continue

        set_model_max_new_tokens(model, args.support_max_new_tokens)
        model_fingerprint = model.fingerprint()
        decoding_config = model.generation_config()
        cache_key = make_h4_cache_key("support", row, logical_paths, prompt, H4_SUPPORT_PROMPT_VERSION, model_fingerprint, decoding_config)
        if cache_key in support_completed:
            support_counts["skipped_cached"] += 1
            continue
        if args.dry_run:
            support_counts["dry_run"] += 1
            continue
        missing = [path for path in runtime_paths if not Path(path).exists()]
        if missing:
            support_counts["ERROR"] += 1
            missing_frame_errors += 1
            append_jsonl(
                errors_path,
                {
                    "stage": "support",
                    "candidate_id": row["candidate_id"],
                    "qa_id": row["qa_id"],
                    "error_type": "MISSING_FRAMES",
                    "missing_frame_paths": missing[:20],
                    "cache_key": cache_key,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            continue
        try:
            raw = model.infer(runtime_paths, prompt)
            parsed = parse_yes_no(raw)
            record = {
                "protocol_version": H4_PROTOCOL_VERSION,
                "stage": "support",
                "study_stage": args.study_stage,
                "qa_id": row["qa_id"],
                "clip_id": row["clip_id"],
                "candidate_id": row["candidate_id"],
                "window_id": row["window_id"],
                "dataset_name": row["dataset_name"],
                "human_question": row["human_question"],
                "model_name": model.model_name,
                "model_revision": model.model_revision,
                "model_identity_hash": model_fingerprint.get("model_identity_hash"),
                "prompt_version": H4_SUPPORT_PROMPT_VERSION,
                "prompt_hash": prompt_sha256(prompt),
                "decoding_config": decoding_config,
                "logical_frame_paths": logical_paths,
                "frame_paths": runtime_paths,
                "n_frames": len(runtime_paths),
                "raw_response": raw,
                "parsed_prediction": parsed,
                "cache_key": cache_key,
                "processor_metadata": getattr(model, "last_processor_metadata", {}) or {},
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_jsonl(support_path, record)
            support_completed.add(cache_key)
            support_counts[parsed] += 1
            support_new += 1
        except Exception as exc:
            support_counts["ERROR"] += 1
            processor_failures += 1
            append_jsonl(errors_path, {"stage": "support", "candidate_id": row["candidate_id"], "qa_id": row["qa_id"], "error_type": exc.__class__.__name__, "error": repr(exc), "cache_key": cache_key})
        if i % 50 == 0:
            logger.info("support progress=%d/%d YES=%d NO=%d INVALID=%d ERROR=%d", i, len(rows), support_counts["YES"], support_counts["NO"], support_counts["INVALID"], support_counts["ERROR"])

    support_by_candidate = latest_by_candidate(support_path)
    supporting_rows = [
        {**row, "support_prediction": support_by_candidate[row["candidate_id"]]["parsed_prediction"]}
        for row in rows
        if row["candidate_id"] in support_by_candidate and support_by_candidate[row["candidate_id"]].get("parsed_prediction") == "YES"
    ]
    supporting_manifest = []
    for row in supporting_rows:
        allowed = dict(row)
        supporting_manifest.append(allowed)
    supporting_leakage = h4_gt_leakage_audit(supporting_manifest)
    if supporting_leakage["gt_leakage"]:
        raise RuntimeError(f"H4 supporting manifest contains GT leakage: {supporting_leakage}")
    write_jsonl(out_dir / "manifest" / "h4_supporting_candidates_gt_free.jsonl", supporting_manifest)

    logger.info("H4 %s spatial pointer requests=%d completed_cache=%d", args.study_stage, len(supporting_rows), len(pointer_completed))
    for i, row in enumerate(supporting_rows, start=1):
        prompt = build_spatial_pointer_prompt(row["human_question"])
        try:
            logical_paths, runtime_paths = resolve_runtime_paths(row, args.frame_root, args.source_frame_prefix)
        except Exception as exc:
            pointer_counts["ERROR"] += 1
            missing_frame_errors += 1
            append_jsonl(errors_path, {"stage": "spatial_pointer", "candidate_id": row.get("candidate_id"), "error_type": exc.__class__.__name__, "error": repr(exc)})
            continue
        set_model_max_new_tokens(model, args.pointer_max_new_tokens)
        model_fingerprint = model.fingerprint()
        decoding_config = model.generation_config()
        cache_key = make_h4_cache_key("spatial_pointer", row, logical_paths, prompt, SPATIAL_POINTER_PROMPT_VERSION, model_fingerprint, decoding_config)
        if cache_key in pointer_completed:
            pointer_counts["skipped_cached"] += 1
            continue
        if args.dry_run:
            pointer_counts["dry_run"] += 1
            continue
        missing = [path for path in runtime_paths if not Path(path).exists()]
        if missing:
            pointer_counts["ERROR"] += 1
            missing_frame_errors += 1
            append_jsonl(errors_path, {"stage": "spatial_pointer", "candidate_id": row["candidate_id"], "qa_id": row["qa_id"], "error_type": "MISSING_FRAMES", "missing_frame_paths": missing[:20], "cache_key": cache_key})
            continue
        try:
            raw = model.infer(runtime_paths, prompt)
            parsed = parse_normalized_bbox_json(raw)
            record = {
                "protocol_version": H4_PROTOCOL_VERSION,
                "stage": "spatial_pointer",
                "study_stage": args.study_stage,
                "qa_id": row["qa_id"],
                "clip_id": row["clip_id"],
                "candidate_id": row["candidate_id"],
                "window_id": row["window_id"],
                "dataset_name": row["dataset_name"],
                "human_question": row["human_question"],
                "model_name": model.model_name,
                "model_revision": model.model_revision,
                "model_identity_hash": model_fingerprint.get("model_identity_hash"),
                "prompt_version": SPATIAL_POINTER_PROMPT_VERSION,
                "prompt_hash": prompt_sha256(prompt),
                "decoding_config": decoding_config,
                "logical_frame_paths": logical_paths,
                "frame_paths": runtime_paths,
                "n_frames": len(runtime_paths),
                "support_prediction": "YES",
                "raw_response": raw,
                "bbox_valid": bool(parsed["bbox_valid"]),
                "predicted_bbox_norm": parsed.get("bbox"),
                "bbox_invalid_reason": None if parsed["bbox_valid"] else parsed.get("reason"),
                "bbox_area_fraction": parsed.get("bbox_area_fraction") if parsed["bbox_valid"] else None,
                "cache_key": cache_key,
                "processor_metadata": getattr(model, "last_processor_metadata", {}) or {},
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_jsonl(pointer_path, record)
            pointer_completed.add(cache_key)
            pointer_counts["VALID_BBOX" if parsed["bbox_valid"] else "INVALID_BBOX"] += 1
            pointer_new += 1
        except Exception as exc:
            pointer_counts["ERROR"] += 1
            processor_failures += 1
            append_jsonl(errors_path, {"stage": "spatial_pointer", "candidate_id": row["candidate_id"], "qa_id": row["qa_id"], "error_type": exc.__class__.__name__, "error": repr(exc), "cache_key": cache_key})

    pointer_rows = list(latest_by_candidate(pointer_path).values())
    bbox_diag = bbox_diagnostic(pointer_rows)
    write_json(out_dir / "summary" / "h4_bbox_diagnostic.json", bbox_diag)
    write_json(out_dir / "audit" / "gt_leakage_audit_probe.json", {"input_manifest": leakage, "supporting_manifest": supporting_leakage})
    repo = {
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
    }
    summary = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "study_stage": args.study_stage,
        "manifest": args.manifest,
        "output_dir": str(out_dir),
        "model_backend": args.model_backend,
        "model_path": args.model_path,
        "model_fingerprint": model.fingerprint(),
        "generation_config": model.generation_config(),
        "support": {
            "n_windows": len(rows),
            "counts": dict(support_counts),
            "new_inference_count": support_new,
            "skipped_cached_count": support_counts["skipped_cached"],
            "support_predictions": str(support_path),
        },
        "spatial_pointer": {
            "n_requests": len(supporting_rows),
            "counts": dict(pointer_counts),
            "new_inference_count": pointer_new,
            "skipped_cached_count": pointer_counts["skipped_cached"],
            "spatial_pointer_predictions": str(pointer_path),
            "bbox_diagnostic": bbox_diag,
        },
        "gt_leakage_audit": {"gt_leakage": leakage["gt_leakage"] + supporting_leakage["gt_leakage"]},
        "missing_frame_errors": missing_frame_errors,
        "processor_failures": processor_failures,
        "repo": repo,
        "model_inference_executed": not args.dry_run,
        "spatial_pointing_executed": not args.dry_run,
        "interventions_generated": False,
        "discovery_executed": args.study_stage == "discovery" and not args.dry_run,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    summary_stem = f"h4_spatial_{args.study_stage}_probe_summary"
    write_json(out_dir / "summary" / f"{summary_stem}.json", summary)
    write_summary_md(out_dir / "summary" / f"{summary_stem}.md", summary)
    (out_dir / "provenance" / "environment_snapshot_probe.txt").write_text(environment_snapshot(), encoding="utf-8")
    (out_dir / "provenance" / "git_status_probe.txt").write_text(repo["git_status_short"] + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
