#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.cache import make_h3_dynamic_probe_cache_key, stable_hash  # noqa: E402
from evidence_stability.dynamic import (  # noqa: E402
    BLOCK_ORDER,
    FREEZE_MID_IDX,
    GLOBAL_SEED,
    INTERVENTION_TYPES,
    PRIMARY_INTERVENTIONS,
    PROTOCOL_VERSION,
    assert_model_manifest_gt_free,
    final_records_by_intervention,
    ordered_frame_hash,
    prompt_hash_for_target,
)
from evidence_stability.model_interface import build_model  # noqa: E402
from evidence_stability.phase_d2 import (  # noqa: E402
    frame_count_processor_audit,
    projection_path_audit,
    remap_projection_frame_paths,
    sanitize_processor_metadata,
)
from evidence_stability.prompts import PROMPT_VERSION, build_evidence_presence_prompt, parse_yes_no  # noqa: E402
from evidence_stability.utils import append_jsonl, read_json, read_jsonl, write_json  # noqa: E402


logger = logging.getLogger("h3_dynamic_probe")
CORE_FINGERPRINT_FIELDS = (
    "config_sha256",
    "generation_config_sha256",
    "architectures",
    "model_type",
    "processor_class",
    "model_class",
    "dtype",
    "do_sample",
    "max_new_tokens",
    "enable_thinking",
    "processor_min_pixels",
    "processor_max_pixels",
)
SCIENTIFIC_ENVIRONMENT_FIELDS = ("transformers_version", "torch_version")


def setup_logging(path: str | None) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)


def normalize_model_hash(fingerprint: dict[str, Any], revision: str | None) -> str:
    return str(fingerprint.get("model_identity_hash") or revision or stable_hash(fingerprint))


def decoding_is_frozen(config: dict[str, Any]) -> bool:
    return config.get("do_sample") is False and int(config.get("max_new_tokens", -1)) == 8 and config.get("enable_thinking") is False


def fingerprint_is_frozen(fingerprint: dict[str, Any]) -> dict[str, bool]:
    return {
        "architecture": "Qwen3VLForConditionalGeneration" in (fingerprint.get("architectures") or []),
        "model_class": fingerprint.get("model_class") == "Qwen3VLForConditionalGeneration",
        "processor_class": fingerprint.get("processor_class") == "Qwen3VLProcessor",
        "dtype": fingerprint.get("dtype") == "bfloat16",
        "do_sample": fingerprint.get("do_sample") is False,
        "max_new_tokens": int(fingerprint.get("max_new_tokens", -1)) == 8,
        "enable_thinking": fingerprint.get("enable_thinking") is False,
    }


def phase_c_consistency(
    phase_c_summary_path: str,
    phase_c_probe_results_path: str,
    model_fingerprint: dict[str, Any],
    decoding_config: dict[str, Any],
    projections: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = read_json(phase_c_summary_path)
    phase_c_fp = summary.get("model_fingerprint") or {}
    differences: dict[str, Any] = {}
    core_matches = []
    for field in CORE_FINGERPRINT_FIELDS:
        old = phase_c_fp.get(field)
        new = model_fingerprint.get(field)
        same = old == new
        core_matches.append(same)
        if not same:
            differences[field] = {"phase_c": old, "h3": new}
    env_matches = []
    for field in SCIENTIFIC_ENVIRONMENT_FIELDS:
        old = phase_c_fp.get(field)
        new = model_fingerprint.get(field)
        same = old == new
        env_matches.append(same)
        if not same:
            differences[field] = {"phase_c": old, "h3": new, "scientific_environment_field": True}
    prompt_index = {}
    if phase_c_probe_results_path and Path(phase_c_probe_results_path).exists():
        for row in read_jsonl(phase_c_probe_results_path):
            if row.get("qa_id") and row.get("window_id") and row.get("prompt_hash"):
                prompt_index[(str(row["qa_id"]), str(row["window_id"]))] = str(row["prompt_hash"])
    prompt_checks = []
    for row in projections:
        prompt_hash = prompt_hash_for_target(str(row["target_action"]))
        old_hash = prompt_index.get((str(row["qa_id"]), str(row["candidate_id"])))
        if old_hash is not None:
            prompt_checks.append(old_hash == prompt_hash)
    return {
        "phase_c_summary": phase_c_summary_path,
        "phase_c_probe_results": phase_c_probe_results_path,
        "prompt_version_matches": summary.get("prompt_version") == PROMPT_VERSION,
        "core_model_fingerprint_matches": all(core_matches),
        "scientific_environment_matches": all(env_matches),
        "decoding_config_matches": summary.get("decoding_config") == decoding_config,
        "per_window_prompt_hash_matches": all(prompt_checks) if prompt_checks else None,
        "model_fingerprint_differences": differences,
        "phase_c_model_identity_hash": phase_c_fp.get("model_identity_hash"),
        "h3_model_identity_hash": model_fingerprint.get("model_identity_hash"),
    }


def assert_consistency(checks: dict[str, Any], allow_env_mismatch: bool) -> None:
    required = ["prompt_version_matches", "core_model_fingerprint_matches", "decoding_config_matches"]
    if not allow_env_mismatch:
        required.append("scientific_environment_matches")
    failed = [key for key in required if checks.get(key) is not True]
    if checks.get("per_window_prompt_hash_matches") is False:
        failed.append("per_window_prompt_hash_matches")
    if failed:
        raise RuntimeError(f"H3 frozen model/prompt consistency check failed: {failed}")


def intervention_protocol_for_cache(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("intervention_provenance") or {}
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "intervention_type": row["intervention_type"],
    }
    if row["intervention_type"] == "FULL_SHUFFLE_V1":
        protocol["permutation"] = provenance.get("permutation")
        protocol["seed"] = provenance.get("seed")
    if row["intervention_type"] == "BLOCK_SHUFFLE_4X4_V1":
        protocol["block_order"] = provenance.get("block_order", BLOCK_ORDER)
    if row["intervention_type"] == "FREEZE_MID_V1":
        protocol["mid_idx"] = provenance.get("mid_idx", FREEZE_MID_IDX)
    return protocol


def cache_keys_for(
    projections: list[dict[str, Any]],
    model_hash: str,
    decoding_config: dict[str, Any],
) -> dict[str, str]:
    out = {}
    for row in projections:
        prompt_hash = prompt_hash_for_target(str(row["target_action"]))
        logical_paths = row.get("logical_relative_frame_paths") or row["ordered_frame_paths"]
        ofh = ordered_frame_hash(list(logical_paths))
        out[row["intervention_id"]] = make_h3_dynamic_probe_cache_key(
            model_hash,
            prompt_hash,
            row["qa_id"],
            row["candidate_id"],
            row["intervention_id"],
            row["intervention_type"],
            list(logical_paths),
            ofh,
            row["target_action"],
            decoding_config,
            intervention_protocol_for_cache(row),
        )
    if len(set(out.values())) != len(out):
        raise RuntimeError("H3 cache key collision detected.")
    return out


def completed_cache_keys(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {str(row["cache_key"]) for row in read_jsonl(p) if row.get("cache_key")}


def project_manifest_row(row: dict[str, Any], frame_root: str | None, source_frame_prefix: str) -> dict[str, Any]:
    assert_model_manifest_gt_free(row)
    projection = remap_projection_frame_paths(row, frame_root, source_frame_prefix)
    if projection["n_frames"] != 16 or len(projection["ordered_frame_paths"]) != 16:
        raise RuntimeError(f"H3 dynamic intervention must contain 16 frames: {row['intervention_id']}")
    return projection


def build_raw_result(
    projection: dict[str, Any],
    model: Any,
    model_fingerprint: dict[str, Any],
    model_hash: str,
    decoding_config: dict[str, Any],
    cache_key: str,
    raw_response: str,
    prediction: str,
    runtime_sec: float,
) -> dict[str, Any]:
    logical_paths = projection.get("logical_relative_frame_paths") or projection["ordered_frame_paths"]
    processor_meta = sanitize_processor_metadata(
        getattr(model, "last_processor_metadata", {}) or {},
        expected_frame_count=projection["n_frames"],
    )
    record = {
        "intervention_id": projection["intervention_id"],
        "candidate_id": projection["candidate_id"],
        "qa_id": projection["qa_id"],
        "clip_id": projection.get("clip_id"),
        "dataset_name": projection["dataset_name"],
        "target_action": projection["target_action"],
        "target_field": projection.get("target_field"),
        "intervention_type": projection["intervention_type"],
        "n_frames": projection["n_frames"],
        "ordered_frame_hash": ordered_frame_hash(list(logical_paths)),
        "prompt_hash": prompt_hash_for_target(str(projection["target_action"])),
        "prompt_version": PROMPT_VERSION,
        "model_fingerprint": model_fingerprint,
        "model_fingerprint_core_hash": stable_hash({field: model_fingerprint.get(field) for field in CORE_FINGERPRINT_FIELDS}),
        "model_identity_hash": model_hash,
        "decoding_config": decoding_config,
        "raw_response": raw_response,
        "prediction": prediction,
        "inference_valid": prediction in {"YES", "NO"},
        "error": None if prediction in {"YES", "NO"} else "UNPARSEABLE_RESPONSE",
        "runtime": {"seconds": runtime_sec, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "cache_key": cache_key,
        "processor_metadata": processor_meta,
    }
    assert_model_manifest_gt_free(record, require_declaration=False)
    return record


def select_rows(
    rows: list[dict[str, Any]],
    smoke_candidate_ids_path: str | None,
    run_all: bool,
) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("generation_valid")]
    if run_all:
        return valid
    if not smoke_candidate_ids_path:
        raise ValueError("--smoke_candidate_ids is required unless --run_all is set")
    smoke = read_json(smoke_candidate_ids_path)
    ids = set(smoke.get("candidate_ids") or [])
    selected = [row for row in valid if row["candidate_id"] in ids]
    expected = len(ids) * len(INTERVENTION_TYPES)
    if len(selected) != expected:
        raise RuntimeError(f"H3 smoke selection count mismatch: got {len(selected)} expected {expected}")
    return selected


def processor_frame_audit(debug_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in debug_records:
        by_type[str(row["intervention_type"])].append(row)
    base = frame_count_processor_audit(debug_records)
    checks = {}
    for intervention_type in INTERVENTION_TYPES:
        rows = by_type.get(intervention_type, [])
        checks[intervention_type] = {
            "n_checked": len(rows),
            "all_expected_16": all((row.get("processor_metadata") or {}).get("expected_frame_count") == 16 for row in rows),
            "all_input_frame_count_16": all((row.get("processor_metadata") or {}).get("input_frame_count") == 16 for row in rows),
        }
    checks["frame_count_processor_audit"] = base
    return checks


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase F-D H3 Dynamic Probe Summary",
        "",
        f"- smoke mode: {summary['smoke_mode']}",
        f"- full discovery executed: {summary['full_discovery_executed']}",
        f"- selected interventions: {summary['n_selected_interventions']}",
        f"- completed: {summary['n_completed']}",
        f"- new inference: {summary['new_inference_count']}",
        f"- skipped cached: {summary['skipped_cached_count']}",
        f"- YES: {summary['prediction_counts'].get('YES', 0)}",
        f"- NO: {summary['prediction_counts'].get('NO', 0)}",
        f"- INVALID: {summary['prediction_counts'].get('INVALID', 0)}",
        f"- ERROR: {summary['prediction_counts'].get('ERROR', 0)}",
        f"- prompt hash matches H2: {summary['phase_c_consistency'].get('per_window_prompt_hash_matches')}",
        f"- core fingerprint matches H2: {summary['phase_c_consistency'].get('core_model_fingerprint_matches')}",
        f"- scientific environment matches H2: {summary['phase_c_consistency'].get('scientific_environment_matches')}",
        "",
        "Descriptive H3 probing only. No formal statistics or independent confirmation was run.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe Phase F-D H3 dynamic interventions.")
    p.add_argument("--manifest", default="outputs/tal_pilot/phase_f/h3_dynamic_v1/manifest/h3_model_intervention_manifest_gt_free.jsonl")
    p.add_argument("--smoke_candidate_ids", default="outputs/tal_pilot/phase_f/h3_dynamic_v1/manifest/smoke_candidate_ids.json")
    p.add_argument("--phase_c_probe_summary", default="outputs/tal_pilot/phase_c_server/qwen3_vl_8b_pilot50/phase_c_pilot_probe_summary.json")
    p.add_argument("--phase_c_probe_results", default="outputs/tal_pilot/phase_c_server/qwen3_vl_8b_pilot50/probe_results.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_f/h3_dynamic_v1/predictions/smoke")
    p.add_argument("--probe_results", default=None)
    p.add_argument("--errors", default=None)
    p.add_argument("--run_all", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--allow_scientific_environment_mismatch", action="store_true")
    p.add_argument("--model_backend", choices=["dummy", "openai_compatible", "qwen3_vl"], default="qwen3_vl")
    p.add_argument("--model_name", default="dummy-video-vlm")
    p.add_argument("--model_revision", default="v1")
    p.add_argument("--model_path", default="/mnt/hdd3/huihui/models/Qwen3-VL-8B-Instruct")
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
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_results = args.probe_results or str(out_dir / "h3_dynamic_probe_results.jsonl")
    errors_path = args.errors or str(out_dir / "h3_dynamic_errors.jsonl")
    Path(probe_results).parent.mkdir(parents=True, exist_ok=True)
    Path(probe_results).touch(exist_ok=True)
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    Path(errors_path).touch(exist_ok=True)
    setup_logging(args.log_path or str(out_dir / "probe.log"))

    rows = read_jsonl(args.manifest)
    selected = select_rows(rows, args.smoke_candidate_ids, args.run_all)
    projections = [project_manifest_row(row, args.frame_root, args.source_frame_prefix) for row in selected]
    ids = [row["intervention_id"] for row in projections]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate H3 intervention IDs: {duplicates[:5]}")
    path_audit = projection_path_audit(projections)
    if path_audit["n_missing_frame_references"] and not args.dry_run:
        summary = {
            "stop_reason": "Missing frame references detected before model loading.",
            "path_audit": path_audit,
            "real_model_inference_executed": False,
        }
        write_json(out_dir / ("h3_dynamic_full_summary.json" if args.run_all else "h3_dynamic_smoke_summary.json"), summary)
        raise RuntimeError(summary["stop_reason"])

    model = build_model(args)
    model_fingerprint = model.fingerprint()
    decoding_config = model.generation_config()
    model_hash = normalize_model_hash(model_fingerprint, model.model_revision)
    frozen_checks = fingerprint_is_frozen(model_fingerprint) if args.model_backend == "qwen3_vl" else {}
    if args.model_backend == "qwen3_vl":
        failed = [key for key, ok in frozen_checks.items() if not ok]
        if failed:
            raise RuntimeError(f"H3 frozen Qwen config check failed: {failed}")
        if not decoding_is_frozen(decoding_config):
            raise RuntimeError(f"H3 decoding config is not frozen: {decoding_config}")
    consistency = phase_c_consistency(
        args.phase_c_probe_summary,
        args.phase_c_probe_results,
        model_fingerprint,
        decoding_config,
        projections,
    )
    assert_consistency(consistency, args.allow_scientific_environment_mismatch)

    cache_keys = cache_keys_for(projections, model_hash, decoding_config)
    done_cache = completed_cache_keys(probe_results)
    completed_cache_at_start = len(done_cache)
    if args.preflight_only:
        summary = {
            "smoke_mode": not args.run_all,
            "full_discovery_executed": False,
            "n_selected_interventions": len(projections),
            "path_audit": path_audit,
            "frozen_qwen_config_checks": frozen_checks,
            "phase_c_consistency": consistency,
            "completed_cache_at_start": completed_cache_at_start,
            "real_model_inference_executed": False,
        }
        write_json(out_dir / ("h3_dynamic_full_summary.json" if args.run_all else "h3_dynamic_smoke_summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    counts = Counter()
    debug_records: list[dict[str, Any]] = []
    logger.info("H3 dynamic probing selected=%d completed_cache=%d", len(projections), completed_cache_at_start)
    for projection in projections:
        cache_key = cache_keys[projection["intervention_id"]]
        if cache_key in done_cache:
            counts["skipped_cached"] += 1
            continue
        if args.dry_run:
            counts["dry_run"] += 1
            continue
        prompt = build_evidence_presence_prompt(str(projection["target_action"]))
        try:
            started = time.time()
            raw = model.infer(projection["ordered_frame_paths"], prompt)
            runtime_sec = time.time() - started
            prediction = parse_yes_no(raw)
            record = build_raw_result(
                projection,
                model,
                model_fingerprint,
                model_hash,
                decoding_config,
                cache_key,
                raw,
                prediction,
                runtime_sec,
            )
            append_jsonl(probe_results, record)
            done_cache.add(cache_key)
            counts[prediction] += 1
            if len(debug_records) < 18 or sum(r["intervention_type"] == projection["intervention_type"] for r in debug_records) < 3:
                debug_records.append(
                    {
                        "intervention_id": projection["intervention_id"],
                        "intervention_type": projection["intervention_type"],
                        "processor_metadata": record["processor_metadata"],
                    }
                )
        except Exception as exc:
            counts["ERROR"] += 1
            append_jsonl(
                errors_path,
                {
                    "intervention_id": projection["intervention_id"],
                    "candidate_id": projection["candidate_id"],
                    "qa_id": projection["qa_id"],
                    "dataset_name": projection["dataset_name"],
                    "target_action": projection["target_action"],
                    "target_field": projection.get("target_field"),
                    "intervention_type": projection["intervention_type"],
                    "prediction": "ERROR",
                    "cache_key": cache_key,
                    "error_type": exc.__class__.__name__,
                    "error": repr(exc),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            if "out of memory" in repr(exc).lower() or "cuda oom" in repr(exc).lower():
                raise

    expected_ids = {row["intervention_id"] for row in projections}
    final_results, completion_audit = final_records_by_intervention(read_jsonl(probe_results), read_jsonl(errors_path), expected_ids)
    final_counts = Counter(row.get("prediction", "INVALID") for row in final_results.values())
    final_counts["ERROR"] = completion_audit["unresolved_errors"]
    n_completed = completion_audit["n_final_success_results"] + completion_audit["unresolved_errors"]
    processor_audit = processor_frame_audit(debug_records)
    if not args.run_all and not args.dry_run:
        if len(projections) != 36:
            raise RuntimeError(f"H3 smoke expected 36 interventions, got {len(projections)}")
        if n_completed != 36:
            raise RuntimeError(f"H3 smoke incomplete: {completion_audit}")
        if final_counts["INVALID"] or final_counts["ERROR"]:
            raise RuntimeError(f"H3 smoke invalid/error predictions: {dict(final_counts)}")
        freeze = processor_audit.get("FREEZE_MID_V1", {})
        if freeze.get("n_checked", 0) and not freeze.get("all_input_frame_count_16"):
            raise RuntimeError(f"FREEZE processor did not preserve 16 logical entries: {freeze}")
    if args.run_all and not args.dry_run:
        if len(projections) != 474:
            raise RuntimeError(f"H3 full expected 474 interventions, got {len(projections)}")
        if n_completed != 474:
            raise RuntimeError(f"H3 full incomplete: {completion_audit}")

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "study_stage": "h3_dynamic_discovery",
        "smoke_mode": not args.run_all,
        "full_discovery_executed": bool(args.run_all and not args.dry_run),
        "manifest": args.manifest,
        "probe_results": probe_results,
        "errors": errors_path,
        "n_selected_interventions": len(projections),
        "n_completed": n_completed,
        "completed_cache_at_start": completed_cache_at_start,
        "new_inference_count": counts["YES"] + counts["NO"] + counts["INVALID"],
        "skipped_cached_count": counts["skipped_cached"],
        "prediction_counts": dict(final_counts),
        "new_prediction_counts_this_run": dict(counts),
        "intervention_type_counts": dict(Counter(row["intervention_type"] for row in projections)),
        "dataset_counts": dict(Counter(row["dataset_name"] for row in projections)),
        "target_field_counts": dict(Counter(row.get("target_field") for row in projections)),
        "path_audit": path_audit,
        "model_fingerprint": model_fingerprint,
        "model_identity_hash": model_hash,
        "decoding_config": decoding_config,
        "frozen_qwen_config_checks": frozen_checks,
        "phase_c_consistency": consistency,
        "processor_frame_audit": processor_audit,
        "completion_audit": completion_audit,
        "gt_information_used_in_model_inference": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
    }
    summary_name = "h3_dynamic_full_summary.json" if args.run_all else "h3_dynamic_smoke_summary.json"
    write_json(out_dir / summary_name, summary)
    write_summary_md(out_dir / summary_name.replace(".json", ".md"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
