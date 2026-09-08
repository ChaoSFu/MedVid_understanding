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

from evidence_stability.cache import make_intervention_probe_cache_key, stable_hash  # noqa: E402
from evidence_stability.model_interface import build_model  # noqa: E402
from evidence_stability.phase_d2 import (  # noqa: E402
    assert_no_forbidden_model_fields,
    assert_raw_result_gt_free,
    audit_cache_collisions,
    deterministic_smoke_select,
    frame_count_distribution,
    frame_count_processor_audit,
    memory_leak_flag,
    project_intervention_for_model,
    projection_path_audit,
    remap_projection_frame_paths,
    sanitize_processor_metadata,
    select_generation_valid_interventions,
    smoke_selection_artifact,
)
from evidence_stability.prompts import PROMPT_VERSION, build_evidence_presence_prompt, parse_yes_no  # noqa: E402
from evidence_stability.utils import append_jsonl, read_json, read_jsonl, write_json  # noqa: E402


logger = logging.getLogger("phase_d2_probe")
FROZEN_QWEN_BACKEND = "qwen3_vl"
FROZEN_QWEN_MODEL_PATH = "/mnt/hdd3/huihui/models/Qwen3-VL-8B-Instruct"
EXPECTED_FRAME_COUNTS = {8, 12, 16, 24}
CORE_PHASE_C_FINGERPRINT_FIELDS = (
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
SCIENTIFIC_ENVIRONMENT_FIELDS = (
    "transformers_version",
    "torch_version",
)


def setup_logging(log_path: str | None) -> None:
    logger.setLevel(logging.INFO)
    for existing_handler in logger.handlers:
        existing_handler.close()
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)


def load_completed_cache_keys(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    keys: set[str] = set()
    for row in read_jsonl(p):
        cache_key = row.get("cache_key")
        if cache_key:
            keys.add(str(cache_key))
    return keys


def assert_unique_field(rows: list[dict[str, Any]], field: str, context: str) -> None:
    counts = Counter(str(row.get(field)) for row in rows)
    duplicates = [value for value, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate {field} in {context}: {duplicates[:5]}")


def load_final_completion_records(
    probe_results: str | Path,
    errors_path: str | Path,
    expected_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_result_rows = read_jsonl(probe_results)
    extra_result_ids = sorted(
        {str(row.get("intervention_id")) for row in all_result_rows}
        - expected_ids
    )
    result_rows = [
        row for row in all_result_rows
        if str(row.get("intervention_id")) in expected_ids
    ]
    result_ids = [str(row.get("intervention_id")) for row in result_rows]
    duplicate_result_ids = [value for value, count in Counter(result_ids).items() if count > 1]
    if duplicate_result_ids:
        raise RuntimeError(f"Duplicate intervention results: {duplicate_result_ids[:5]}")

    for row in result_rows:
        assert_raw_result_gt_free(row)

    result_id_set = set(result_ids)
    latest_errors: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(errors_path):
        intervention_id = str(row.get("intervention_id"))
        if intervention_id in expected_ids and intervention_id not in result_id_set:
            latest_errors[intervention_id] = row
    final_error_rows = list(latest_errors.values())
    completed_ids = result_id_set | set(latest_errors)
    audit = {
        "n_completed_results": len(result_rows),
        "n_completed_errors_without_result": len(final_error_rows),
        "n_completed_final_records": len(completed_ids),
        "missing_intervention_ids": sorted(expected_ids - completed_ids)[:20],
        "extra_result_ids": extra_result_ids[:20],
        "duplicate_result_ids": duplicate_result_ids,
    }
    return result_rows, final_error_rows, audit


def final_prediction_counts(result_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]]) -> Counter:
    counts = Counter(str(row.get("parsed_prediction", "INVALID")) for row in result_rows)
    counts["ERROR"] += len(error_rows)
    return counts


def prediction_distributions(
    result_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    projections: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    projection_by_id = {str(row["intervention_id"]): row for row in projections}
    final_rows = []
    for row in result_rows:
        final_rows.append((str(row["intervention_id"]), str(row.get("parsed_prediction", "INVALID"))))
    for row in error_rows:
        final_rows.append((str(row["intervention_id"]), "ERROR"))

    specs = {
        "by_intervention_type": lambda p: str(p["intervention_type"]),
        "by_intervention_family": lambda p: str(p["intervention_family"]),
        "by_dataset": lambda p: str(p["dataset_name"]),
        "by_target_field": lambda p: str(p.get("target_field")),
        "by_frame_count": lambda p: str(p["n_frames"]),
    }
    out: dict[str, dict[str, dict[str, int]]] = {}
    for name, getter in specs.items():
        table: dict[str, Counter] = defaultdict(Counter)
        for intervention_id, prediction in final_rows:
            projection = projection_by_id.get(intervention_id)
            if projection:
                table[getter(projection)][prediction] += 1
        out[name] = {group: dict(counts) for group, counts in sorted(table.items())}
    return out


def model_identity_hash(model_fingerprint: dict[str, Any], model_revision: str | None) -> str:
    return str(model_fingerprint.get("model_identity_hash") or model_revision or stable_hash(model_fingerprint))


def normalize_model_path(path: Any) -> str | None:
    if path is None:
        return None
    return str(Path(str(path)).expanduser().resolve(strict=False))


def core_fingerprint_value_matches(field: str, old_value: Any, new_value: Any) -> bool:
    return old_value == new_value


def image_sizes(paths: list[str]) -> list[list[int]]:
    try:
        from PIL import Image
    except ImportError:
        return []
    sizes: list[list[int]] = []
    for path in paths:
        with Image.open(path) as img:
            sizes.append([int(img.size[0]), int(img.size[1])])
    return sizes


def resolve_projection_frame_paths(
    projection: dict[str, Any],
    frame_root: str | None,
    source_frame_prefix: str,
) -> dict[str, Any]:
    return remap_projection_frame_paths(projection, frame_root, source_frame_prefix)


def prompt_for_projection(projection: dict[str, Any]) -> tuple[str, str]:
    prompt = build_evidence_presence_prompt(str(projection["target_action"]))
    forbidden_terms = ["gt span", "ground truth", "gt_alignment", "candidate_label", "true_support", "spurious_support"]
    if any(term in prompt.lower() for term in forbidden_terms):
        raise RuntimeError(f"Forbidden GT/label term found in prompt for {projection['intervention_id']}")
    return prompt, stable_hash({"prompt": prompt})


def decoding_config_is_frozen(decoding_config: dict[str, Any]) -> bool:
    return (
        decoding_config.get("do_sample") is False
        and int(decoding_config.get("max_new_tokens", -1)) == 8
        and decoding_config.get("enable_thinking") is False
    )


def fingerprint_is_frozen_qwen(args: argparse.Namespace, fingerprint: dict[str, Any]) -> dict[str, bool]:
    return {
        "model_backend": args.model_backend == FROZEN_QWEN_BACKEND,
        "model_path_argument_matches_loaded_model": normalize_model_path(args.model_path) == normalize_model_path(fingerprint.get("model_path")),
        "architecture": "Qwen3VLForConditionalGeneration" in (fingerprint.get("architectures") or []),
        "model_class": fingerprint.get("model_class") == "Qwen3VLForConditionalGeneration",
        "processor_class": fingerprint.get("processor_class") == "Qwen3VLProcessor",
        "dtype": fingerprint.get("dtype") == "bfloat16",
        "do_sample": fingerprint.get("do_sample") is False,
        "max_new_tokens": int(fingerprint.get("max_new_tokens", -1)) == 8,
        "enable_thinking": fingerprint.get("enable_thinking") is False,
    }


def load_phase_c_prompt_index(path: str | None) -> dict[tuple[str, str], str]:
    if not path or not Path(path).exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    for row in read_jsonl(path):
        qa_id = row.get("qa_id")
        window_id = row.get("window_id")
        prompt_hash = row.get("prompt_hash")
        if qa_id and window_id and prompt_hash:
            out[(str(qa_id), str(window_id))] = str(prompt_hash)
    return out


def load_smoke_selection_artifact(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return read_json(p)


def select_from_smoke_artifact(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    requested_ids = [
        str(item["intervention_id"])
        for item in artifact.get("interventions", [])
    ]
    if not requested_ids:
        raise RuntimeError("Smoke selection artifact contains no intervention IDs.")
    by_id = {str(row["intervention_id"]): row for row in rows}
    missing = [intervention_id for intervention_id in requested_ids if intervention_id not in by_id]
    if missing:
        raise RuntimeError(f"Smoke selection artifact references missing intervention IDs: {missing[:5]}")
    selected = [by_id[intervention_id] for intervention_id in requested_ids]
    if any(not row.get("generation_valid") for row in selected):
        raise RuntimeError("Smoke selection artifact contains generation-invalid interventions.")
    return selected


def phase_c_consistency_check(
    args: argparse.Namespace,
    model_fingerprint: dict[str, Any],
    decoding_config: dict[str, Any],
    projections: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "checked": False,
        "summary_path": args.phase_c_probe_summary,
        "probe_results_path": args.phase_c_probe_results,
        "prompt_version_matches": None,
        "model_identity_hash_matches": None,
        "core_model_fingerprint_matches": None,
        "scientific_environment_matches": None,
        "model_fingerprint_differences": {},
        "decoding_config_matches": None,
        "per_window_prompt_hash_matches": None,
    }
    if args.skip_phase_c_consistency:
        checks["skipped"] = True
        return checks

    summary = read_json(args.phase_c_probe_summary)
    checks["checked"] = True
    checks["prompt_version_matches"] = summary.get("prompt_version") == args.prompt_version
    phase_c_fp = summary.get("model_fingerprint") or {}
    checks["model_identity_hash_matches"] = (
        phase_c_fp.get("model_identity_hash") == model_fingerprint.get("model_identity_hash")
    )
    differences = {}
    core_matches = []
    for field in CORE_PHASE_C_FINGERPRINT_FIELDS:
        old_value = phase_c_fp.get(field)
        new_value = model_fingerprint.get(field)
        same = core_fingerprint_value_matches(field, old_value, new_value)
        core_matches.append(same)
        if not same:
            differences[field] = {
                "phase_c": old_value,
                "phase_d2": new_value,
            }
    env_matches = []
    for field in SCIENTIFIC_ENVIRONMENT_FIELDS:
        old_value = phase_c_fp.get(field)
        new_value = model_fingerprint.get(field)
        same = old_value == new_value
        env_matches.append(same)
        if not same:
            differences[field] = {
                "phase_c": old_value,
                "phase_d2": new_value,
                "scientific_environment_field": True,
            }
    runtime_fields = sorted(
        (set(phase_c_fp) | set(model_fingerprint))
        - set(CORE_PHASE_C_FINGERPRINT_FIELDS)
        - set(SCIENTIFIC_ENVIRONMENT_FIELDS)
    )
    for field in runtime_fields:
        old_value = phase_c_fp.get(field)
        new_value = model_fingerprint.get(field)
        if old_value != new_value:
            differences[field] = {
                "phase_c": old_value,
                "phase_d2": new_value,
                "runtime_or_hash_field": True,
            }
    checks["core_model_fingerprint_matches"] = all(core_matches)
    checks["scientific_environment_matches"] = all(env_matches)
    checks["model_fingerprint_differences"] = differences
    checks["decoding_config_matches"] = summary.get("decoding_config") == decoding_config

    prompt_index = load_phase_c_prompt_index(args.phase_c_probe_results)
    if prompt_index:
        prompt_checks = []
        for projection in projections:
            _, prompt_hash = prompt_for_projection(projection)
            old_hash = prompt_index.get((projection["qa_id"], projection["window_id"]))
            if old_hash is not None:
                prompt_checks.append(old_hash == prompt_hash)
        checks["per_window_prompt_hash_matches"] = all(prompt_checks) if prompt_checks else None
    else:
        checks["per_window_prompt_hash_matches"] = None
    return checks


def assert_phase_c_consistency(checks: dict[str, Any], allow_scientific_environment_mismatch: bool = False) -> None:
    if checks.get("skipped"):
        return
    required = ["prompt_version_matches", "core_model_fingerprint_matches", "decoding_config_matches"]
    if not allow_scientific_environment_mismatch:
        required.append("scientific_environment_matches")
    failed = [name for name in required if checks.get(name) is not True]
    if checks.get("per_window_prompt_hash_matches") is False:
        failed.append("per_window_prompt_hash_matches")
    if failed:
        raise RuntimeError(f"Phase C consistency check failed: {failed}")


def assert_smoke_selection_coverage(projections: list[dict[str, Any]]) -> dict[str, Any]:
    frame_counts = set(frame_count_distribution(projections))
    datasets = {p["dataset_name"] for p in projections}
    target_fields = {p.get("target_field") for p in projections}
    intervention_types = {p["intervention_type"] for p in projections}
    coverage = {
        "covers_8_12_16_24_frames": EXPECTED_FRAME_COUNTS <= frame_counts,
        "covers_at_least_two_datasets": len(datasets) >= 2,
        "covers_action_and_phase": {"action", "phase"} <= target_fields,
        "covers_shift": any(t.startswith("SHIFT_") for t in intervention_types),
        "covers_resample_50": "RESAMPLE_50" in intervention_types,
        "covers_resample_75": "RESAMPLE_75" in intervention_types,
        "covers_context": "CONTEXT_1P5X" in intervention_types,
    }
    failed = [k for k, ok in coverage.items() if not ok]
    if failed:
        raise RuntimeError(f"Smoke selection coverage failed: {failed}")
    return coverage


def cache_key_audit(
    projections: list[dict[str, Any]],
    model_hash: str,
    decoding_config: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    keys: dict[str, str] = {}
    stable = True
    for projection in projections:
        _, prompt_hash = prompt_for_projection(projection)
        key_a = make_intervention_probe_cache_key(
            model_hash,
            prompt_hash,
            projection["qa_id"],
            projection["window_id"],
            projection["intervention_id"],
            projection.get("logical_relative_frame_paths", projection["ordered_frame_paths"]),
            projection["n_frames"],
            decoding_config,
        )
        key_b = make_intervention_probe_cache_key(
            model_hash,
            prompt_hash,
            projection["qa_id"],
            projection["window_id"],
            projection["intervention_id"],
            projection.get("logical_relative_frame_paths", projection["ordered_frame_paths"]),
            projection["n_frames"],
            decoding_config,
        )
        stable = stable and key_a == key_b
        keys[projection["intervention_id"]] = key_a
    audit = audit_cache_collisions(projections, keys)
    audit["same_rerun_cache_key_stable"] = stable
    audit["different_intervention_ids_have_different_cache_keys"] = len(set(keys.values())) == len(keys)
    return keys, audit


def build_raw_result(
    projection: dict[str, Any],
    model: Any,
    model_fingerprint: dict[str, Any],
    model_hash: str,
    decoding_config: dict[str, Any],
    prompt_hash: str,
    cache_key: str,
    raw_response: str,
    parsed: str,
) -> dict[str, Any]:
    processor_meta = sanitize_processor_metadata(
        getattr(model, "last_processor_metadata", {}) or {},
        expected_frame_count=projection["n_frames"],
    )
    record = {
        "qa_id": projection["qa_id"],
        "clip_id": projection["clip_id"],
        "window_id": projection["window_id"],
        "intervention_id": projection["intervention_id"],
        "dataset_name": projection["dataset_name"],
        "target_action": projection["target_action"],
        "intervention_family": projection["intervention_family"],
        "intervention_type": projection["intervention_type"],
        "model_name": model.model_name,
        "model_revision": model.model_revision,
        "model_identity_hash": model_hash,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash,
        "decoding_config": decoding_config,
        "ordered_frame_paths": projection["ordered_frame_paths"],
        "n_frames": projection["n_frames"],
        "n_unique_frames": projection["n_unique_frames"],
        "raw_response": raw_response,
        "parsed_prediction": parsed,
        "cache_key": cache_key,
        "processor_metadata": processor_meta,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    assert_raw_result_gt_free(record)
    return record


def write_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    full_label = "Smoke" if summary.get("smoke_mode") else "Full"
    lines = [
        f"# Phase D.2 {full_label} Probe Summary",
        "",
        "## Scope",
        f"- smoke mode: {summary.get('smoke_mode')}",
        f"- generation-valid interventions available: {summary.get('n_generation_valid')}",
        f"- generation-invalid interventions: {summary.get('n_generation_invalid')}",
        f"- selected interventions: {summary.get('n_selected_interventions')}",
        f"- full run executed: {summary.get('full_run_executed')}",
        "",
        "## Engineering Counts",
        f"- frame count distribution: {summary.get('frame_count_distribution')}",
        f"- intervention type distribution: {summary.get('intervention_type_distribution')}",
        f"- dataset distribution: {summary.get('dataset_distribution')}",
        f"- target field distribution: {summary.get('target_field_distribution')}",
        "",
        "## Inference",
        f"- completed final records: {summary.get('n_completed')}",
        f"- remaining interventions at start: {summary.get('remaining_interventions_at_start')}",
        f"- remaining interventions at completion: {summary.get('remaining_interventions')}",
        f"- new inference count: {summary.get('new_inference_count')}",
        f"- skipped cached count: {summary.get('skipped_cached_count')}",
        f"- YES: {summary.get('prediction_counts', {}).get('YES', 0)}",
        f"- NO: {summary.get('prediction_counts', {}).get('NO', 0)}",
        f"- INVALID: {summary.get('prediction_counts', {}).get('INVALID', 0)}",
        f"- ERROR: {summary.get('prediction_counts', {}).get('ERROR', 0)}",
        "",
        "## Checks",
        f"- generation_valid only selection: {summary.get('generation_valid_only_selection')}",
        f"- GT information used in model inference: {summary.get('gt_information_used_in_model_inference')}",
        f"- frozen Qwen config checks: {summary.get('frozen_qwen_config_checks')}",
        f"- Phase C consistency: {summary.get('phase_c_consistency')}",
        f"- processor frame-count audit: {summary.get('processor_frame_count_audit')}",
        f"- path audit: {summary.get('path_audit')}",
        f"- unexpected temporal resampling: {summary.get('unexpected_temporal_resampling')}",
        f"- cache audit: {summary.get('cache_audit')}",
        f"- possible memory leak: {summary.get('possible_memory_leak')}",
        "",
        "Phase D.2 stops here. No Phase D.3 join, Phase E stability analysis, or H2 interpretation was executed.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D.2 frozen-Qwen intervention probing.")
    p.add_argument("--interventions", default="outputs/tal_pilot/phase_d/intervention_pilot/interventions.jsonl")
    p.add_argument("--phase_c_probe_summary", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50/phase_c_pilot_probe_summary.json")
    p.add_argument("--phase_c_probe_results", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50/probe_results.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_d/qwen3_vl_8b_intervention_smoke")
    p.add_argument("--probe_results", default=None)
    p.add_argument("--errors", default=None)
    p.add_argument("--selection_output", default=None)
    p.add_argument("--reuse_selection_from", default=None)
    p.add_argument("--summary_output", default=None)
    p.add_argument("--log_path", default=None)
    p.add_argument("--max_interventions", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected_generation_valid", type=int, default=777)
    p.add_argument("--run_all_generation_valid", action="store_true")
    p.add_argument("--skip_phase_c_consistency", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--allow_scientific_environment_mismatch", action="store_true")
    p.add_argument("--prompt_version", default=PROMPT_VERSION)
    p.add_argument("--model_backend", choices=["dummy", "openai_compatible", "qwen3_vl", "local_hf_vlm"], default=FROZEN_QWEN_BACKEND)
    p.add_argument("--model_name", default="dummy-video-vlm")
    p.add_argument("--model_revision", default="v1")
    p.add_argument("--model_path", default=FROZEN_QWEN_MODEL_PATH)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_map", choices=["auto", "balanced", "balanced_low_0", "sequential"], default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--processor_min_pixels", type=int, default=None)
    p.add_argument("--processor_max_pixels", type=int, default=None)
    p.add_argument("--frame_root", default=None)
    p.add_argument(
        "--source_frame_prefix",
        default="/root/data,/mnt/hdd3/huihui/hh_datas/MedVidU/valdata,/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
        help="Comma-separated known logical roots from Phase C/D artifacts.",
    )
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_results = args.probe_results or str(out_dir / "intervention_probe_results.jsonl")
    errors_path = args.errors or str(out_dir / "errors.jsonl")
    selection_output = args.selection_output or str(out_dir / "phase_d2_smoke_intervention_ids.json")
    default_summary_name = "phase_d2_full_summary.json" if args.run_all_generation_valid else "phase_d2_smoke_summary.json"
    summary_output = args.summary_output or str(out_dir / default_summary_name)
    summary_md_output = str(Path(summary_output).with_suffix(".md"))
    Path(probe_results).parent.mkdir(parents=True, exist_ok=True)
    Path(probe_results).touch(exist_ok=True)
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    Path(errors_path).touch(exist_ok=True)
    setup_logging(args.log_path or str(out_dir / "probe.log"))

    if args.prompt_version != PROMPT_VERSION:
        raise RuntimeError(f"Prompt version mismatch: {args.prompt_version} != {PROMPT_VERSION}")
    if args.model_backend in {"qwen3_vl", "local_hf_vlm"} and not args.model_path:
        raise ValueError("--model_path is required when using a local Hugging Face VLM backend")

    raw_rows = read_jsonl(args.interventions)
    assert_unique_field(raw_rows, "intervention_id", "Phase D.1 intervention manifest")
    generation_valid_rows = select_generation_valid_interventions(raw_rows)
    assert_unique_field(generation_valid_rows, "intervention_id", "generation-valid intervention selection")
    n_generation_valid = len(generation_valid_rows)
    n_generation_invalid = len(raw_rows) - n_generation_valid
    if args.expected_generation_valid >= 0 and n_generation_valid != args.expected_generation_valid:
        raise RuntimeError(
            f"generation_valid count mismatch: got {n_generation_valid}, expected {args.expected_generation_valid}"
        )

    reused_selection_artifact = None
    if args.run_all_generation_valid:
        source_selection = generation_valid_rows
    else:
        reused_selection_artifact = (
            load_smoke_selection_artifact(args.reuse_selection_from)
            or load_smoke_selection_artifact(selection_output)
        )
        if reused_selection_artifact:
            source_selection = select_from_smoke_artifact(generation_valid_rows, reused_selection_artifact)
        else:
            source_selection = deterministic_smoke_select(
                generation_valid_rows,
                n=args.max_interventions,
                seed=args.seed,
            )
    projections = [
        resolve_projection_frame_paths(
            project_intervention_for_model(row),
            args.frame_root,
            args.source_frame_prefix,
        )
        for row in source_selection
    ]
    if not args.run_all_generation_valid:
        assert_smoke_selection_coverage(projections)
        write_json(selection_output, reused_selection_artifact or smoke_selection_artifact(source_selection, args.seed))

    for projection in projections:
        assert_no_forbidden_model_fields(projection, "model-side intervention projection")
        if projection["n_frames"] not in EXPECTED_FRAME_COUNTS:
            raise RuntimeError(f"Unexpected frame count in {projection['intervention_id']}: {projection['n_frames']}")
        if not projection["ordered_frame_paths"]:
            raise RuntimeError(f"Empty frame sequence for {projection['intervention_id']}")
        if projection["n_frames"] != len(projection["ordered_frame_paths"]):
            raise RuntimeError(f"Frame count mismatch for {projection['intervention_id']}")
    path_audit = projection_path_audit(projections)
    if not args.dry_run and path_audit["n_missing_frame_references"]:
        preflight_summary = {
            "report_title": "Phase D.2 Preflight Frame Audit Stop Summary",
            "smoke_mode": not args.run_all_generation_valid,
            "full_run_executed": False,
            "interventions_input": args.interventions,
            "n_total_interventions": len(raw_rows),
            "n_generation_valid": n_generation_valid,
            "n_generation_invalid": n_generation_invalid,
            "n_selected_interventions": len(projections),
            "frame_count_distribution": frame_count_distribution(projections),
            "intervention_type_distribution": dict(Counter(p["intervention_type"] for p in projections)),
            "intervention_family_distribution": dict(Counter(p["intervention_family"] for p in projections)),
            "dataset_distribution": dict(Counter(p["dataset_name"] for p in projections)),
            "target_field_distribution": dict(Counter(str(p.get("target_field")) for p in projections)),
            "frame_root": args.frame_root,
            "source_frame_prefix": args.source_frame_prefix,
            "path_audit": path_audit,
            "real_model_inference_executed": False,
            "stop_reason": "Missing frame references detected before model loading.",
        }
        write_json(summary_output, preflight_summary)
        write_summary_md(summary_md_output, preflight_summary)
        raise RuntimeError(
            "Missing frame references detected before model loading: "
            f"{path_audit['n_missing_frame_references']} frame references across "
            f"{path_audit['n_interventions_with_missing_frames']} interventions."
        )

    model = build_model(args)
    model_fingerprint = model.fingerprint()
    decoding_config = model.generation_config()
    model_hash = model_identity_hash(model_fingerprint, model.model_revision)
    frozen_checks = fingerprint_is_frozen_qwen(args, model_fingerprint) if args.model_backend == "qwen3_vl" else {}
    if args.model_backend == "qwen3_vl":
        failed = [name for name, ok in frozen_checks.items() if not ok]
        if failed:
            raise RuntimeError(f"Frozen Qwen config check failed: {failed}")
        if not decoding_config_is_frozen(decoding_config):
            raise RuntimeError(f"Frozen decoding config check failed: {decoding_config}")

    consistency = phase_c_consistency_check(args, model_fingerprint, decoding_config, projections)
    try:
        assert_phase_c_consistency(
            consistency,
            allow_scientific_environment_mismatch=args.allow_scientific_environment_mismatch,
        )
    except RuntimeError:
        preflight_summary = {
            "report_title": "Phase D.2 Preflight Stop Summary",
            "smoke_mode": not args.run_all_generation_valid,
            "full_run_executed": False,
            "interventions_input": args.interventions,
            "n_total_interventions": len(raw_rows),
            "n_generation_valid": n_generation_valid,
            "n_generation_invalid": n_generation_invalid,
            "n_selected_interventions": len(projections),
            "frame_root": args.frame_root,
            "source_frame_prefix": args.source_frame_prefix,
            "path_audit": path_audit,
            "model_backend": args.model_backend,
            "model_path": args.model_path,
            "model_fingerprint": model_fingerprint,
            "decoding_config": decoding_config,
            "frozen_qwen_config_checks": frozen_checks,
            "phase_c_consistency": consistency,
            "real_model_inference_executed": False,
            "stop_reason": "Phase C/D.2 consistency check failed before model inference.",
        }
        write_json(summary_output, preflight_summary)
        write_summary_md(summary_md_output, preflight_summary)
        if args.preflight_only:
            print(json.dumps(preflight_summary, ensure_ascii=False, indent=2))
            return
        raise
    if args.preflight_only:
        summary = {
            "report_title": "Phase D.2 Preflight Summary",
            "smoke_mode": not args.run_all_generation_valid,
            "full_run_executed": False,
            "interventions_input": args.interventions,
            "n_total_interventions": len(raw_rows),
            "n_generation_valid": n_generation_valid,
            "n_generation_invalid": n_generation_invalid,
            "n_selected_interventions": len(projections),
            "frame_root": args.frame_root,
            "source_frame_prefix": args.source_frame_prefix,
            "path_audit": path_audit,
            "model_backend": args.model_backend,
            "model_path": args.model_path,
            "model_fingerprint": model_fingerprint,
            "decoding_config": decoding_config,
            "frozen_qwen_config_checks": frozen_checks,
            "phase_c_consistency": consistency,
            "real_model_inference_executed": False,
            "stop_note": "Preflight only: no Qwen inference was executed.",
        }
        write_json(summary_output, summary)
        write_summary_md(summary_md_output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    cache_keys, cache_audit = cache_key_audit(projections, model_hash, decoding_config)
    completed = load_completed_cache_keys(probe_results)
    completed_cache_at_start = len(completed)
    remaining_interventions_at_start = sum(
        1
        for projection in projections
        if cache_keys[projection["intervention_id"]] not in completed
    )
    counts = Counter()
    memory_rows: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    logger.info(
        "Phase D.2 probing selected=%d generation_valid=%d completed_cache=%d",
        len(projections),
        n_generation_valid,
        completed_cache_at_start,
    )
    started = time.time()

    for i, projection in enumerate(projections, start=1):
        prompt, prompt_hash = prompt_for_projection(projection)
        cache_key = cache_keys[projection["intervention_id"]]
        if cache_key in completed:
            counts["skipped_cached"] += 1
            continue
        if args.dry_run:
            counts["dry_run"] += 1
            continue

        missing = [path for path in projection["ordered_frame_paths"] if not Path(path).exists()]
        if missing:
            counts["ERROR"] += 1
            append_jsonl(
                errors_path,
                {
                    "qa_id": projection["qa_id"],
                    "clip_id": projection["clip_id"],
                    "window_id": projection["window_id"],
                    "intervention_id": projection["intervention_id"],
                    "dataset_name": projection["dataset_name"],
                    "model_name": model.model_name,
                    "model_revision": model.model_revision,
                    "prompt_version": args.prompt_version,
                    "cache_key": cache_key,
                    "error_type": "MISSING_FRAMES",
                    "missing_frame_paths": missing,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            continue

        try:
            raw_response = model.infer(projection["ordered_frame_paths"], prompt)
            parsed = parse_yes_no(raw_response)
            record = build_raw_result(
                projection,
                model,
                model_fingerprint,
                model_hash,
                decoding_config,
                prompt_hash,
                cache_key,
                raw_response,
                parsed,
            )
            append_jsonl(probe_results, record)
            completed.add(cache_key)
            counts[parsed] += 1
            gpu_memory = model.gpu_memory_stats()
            if gpu_memory:
                memory_rows.append(
                    {
                        "intervention_index": i,
                        "intervention_id": projection["intervention_id"],
                        "n_frames": projection["n_frames"],
                        **gpu_memory,
                    }
                )
            represented_frame_counts = {record["expected_frame_count"] for record in debug_records}
            if len(debug_records) < 12 or projection["n_frames"] not in represented_frame_counts:
                debug_records.append(
                    {
                        "intervention_id": projection["intervention_id"],
                        "expected_frame_count": projection["n_frames"],
                        "processor_metadata": sanitize_processor_metadata(
                            getattr(model, "last_processor_metadata", {}) or {},
                            expected_frame_count=projection["n_frames"],
                        ),
                    }
                )
        except RuntimeError as exc:
            if "out of memory" in repr(exc).lower() or "cuda oom" in repr(exc).lower():
                append_jsonl(
                    errors_path,
                    {
                        "qa_id": projection["qa_id"],
                        "clip_id": projection["clip_id"],
                        "window_id": projection["window_id"],
                        "intervention_id": projection["intervention_id"],
                        "dataset_name": projection["dataset_name"],
                        "model_name": model.model_name,
                        "model_revision": model.model_revision,
                        "prompt_version": args.prompt_version,
                        "cache_key": cache_key,
                        "error_type": "CUDA_OOM_STOP",
                        "error": repr(exc),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                raise
            counts["ERROR"] += 1
            append_jsonl(
                errors_path,
                {
                    "qa_id": projection["qa_id"],
                    "clip_id": projection["clip_id"],
                    "window_id": projection["window_id"],
                    "intervention_id": projection["intervention_id"],
                    "dataset_name": projection["dataset_name"],
                    "model_name": model.model_name,
                    "model_revision": model.model_revision,
                    "prompt_version": args.prompt_version,
                    "cache_key": cache_key,
                    "error_type": exc.__class__.__name__,
                    "error": repr(exc),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        except Exception as exc:
            counts["ERROR"] += 1
            append_jsonl(
                errors_path,
                {
                    "qa_id": projection["qa_id"],
                    "clip_id": projection["clip_id"],
                    "window_id": projection["window_id"],
                    "intervention_id": projection["intervention_id"],
                    "dataset_name": projection["dataset_name"],
                    "model_name": model.model_name,
                    "model_revision": model.model_revision,
                    "prompt_version": args.prompt_version,
                    "cache_key": cache_key,
                    "error_type": exc.__class__.__name__,
                    "error": repr(exc),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

    processor_audit = frame_count_processor_audit(debug_records)
    represented_values = [v for v in processor_audit.values() if v is not None]
    unexpected_temporal_resampling = any(v is False for v in represented_values)
    if unexpected_temporal_resampling:
        raise RuntimeError(f"Processor did not represent all frames: {processor_audit}")

    expected_ids = {str(projection["intervention_id"]) for projection in projections}
    result_rows, final_error_rows, completion_audit = load_final_completion_records(
        probe_results,
        errors_path,
        expected_ids,
    )
    completed_prediction_counts = final_prediction_counts(result_rows, final_error_rows)
    completed_count = completion_audit["n_completed_final_records"]
    remaining_interventions = len(projections) - completed_count
    prediction_count_total = sum(
        completed_prediction_counts[label]
        for label in ("YES", "NO", "INVALID", "ERROR")
    )
    if args.run_all_generation_valid and not args.dry_run:
        if completed_count != len(projections):
            raise RuntimeError(f"Full D.2b completion audit failed: {completion_audit}")
        if prediction_count_total != completed_count:
            raise RuntimeError(
                "Full D.2b prediction count audit failed: "
                f"prediction_total={prediction_count_total} completed={completed_count}"
            )
        if completion_audit["extra_result_ids"] or completion_audit["duplicate_result_ids"]:
            raise RuntimeError(f"Full D.2b result identity audit failed: {completion_audit}")

    summary = {
        "report_title": (
            "Phase D.2b Full Frozen-Qwen Intervention Probe Summary"
            if args.run_all_generation_valid
            else "Phase D.2 Frozen-Qwen Intervention Smoke Summary"
        ),
        "smoke_mode": not args.run_all_generation_valid,
        "full_run_executed": bool(args.run_all_generation_valid and not args.dry_run),
        "interventions_input": args.interventions,
        "n_total_interventions": len(raw_rows),
        "n_generation_valid": n_generation_valid,
        "n_generation_invalid": n_generation_invalid,
        "n_selected_interventions": len(projections),
        "n_completed": completed_count,
        "n_completed_results": completion_audit["n_completed_results"],
        "n_completed_errors_without_result": completion_audit["n_completed_errors_without_result"],
        "remaining_interventions": remaining_interventions,
        "completion_audit": completion_audit,
        "generation_valid_only_selection": all(p["generation_valid"] for p in projections),
        "frame_count_distribution": frame_count_distribution(projections),
        "intervention_type_distribution": dict(Counter(p["intervention_type"] for p in projections)),
        "intervention_family_distribution": dict(Counter(p["intervention_family"] for p in projections)),
        "dataset_distribution": dict(Counter(p["dataset_name"] for p in projections)),
        "target_field_distribution": dict(Counter(str(p.get("target_field")) for p in projections)),
        "model_backend": args.model_backend,
        "model_name": model.model_name,
        "model_revision": model.model_revision,
        "model_path": args.model_path,
        "frame_root": args.frame_root,
        "source_frame_prefix": args.source_frame_prefix,
        "path_audit": path_audit,
        "model_fingerprint": model_fingerprint,
        "model_identity_hash": model_hash,
        "prompt_version": args.prompt_version,
        "unique_prompt_hash_count": len({prompt_for_projection(p)[1] for p in projections}),
        "decoding_config": decoding_config,
        "frozen_qwen_config_checks": frozen_checks,
        "phase_c_consistency": consistency,
        "cache_audit": cache_audit,
        "completed_cache_at_start": completed_cache_at_start,
        "remaining_interventions_at_start": remaining_interventions_at_start,
        "skipped_cached_count": counts["skipped_cached"],
        "new_inference_count": counts["YES"] + counts["NO"] + counts["INVALID"],
        "dry_run_count": counts["dry_run"],
        "prediction_counts": {
            "YES": completed_prediction_counts["YES"],
            "NO": completed_prediction_counts["NO"],
            "INVALID": completed_prediction_counts["INVALID"],
            "ERROR": completed_prediction_counts["ERROR"],
        },
        "new_prediction_counts_this_run": {
            "YES": counts["YES"],
            "NO": counts["NO"],
            "INVALID": counts["INVALID"],
            "ERROR": counts["ERROR"],
        },
        "prediction_distributions": prediction_distributions(result_rows, final_error_rows, projections),
        "probe_results": probe_results,
        "errors": errors_path,
        "selection_output": selection_output,
        "reuse_selection_from": args.reuse_selection_from,
        "reused_existing_smoke_selection": reused_selection_artifact is not None,
        "processor_frame_count_audit": processor_audit,
        "unexpected_temporal_resampling": unexpected_temporal_resampling,
        "memory_by_frame_count": {
            str(k): {
                "n_observations": len(v),
                "max_peak_allocated_bytes": max((r.get("peak_allocated_bytes") or 0 for r in v), default=0),
                "max_peak_reserved_bytes": max((r.get("peak_reserved_bytes") or 0 for r in v), default=0),
                "max_current_allocated_bytes": max((r.get("current_allocated_bytes") or 0 for r in v), default=0),
            }
            for k, v in {
                n: [r for r in memory_rows if r["n_frames"] == n]
                for n in sorted({r["n_frames"] for r in memory_rows})
            }.items()
        },
        "final_gpu_memory": model.gpu_memory_stats(),
        "memory_after_interventions": memory_rows[:20],
        "possible_memory_leak": memory_leak_flag(memory_rows),
        "gt_information_used_in_model_inference": False,
        "model_side_projection_gt_free": True,
        "raw_result_schema_gt_free": True,
        "original_phase_c_response_used_as_model_input": False,
        "prompt_mentions_intervention_type": False,
        "elapsed_sec": round(time.time() - started, 3),
        "stop_note": "Phase D.2 smoke stops here. No full intervention run or H2 stability analysis was executed.",
    }
    write_json(summary_output, summary)
    write_summary_md(summary_md_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
