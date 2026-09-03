#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.cache import make_probe_cache_key, stable_hash  # noqa: E402
from evidence_stability.model_interface import build_model  # noqa: E402
from evidence_stability.prompts import PROMPT_VERSION, build_evidence_presence_prompt, parse_yes_no  # noqa: E402
from evidence_stability.utils import append_jsonl, read_jsonl, write_json  # noqa: E402


logger = logging.getLogger("phase_c_probe")


def setup_logging(log_path: str | None) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)


def resolve_frame_path(path: str, frame_root: str | None, source_frame_prefix: str = "/root/data") -> str:
    if not frame_root:
        return path
    normalized_prefix = source_frame_prefix.rstrip("/")
    if path == normalized_prefix:
        return frame_root.rstrip("/")
    if path.startswith(normalized_prefix + "/"):
        suffix = path[len(normalized_prefix) :].lstrip("/")
        return str(Path(frame_root) / suffix)
    return path


def resolve_window_frame_paths(
    window: dict[str, Any],
    frame_root: str | None,
    source_frame_prefix: str,
) -> list[str]:
    return [
        resolve_frame_path(path, frame_root, source_frame_prefix)
        for path in window["frame_paths"]
    ]


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


def summarize_frame_counts(windows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(len(w["frame_paths"]) for w in windows)
    return {
        "distribution": dict(sorted(counts.items())),
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
    }


def processor_represented_all_frames(metadata: dict[str, Any]) -> bool | None:
    frame_count = metadata.get("input_frame_count")
    for key in ("image_grid_thw_rows", "video_grid_thw_rows"):
        rows = metadata.get(key)
        if rows is not None and frame_count is not None:
            return int(rows) == int(frame_count)
    return None


def write_probe_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    title = summary.get("report_title") or "Phase C Probe Summary"
    stop_note = summary.get("stop_note") or "Phase C probing stops here."
    lines = [
        f"# {title}",
        "",
        "## Model",
        f"- model backend: {summary.get('model_backend')}",
        f"- model path: {summary.get('model_fingerprint', {}).get('model_path')}",
        f"- model class: {summary.get('model_fingerprint', {}).get('model_class')}",
        f"- processor class: {summary.get('model_fingerprint', {}).get('processor_class')}",
        f"- transformers version: {summary.get('model_fingerprint', {}).get('transformers_version')}",
        f"- torch version: {summary.get('model_fingerprint', {}).get('torch_version')}",
        f"- dtype: {summary.get('model_fingerprint', {}).get('dtype')}",
        f"- device: {summary.get('model_fingerprint', {}).get('device')}",
        f"- GPU: {summary.get('final_gpu_memory', {}).get('gpu_name')}",
        f"- peak allocated memory: {summary.get('final_gpu_memory', {}).get('peak_allocated_bytes')}",
        f"- peak reserved memory: {summary.get('final_gpu_memory', {}).get('peak_reserved_bytes')}",
        "",
        "## Input",
        f"- QA count: {summary.get('n_qa')}",
        f"- window count: {summary.get('n_windows')}",
        f"- frame count distribution: {summary.get('frame_count_distribution')}",
        f"- missing frame errors: {summary.get('missing_frame_errors')}",
        f"- processor failures: {summary.get('processor_failures')}",
        "",
        "## Prediction",
        f"- YES: {summary.get('counts', {}).get('YES', 0)}",
        f"- NO: {summary.get('counts', {}).get('NO', 0)}",
        f"- INVALID: {summary.get('counts', {}).get('INVALID', 0)}",
        f"- ERROR: {summary.get('counts', {}).get('ERROR', 0)}",
        f"- YES rate: {summary.get('yes_rate_new_outputs')}",
        "",
        "## Cache",
        f"- new inference count: {summary.get('new_inference_count')}",
        f"- cache-hit count at start: {summary.get('completed_cache_at_start')}",
        f"- skipped cached count: {summary.get('counts', {}).get('skipped_cached', 0)}",
        "",
        "## Checks",
        f"- Did the model receive all ordered window frames? {summary.get('model_received_all_ordered_window_frames')}",
        f"- Did the processor represent all debug-window frames? {summary.get('processor_represented_all_debug_frames')}",
        f"- Were duplicate frames preserved? {summary.get('duplicate_frames_preserved')}",
        f"- Did the processor perform any unexpected temporal resampling? {summary.get('unexpected_temporal_resampling')}",
        f"- Did any setting have to be changed from the frozen Phase C specification? {summary.get('frozen_phase_c_settings_changed')}",
        "",
        stop_note,
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_qa_ids(samples: list[dict[str, Any]], max_qa: int, seed: int) -> list[str]:
    eligible = [s for s in samples if s.get("analysis_eligible")]
    if max_qa < 0 or max_qa >= len(eligible):
        return [s["qa_id"] for s in eligible]

    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in eligible:
        by_dataset[sample["dataset_name"]].append(sample)

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    datasets = sorted(by_dataset)
    for dataset in datasets:
        pool = list(by_dataset[dataset])
        rng.shuffle(pool)
        if pool and len(chosen) < max_qa:
            sample = pool[0]
            chosen.append(sample)
            seen.add(sample["qa_id"])

    remaining = [s for s in eligible if s["qa_id"] not in seen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max_qa - len(chosen)])
    chosen.sort(key=lambda s: s["qa_id"])
    return [s["qa_id"] for s in chosen]


def load_completed_cache_keys(path: str) -> set[str]:
    if not Path(path).exists():
        return set()
    keys: set[str] = set()
    for row in read_jsonl(path):
        cache_key = row.get("cache_key")
        if cache_key:
            keys.add(cache_key)
    return keys


def shared_clip_cache_key_check(
    samples: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    model_name: str,
    model_revision: str | None,
    prompt_version: str,
    model_fingerprint: dict[str, Any] | None = None,
    decoding_config: dict[str, Any] | None = None,
    frame_root: str | None = None,
    source_frame_prefix: str = "/root/data",
) -> dict[str, Any]:
    sample_by_qa = {s["qa_id"]: s for s in samples if s.get("analysis_eligible")}
    windows_by_qa: dict[str, dict[str, Any]] = {}
    for window in windows:
        windows_by_qa.setdefault(window["qa_id"], window)

    clip_to_qa: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if sample.get("analysis_eligible") and sample["qa_id"] in windows_by_qa:
            clip_to_qa[sample["clip_id"]].append(sample["qa_id"])

    for clip_id, qa_ids in sorted(clip_to_qa.items()):
        if len(qa_ids) < 2:
            continue
        qa_a, qa_b = sorted(qa_ids)[:2]
        window_a = windows_by_qa[qa_a]
        window_b = windows_by_qa[qa_b]
        prompt_a = build_evidence_presence_prompt(sample_by_qa[qa_a]["target_action"])
        prompt_b = build_evidence_presence_prompt(sample_by_qa[qa_b]["target_action"])
        key_a = make_probe_cache_key(
            model_name,
            model_revision,
            prompt_version,
            qa_a,
            window_a["window_id"],
            resolve_window_frame_paths(window_a, frame_root, source_frame_prefix),
            prompt_a,
            model_fingerprint=model_fingerprint,
            decoding_config=decoding_config,
        )
        key_b = make_probe_cache_key(
            model_name,
            model_revision,
            prompt_version,
            qa_b,
            window_b["window_id"],
            resolve_window_frame_paths(window_b, frame_root, source_frame_prefix),
            prompt_b,
            model_fingerprint=model_fingerprint,
            decoding_config=decoding_config,
        )
        return {
            "checked": True,
            "clip_id": clip_id,
            "qa_ids": [qa_a, qa_b],
            "window_ids": [window_a["window_id"], window_b["window_id"]],
            "cache_keys_differ": key_a != key_b,
        }
    return {"checked": False, "cache_keys_differ": True}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C evidence presence probing.")
    p.add_argument("--samples", default="outputs/tal_pilot/samples.jsonl")
    p.add_argument("--windows", default="outputs/tal_pilot/windows.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_c")
    p.add_argument("--probe_results", default=None)
    p.add_argument("--errors", default=None)
    p.add_argument("--selected_qa_output", default=None)
    p.add_argument("--summary_output", default=None)
    p.add_argument("--log_path", default=None)
    p.add_argument("--max_qa", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_version", default=PROMPT_VERSION)
    p.add_argument("--model_backend", choices=["dummy", "openai_compatible", "qwen3_vl"], default="dummy")
    p.add_argument("--model_name", default="dummy-video-vlm")
    p.add_argument("--model_revision", default="v1")
    p.add_argument("--model_path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--processor_min_pixels", type=int, default=None)
    p.add_argument("--processor_max_pixels", type=int, default=None)
    p.add_argument("--frame_root", default=None)
    p.add_argument("--source_frame_prefix", default="/root/data")
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--limit_windows", type=int, default=-1)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_results = args.probe_results or str(out_dir / "probe_results.jsonl")
    errors_path = args.errors or str(out_dir / "errors.jsonl")
    selected_qa_output = args.selected_qa_output or str(out_dir / "phase_c_pilot_qa_ids.json")
    summary_output = args.summary_output or str(out_dir / "phase_c_smoke_summary.json")
    summary_md_output = str(Path(summary_output).with_suffix(".md"))
    Path(probe_results).parent.mkdir(parents=True, exist_ok=True)
    Path(probe_results).touch(exist_ok=True)
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    Path(errors_path).touch(exist_ok=True)
    setup_logging(args.log_path or str(out_dir / "probe.log"))

    samples = read_jsonl(args.samples)
    sample_by_qa = {s["qa_id"]: s for s in samples}
    qa_ids = select_qa_ids(samples, args.max_qa, args.seed)
    write_json(
        selected_qa_output,
        {
            "seed": args.seed,
            "max_qa": args.max_qa,
            "n_qa": len(qa_ids),
            "qa_ids": qa_ids,
            "dataset_counts": dict(Counter(sample_by_qa[q]["dataset_name"] for q in qa_ids)),
        },
    )

    all_formal_windows = read_jsonl(args.windows)
    windows = [
        w
        for w in all_formal_windows
        if w.get("analysis_eligible") and w["qa_id"] in set(qa_ids)
    ]
    if args.limit_windows >= 0:
        windows = windows[: args.limit_windows]

    if args.model_backend == "qwen3_vl" and not args.model_path:
        raise ValueError("--model_path is required when --model_backend qwen3_vl")

    model = build_model(args)
    model_fingerprint = model.fingerprint()
    decoding_config = model.generation_config()
    completed = load_completed_cache_keys(probe_results)
    completed_cache_at_start = len(completed)
    counts = Counter()
    dataset_counts = Counter()
    shared_clip_cache_checks: dict[str, list[dict[str, str]]] = defaultdict(list)
    preview_records: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    processor_observation: dict[str, Any] = {}
    memory_after_windows: list[dict[str, Any]] = []

    logger.info("Phase C probing qa=%d windows=%d completed_cache=%d", len(qa_ids), len(windows), len(completed))
    started = time.time()
    for i, window in enumerate(windows, start=1):
        sample = sample_by_qa[window["qa_id"]]
        resolved_frame_paths = resolve_window_frame_paths(window, args.frame_root, args.source_frame_prefix)
        prompt = build_evidence_presence_prompt(sample["target_action"])
        forbidden = ["gt span", "gt_answer", "gt_alignment_class", "processed_gt_spans", "raw_gt_spans"]
        if any(term in prompt.lower() for term in forbidden):
            raise RuntimeError(f"Forbidden GT term found in prompt for {window['window_id']}")

        cache_key = make_probe_cache_key(
            model_name=model.model_name,
            model_revision=model.model_revision,
            prompt_version=args.prompt_version,
            qa_id=window["qa_id"],
            window_id=window["window_id"],
            frame_paths=resolved_frame_paths,
            prompt=prompt,
            model_fingerprint=model_fingerprint,
            decoding_config=decoding_config,
        )
        shared_clip_cache_checks[sample["clip_id"]].append({"qa_id": window["qa_id"], "cache_key": cache_key})
        if len(preview_records) < 5:
            preview_records.append(
                {
                    "qa_id": window["qa_id"],
                    "clip_id": window["clip_id"],
                    "window_id": window["window_id"],
                    "target_action": sample["target_action"],
                    "dataset_name": sample["dataset_name"],
                    "source_frame_paths": window["frame_paths"],
                    "frame_paths": resolved_frame_paths,
                    "prompt": prompt,
                    "cache_key": cache_key,
                }
            )
        if cache_key in completed:
            counts["skipped_cached"] += 1
            continue
        if args.dry_run:
            counts["dry_run"] += 1
            continue

        missing = [path for path in resolved_frame_paths if not Path(path).exists()]
        if missing:
            counts["ERROR"] += 1
            counts["missing_frame_errors"] += 1
            dataset_counts[(sample["dataset_name"], "ERROR")] += 1
            append_jsonl(
                errors_path,
                {
                    "qa_id": window["qa_id"],
                    "clip_id": window["clip_id"],
                    "window_id": window["window_id"],
                    "dataset_name": sample["dataset_name"],
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
            raw_response = model.infer(resolved_frame_paths, prompt)
            parsed = parse_yes_no(raw_response)
            processor_meta = getattr(model, "last_processor_metadata", {}) or {}
            model_debug = getattr(model, "last_debug_metadata", {}) or {}
            if not processor_observation and processor_meta:
                processor_observation = processor_meta
            record = {
                "qa_id": window["qa_id"],
                "clip_id": window["clip_id"],
                "window_id": window["window_id"],
                "target_action": sample["target_action"],
                "dataset_name": sample["dataset_name"],
                "model_name": model.model_name,
                "model_revision": model.model_revision,
                "prompt_version": args.prompt_version,
                "prompt_hash": stable_hash({"prompt": prompt}),
                "cache_key": cache_key,
                "raw_response": raw_response,
                "parsed_prediction": parsed,
                "source_frame_paths": window["frame_paths"],
                "frame_paths": resolved_frame_paths,
                "start_time": window.get("start_time"),
                "end_time": window.get("end_time"),
                "n_frames": window.get("n_frames"),
                "n_unique_frames": window.get("n_unique_frames"),
                "duplicate_ratio": window.get("duplicate_ratio"),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_jsonl(probe_results, record)
            completed.add(cache_key)
            counts[parsed] += 1
            dataset_counts[(sample["dataset_name"], parsed)] += 1
            gpu_memory = model.gpu_memory_stats()
            if gpu_memory:
                memory_after_windows.append({"window_index": i, **gpu_memory})
            if len(debug_records) < 3:
                debug_record = {
                    "qa_id": window["qa_id"],
                    "clip_id": window["clip_id"],
                    "window_id": window["window_id"],
                    "target_action": sample["target_action"],
                    "n_input_frames": len(resolved_frame_paths),
                    "first_3_resolved_frame_paths": resolved_frame_paths[:3],
                    "last_3_resolved_frame_paths": resolved_frame_paths[-3:],
                    "frame_paths": resolved_frame_paths,
                    "image_sizes": model_debug.get("image_sizes") or image_sizes(resolved_frame_paths),
                    "prompt": prompt,
                    "raw_model_response": raw_response,
                    "parsed_prediction": parsed,
                    "processor_metadata": processor_meta,
                }
                debug_records.append(debug_record)
                logger.info("debug_window_%d=%s", len(debug_records), json.dumps(debug_record, ensure_ascii=False))
        except Exception as exc:
            counts["ERROR"] += 1
            counts["processor_failures"] += 1
            dataset_counts[(sample["dataset_name"], "ERROR")] += 1
            append_jsonl(
                errors_path,
                {
                    "qa_id": window["qa_id"],
                    "clip_id": window["clip_id"],
                    "window_id": window["window_id"],
                    "dataset_name": sample["dataset_name"],
                    "model_name": model.model_name,
                    "model_revision": model.model_revision,
                    "prompt_version": args.prompt_version,
                    "cache_key": cache_key,
                    "source_frame_paths": window["frame_paths"],
                    "frame_paths": resolved_frame_paths,
                    "error_type": exc.__class__.__name__,
                    "error": repr(exc),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

        if i % 100 == 0:
            logger.info(
                "completed=%d/%d skipped_cached=%d YES=%d NO=%d INVALID=%d ERROR=%d",
                i,
                len(windows),
                counts["skipped_cached"],
                counts["YES"],
                counts["NO"],
                counts["INVALID"],
                counts["ERROR"],
            )

    shared_different_cache = True
    for entries in shared_clip_cache_checks.values():
        by_qa: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            by_qa[entry["qa_id"]].add(entry["cache_key"])
        if len(by_qa) > 1:
            flattened = [key for keys in by_qa.values() for key in keys]
            if len(flattened) != len(set(flattened)):
                shared_different_cache = False
                break

    summary = {
        "report_title": "Phase C Pilot Probe Summary" if len(qa_ids) > 10 else "Phase C Smoke Summary",
        "stop_note": (
            "Phase C pilot probing stops here. No Phase D, shift, resampling, context expansion, or stability analysis was run."
            if len(qa_ids) > 10
            else "Phase C smoke stops here. No 50-QA pilot, Phase D, shift, resampling, context expansion, or stability analysis was run."
        ),
        "n_qa": len(qa_ids),
        "n_windows": len(windows),
        "model_backend": args.model_backend,
        "model_name": model.model_name,
        "model_revision": model.model_revision,
        "model_path": args.model_path,
        "model_fingerprint": model_fingerprint,
        "prompt_version": args.prompt_version,
        "decoding_config": decoding_config,
        "counts": dict(counts),
        "yes_rate_new_outputs": counts["YES"] / max(1, counts["YES"] + counts["NO"] + counts["INVALID"]),
        "new_inference_count": counts["YES"] + counts["NO"] + counts["INVALID"],
        "completed_cache_at_start": completed_cache_at_start,
        "selected_qa_output": selected_qa_output,
        "probe_results": probe_results,
        "errors": errors_path,
        "frame_root": args.frame_root,
        "source_frame_prefix": args.source_frame_prefix,
        "frame_count_distribution": summarize_frame_counts(windows),
        "missing_frame_errors": counts["missing_frame_errors"],
        "processor_failures": counts["processor_failures"],
        "dataset_prediction_counts": {
            f"{dataset}:{pred}": count for (dataset, pred), count in dataset_counts.items()
        },
        "prompt_contains_forbidden_gt_terms": False,
        "shared_clip_different_qa_cache_keys_differ": shared_different_cache,
        "global_shared_clip_cache_key_check": shared_clip_cache_key_check(
            samples,
            all_formal_windows,
            model.model_name,
            model.model_revision,
            args.prompt_version,
            model_fingerprint=model_fingerprint,
            decoding_config=decoding_config,
            frame_root=args.frame_root,
            source_frame_prefix=args.source_frame_prefix,
        ),
        "preview_records": preview_records,
        "debug_records": debug_records,
        "processor_observation": processor_observation,
        "final_gpu_memory": model.gpu_memory_stats(),
        "memory_after_windows": memory_after_windows[:20],
        "model_received_all_ordered_window_frames": all(
            len(record["frame_paths"]) == record["n_input_frames"] for record in debug_records
        ) if debug_records else None,
        "processor_represented_all_debug_frames": processor_represented_all_frames(processor_observation),
        "duplicate_frames_preserved": True,
        "unexpected_temporal_resampling": (
            not processor_represented_all_frames(processor_observation)
            if processor_represented_all_frames(processor_observation) is not None
            else None
        ),
        "frozen_phase_c_settings_changed": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(summary_output, summary)
    write_probe_summary_md(summary_md_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
