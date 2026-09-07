from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
from statistics import median
from typing import Any

from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MASK_TO_BBOX_VERSION,
    PROMPT_VERSION,
    RunConfig,
)
from .frame_adapter import logical_frame_identities, path_exists_audit, select_model_frames
from .io_utils import assert_no_gt_leak, read_json, read_jsonl, sha256_json, sha256_text, write_json, write_jsonl
from .stg_time_spec import build_stg_target_alignment, parse_stg_target_schedule
from .temporal_mapper import TemporalMapper, audit_uniform_spacing


TASK_ALIASES = {
    "stg": "stg",
    "region_caption_gpt": "rc",
    "region_caption_gemini": "rc",
    "region_caption": "rc",
    "cvs_assessment": "cvs",
    "cvs": "cvs",
}


def canonical_task(qa_type: str | None) -> str | None:
    return TASK_ALIASES.get(str(qa_type or ""))


def extract_human_question(sample: dict[str, Any]) -> str:
    for turn in sample.get("conversations") or []:
        if turn.get("from") in {"human", "user"}:
            return str(turn.get("value", "") or "")
    return str(sample.get("question", "") or "")


def make_sample_id(sample: dict[str, Any], original_index: int, task: str) -> str:
    return f"{original_index:06d}::{sample.get('id', '')}::{task}"


def remap_path(path: str, old_root: str, new_root: str | None) -> str:
    if not new_root:
        return path
    old = old_root.rstrip("/")
    new = new_root.rstrip("/")
    if path == old:
        return new
    if path.startswith(old + "/"):
        return new + path[len(old):]
    return path


def _gt_free_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(sample.get("metadata") or {})
    allowed = ("video_id", "fps", "input_video_start_time", "input_video_end_time", "input_video_start_frame", "input_video_end_frame", "start_frame", "end_frame")
    return {key: metadata[key] for key in allowed if key in metadata}


def _provided_region(sample: dict[str, Any], old_data_root: str, new_data_root: str | None) -> dict[str, Any] | None:
    rc_info = sample.get("RC_info")
    if not isinstance(rc_info, dict):
        return None
    region = {
        "provided_region_is_task_input": True,
        "start_frame": remap_path(str(rc_info["start_frame"]), old_data_root, new_data_root) if rc_info.get("start_frame") else None,
        "start_frame_bbox": [float(x) for x in rc_info.get("start_frame_bbox", [])],
        "schema": "frame_path_plus_xyxy_bbox",
    }
    return region


def build_gt_free_row(
    sample: dict[str, Any],
    original_index: int,
    old_data_root: str = DEFAULT_OLD_DATA_ROOT,
    new_data_root: str | None = DEFAULT_NEW_DATA_ROOT,
    fps: float = 1.0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    task = canonical_task(sample.get("qa_type"))
    if task is None:
        raise ValueError(f"unsupported qa_type: {sample.get('qa_type')}")
    raw_video = list(sample.get("video") or [])
    sampled_frames = [int(x) for x in sample.get("sampled_video_frames") or []]
    if len(raw_video) != len(sampled_frames):
        raise ValueError(f"sample {original_index} has non-aligned video and sampled_video_frames")
    if not raw_video:
        raise ValueError(f"sample {original_index} has no benchmark frames")
    frame_paths = [remap_path(str(path), old_data_root, new_data_root) for path in raw_video]
    metadata = _gt_free_metadata(sample)
    dataset = str(sample.get("dataset_name") or sample.get("data_source") or "")
    mapping = TemporalMapper.map(dataset, sampled_frames, frame_paths, metadata)
    local_times = [float(obs.local_time) for obs in mapping.observations]
    spacing = audit_uniform_spacing(local_times)
    sampling = select_model_frames(frame_paths, local_times, mapping.clip_duration, fps=fps, max_frames=max_frames)
    question = extract_human_question(sample)
    row: dict[str, Any] = {
        "sample_id": make_sample_id(sample, original_index, task),
        "qa_id": sample.get("id"),
        "original_index": original_index,
        "task": task,
        "qa_type": sample.get("qa_type"),
        "dataset_name": sample.get("dataset_name"),
        "data_source": sample.get("data_source"),
        "clip_id": metadata.get("video_id") or sample.get("id"),
        "human_question": question,
        "ordered_frame_paths": frame_paths,
        "source_frame_indices": sampled_frames,
        "logical_frame_identities": logical_frame_identities(frame_paths, sampled_frames),
        "local_timestamps": local_times,
        "clip_local_duration": mapping.clip_duration,
        "clip_metadata_gt_free": metadata,
        "n_benchmark_frames": len(frame_paths),
        "model_sampling": sampling.to_dict(),
        "path_exists_audit": path_exists_audit(frame_paths),
        "time_mapping_method": mapping.time_mapping_method,
        "source_timebase_hz": mapping.source_timebase_hz,
        "clip_duration_source": mapping.clip_duration_source,
        "timestamp_spacing_audit": spacing.to_dict(),
        "human_question_sha256": sha256_text(question),
        "ordered_benchmark_frame_hash": sha256_json(logical_frame_identities(frame_paths, sampled_frames)),
        "selected_frame_hash": sha256_json(sampling.to_dict()),
        "timestamp_hash": sha256_json(local_times),
        "prompt_version": PROMPT_VERSION,
        "mask_to_bbox_version": MASK_TO_BBOX_VERSION,
        "gt_information_available_to_model": False,
        "gt_fields_present": False,
        "source_video_accessed": False,
        "fake_mp4_used": False,
        "path_mapping": {"old_data_root": old_data_root, "new_data_root": new_data_root},
    }
    if task == "stg":
        schedule = parse_stg_target_schedule(question)
        target_alignment = build_stg_target_alignment(local_times, sampled_frames, frame_paths, schedule)
        row["stg_target_schedule"] = schedule.to_dict()
        row["stg_target_alignment"] = target_alignment
        row["stg_target_schedule_hash"] = sha256_json(
            {"schedule": row["stg_target_schedule"], "alignment": target_alignment}
        )
    if task == "rc":
        region = _provided_region(sample, old_data_root, new_data_root)
        if region is None:
            raise ValueError("RC sample lacks provided RC_info")
        row["provided_region"] = region
        row["provided_region_hash"] = sha256_json(region)
    assert_no_gt_leak(row, allowed_paths={"$.provided_region", "$.provided_region.start_frame_bbox"})
    return row


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": float(median(values)), "max": max(values)}


def inventory_samples(data: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    datasets: dict[str, Counter[str]] = defaultdict(Counter)
    frame_counts: dict[str, list[int]] = defaultdict(list)
    resolutions: dict[str, Counter[str]] = defaultdict(Counter)
    question_examples: dict[str, list[str]] = defaultdict(list)
    schema: dict[str, dict[str, Any]] = {}
    for sample in data:
        if not isinstance(sample, dict):
            continue
        task = canonical_task(sample.get("qa_type"))
        if task is None:
            continue
        counts[task] += 1
        dataset = str(sample.get("dataset_name") or sample.get("data_source") or "Unknown")
        datasets[task][dataset] += 1
        frame_counts[task].append(len(sample.get("video") or []))
        if len(question_examples[task]) < 3:
            question_examples[task].append(extract_human_question(sample))
        if task not in schema:
            schema[task] = {
                "qa_type_examples": sorted({str(sample.get("qa_type"))}),
                "keys": sorted(sample.keys()),
                "has_struc_info": "struc_info" in sample,
                "has_RC_info": "RC_info" in sample,
                "rc_schema": "frame_path_plus_xyxy_bbox" if task == "rc" and isinstance(sample.get("RC_info"), dict) else None,
                "official_evaluator_schema": {
                    "stg": "leaderboard JSON records with answer text containing '<seconds> seconds: [x1, y1, x2, y2]'",
                    "rc": "LLM judge / semantic text comparison over answer vs gnd",
                    "cvs": "component text: Two structures, Cystic plate, Hepatocystic triangle scores 0/1/2",
                }[task],
            }
        meta = sample.get("metadata") or {}
        if "width" in meta and "height" in meta:
            resolutions[task][f"{meta['width']}x{meta['height']}"] += 1
    return {
        "n_samples": dict(sorted(counts.items())),
        "datasets": {task: dict(sorted(counter.items())) for task, counter in sorted(datasets.items())},
        "frame_count_distribution": {
            task: {
                "min": min(vals) if vals else None,
                "median": float(median(vals)) if vals else None,
                "max": max(vals) if vals else None,
            }
            for task, vals in sorted(frame_counts.items())
        },
        "resolution_distribution": {task: dict(counter) for task, counter in sorted(resolutions.items())},
        "question_examples": dict(question_examples),
        "task_schema": schema,
        "notes": {
            "stg": "Trainval STG includes GT in conversations assistant and struc_info; GT-free manifest strips both.",
            "rc": "RC_info is benchmark-provided task input and retained as provided_region; target captions are stripped.",
            "cvs": "CVS question/options are task input; component scores in assistant/struc_info are stripped.",
        },
    }


def build_manifests(cfg: RunConfig) -> dict[str, Any]:
    cfg.make_dirs()
    data = read_json(cfg.data_path)
    if not isinstance(data, list):
        raise TypeError("MedVidU JSON must be a list")
    rows_by_task: dict[str, list[dict[str, Any]]] = {"stg": [], "rc": [], "cvs": []}
    failures: list[dict[str, Any]] = []
    for i, sample in enumerate(data):
        if not isinstance(sample, dict) or canonical_task(sample.get("qa_type")) is None:
            continue
        try:
            row = build_gt_free_row(sample, i, old_data_root=cfg.old_data_root, new_data_root=cfg.new_data_root, fps=cfg.fps, max_frames=cfg.max_frames)
            rows_by_task[row["task"]].append(row)
        except Exception as exc:
            failures.append({"original_index": i, "id": sample.get("id") if isinstance(sample, dict) else None, "qa_type": sample.get("qa_type") if isinstance(sample, dict) else None, "error": repr(exc)})
    for task, rows in rows_by_task.items():
        write_jsonl(cfg.manifest_dir / f"{task}_gt_free.jsonl", rows)
    inventory = inventory_samples([x for x in data if isinstance(x, dict)])
    inventory.update(
        {
            "data_path": str(cfg.data_path),
            "manifest_rows": {task: len(rows) for task, rows in rows_by_task.items()},
            "manifest_failures": failures,
        }
    )
    write_json(cfg.manifest_dir / "medvidu_task_inventory.json", inventory)
    write_json(
        cfg.audit_dir / "frame_sampling_policy.json",
        {
            "upstream_official_policy": "fps=1.0, max_frames=128, sample_frames=int(video_duration*fps), uniform over clip",
            "medvidu_policy": "all benchmark-provided frames are passed to the model in listed order; no temporal downsampling and no source frames are accessed",
            "sampling_version": rows_by_task["stg"][0]["model_sampling"]["policy"] if rows_by_task["stg"] else None,
            "additional_sampling": False,
            "fake_mp4_used": False,
        },
    )
    write_json(
        cfg.audit_dir / "timestamp_mapping.json",
        {
            task: {
                "n_rows": len(rows),
                "status_counts": dict(Counter(row["timestamp_spacing_audit"]["status"] for row in rows)),
                "first_timestamp": _numeric_summary([row["local_timestamps"][0] for row in rows if row["local_timestamps"]]),
                "last_timestamp": _numeric_summary([row["local_timestamps"][-1] for row in rows if row["local_timestamps"]]),
            }
            for task, rows in rows_by_task.items()
        },
    )
    stg_alignment_errors = [
        float(item["absolute_timing_error_seconds"])
        for row in rows_by_task["stg"]
        for item in row.get("stg_target_alignment", [])
    ]
    write_json(
        cfg.audit_dir / "stg_target_timestamp_alignment.json",
        {
            "version": "stg_question_schedule_and_nearest_frame_v1",
            "n_stg_rows": len(rows_by_task["stg"]),
            "n_target_timestamps": len(stg_alignment_errors),
            "max_absolute_timing_error_seconds": max(stg_alignment_errors) if stg_alignment_errors else None,
            "mean_absolute_timing_error_seconds": (sum(stg_alignment_errors) / len(stg_alignment_errors)) if stg_alignment_errors else None,
            "policy": "Parse STG target times from the human task question and pair each with the nearest GT-free benchmark frame.",
        },
    )
    write_json(
        cfg.audit_dir / "gt_leakage.json",
        {
            "stg": {"gt_visible": False},
            "cvs": {"gt_visible": False},
            "rc": {"provided_region_visible": True, "provided_region_reason": "benchmark task input", "target_caption_visible": False},
            "rows_checked": {task: len(rows) for task, rows in rows_by_task.items()},
        },
    )
    return inventory


def choose_smoke_rows(rows: list[dict[str, Any]], size: int = 10, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset_name") or "Unknown")].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for dataset, ds_rows in sorted(by_dataset.items()):
        ordered = sorted(ds_rows, key=lambda r: (-int(r["n_benchmark_frames"]), r["sample_id"]))
        if ordered:
            selected[ordered[0]["sample_id"]] = {**ordered[0], "_smoke_reason": f"{dataset}:max_frame_count"}
        ordered = sorted(ds_rows, key=lambda r: (-float(r["clip_local_duration"]), r["sample_id"]))
        if ordered:
            selected[ordered[0]["sample_id"]] = {**ordered[0], "_smoke_reason": f"{dataset}:max_clip_duration"}
    remaining = [row for row in rows if row["sample_id"] not in selected]
    rng.shuffle(remaining)
    remaining.sort(key=lambda r: (sum(1 for x in selected.values() if x.get("dataset_name") == r.get("dataset_name")), r["sample_id"]))
    for row in remaining:
        if len(selected) >= size:
            break
        selected[row["sample_id"]] = {**row, "_smoke_reason": "seeded_dataset_balanced_fill"}
    out = list(selected.values())
    out.sort(key=lambda r: (str(r.get("dataset_name") or ""), r["sample_id"]))
    return out[:size]


def write_smoke_manifests(cfg: RunConfig) -> dict[str, Any]:
    payload = {"seed": cfg.smoke_seed, "requested_size_per_task": cfg.smoke_size, "selection_policy": "GT-blind: dataset coverage, max frame count, max clip duration, seeded balanced fill", "tasks": {}}
    for task in ("stg", "rc", "cvs"):
        rows = read_jsonl(cfg.manifest_dir / f"{task}_gt_free.jsonl")
        smoke = choose_smoke_rows(rows, size=cfg.smoke_size, seed=cfg.smoke_seed)
        write_jsonl(cfg.manifest_dir / f"{task}_gt_free.smoke.jsonl", smoke)
        payload["tasks"][task] = {
            "actual_size": len(smoke),
            "sample_ids": [row["sample_id"] for row in smoke],
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "qa_id": row.get("qa_id"),
                    "dataset_name": row.get("dataset_name"),
                    "n_benchmark_frames": row.get("n_benchmark_frames"),
                    "n_model_input_frames": row.get("model_sampling", {}).get("n_selected"),
                    "clip_local_duration": row.get("clip_local_duration"),
                    "smoke_reason": row.get("_smoke_reason"),
                }
                for row in smoke
            ],
        }
    write_json(cfg.manifest_dir / "smoke_sample_ids.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GT-free E-VQA MedVidU manifests.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--old-data-root", default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new-data-root", default=DEFAULT_NEW_DATA_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root, old_data_root=args.old_data_root, new_data_root=args.new_data_root, smoke_size=args.smoke_size, smoke_seed=args.seed)
    inventory = build_manifests(cfg)
    if args.smoke:
        inventory["smoke"] = write_smoke_manifests(cfg)
        write_json(cfg.manifest_dir / "medvidu_task_inventory.json", inventory)
    print(inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
