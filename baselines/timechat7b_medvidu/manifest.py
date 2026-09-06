from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
from statistics import median
from typing import Any

from baselines.timelens_medvidu.temporal_mapper import TemporalMapper, audit_uniform_spacing

from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    FRAME_LOADER_VERSION,
    PROMPT_VERSION,
    QUERY_ADAPTER_VERSION,
    TIMESTAMP_ADAPTER_VERSION,
    RunConfig,
)
from .frame_loader import select_frame_inputs
from .io_utils import assert_no_gt_leak, read_json, read_jsonl, sha256_json, sha256_text, write_json, write_jsonl


def extract_human_question(sample: dict[str, Any]) -> str:
    for turn in sample.get("conversations") or []:
        if turn.get("from") in {"human", "user"}:
            return str(turn.get("value", "") or "")
    return str(sample.get("question", "") or "")


def make_sample_id(sample: dict[str, Any], original_index: int) -> str:
    return f"{original_index:06d}::{sample.get('id', '')}::tal"


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
    allowed = ("video_id", "fps", "input_video_start_time", "input_video_end_time")
    return {key: metadata[key] for key in allowed if key in metadata}


def build_gt_free_row(
    sample: dict[str, Any],
    original_index: int,
    old_data_root: str = DEFAULT_OLD_DATA_ROOT,
    new_data_root: str | None = DEFAULT_NEW_DATA_ROOT,
) -> dict[str, Any]:
    if sample.get("qa_type") != "tal":
        raise ValueError(f"sample {original_index} qa_type is not TAL: {sample.get('qa_type')}")
    raw_video = list(sample.get("video") or [])
    sampled_frames = [int(x) for x in sample.get("sampled_video_frames") or []]
    if len(raw_video) != len(sampled_frames):
        raise ValueError(f"sample {original_index} has non-aligned video and sampled_video_frames")
    video = [remap_path(path, old_data_root, new_data_root) for path in raw_video]
    dataset_name = str(sample.get("dataset_name") or sample.get("data_source") or "")
    clip_metadata = _gt_free_metadata(sample)
    mapping = TemporalMapper.map(dataset_name, sampled_frames, video, clip_metadata)
    local_times = [float(obs.local_time) for obs in mapping.observations]
    spacing = audit_uniform_spacing(local_times)
    selected = select_frame_inputs(video, local_times)
    frame_identities = [
        {
            "logical_position": i,
            "source_frame_index": sampled_frames[i],
            "frame_path": video[i],
            "frame_identity_hash": sha256_json({"logical_position": i, "source_frame_index": sampled_frames[i], "frame_path": video[i]}),
        }
        for i in range(len(video))
    ]
    row = {
        "sample_id": make_sample_id(sample, original_index),
        "qa_id": sample.get("id"),
        "original_index": original_index,
        "qa_type": "tal",
        "dataset_name": sample.get("dataset_name"),
        "data_source": sample.get("data_source"),
        "clip_id": clip_metadata.get("video_id") or sample.get("id"),
        "human_question": extract_human_question(sample),
        "ordered_frame_paths": video,
        "logical_frame_identities": frame_identities,
        "source_frame_indices": sampled_frames,
        "local_timestamps": local_times,
        "clip_local_duration": mapping.clip_duration,
        "clip_metadata_gt_free": clip_metadata,
        "n_input_frames": len(video),
        "selected_logical_indices": selected["selection_audit"]["selected_logical_indices"],
        "selected_local_timestamps": selected["selected_local_timestamps"],
        "selected_rounded_timestamps": selected["selected_rounded_timestamps"],
        "timestamp_texts": selected["timestamp_texts"],
        "timechat_msg": selected["msg"],
        "time_mapping_method": mapping.time_mapping_method,
        "source_timebase_hz": mapping.source_timebase_hz,
        "clip_duration_source": mapping.clip_duration_source,
        "timestamp_spacing_audit": spacing.to_dict(),
        "selection_audit": selected["selection_audit"],
        "query_adapter_version": QUERY_ADAPTER_VERSION,
        "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
        "frame_loader_version": FRAME_LOADER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "human_question_sha256": sha256_text(extract_human_question(sample)),
        "gt_information_available_to_model": False,
        "gt_fields_present": False,
        "source_video_accessed": False,
        "path_mapping": {"old_data_root": old_data_root, "new_data_root": new_data_root},
    }
    assert_no_gt_leak(row)
    return row


def build_manifest(data_path: Path, output_root: Path, old_data_root: str = DEFAULT_OLD_DATA_ROOT, new_data_root: str | None = DEFAULT_NEW_DATA_ROOT) -> dict[str, Any]:
    cfg = RunConfig(data_path=data_path, output_root=output_root, old_data_root=old_data_root, new_data_root=new_data_root)
    cfg.make_dirs()
    data = read_json(data_path)
    if not isinstance(data, list):
        raise TypeError("MedVidU trainval JSON must be a list")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_by_dataset: Counter[str] = Counter()
    tal_by_dataset: Counter[str] = Counter()
    mapping_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    timestamp_values_by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            failures.append({"original_index": i, "error": "not a JSON object"})
            continue
        dataset = str(sample.get("dataset_name") or sample.get("data_source") or "Unknown")
        total_by_dataset[dataset] += 1
        if sample.get("qa_type") != "tal":
            continue
        tal_by_dataset[dataset] += 1
        try:
            row = build_gt_free_row(sample, i, old_data_root=old_data_root, new_data_root=new_data_root)
            rows.append(row)
            mapping_by_dataset[dataset][row["timestamp_spacing_audit"]["status"]] += 1
            times = [float(t) for t in row["local_timestamps"]]
            deltas = [times[j + 1] - times[j] for j in range(len(times) - 1)]
            timestamp_values_by_dataset[dataset]["first"].append(times[0])
            timestamp_values_by_dataset[dataset]["last"].append(times[-1])
            timestamp_values_by_dataset[dataset]["positive_delta"].extend([d for d in deltas if d > 0])
            timestamp_values_by_dataset[dataset]["non_monotonic"].append(1.0 if any(d < 0 for d in deltas) else 0.0)
            timestamp_values_by_dataset[dataset]["duplicate"].append(1.0 if any(d == 0 for d in deltas) else 0.0)
        except Exception as exc:
            failures.append({"original_index": i, "id": sample.get("id"), "dataset_name": dataset, "error": repr(exc)})
    manifest_path = cfg.manifest_dir / "medvidu_tal_timechat_gt_free.jsonl"
    inventory = {
        "data_path": str(data_path),
        "n_total": len(data),
        "n_tal": sum(1 for sample in data if isinstance(sample, dict) and sample.get("qa_type") == "tal"),
        "manifest_rows": len(rows),
        "n_by_dataset": dict(sorted(tal_by_dataset.items())),
        "total_by_dataset": dict(sorted(total_by_dataset.items())),
        "mapping_failures": failures,
        "gt_fields_present": False,
        "manifest_path": str(manifest_path),
    }
    write_jsonl(manifest_path, rows)
    write_json(cfg.manifest_dir / "medvidu_tal_inventory.json", inventory)
    write_json(
        cfg.audit_dir / "timestamp_mapping_by_dataset.json",
        {
            dataset: {
                "n_samples": sum(counter.values()),
                "status_counts": dict(sorted(counter.items())),
                "first_timestamp": _numeric_summary(timestamp_values_by_dataset[dataset]["first"]),
                "last_timestamp": _numeric_summary(timestamp_values_by_dataset[dataset]["last"]),
                "median_positive_delta": (
                    median(timestamp_values_by_dataset[dataset]["positive_delta"])
                    if timestamp_values_by_dataset[dataset]["positive_delta"]
                    else None
                ),
                "non_monotonic_count": int(sum(timestamp_values_by_dataset[dataset]["non_monotonic"])),
                "duplicate_timestamp_count": int(sum(timestamp_values_by_dataset[dataset]["duplicate"])),
                "mapping_failures": [failure for failure in failures if failure.get("dataset_name") == dataset],
            }
            for dataset, counter in sorted(mapping_by_dataset.items())
        },
    )
    write_json(cfg.audit_dir / "gt_leakage_audit.json", {"gt_fields_present": False, "rows_checked": len(rows), "leak_count": 0})
    return inventory


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": float(median(values)), "max": max(values)}


def choose_smoke_rows(rows: list[dict[str, Any]], size: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset_name") or "Unknown")].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for dataset, ds_rows in sorted(by_dataset.items()):
        for key, reason in (
            (lambda r: (-int(r["n_input_frames"]), r["sample_id"]), "max_frame_count"),
            (lambda r: (-float(r["clip_local_duration"]), r["sample_id"]), "max_clip_duration"),
        ):
            ordered = sorted(ds_rows, key=key)
            if ordered:
                selected[ordered[0]["sample_id"]] = {**ordered[0], "_smoke_reason": f"{dataset}:{reason}"}
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


def write_smoke_manifest(output_root: Path, size: int = 20, seed: int = 42) -> dict[str, Any]:
    cfg = RunConfig(output_root=output_root, smoke_size=size, smoke_seed=seed)
    rows = read_jsonl(cfg.manifest_dir / "medvidu_tal_timechat_gt_free.jsonl")
    smoke = choose_smoke_rows(rows, size=size, seed=seed)
    smoke_path = cfg.manifest_dir / "medvidu_tal_timechat_gt_free.smoke.jsonl"
    write_jsonl(smoke_path, smoke)
    payload = {
        "seed": seed,
        "requested_size": size,
        "actual_size": len(smoke),
        "selection_policy": "GT-blind: dataset coverage, frame count, clip duration, sample-id deterministic fill",
        "sample_ids": [row["sample_id"] for row in smoke],
        "samples": [
            {
                "sample_id": row["sample_id"],
                "qa_id": row.get("qa_id"),
                "dataset_name": row.get("dataset_name"),
                "n_input_frames": row.get("n_input_frames"),
                "n_selected_frames": row.get("selection_audit", {}).get("n_selected_frames"),
                "clip_local_duration": row.get("clip_local_duration"),
                "smoke_reason": row.get("_smoke_reason"),
            }
            for row in smoke
        ],
    }
    write_json(cfg.manifest_dir / "smoke_sample_ids.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the GT-free TimeChat-7B ActivityNet VTune MedVidU TAL manifest.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--old-data-root", default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new-data-root", default=DEFAULT_NEW_DATA_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_manifest(args.data_path, args.output_root, old_data_root=args.old_data_root, new_data_root=args.new_data_root)
    if args.smoke:
        inventory["smoke"] = write_smoke_manifest(args.output_root, size=args.smoke_size, seed=args.seed)
        write_json(Path(args.output_root) / "manifest" / "medvidu_tal_inventory.json", inventory)
    print(inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
