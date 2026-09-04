from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
from typing import Any

from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GROUNDER_PROMPT,
    PROMPT_VERSION,
    RunConfig,
    TIMESTAMP_ADAPTER_VERSION,
)
from .io_utils import assert_no_gt_leak, read_json, read_jsonl, sha256_text, write_json, write_jsonl
from .temporal_mapper import TemporalMapper, audit_uniform_spacing


def extract_question(sample: dict[str, Any]) -> str:
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
        raise ValueError(
            f"sample {original_index} len(video)={len(raw_video)} "
            f"!= len(sampled_video_frames)={len(sampled_frames)}"
        )
    video = [remap_path(path, old_data_root, new_data_root) for path in raw_video]
    metadata = dict(sample.get("metadata") or {})
    dataset_name = str(sample.get("dataset_name") or sample.get("data_source") or "")
    mapping = TemporalMapper.map(dataset_name, sampled_frames, video, metadata)
    local_times = [obs.local_time for obs in mapping.observations]
    spacing = audit_uniform_spacing(local_times)

    human_question = extract_question(sample)
    row: dict[str, Any] = {
        "sample_id": make_sample_id(sample, original_index),
        "qa_id": sample.get("id"),
        "original_index": original_index,
        "qa_type": "tal",
        "dataset_name": sample.get("dataset_name"),
        "data_source": sample.get("data_source"),
        "clip_id": metadata.get("video_id") or sample.get("id"),
        "human_question": human_question,
        "video": video,
        "sampled_video_frames": sampled_frames,
        "metadata": metadata,
        "n_medvidu_frames": len(video),
        "clip_duration": mapping.clip_duration,
        "first_local_time": local_times[0] if local_times else None,
        "last_local_time": local_times[-1] if local_times else None,
        "time_mapping_method": mapping.time_mapping_method,
        "source_timebase_hz": mapping.source_timebase_hz,
        "clip_duration_source": mapping.clip_duration_source,
        "frame_observations": [obs.to_dict() for obs in mapping.observations],
        "timestamp_spacing_audit": spacing.to_dict(),
        "effective_fps": spacing.effective_fps,
        "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(GROUNDER_PROMPT.format(human_question)),
        "gt_information_available_to_model": False,
        "path_mapping": {
            "old_data_root": old_data_root,
            "new_data_root": new_data_root,
        },
    }
    assert_no_gt_leak(row)
    return row


def build_manifest(
    data_path: Path,
    output_path: Path,
    old_data_root: str = DEFAULT_OLD_DATA_ROOT,
    new_data_root: str | None = DEFAULT_NEW_DATA_ROOT,
) -> dict[str, Any]:
    data = read_json(data_path)
    if not isinstance(data, list):
        raise TypeError("MedVidU input JSON must be a list")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_by_dataset: Counter[str] = Counter()
    tal_by_dataset: Counter[str] = Counter()
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
            rows.append(build_gt_free_row(sample, i, old_data_root=old_data_root, new_data_root=new_data_root))
        except Exception as exc:
            failures.append(
                {
                    "original_index": i,
                    "id": sample.get("id"),
                    "qa_type": sample.get("qa_type"),
                    "dataset_name": sample.get("dataset_name"),
                    "error": repr(exc),
                }
            )
    write_jsonl(output_path, rows)
    report = {
        "data_path": str(data_path),
        "output_path": str(output_path),
        "path_mapping": {"old_data_root": old_data_root, "new_data_root": new_data_root},
        "n_total_samples": len(data),
        "n_tal_samples": sum(1 for sample in data if isinstance(sample, dict) and sample.get("qa_type") == "tal"),
        "manifest_rows": len(rows),
        "failures": failures,
        "counts_by_dataset": dict(sorted(total_by_dataset.items())),
        "tal_counts_by_dataset": dict(sorted(tal_by_dataset.items())),
        "timestamp_status_counts": dict(sorted(Counter(row["timestamp_spacing_audit"]["status"] for row in rows).items())),
    }
    write_json(output_path.with_suffix(".summary.json"), report)
    return report


def choose_smoke_rows(rows: list[dict[str, Any]], size: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset_name") or "Unknown")].append(row)

    selected: dict[str, dict[str, Any]] = {}
    for dataset, ds_rows in sorted(by_dataset.items()):
        ordered = sorted(ds_rows, key=lambda r: (-int(r["n_medvidu_frames"]), r["sample_id"]))
        if ordered:
            selected[ordered[0]["sample_id"]] = {**ordered[0], "_smoke_reason": f"{dataset}:max_frame_count"}
        ordered = sorted(ds_rows, key=lambda r: (-float(r["clip_duration"]), r["sample_id"]))
        if ordered:
            selected[ordered[0]["sample_id"]] = {**ordered[0], "_smoke_reason": f"{dataset}:max_clip_duration"}

    remaining = [row for row in rows if row["sample_id"] not in selected]
    remaining.sort(key=lambda r: (str(r.get("dataset_name") or ""), int(r["n_medvidu_frames"]), float(r["clip_duration"]), r["sample_id"]))
    rng.shuffle(remaining)
    remaining.sort(key=lambda r: (sum(1 for s in selected.values() if s.get("dataset_name") == r.get("dataset_name")), r["sample_id"]))
    for row in remaining:
        if len(selected) >= size:
            break
        selected[row["sample_id"]] = {**row, "_smoke_reason": "seeded_dataset_balanced_fill"}

    out = list(selected.values())
    out.sort(key=lambda r: (str(r.get("dataset_name") or ""), r["sample_id"]))
    return out[:size]


def write_smoke_manifest(rows: list[dict[str, Any]], output_path: Path, size: int = 20, seed: int = 42) -> dict[str, Any]:
    smoke = choose_smoke_rows(rows, size=size, seed=seed)
    write_jsonl(output_path, smoke)
    payload = {
        "seed": seed,
        "requested_size": size,
        "actual_size": len(smoke),
        "selection_policy": "GT-blind: dataset coverage, max frame count, max clip duration, seeded balanced fill",
        "sample_ids": [row["sample_id"] for row in smoke],
        "samples": [
            {
                "sample_id": row["sample_id"],
                "qa_id": row.get("qa_id"),
                "dataset_name": row.get("dataset_name"),
                "n_medvidu_frames": row.get("n_medvidu_frames"),
                "clip_duration": row.get("clip_duration"),
                "timestamp_status": row.get("timestamp_spacing_audit", {}).get("status"),
                "smoke_reason": row.get("_smoke_reason"),
            }
            for row in smoke
        ],
    }
    write_json(output_path.parent / "smoke_sample_ids.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GT-free TimeLens-8B manifest for MedVidU TAL.")
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
    cfg = RunConfig(
        data_path=args.data_path,
        output_root=args.output_root,
        old_data_root=args.old_data_root,
        new_data_root=args.new_data_root,
        smoke_size=args.smoke_size,
        smoke_seed=args.seed,
    )
    cfg.make_dirs()
    manifest_path = cfg.manifest_dir / "medvidu_tal_timelens_manifest_gt_free.jsonl"
    report = build_manifest(cfg.data_path, manifest_path, old_data_root=cfg.old_data_root, new_data_root=cfg.new_data_root)
    if args.smoke:
        report["smoke"] = write_smoke_manifest(
            read_jsonl(manifest_path),
            cfg.manifest_dir / "medvidu_tal_timelens_manifest_gt_free.smoke.jsonl",
            size=cfg.smoke_size,
            seed=cfg.smoke_seed,
        )
        write_json(manifest_path.with_suffix(".summary.json"), report)
    write_json(cfg.provenance_dir / "run_config.json", cfg.to_jsonable())
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
