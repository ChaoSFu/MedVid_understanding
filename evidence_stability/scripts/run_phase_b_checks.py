#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.utils import read_jsonl, write_json  # noqa: E402


def old_position_mapping_oob(sample: dict) -> bool:
    old_duration = (len(sample["sampled_frame_indices"]) - 1) / sample["metadata_fps"]
    tolerance = max(1.0, 1.0 / sample["metadata_fps"])
    max_gt = max((sp["end"] for sp in sample["raw_gt_spans"]), default=0.0)
    return max_gt > old_duration + tolerance


def choose_inspection_samples(samples: list[dict], n: int, seed: int) -> list[dict]:
    clip_counts = Counter(s["clip_id"] for s in samples)
    shared_clips = {clip_id for clip_id, count in clip_counts.items() if count > 1}

    required = [
        lambda s: s["clip_id"] in shared_clips,
        lambda s: s["zero_duration_span_count"] > 0,
        lambda s: any((sp["end"] - sp["start"]) < 2 for sp in s["gt_spans"]),
        lambda s: s["duplicate_count"] > 0,
        lambda s: s["gt_num_spans"] > 1,
        lambda s: s["dataset_name"] == "AVOS" and old_position_mapping_oob(s),
        lambda s: s["dataset_name"] == "CholecT50" and s["metadata_fps"] <= 0.25,
        lambda s: s["dataset_name"] == "CoPESD" and s["temporal_status"] != "OK",
        lambda s: s["gt_boundary_clipped"],
    ]
    datasets = ["AVOS", "CholecT50", "CoPESD", "EgoSurgery", "NurViD"]
    chosen: list[dict] = []
    seen = set()

    def add(sample: dict | None) -> None:
        if sample and sample["qa_id"] not in seen:
            chosen.append(sample)
            seen.add(sample["qa_id"])

    rng = random.Random(seed)
    if shared_clips:
        shared_clip = sorted(shared_clips)[0]
        shared_pool = [s for s in samples if s["clip_id"] == shared_clip]
        shared_pool.sort(key=lambda s: s["qa_id"])
        for sample in shared_pool[:2]:
            add(sample)

    for predicate in required:
        pool = [s for s in samples if predicate(s)]
        rng.shuffle(pool)
        add(pool[0] if pool else None)
    for dataset in datasets:
        pool = [s for s in samples if s.get("dataset_name") == dataset]
        rng.shuffle(pool)
        add(pool[0] if pool else None)

    pool = list(samples)
    rng.shuffle(pool)
    for sample in pool:
        if len(chosen) >= n:
            break
        add(sample)
    return chosen[:n]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write Phase B real-sample alignment inspection.")
    p.add_argument("--samples", default="outputs/tal_pilot/samples.jsonl")
    p.add_argument("--alignment_audit", default="outputs/tal_pilot/window_alignment_audit.jsonl")
    p.add_argument("--output", default="outputs/tal_pilot/phase_b1_targeted_alignment_inspection.json")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_windows_per_sample", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.samples)
    audit_rows = read_jsonl(args.alignment_audit)
    windows_by_qa: dict[str, list[dict]] = {}
    for row in audit_rows:
        windows_by_qa.setdefault(row["qa_id"], []).append(row)

    chosen = choose_inspection_samples(samples, args.n, args.seed)
    output = []
    for sample in chosen:
        obs = sample["frame_observations"]
        cells = sample["temporal_cells"]
        windows = windows_by_qa.get(sample["qa_id"], [])[: args.max_windows_per_sample]
        output.append({
            "qa_id": sample["qa_id"],
            "clip_id": sample["clip_id"],
            "sample_id": sample["sample_id"],
            "dataset_name": sample["dataset_name"],
            "metadata_fps": sample["metadata_fps"],
            "target_action": sample["target_action"],
            "time_mapping_method": sample["time_mapping_method"],
            "source_timebase_hz": sample["source_timebase_hz"],
            "clip_duration_source": sample["clip_duration_source"],
            "clip_duration": sample["clip_duration"],
            "old_position_mapping_oob": old_position_mapping_oob(sample),
            "temporal_status": sample["temporal_status"],
            "analysis_eligible": sample["analysis_eligible"],
            "gt_boundary_clipped": sample["gt_boundary_clipped"],
            "raw_gt_spans": sample["raw_gt_spans"],
            "processed_gt_spans": sample["processed_gt_spans"],
            "invalid_gt_spans": sample["invalid_gt_spans"],
            "gt_visible_source_frame_indices": sample["gt_visible_source_frame_indices"],
            "frame_timestamps_preview": obs[:5] + [{"omitted": max(0, len(obs) - 10)}] + obs[-5:] if len(obs) > 10 else obs,
            "temporal_cells_preview": cells[:5] + [{"omitted": max(0, len(cells) - 10)}] + cells[-5:] if len(cells) > 10 else cells,
            "generated_windows_with_alignment_metrics": windows,
        })
    write_json(args.output, output)
    print(args.output)


if __name__ == "__main__":
    main()
