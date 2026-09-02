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

from evidence_stability.dataset import normalize_tal_sample  # noqa: E402
from evidence_stability.utils import append_jsonl, read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare normalized MedVidU TAL samples with Mapping C.")
    p.add_argument("--data_json", default="data_json/init_datas/medvidu_eccv2026_trainval.json")
    p.add_argument("--frame_root", default=None)
    p.add_argument("--old_frame_root", default="/root/data")
    p.add_argument("--output", default="outputs/tal_pilot/samples.jsonl")
    p.add_argument("--parse_failures", default="outputs/tal_pilot/parse_failures.jsonl")
    p.add_argument("--excluded_temporal_output", default="outputs/tal_pilot/excluded_temporal_samples.jsonl")
    p.add_argument("--summary_output", default="outputs/tal_pilot/prepare_summary.json")
    p.add_argument("--max_samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_verify_paths", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = read_json(args.data_json)
    tal = [(i, sample) for i, sample in enumerate(data) if sample.get("qa_type") == "tal"]

    if args.max_samples >= 0 and args.max_samples < len(tal):
        rng = random.Random(args.seed)
        chosen_positions = sorted(rng.sample(range(len(tal)), args.max_samples))
        tal = [tal[i] for i in chosen_positions]

    for path in [args.output, args.parse_failures, args.excluded_temporal_output]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")

    counts = Counter()
    temporal_counts = Counter()
    dataset_counts = Counter()
    missing_path_samples = 0
    examples = []

    for original_idx, sample in tal:
        normalized, failure = normalize_tal_sample(
            sample,
            original_index=original_idx,
            frame_root=args.frame_root,
            old_frame_root=args.old_frame_root,
            verify_paths=not args.no_verify_paths,
        )
        if failure is not None:
            append_jsonl(args.parse_failures, failure)
            counts["parse_failed"] += 1
            continue

        append_jsonl(args.output, normalized)
        counts["ok"] += 1
        temporal_counts[normalized["temporal_status"]] += 1
        dataset_counts[normalized["dataset_name"]] += 1
        if normalized.get("first_missing_frame"):
            missing_path_samples += 1
        if normalized["temporal_status"] in {"GT_OUT_OF_RANGE", "GT_INVALID", "NO_VISIBLE_GT_FRAME"}:
            append_jsonl(args.excluded_temporal_output, normalized)
        if len(examples) < 10:
            examples.append({
                "sample_id": normalized["sample_id"],
                "dataset_name": normalized["dataset_name"],
                "fps": normalized["fps"],
                "clip_duration": normalized["clip_duration"],
                "frame_index_range": [
                    normalized["sampled_frame_indices"][0],
                    normalized["sampled_frame_indices"][-1],
                ],
                "time_range": [
                    normalized["frame_observations"][0]["local_time"],
                    normalized["frame_observations"][-1]["local_time"],
                ],
                "gt_spans": normalized["gt_spans"],
                "n_gt_visible_total": normalized["n_gt_visible_total"],
                "temporal_status": normalized["temporal_status"],
                "gt_boundary_clipped": normalized["gt_boundary_clipped"],
                "duplicate_count": normalized["duplicate_count"],
            })

    summary = {
        "data_json": args.data_json,
        "frame_root": args.frame_root,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "counts": dict(counts),
        "dataset_counts": dict(dataset_counts),
        "temporal_status_counts": dict(temporal_counts),
        "missing_path_samples": missing_path_samples,
        "first_10_sanity_examples": examples,
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
