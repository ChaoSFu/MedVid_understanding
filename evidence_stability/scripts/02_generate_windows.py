#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.utils import append_jsonl, read_jsonl, write_json  # noqa: E402
from evidence_stability.windows import compute_window_alignment, generate_position_windows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate position-based TAL sliding windows.")
    p.add_argument("--samples", default="outputs/tal_pilot/samples.jsonl")
    p.add_argument("--output", default="outputs/tal_pilot/windows.jsonl")
    p.add_argument("--stats_output", default="outputs/tal_pilot/window_stats.json")
    p.add_argument("--alignment_audit_output", default="outputs/tal_pilot/window_alignment_audit.jsonl")
    p.add_argument("--window_size", type=int, default=16, choices=[8, 16, 24])
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--drop_last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write_alignment_audit", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.samples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("", encoding="utf-8")
    if args.write_alignment_audit:
        Path(args.alignment_audit_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.alignment_audit_output).write_text("", encoding="utf-8")

    counts_by_sample = []
    durations = []
    dataset_counts = Counter()
    total_windows = 0
    status_counts = Counter()

    for sample in samples:
        windows = generate_position_windows(
            sample,
            window_size=args.window_size,
            stride=args.stride,
            drop_last=args.drop_last,
        )
        counts_by_sample.append(len(windows))
        dataset_counts[sample.get("dataset_name")] += len(windows)
        status_counts[sample.get("temporal_status")] += len(windows)
        total_windows += len(windows)
        for window in windows:
            append_jsonl(args.output, window)
            if window["start_time"] is not None and window["end_time"] is not None:
                durations.append(window["end_time"] - window["start_time"])
            if args.write_alignment_audit:
                audit = {
                    **{k: window[k] for k in [
                        "window_id",
                        "sample_id",
                        "dataset_name",
                        "start_pos",
                        "end_pos",
                        "start_time",
                        "end_time",
                        "n_frames",
                        "n_unique_frames",
                        "duplicate_ratio",
                    ]},
                    "temporal_status": sample.get("temporal_status"),
                    "gt_spans": sample.get("gt_spans"),
                    "gt_visible_source_frame_indices": sample.get("gt_visible_source_frame_indices"),
                    **compute_window_alignment(sample, window),
                }
                append_jsonl(args.alignment_audit_output, audit)

    stats = {
        "samples": len(samples),
        "total_windows": total_windows,
        "window_size": args.window_size,
        "stride": args.stride,
        "drop_last": args.drop_last,
        "avg_windows_per_sample": mean(counts_by_sample) if counts_by_sample else 0,
        "median_windows_per_sample": median(counts_by_sample) if counts_by_sample else 0,
        "window_duration_min": min(durations) if durations else None,
        "window_duration_median": median(durations) if durations else None,
        "window_duration_max": max(durations) if durations else None,
        "windows_by_dataset": dict(dataset_counts),
        "windows_by_temporal_status": dict(status_counts),
    }
    write_json(args.stats_output, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
