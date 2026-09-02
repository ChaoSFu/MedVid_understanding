#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.utils import read_json, read_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize and assert Phase B.1 outputs.")
    p.add_argument("--samples", default="outputs/tal_pilot/samples.jsonl")
    p.add_argument("--windows", default="outputs/tal_pilot/windows.jsonl")
    p.add_argument("--alignment_audit", default="outputs/tal_pilot/window_alignment_audit.jsonl")
    p.add_argument("--window_stats", default="outputs/tal_pilot/window_stats.json")
    p.add_argument("--output_json", default="outputs/tal_pilot/phase_b1_summary.json")
    p.add_argument("--output_md", default="outputs/tal_pilot/phase_b1_summary.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.samples)
    windows = read_jsonl(args.windows)
    audit = read_jsonl(args.alignment_audit)
    window_stats = read_json(args.window_stats)

    qa_ids = [s["qa_id"] for s in samples]
    clip_ids = [s["clip_id"] for s in samples]
    audit_window_ids = [w["window_id"] for w in audit]
    window_ids = [w["window_id"] for w in windows]
    class_counts = Counter(row["gt_alignment_class"] for row in audit)
    temporal_by_dataset = defaultdict(Counter)
    for sample in samples:
        temporal_by_dataset[sample["dataset_name"]][sample["temporal_status"]] += 1

    inverted_processed = []
    for sample in samples:
        for span in sample.get("processed_gt_spans", []):
            if not (0 <= span["start"] <= span["end"] <= sample["clip_duration"]):
                inverted_processed.append({
                    "qa_id": sample["qa_id"],
                    "clip_id": sample["clip_id"],
                    "span": span,
                    "clip_duration": sample["clip_duration"],
                })

    qa_window_classes = defaultdict(set)
    for row in audit:
        qa_window_classes[row["qa_id"]].add(row["gt_alignment_class"])

    qa_with_strong_and_no_overlap = sum(
        1
        for classes in qa_window_classes.values()
        if "STRONG_GT_ALIGNED" in classes and "NO_GT_OVERLAP" in classes
    )

    duplicate_window_ids = len(audit_window_ids) - len(set(audit_window_ids))
    duplicate_output_window_ids = len(window_ids) - len(set(window_ids))
    duplicate_qa_ids = len(qa_ids) - len(set(qa_ids))

    checks = {
        "qa_id_unique": duplicate_qa_ids == 0,
        "window_id_unique": duplicate_window_ids == 0,
        "formal_windows_are_analysis_eligible": all(w["analysis_eligible"] for w in windows),
        "no_processed_gt_span_invariant_violations": not inverted_processed,
        "no_aligned_window_for_ineligible_sample": all(
            not (
                row["gt_alignment_class"] in {"STRONG_GT_ALIGNED", "WEAK_GT_ALIGNED"}
                and not row["analysis_eligible"]
            )
            for row in audit
        ),
        "no_gt_overlap_has_zero_visible": all(
            row["n_gt_visible_in_window"] == 0
            for row in audit
            if row["gt_alignment_class"] == "NO_GT_OVERLAP"
        ),
        "strong_has_visible_gt": all(
            row["n_gt_visible_in_window"] > 0
            for row in audit
            if row["gt_alignment_class"] == "STRONG_GT_ALIGNED"
        ),
        "eligible_qa_has_visible_gt": all(
            s["n_gt_visible_total"] > 0
            for s in samples
            if s["analysis_eligible"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"Phase B.1 checks failed: {failed}")

    clip_counter = Counter(clip_ids)
    shared_clip_ids = {clip_id: n for clip_id, n in clip_counter.items() if n > 1}
    summary = {
        "total_qa_records": len(samples),
        "unique_qa_id": len(set(qa_ids)),
        "duplicate_qa_ids": duplicate_qa_ids,
        "unique_clip_id": len(set(clip_ids)),
        "shared_clip_ids": len(shared_clip_ids),
        "temporal_status_by_dataset": {
            dataset: dict(counter) for dataset, counter in sorted(temporal_by_dataset.items())
        },
        "analysis_eligible_qa_count": sum(s["analysis_eligible"] for s in samples),
        "total_windows": window_stats.get("total_windows"),
        "total_generated_windows": window_stats.get("total_windows"),
        "formal_inference_windows": len(windows),
        "analysis_eligible_windows": sum(w["analysis_eligible"] for w in windows),
        "gt_alignment_class_counts": dict(class_counts),
        "qa_with_strong_gt_aligned_and_no_gt_overlap": qa_with_strong_and_no_overlap,
        "invalid_or_inverted_processed_spans": len(inverted_processed),
        "invalid_gt_span_records": sum(len(s.get("invalid_gt_spans", [])) for s in samples),
        "duplicate_window_ids": duplicate_window_ids,
        "duplicate_output_window_ids": duplicate_output_window_ids,
        "window_stats": window_stats,
        "checks": checks,
    }
    write_json(args.output_json, summary)

    lines = [
        "# Phase B.1 Summary",
        "",
        f"- total QA records: {summary['total_qa_records']}",
        f"- unique qa_id: {summary['unique_qa_id']}",
        f"- unique clip_id: {summary['unique_clip_id']}",
        f"- shared clip IDs: {summary['shared_clip_ids']}",
        f"- analysis eligible QA: {summary['analysis_eligible_qa_count']}",
        f"- total generated windows: {summary['total_generated_windows']}",
        f"- formal inference windows: {summary['formal_inference_windows']}",
        f"- analysis eligible windows: {summary['analysis_eligible_windows']}",
        f"- duplicate window IDs: {summary['duplicate_window_ids']}",
        f"- invalid/inverted processed spans: {summary['invalid_or_inverted_processed_spans']}",
        f"- invalid GT span records: {summary['invalid_gt_span_records']}",
        "",
        "## Temporal Status By Dataset",
        "",
        "| dataset | statuses |",
        "|---|---|",
    ]
    for dataset, counts in summary["temporal_status_by_dataset"].items():
        lines.append(f"| {dataset} | `{counts}` |")
    lines.extend([
        "",
        "## GT Alignment Class Counts",
        "",
        f"`{summary['gt_alignment_class_counts']}`",
        "",
        f"QA containing both STRONG_GT_ALIGNED and NO_GT_OVERLAP: {qa_with_strong_and_no_overlap}",
        "",
        "## Checks",
        "",
    ])
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.append("")
    lines.append("Stopped before Phase C/VLM inference.")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
