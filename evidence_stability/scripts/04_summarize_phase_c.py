#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from evidence_stability.phase_c import phase_c_label  # noqa: E402


PREDICTIONS = ["YES", "NO", "INVALID", "ERROR"]
ALIGNMENT_CLASSES = ["STRONG_GT_ALIGNED", "WEAK_GT_ALIGNED", "NO_GT_OVERLAP"]


def load_selected_qa_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    data = read_json(path)
    return set(data.get("qa_ids", []))


def dedupe_probe_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        cache_key = row.get("cache_key")
        if cache_key and cache_key in seen:
            continue
        if cache_key:
            seen.add(cache_key)
        out.append(row)
    return out


def write_confusion_csv(path: str, labeled: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["dataset_name", "gt_alignment_class", "vlm_yes", "vlm_no", "vlm_invalid", "vlm_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        groups = [("ALL", labeled)]
        for dataset in sorted({r["dataset_name"] for r in labeled}):
            groups.append((dataset, [r for r in labeled if r["dataset_name"] == dataset]))
        for dataset, rows in groups:
            for alignment in ALIGNMENT_CLASSES:
                sub = [r for r in rows if r["gt_alignment_class"] == alignment]
                counter = Counter(r["parsed_prediction"] for r in sub)
                writer.writerow(
                    {
                        "dataset_name": dataset,
                        "gt_alignment_class": alignment,
                        "vlm_yes": counter["YES"],
                        "vlm_no": counter["NO"],
                        "vlm_invalid": counter["INVALID"],
                        "vlm_error": counter["ERROR"],
                    }
                )


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_counts = Counter(r["parsed_prediction"] for r in rows)
    label_counts = Counter(r["phase_c_label"] for r in rows)
    n = len(rows)
    yes_denom = pred_counts["YES"] + pred_counts["NO"] + pred_counts["INVALID"]
    return {
        "n_qa": len({r["qa_id"] for r in rows}),
        "n_windows": n,
        "YES": pred_counts["YES"],
        "NO": pred_counts["NO"],
        "INVALID": pred_counts["INVALID"],
        "ERROR": pred_counts["ERROR"],
        "YES_rate": pred_counts["YES"] / yes_denom if yes_denom else 0.0,
        "INVALID_or_error_rate": (pred_counts["INVALID"] + pred_counts["ERROR"]) / n if n else 0.0,
        "TRUE_SUPPORT": label_counts["TRUE_SUPPORT"],
        "WEAK_TRUE_SUPPORT": label_counts["WEAK_TRUE_SUPPORT"],
        "SPURIOUS_SUPPORT": label_counts["SPURIOUS_SUPPORT"],
        "TRUE_NEGATIVE_CONTROL": label_counts["TRUE_NEGATIVE_CONTROL"],
        "NO_ON_STRONG_GT_ALIGNED": label_counts["NO_ON_STRONG_GT_ALIGNED"],
    }


def write_by_dataset_csv(path: str, labeled: list[dict[str, Any]]) -> None:
    rows = []
    rows.append({"dataset_name": "ALL", **summarize_group(labeled)})
    for dataset in sorted({r["dataset_name"] for r in labeled}):
        rows.append({"dataset_name": dataset, **summarize_group([r for r in labeled if r["dataset_name"] == dataset])})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_qualitative_html(path: str, cases: dict[str, list[dict[str, Any]]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} .case{border:1px solid #ccc;padding:12px;margin:12px 0} .frames{display:flex;gap:4px;overflow-x:auto}.frames img{height:96px;border:1px solid #ddd}</style>",
        "<title>Phase C Qualitative Cases</title></head><body>",
        "<h1>Phase C Qualitative Cases</h1>",
    ]
    for label, rows in cases.items():
        parts.append(f"<h2>{html.escape(label)}</h2>")
        for row in rows:
            parts.append("<div class='case'>")
            parts.append(
                "<p>"
                f"<b>qa_id</b>: {html.escape(row['qa_id'])}<br>"
                f"<b>window_id</b>: {html.escape(row['window_id'])}<br>"
                f"<b>dataset</b>: {html.escape(row['dataset_name'])}<br>"
                f"<b>target</b>: {html.escape(row['target_action'])}<br>"
                f"<b>time</b>: {row.get('start_time')} - {row.get('end_time')}<br>"
                f"<b>raw</b>: {html.escape(str(row.get('raw_response', '')))}<br>"
                f"<b>parsed</b>: {html.escape(row['parsed_prediction'])}<br>"
                f"<b>GT alignment</b>: {html.escape(row['gt_alignment_class'])}"
                "</p>"
            )
            parts.append("<div class='frames'>")
            for frame in row.get("frame_paths", [])[:16]:
                parts.append(f"<img src='{html.escape(frame)}' title='{html.escape(frame)}'>")
            parts.append("</div></div>")
    parts.append("</body></html>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc Phase C probe labeling and summaries.")
    p.add_argument("--probe_results", default="outputs/tal_pilot/phase_c/probe_results.jsonl")
    p.add_argument("--errors", default="outputs/tal_pilot/phase_c/errors.jsonl")
    p.add_argument("--alignment_audit", default="outputs/tal_pilot/window_alignment_audit.jsonl")
    p.add_argument("--selected_qa", default="outputs/tal_pilot/phase_c/phase_c_pilot_qa_ids.json")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_c")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = load_selected_qa_ids(args.selected_qa)

    probe_rows = dedupe_probe_results(read_jsonl(args.probe_results) if Path(args.probe_results).exists() else [])
    error_rows = read_jsonl(args.errors) if Path(args.errors).exists() else []
    if selected is not None:
        probe_rows = [r for r in probe_rows if r["qa_id"] in selected]
        error_rows = [r for r in error_rows if r["qa_id"] in selected]

    audit_by_key = {
        (row["qa_id"], row["window_id"]): row
        for row in read_jsonl(args.alignment_audit)
        if row.get("analysis_eligible")
    }

    labeled = []
    for row in probe_rows:
        key = (row["qa_id"], row["window_id"])
        audit = audit_by_key.get(key)
        if not audit:
            continue
        merged = {
            **row,
            "gt_alignment_class": audit["gt_alignment_class"],
            "n_gt_visible_in_window": audit["n_gt_visible_in_window"],
            "n_gt_visible_total": audit["n_gt_visible_total"],
            "evidence_density": audit["evidence_density"],
            "gt_evidence_recall": audit["gt_evidence_recall"],
        }
        merged["phase_c_label"] = phase_c_label(merged["parsed_prediction"], merged["gt_alignment_class"])
        labeled.append(merged)

    for err in error_rows:
        key = (err["qa_id"], err["window_id"])
        audit = audit_by_key.get(key)
        if not audit:
            continue
        merged = {
            **err,
            "raw_response": "",
            "parsed_prediction": "ERROR",
            "gt_alignment_class": audit["gt_alignment_class"],
            "n_gt_visible_in_window": audit["n_gt_visible_in_window"],
            "n_gt_visible_total": audit["n_gt_visible_total"],
            "evidence_density": audit["evidence_density"],
            "gt_evidence_recall": audit["gt_evidence_recall"],
            "phase_c_label": "ERROR",
        }
        labeled.append(merged)

    write_jsonl(out_dir / "probe_labeled_results.jsonl", labeled)
    write_confusion_csv(str(out_dir / "phase_c_confusion_matrix.csv"), labeled)
    write_by_dataset_csv(str(out_dir / "phase_c_by_dataset.csv"), labeled)

    by_qa = defaultdict(list)
    for row in labeled:
        by_qa[row["qa_id"]].append(row)
    paired_rows = []
    true_counts = []
    spurious_counts = []
    for qa_id, rows in sorted(by_qa.items()):
        n_true = sum(r["phase_c_label"] == "TRUE_SUPPORT" for r in rows)
        n_spurious = sum(r["phase_c_label"] == "SPURIOUS_SUPPORT" for r in rows)
        true_counts.append(n_true)
        spurious_counts.append(n_spurious)
        if n_true > 0 and n_spurious > 0:
            paired_rows.append(
                {
                    "qa_id": qa_id,
                    "clip_id": rows[0]["clip_id"],
                    "dataset_name": rows[0]["dataset_name"],
                    "target_action": rows[0]["target_action"],
                    "n_true_support": n_true,
                    "n_spurious_support": n_spurious,
                    "paired_h2_eligible": True,
                }
            )
    write_jsonl(out_dir / "paired_h2_qa.jsonl", paired_rows)

    case_specs = {
        "TRUE_SUPPORT": lambda r: r["phase_c_label"] == "TRUE_SUPPORT",
        "WEAK_TRUE_SUPPORT": lambda r: r["phase_c_label"] == "WEAK_TRUE_SUPPORT",
        "SPURIOUS_SUPPORT": lambda r: r["phase_c_label"] == "SPURIOUS_SUPPORT",
        "NO_ON_STRONG_GT_ALIGNED": lambda r: r["phase_c_label"] == "NO_ON_STRONG_GT_ALIGNED",
    }
    cases = {
        name: [r for r in labeled if predicate(r)][:20]
        for name, predicate in case_specs.items()
    }
    qual_dir = out_dir / "qualitative_cases"
    write_json(qual_dir / "cases.json", cases)
    write_qualitative_html(str(qual_dir / "index.html"), cases)

    summary = {
        **summarize_group(labeled),
        "model_name": probe_rows[0].get("model_name") if probe_rows else None,
        "model_revision": probe_rows[0].get("model_revision") if probe_rows else None,
        "prompt_version": probe_rows[0].get("prompt_version") if probe_rows else None,
        "n_paired_h2_eligible_qa": len(paired_rows),
        "n_true_support_per_qa": {
            "mean": mean(true_counts) if true_counts else 0,
            "max": max(true_counts) if true_counts else 0,
            "distribution": dict(Counter(true_counts)),
        },
        "n_spurious_support_per_qa": {
            "mean": mean(spurious_counts) if spurious_counts else 0,
            "max": max(spurious_counts) if spurious_counts else 0,
            "distribution": dict(Counter(spurious_counts)),
        },
        "p_yes_given_strong_gt_aligned": _p_yes_given(labeled, "STRONG_GT_ALIGNED"),
        "p_yes_given_no_gt_overlap": _p_yes_given(labeled, "NO_GT_OVERLAP"),
    }
    write_json(out_dir / "phase_c_summary.json", summary)
    write_summary_md(out_dir / "phase_c_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _p_yes_given(rows: list[dict[str, Any]], alignment_class: str) -> float | None:
    sub = [r for r in rows if r["gt_alignment_class"] == alignment_class and r["parsed_prediction"] in {"YES", "NO", "INVALID"}]
    if not sub:
        return None
    return sum(r["parsed_prediction"] == "YES" for r in sub) / len(sub)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase C Summary",
        "",
        f"- number of QA evaluated: {summary['n_qa']}",
        f"- number of windows evaluated: {summary['n_windows']}",
        f"- YES rate: {summary['YES_rate']:.4f}",
        f"- TRUE_SUPPORT count: {summary['TRUE_SUPPORT']}",
        f"- WEAK_TRUE_SUPPORT count: {summary['WEAK_TRUE_SUPPORT']}",
        f"- SPURIOUS_SUPPORT count: {summary['SPURIOUS_SUPPORT']}",
        f"- TRUE_NEGATIVE_CONTROL count: {summary['TRUE_NEGATIVE_CONTROL']}",
        f"- paired_h2_eligible QA: {summary['n_paired_h2_eligible_qa']}",
        f"- INVALID/error rate: {summary['INVALID_or_error_rate']:.4f}",
        f"- P(YES | STRONG_GT_ALIGNED): {summary['p_yes_given_strong_gt_aligned']}",
        f"- P(YES | NO_GT_OVERLAP): {summary['p_yes_given_no_gt_overlap']}",
        "",
        "Phase C stops here. No stress testing or stability analysis was run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
