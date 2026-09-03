#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402


ALIGNMENT_CLASSES = ["STRONG_GT_ALIGNED", "WEAK_GT_ALIGNED", "NO_GT_OVERLAP"]
PREDICTIONS = ["YES", "NO", "INVALID"]
CANDIDATE_LABELS = [
    "TRUE_SUPPORT",
    "WEAK_TRUE_SUPPORT",
    "SPURIOUS_SUPPORT",
    "NO_STRONG_GT",
    "NO_WEAK_GT",
    "TRUE_NEGATIVE_CONTROL",
    "INVALID_MODEL_OUTPUT",
]


def candidate_label(parsed_prediction: str, gt_alignment_class: str) -> str:
    if parsed_prediction == "YES" and gt_alignment_class == "STRONG_GT_ALIGNED":
        return "TRUE_SUPPORT"
    if parsed_prediction == "YES" and gt_alignment_class == "WEAK_GT_ALIGNED":
        return "WEAK_TRUE_SUPPORT"
    if parsed_prediction == "YES" and gt_alignment_class == "NO_GT_OVERLAP":
        return "SPURIOUS_SUPPORT"
    if parsed_prediction == "NO" and gt_alignment_class == "STRONG_GT_ALIGNED":
        return "NO_STRONG_GT"
    if parsed_prediction == "NO" and gt_alignment_class == "WEAK_GT_ALIGNED":
        return "NO_WEAK_GT"
    if parsed_prediction == "NO" and gt_alignment_class == "NO_GT_OVERLAP":
        return "TRUE_NEGATIVE_CONTROL"
    return "INVALID_MODEL_OUTPUT"


def load_selected_qa_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    data = read_json(path)
    return set(data.get("qa_ids", []))


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["qa_id"]), str(row["window_id"])


def index_unique(rows: list[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str]] = []
    for row in rows:
        row_key = key(row)
        if row_key in indexed:
            duplicates.append(row_key)
        indexed[row_key] = row
    if duplicates:
        raise RuntimeError(f"Duplicate qa_id+window_id in {name}: {duplicates[:5]} total={len(duplicates)}")
    return indexed


def read_window_metadata(path: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if rows and "gt_alignment_class" not in rows[0]:
        candidate = Path(path).with_name("window_alignment_audit.jsonl")
        if candidate.exists():
            rows = read_jsonl(candidate)
        else:
            raise RuntimeError(
                f"{path} does not contain gt_alignment_class; pass --windows outputs/tal_pilot/window_alignment_audit.jsonl"
            )
    required = {
        "qa_id",
        "window_id",
        "gt_alignment_class",
        "evidence_density",
        "gt_evidence_recall",
        "n_gt_visible_in_window",
        "n_gt_visible_total",
        "analysis_eligible",
        "dataset_name",
    }
    missing = sorted(required - set(rows[0].keys())) if rows else sorted(required)
    if missing:
        raise RuntimeError(f"Window metadata missing required fields: {missing}")
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def distribution_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0, "iqr": 0, "distribution": {}}
    vals = [float(v) for v in values]
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
        "iqr": percentile(vals, 0.75) - percentile(vals, 0.25),
        "distribution": dict(Counter(values)),
    }


def gt_duration_bin(duration: float | None) -> str:
    if duration is None:
        return "unknown"
    if duration < 2:
        return "<2 sec"
    if duration < 4:
        return "2-4 sec"
    if duration < 10:
        return "4-10 sec"
    if duration < 30:
        return "10-30 sec"
    return ">=30 sec"


def p_yes_given(rows: list[dict[str, Any]], alignment_class: str) -> float | None:
    sub = [r for r in rows if r["gt_alignment_class"] == alignment_class and r["parsed_prediction"] in PREDICTIONS]
    if not sub:
        return None
    return sum(r["parsed_prediction"] == "YES" for r in sub) / len(sub)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred = Counter(r["parsed_prediction"] for r in rows)
    align = Counter(r["gt_alignment_class"] for r in rows)
    labels = Counter(r["candidate_label"] for r in rows)
    denom = pred["YES"] + pred["NO"] + pred["INVALID"]
    return {
        "n_qa": len({r["qa_id"] for r in rows}),
        "n_windows": len(rows),
        "YES": pred["YES"],
        "NO": pred["NO"],
        "INVALID": pred["INVALID"],
        "YES_rate": pred["YES"] / denom if denom else 0.0,
        "STRONG_GT_ALIGNED": align["STRONG_GT_ALIGNED"],
        "WEAK_GT_ALIGNED": align["WEAK_GT_ALIGNED"],
        "NO_GT_OVERLAP": align["NO_GT_OVERLAP"],
        "TRUE_SUPPORT": labels["TRUE_SUPPORT"],
        "WEAK_TRUE_SUPPORT": labels["WEAK_TRUE_SUPPORT"],
        "SPURIOUS_SUPPORT": labels["SPURIOUS_SUPPORT"],
        "NO_STRONG_GT": labels["NO_STRONG_GT"],
        "NO_WEAK_GT": labels["NO_WEAK_GT"],
        "TRUE_NEGATIVE_CONTROL": labels["TRUE_NEGATIVE_CONTROL"],
        "INVALID_MODEL_OUTPUT": labels["INVALID_MODEL_OUTPUT"],
        "p_yes_given_strong_gt_aligned": p_yes_given(rows, "STRONG_GT_ALIGNED"),
        "p_yes_given_weak_gt_aligned": p_yes_given(rows, "WEAK_GT_ALIGNED"),
        "p_yes_given_no_gt_overlap": p_yes_given(rows, "NO_GT_OVERLAP"),
    }


def write_confusion_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    out = []
    for alignment in ALIGNMENT_CLASSES:
        sub = [r for r in rows if r["gt_alignment_class"] == alignment]
        pred = Counter(r["parsed_prediction"] for r in sub)
        total = len(sub)
        out.append(
            {
                "gt_alignment_class": alignment,
                "VLM_YES": pred["YES"],
                "VLM_NO": pred["NO"],
                "INVALID": pred["INVALID"],
                "TOTAL": total,
                "YES_rate": pred["YES"] / total if total else 0.0,
            }
        )
    write_csv(path, out)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_window_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": row["window_id"],
        "start_pos": row.get("start_pos"),
        "end_pos": row.get("end_pos"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "evidence_density": row.get("evidence_density"),
        "gt_evidence_recall": row.get("gt_evidence_recall"),
    }


def build_per_qa(rows: list[dict[str, Any]], sample_by_qa: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qa[row["qa_id"]].append(row)

    per_qa: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for qa_id, qa_rows in sorted(by_qa.items()):
        pred = Counter(r["parsed_prediction"] for r in qa_rows)
        align = Counter(r["gt_alignment_class"] for r in qa_rows)
        labels = Counter(r["candidate_label"] for r in qa_rows)
        true_windows = [candidate_window_payload(r) for r in qa_rows if r["candidate_label"] == "TRUE_SUPPORT"]
        spurious_windows = [candidate_window_payload(r) for r in qa_rows if r["candidate_label"] == "SPURIOUS_SUPPORT"]
        sample = sample_by_qa.get(qa_id, {})
        record = {
            "qa_id": qa_id,
            "clip_id": qa_rows[0]["clip_id"],
            "dataset_name": qa_rows[0]["dataset_name"],
            "target_action": qa_rows[0]["target_action"],
            "target_field": sample.get("target_field"),
            "gt_duration_total": sample.get("gt_duration_total"),
            "gt_duration_bin": gt_duration_bin(sample.get("gt_duration_total")),
            "n_windows": len(qa_rows),
            "n_yes": pred["YES"],
            "n_no": pred["NO"],
            "n_invalid": pred["INVALID"],
            "n_strong_gt_aligned": align["STRONG_GT_ALIGNED"],
            "n_weak_gt_aligned": align["WEAK_GT_ALIGNED"],
            "n_no_gt_overlap": align["NO_GT_OVERLAP"],
            "n_true_support": labels["TRUE_SUPPORT"],
            "n_weak_true_support": labels["WEAK_TRUE_SUPPORT"],
            "n_spurious_support": labels["SPURIOUS_SUPPORT"],
            "n_no_strong_gt": labels["NO_STRONG_GT"],
            "n_no_weak_gt": labels["NO_WEAK_GT"],
            "n_true_negative_control": labels["TRUE_NEGATIVE_CONTROL"],
            "has_true_support": labels["TRUE_SUPPORT"] >= 1,
            "has_spurious_support": labels["SPURIOUS_SUPPORT"] >= 1,
            "paired_h2_eligible": labels["TRUE_SUPPORT"] >= 1 and labels["SPURIOUS_SUPPORT"] >= 1,
        }
        per_qa.append(record)
        if record["paired_h2_eligible"]:
            paired.append(
                {
                    **record,
                    "true_support_window_ids": [w["window_id"] for w in true_windows],
                    "spurious_support_window_ids": [w["window_id"] for w in spurious_windows],
                    "true_support_windows": true_windows,
                    "spurious_support_windows": spurious_windows,
                }
            )
    return per_qa, paired


def write_by_dataset(path: Path, rows: list[dict[str, Any]], per_qa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qa_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qa in per_qa:
        qa_by_dataset[qa["dataset_name"]].append(qa)

    out = []
    for dataset in ["AVOS", "CholecT50", "CoPESD", "EgoSurgery", "NurViD"]:
        sub = [r for r in rows if r["dataset_name"] == dataset]
        summary = summarize_rows(sub)
        qa_rows = qa_by_dataset.get(dataset, [])
        out.append(
            {
                "dataset_name": dataset,
                "n_qa": summary["n_qa"],
                "n_windows": summary["n_windows"],
                "YES": summary["YES"],
                "NO": summary["NO"],
                "YES_rate": summary["YES_rate"],
                "STRONG_GT_ALIGNED": summary["STRONG_GT_ALIGNED"],
                "WEAK_GT_ALIGNED": summary["WEAK_GT_ALIGNED"],
                "NO_GT_OVERLAP": summary["NO_GT_OVERLAP"],
                "TRUE_SUPPORT": summary["TRUE_SUPPORT"],
                "WEAK_TRUE_SUPPORT": summary["WEAK_TRUE_SUPPORT"],
                "SPURIOUS_SUPPORT": summary["SPURIOUS_SUPPORT"],
                "NO_STRONG_GT": summary["NO_STRONG_GT"],
                "TRUE_NEGATIVE_CONTROL": summary["TRUE_NEGATIVE_CONTROL"],
                "P_YES_given_STRONG_GT_ALIGNED": summary["p_yes_given_strong_gt_aligned"],
                "P_YES_given_WEAK_GT_ALIGNED": summary["p_yes_given_weak_gt_aligned"],
                "P_YES_given_NO_GT_OVERLAP": summary["p_yes_given_no_gt_overlap"],
                "n_QA_with_true_support": sum(q["has_true_support"] for q in qa_rows),
                "n_QA_with_spurious_support": sum(q["has_spurious_support"] for q in qa_rows),
                "n_paired_h2_eligible_QA": sum(q["paired_h2_eligible"] for q in qa_rows),
            }
        )
    write_csv(path, out)
    return out


def write_by_target_type(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(r.get("target_field") for r in rows):
        return []
    out = []
    for target_field in sorted({r.get("target_field") for r in rows if r.get("target_field")}):
        out.append({"target_field": target_field, **summarize_rows([r for r in rows if r.get("target_field") == target_field])})
    write_csv(path, out)
    return out


def write_by_gt_duration(path: Path, rows: list[dict[str, Any]], per_qa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qa_by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qa in per_qa:
        qa_by_bin[qa["gt_duration_bin"]].append(qa)
    out = []
    for duration_bin in ["<2 sec", "2-4 sec", "4-10 sec", "10-30 sec", ">=30 sec", "unknown"]:
        sub = [r for r in rows if r.get("gt_duration_bin") == duration_bin]
        if not sub and not qa_by_bin.get(duration_bin):
            continue
        summary = summarize_rows(sub)
        qa_rows = qa_by_bin.get(duration_bin, [])
        out.append(
            {
                "gt_duration_bin": duration_bin,
                "n_qa": len(qa_rows),
                "n_windows": len(sub),
                "P_YES_given_STRONG_GT_ALIGNED": summary["p_yes_given_strong_gt_aligned"],
                "P_YES_given_NO_GT_OVERLAP": summary["p_yes_given_no_gt_overlap"],
                "paired_h2_eligibility_rate": (
                    sum(q["paired_h2_eligible"] for q in qa_rows) / len(qa_rows) if qa_rows else 0.0
                ),
            }
        )
    write_csv(path, out)
    return out


def write_qualitative_cases(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    case_dir = out_dir / "qualitative_cases"
    specs = ["TRUE_SUPPORT", "SPURIOUS_SUPPORT", "NO_STRONG_GT", "WEAK_TRUE_SUPPORT"]
    cases: dict[str, list[dict[str, Any]]] = {}
    for label in specs:
        picked = []
        seen_datasets: set[str] = set()
        candidates = [r for r in rows if r["candidate_label"] == label]
        for row in candidates:
            if row["dataset_name"] in seen_datasets and len(seen_datasets) < 5:
                continue
            seen_datasets.add(row["dataset_name"])
            picked.append(row)
            if len(picked) >= 10:
                break
        if len(picked) < 10:
            used = {r["window_id"] for r in picked}
            picked.extend([r for r in candidates if r["window_id"] not in used][: 10 - len(picked)])
        cases[label] = [qualitative_payload(r) for r in picked]
    write_json(case_dir / "cases.json", cases)
    write_qualitative_html(case_dir / "index.html", cases)
    return cases


def qualitative_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": row["qa_id"],
        "dataset": row["dataset_name"],
        "target_action": row["target_action"],
        "window_id": row["window_id"],
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "frame_paths": row.get("frame_paths") or [],
        "VLM prediction": row["parsed_prediction"],
        "candidate_label": row["candidate_label"],
        "gt_alignment_class": row["gt_alignment_class"],
        "evidence_density": row.get("evidence_density"),
        "gt_evidence_recall": row.get("gt_evidence_recall"),
    }


def write_qualitative_html(path: Path, cases: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px}.case{border:1px solid #ccc;padding:12px;margin:12px 0}.frames{display:flex;gap:4px;overflow-x:auto}.frames img{height:96px;border:1px solid #ddd}</style>",
        "<title>Phase C.3 Qualitative Cases</title></head><body>",
        "<h1>Phase C.3 Qualitative Cases</h1>",
    ]
    for label, rows in cases.items():
        parts.append(f"<h2>{html.escape(label)}</h2>")
        for row in rows:
            parts.append("<div class='case'>")
            parts.append(
                "<p>"
                f"<b>qa_id</b>: {html.escape(row['qa_id'])}<br>"
                f"<b>dataset</b>: {html.escape(row['dataset'])}<br>"
                f"<b>target</b>: {html.escape(row['target_action'])}<br>"
                f"<b>window_id</b>: {html.escape(row['window_id'])}<br>"
                f"<b>time</b>: {row.get('start_time')} - {row.get('end_time')}<br>"
                f"<b>VLM prediction</b>: {html.escape(row['VLM prediction'])}<br>"
                f"<b>candidate_label</b>: {html.escape(row['candidate_label'])}<br>"
                f"<b>gt_alignment_class</b>: {html.escape(row['gt_alignment_class'])}<br>"
                f"<b>evidence_density</b>: {row.get('evidence_density')}<br>"
                f"<b>gt_evidence_recall</b>: {row.get('gt_evidence_recall')}"
                "</p>"
            )
            parts.append("<div class='frames'>")
            for frame in row.get("frame_paths", [])[:16]:
                parts.append(f"<img src='{html.escape(frame)}' title='{html.escape(frame)}'>")
            parts.append("</div></div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def assert_candidate_validity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spurious_violations = []
    true_violations = []
    for row in rows:
        if row["candidate_label"] == "SPURIOUS_SUPPORT":
            if row["gt_alignment_class"] != "NO_GT_OVERLAP" or row["n_gt_visible_in_window"] != 0:
                spurious_violations.append(key(row))
            if row.get("evidence_density") not in (0, 0.0) or row.get("gt_evidence_recall") not in (0, 0.0):
                spurious_violations.append(key(row))
        if row["candidate_label"] == "TRUE_SUPPORT":
            strong_threshold = (
                row["gt_alignment_class"] == "STRONG_GT_ALIGNED"
                and row["n_gt_visible_in_window"] > 0
                and (row.get("evidence_density", 0) >= 0.25 or row.get("gt_evidence_recall", 0) >= 0.50)
            )
            if not strong_threshold:
                true_violations.append(key(row))
    if spurious_violations:
        raise RuntimeError(f"SPURIOUS_SUPPORT GT violations: {spurious_violations[:5]} total={len(spurious_violations)}")
    if true_violations:
        raise RuntimeError(f"TRUE_SUPPORT GT violations: {true_violations[:5]} total={len(true_violations)}")
    return {
        "spurious_support_gt_violations": 0,
        "true_support_gt_violations": 0,
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    global_s = summary["global"]
    gate = summary["h2_gate"]
    lines = [
        "# Phase C.3 Post-hoc Candidate Labeling and H2 Gate Analysis",
        "",
        "## Global",
        f"- n_qa: {global_s['n_qa']}",
        f"- n_windows: {global_s['n_windows']}",
        f"- YES: {global_s['YES']}",
        f"- NO: {global_s['NO']}",
        f"- TRUE_SUPPORT: {global_s['TRUE_SUPPORT']}",
        f"- WEAK_TRUE_SUPPORT: {global_s['WEAK_TRUE_SUPPORT']}",
        f"- SPURIOUS_SUPPORT: {global_s['SPURIOUS_SUPPORT']}",
        f"- NO_STRONG_GT: {global_s['NO_STRONG_GT']}",
        f"- TRUE_NEGATIVE_CONTROL: {global_s['TRUE_NEGATIVE_CONTROL']}",
        f"- P(YES | STRONG_GT_ALIGNED): {global_s['p_yes_given_strong_gt_aligned']}",
        f"- P(YES | WEAK_GT_ALIGNED): {global_s['p_yes_given_weak_gt_aligned']}",
        f"- P(YES | NO_GT_OVERLAP): {global_s['p_yes_given_no_gt_overlap']}",
        "",
        "## QA Level",
        f"- QA with TRUE_SUPPORT: {gate['n_qa_with_true_support']}",
        f"- QA with SPURIOUS_SUPPORT: {gate['n_qa_with_spurious_support']}",
        f"- paired_h2_eligible QA: {gate['n_paired_h2_eligible_qa']}",
        "",
        "## Audit",
        f"- unmatched joins: {summary['join_audit']['n_unmatched_probe_rows']}",
        f"- duplicate joins: {summary['join_audit']['n_duplicate_join_keys']}",
        f"- spurious-support GT violations: {summary['validity_audit']['spurious_support_gt_violations']}",
        f"- true-support GT violations: {summary['validity_audit']['true_support_gt_violations']}",
        "",
        "No temporal interventions, stability scores, or Phase D analyses were run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C.3 post-hoc candidate labeling and H2 gate analysis.")
    p.add_argument("--probe_results", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50/probe_results.jsonl")
    p.add_argument("--windows", default="outputs/tal_pilot/window_alignment_audit.jsonl")
    p.add_argument("--samples", default="outputs/tal_pilot/samples.jsonl")
    p.add_argument("--qa_ids", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50/phase_c_pilot_qa_ids.json")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = load_selected_qa_ids(args.qa_ids)

    probe_rows = read_jsonl(args.probe_results)
    if selected is not None:
        probe_rows = [r for r in probe_rows if r["qa_id"] in selected]
    raw_model_gt_fields = sorted(
        {
            field
            for row in probe_rows
            for field in row
            if field in {"gt_alignment_class", "gt_spans", "evidence_density", "gt_evidence_recall", "TRUE_SUPPORT", "SPURIOUS_SUPPORT"}
        }
    )
    if raw_model_gt_fields:
        raise RuntimeError(f"Raw probe results contain GT fields: {raw_model_gt_fields}")

    probe_by_key = index_unique(probe_rows, "probe_results")
    window_rows = read_window_metadata(args.windows)
    if selected is not None:
        window_rows = [r for r in window_rows if r["qa_id"] in selected]
    windows_by_key = index_unique(window_rows, "Phase B.1 window metadata")
    sample_by_qa = {row["qa_id"]: row for row in read_jsonl(args.samples)} if Path(args.samples).exists() else {}

    unmatched = [row_key for row_key in probe_by_key if row_key not in windows_by_key]
    if unmatched:
        raise RuntimeError(f"Unmatched probe rows: {unmatched[:5]} total={len(unmatched)}")

    labeled: list[dict[str, Any]] = []
    for row_key, probe in probe_by_key.items():
        window = windows_by_key[row_key]
        if not window.get("analysis_eligible"):
            raise RuntimeError(f"Probe row maps to analysis_eligible=false window: {row_key}")
        sample = sample_by_qa.get(probe["qa_id"], {})
        merged = {
            "qa_id": probe["qa_id"],
            "clip_id": probe["clip_id"],
            "window_id": probe["window_id"],
            "dataset_name": probe["dataset_name"],
            "target_action": probe.get("target_action"),
            "target_field": sample.get("target_field"),
            "parsed_prediction": probe.get("parsed_prediction", "INVALID"),
            "raw_response": probe.get("raw_response", ""),
            "gt_alignment_class": window["gt_alignment_class"],
            "evidence_density": window["evidence_density"],
            "gt_evidence_recall": window["gt_evidence_recall"],
            "n_gt_visible_in_window": window["n_gt_visible_in_window"],
            "n_gt_visible_total": window["n_gt_visible_total"],
            "start_time": window.get("start_time", probe.get("start_time")),
            "end_time": window.get("end_time", probe.get("end_time")),
            "start_pos": window.get("start_pos"),
            "end_pos": window.get("end_pos"),
            "n_frames": window.get("n_frames", probe.get("n_frames")),
            "n_unique_frames": window.get("n_unique_frames", probe.get("n_unique_frames")),
            "duplicate_ratio": window.get("duplicate_ratio", probe.get("duplicate_ratio")),
            "temporal_status": window.get("temporal_status"),
            "analysis_eligible": window.get("analysis_eligible"),
            "gt_duration_total": sample.get("gt_duration_total"),
            "gt_duration_bin": gt_duration_bin(sample.get("gt_duration_total")),
            "time_mapping_method": sample.get("time_mapping_method"),
            "source_timebase_hz": sample.get("source_timebase_hz"),
            "clip_duration_source": sample.get("clip_duration_source"),
            "frame_paths": probe.get("frame_paths") or [],
            "source_frame_paths": probe.get("source_frame_paths"),
            "model_name": probe.get("model_name"),
            "model_revision": probe.get("model_revision"),
            "prompt_version": probe.get("prompt_version"),
            "cache_key": probe.get("cache_key"),
        }
        merged["candidate_label"] = candidate_label(merged["parsed_prediction"], merged["gt_alignment_class"])
        labeled.append(merged)

    joined_by_key = index_unique(labeled, "joined labeled results")
    if len(joined_by_key) != len(labeled):
        raise RuntimeError("Joined result key accounting failed.")

    labels = Counter(r["candidate_label"] for r in labeled)
    pred = Counter(r["parsed_prediction"] for r in labeled)
    yes_sum = labels["TRUE_SUPPORT"] + labels["WEAK_TRUE_SUPPORT"] + labels["SPURIOUS_SUPPORT"]
    no_sum = labels["NO_STRONG_GT"] + labels["NO_WEAK_GT"] + labels["TRUE_NEGATIVE_CONTROL"]
    if yes_sum != pred["YES"]:
        raise RuntimeError(f"YES-side accounting failed: candidates={yes_sum} total_yes={pred['YES']}")
    if no_sum != pred["NO"]:
        raise RuntimeError(f"NO-side accounting failed: candidates={no_sum} total_no={pred['NO']}")

    validity_audit = assert_candidate_validity(labeled)
    per_qa, paired = build_per_qa(labeled, sample_by_qa)
    for row in paired:
        if not (row["n_true_support"] >= 1 and row["n_spurious_support"] >= 1):
            raise RuntimeError(f"Invalid paired_h2_eligible row: {row['qa_id']}")

    write_jsonl(out_dir / "probe_labeled_results.jsonl", labeled)
    write_confusion_matrix(out_dir / "phase_c_confusion_matrix.csv", labeled)
    write_csv(out_dir / "phase_c_per_qa.csv", per_qa)
    write_jsonl(out_dir / "paired_h2_qa.jsonl", paired)
    by_dataset = write_by_dataset(out_dir / "phase_c_by_dataset.csv", labeled, per_qa)
    by_target_type = write_by_target_type(out_dir / "phase_c_by_target_type.csv", labeled)
    by_gt_duration = write_by_gt_duration(out_dir / "phase_c_by_gt_duration.csv", labeled, per_qa)
    write_qualitative_cases(out_dir, labeled)

    true_counts_all = [q["n_true_support"] for q in per_qa]
    spurious_counts_all = [q["n_spurious_support"] for q in per_qa]
    paired_true_counts = [q["n_true_support"] for q in paired]
    paired_spurious_counts = [q["n_spurious_support"] for q in paired]
    paired_dataset_counts = Counter(q["dataset_name"] for q in paired)
    spurious_by_dataset = Counter(r["dataset_name"] for r in labeled if r["candidate_label"] == "SPURIOUS_SUPPORT")
    spurious_by_action = Counter(r["target_action"] for r in labeled if r["candidate_label"] == "SPURIOUS_SUPPORT")

    summary = {
        "inputs": {
            "probe_results": args.probe_results,
            "windows": args.windows,
            "samples": args.samples,
            "qa_ids": args.qa_ids,
        },
        "join_audit": {
            "n_probe_rows": len(probe_rows),
            "n_matched_rows": len(labeled),
            "n_unmatched_probe_rows": len(unmatched),
            "n_duplicate_join_keys": 0,
        },
        "raw_probe_gt_leakage_audit": {
            "raw_model_gt_fields": raw_model_gt_fields,
            "gt_information_needed_to_produce_original_vlm_prediction": False,
        },
        "validity_audit": validity_audit,
        "global": summarize_rows(labeled),
        "candidate_counts": dict(labels),
        "h2_gate": {
            "non_trivial_spurious_support_pool": labels["SPURIOUS_SUPPORT"] > 0,
            "n_qa_with_true_support": sum(q["has_true_support"] for q in per_qa),
            "n_qa_with_spurious_support": sum(q["has_spurious_support"] for q in per_qa),
            "n_paired_h2_eligible_qa": len(paired),
            "paired_dataset_counts": dict(paired_dataset_counts),
            "spurious_support_by_dataset": dict(spurious_by_dataset),
            "spurious_support_top_targets": dict(spurious_by_action.most_common(20)),
        },
        "qa_level_distributions": {
            "n_true_support_per_qa_all": distribution_stats(true_counts_all),
            "n_spurious_support_per_qa_all": distribution_stats(spurious_counts_all),
            "n_true_support_per_qa_paired": distribution_stats(paired_true_counts),
            "n_spurious_support_per_qa_paired": distribution_stats(paired_spurious_counts),
        },
        "by_dataset": by_dataset,
        "by_target_type": by_target_type,
        "by_gt_duration": by_gt_duration,
        "window_level_statistics_warning": (
            "Window counts differ by QA and dataset; use QA-level and dataset-level summaries before drawing conclusions."
        ),
    }
    write_json(out_dir / "phase_c_posthoc_summary.json", summary)
    write_summary_md(out_dir / "phase_c_posthoc_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
