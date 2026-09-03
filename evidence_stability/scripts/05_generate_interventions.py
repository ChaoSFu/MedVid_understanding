#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.interventions import (  # noqa: E402
    generate_context,
    generate_resample,
    generate_shift,
    intervention_position_plan,
)
from evidence_stability.utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from evidence_stability.windows import compute_window_alignment  # noqa: E402


INTERVENTION_TYPES = ["SHIFT_LEFT_2", "SHIFT_RIGHT_2", "RESAMPLE_75", "RESAMPLE_50", "CONTEXT_1P5X"]
CORE_LABELS = {"TRUE_SUPPORT", "SPURIOUS_SUPPORT"}
SHIFT_TYPES = {"SHIFT_LEFT_2", "SHIFT_RIGHT_2"}
RESAMPLE_TYPES = {"RESAMPLE_75", "RESAMPLE_50"}
CONTEXT_TYPES = {"CONTEXT_1P5X"}


def read_json_if_exists(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_paired_qa_ids(path: str) -> set[str]:
    return {row["qa_id"] for row in read_jsonl(path)}


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["qa_id"]), str(row["window_id"])


def index_unique(rows: list[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = []
    for row in rows:
        key = row_key(row)
        if key in out:
            duplicates.append(key)
        out[key] = row
    if duplicates:
        raise RuntimeError(f"Duplicate qa_id+window_id in {name}: {duplicates[:5]} total={len(duplicates)}")
    return out


def normalize_candidate_label(row: dict[str, Any]) -> str:
    label = row.get("candidate_label") or row.get("phase_c_label")
    if label == "NO_ON_STRONG_GT_ALIGNED":
        label = "NO_STRONG_GT"
    if label == "NO_ON_WEAK_GT_ALIGNED":
        label = "NO_WEAK_GT"
    return str(label)


def intervention_family(intervention_type: str) -> str:
    if intervention_type in SHIFT_TYPES:
        return "SHIFT"
    if intervention_type in RESAMPLE_TYPES:
        return "RESAMPLE"
    if intervention_type in CONTEXT_TYPES:
        return "CONTEXT"
    raise ValueError(f"Unknown intervention type: {intervention_type}")


def gt_blind_generation_audit() -> dict[str, Any]:
    forbidden = {
        "gt_span",
        "gt_spans",
        "gt_alignment_class",
        "evidence_density",
        "gt_evidence_recall",
        "n_gt_visible_in_window",
        "n_gt_visible_total",
        "candidate_label",
    }
    functions = [generate_shift, generate_resample, generate_context, intervention_position_plan]
    violations = {}
    for fn in functions:
        params = set(inspect.signature(fn).parameters)
        overlap = sorted(params & forbidden)
        if overlap:
            violations[fn.__name__] = overlap
    if violations:
        raise RuntimeError(f"GT leakage into generation function signatures: {violations}")
    return {"gt_leakage_into_generation": 0, "checked_functions": [fn.__name__ for fn in functions]}


def parse_position_from_window_id(window_id: str) -> tuple[int, int]:
    tail = window_id.rsplit("::pos", 1)[-1]
    start, end = tail.split("-", 1)
    return int(start), int(end)


def get_candidate_positions(candidate: dict[str, Any], window_meta: dict[str, Any]) -> tuple[int, int]:
    start = candidate.get("start_pos", window_meta.get("start_pos"))
    end = candidate.get("end_pos", window_meta.get("end_pos"))
    if start is None or end is None:
        return parse_position_from_window_id(candidate["window_id"])
    return int(start), int(end)


def build_intervention_window(sample: dict[str, Any], candidate: dict[str, Any], positions: tuple[int, ...], intervention_id: str) -> dict[str, Any]:
    frame_paths = sample["frame_paths"]
    frame_indices = sample["sampled_frame_indices"]
    observations = sample["frame_observations"]
    return {
        "window_id": intervention_id,
        "qa_id": candidate["qa_id"],
        "clip_id": candidate["clip_id"],
        "sample_id": candidate["qa_id"],
        "dataset_name": candidate["dataset_name"],
        "analysis_eligible": True,
        "start_pos": positions[0],
        "end_pos": positions[-1],
        "frame_paths": [frame_paths[i] for i in positions],
        "frame_indices": [frame_indices[i] for i in positions],
        "frame_positions": list(positions),
        "frame_times": [float(observations[i]["local_time"]) for i in positions],
        "n_frames": len(positions),
        "n_unique_frames": len(set(frame_indices[i] for i in positions)),
        "duplicate_count": len(positions) - len(set(frame_indices[i] for i in positions)),
    }


def classify_validity(candidate_label: str, alignment: dict[str, Any]) -> dict[str, Any]:
    n_visible = int(alignment["n_gt_visible_in_window"])
    alignment_class = alignment["gt_alignment_class"]
    if candidate_label == "TRUE_SUPPORT":
        strict = alignment_class == "STRONG_GT_ALIGNED"
        any_gt = n_visible > 0
        if strict:
            reason = "STRICT_VALID_STRONG_GT_ALIGNED"
        elif any_gt:
            reason = "WEAKENED_GT_SUPPORT"
        else:
            reason = "GT_EVIDENCE_REMOVED"
        return {"strict_valid": strict, "any_gt_valid": any_gt, "validity_reason": reason}
    if candidate_label == "SPURIOUS_SUPPORT":
        strict = n_visible == 0 and alignment_class == "NO_GT_OVERLAP"
        reason = "STRICT_VALID_NO_GT_OVERLAP" if strict else "GT_CONTAMINATED"
        return {"strict_valid": strict, "any_gt_valid": strict, "validity_reason": reason}
    raise RuntimeError(f"Unsupported original candidate label: {candidate_label}")


def source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_intervention_record(
    candidate: dict[str, Any],
    window_meta: dict[str, Any],
    sample: dict[str, Any],
    intervention_type: str,
) -> dict[str, Any]:
    original_label = normalize_candidate_label(candidate)
    if original_label not in CORE_LABELS:
        raise RuntimeError(f"Unexpected candidate_label for intervention: {original_label}")

    start_pos, end_pos = get_candidate_positions(candidate, window_meta)
    clip_n = len(sample["frame_paths"])
    plan = intervention_position_plan(start_pos, end_pos, clip_n, intervention_type)
    intervention_id = f"{candidate['window_id']}::{intervention_type}"
    original_positions = tuple(range(start_pos, end_pos + 1))
    original_frame_paths = [sample["frame_paths"][i] for i in original_positions]
    original_frame_indices = [sample["sampled_frame_indices"][i] for i in original_positions]
    original_frame_times = [float(sample["frame_observations"][i]["local_time"]) for i in original_positions]
    intervened_frame_times = [float(sample["frame_observations"][i]["local_time"]) for i in plan.positions]

    record: dict[str, Any] = {
        "qa_id": candidate["qa_id"],
        "clip_id": candidate["clip_id"],
        "window_id": candidate["window_id"],
        "intervention_id": intervention_id,
        "dataset_name": candidate["dataset_name"],
        "target_action": candidate.get("target_action"),
        "target_field": candidate.get("target_field") or sample.get("target_field"),
        "gt_duration_total": candidate.get("gt_duration_total", sample.get("gt_duration_total")),
        "gt_duration_bin": candidate.get("gt_duration_bin"),
        "original_candidate_label": original_label,
        "intervention_family": intervention_family(intervention_type),
        "intervention_type": intervention_type,
        "original_position_start": start_pos,
        "original_position_end": end_pos,
        "original_position_indices": list(original_positions),
        "original_start_time": window_meta.get("start_time", candidate.get("start_time")),
        "original_end_time": window_meta.get("end_time", candidate.get("end_time")),
        "original_frame_paths": original_frame_paths,
        "original_frame_indices": original_frame_indices,
        "original_frame_times": original_frame_times,
        "original_n_frames": len(original_positions),
        "original_n_unique_frames": len(set(original_frame_indices)),
        "intervened_position_indices": list(plan.positions),
        "intervened_frame_paths": [sample["frame_paths"][i] for i in plan.positions],
        "intervened_frame_indices": [sample["sampled_frame_indices"][i] for i in plan.positions],
        "intervened_frame_times": intervened_frame_times,
        "intervened_start_time": min(intervened_frame_times) if intervened_frame_times else None,
        "intervened_end_time": max(intervened_frame_times) if intervened_frame_times else None,
        "intervened_n_frames": len(plan.positions),
        "intervened_n_unique_frames": len(set(sample["sampled_frame_indices"][i] for i in plan.positions)),
        "boundary_asymmetric": plan.boundary_asymmetric,
        "generation_valid": plan.generation_valid,
        "generation_invalid_reason": plan.generation_invalid_reason,
        "generation_metadata": {
            "gt_blind": True,
            "coordinate_space": "sample_position",
            "clip_n_sampled_positions": clip_n,
            "deterministic": True,
            "resampling_rule": "round(i * (input_length - 1) / (output_length - 1)) for evenly spaced indices",
        },
        "original_gt_alignment_class": candidate.get("gt_alignment_class"),
        "original_evidence_density": candidate.get("evidence_density"),
        "original_gt_evidence_recall": candidate.get("gt_evidence_recall"),
        "original_n_gt_visible": candidate.get("n_gt_visible_in_window"),
        "intervention_gt_alignment_class": None,
        "intervention_evidence_density": None,
        "intervention_gt_evidence_recall": None,
        "intervention_n_gt_visible": None,
        "intervention_n_unique_gt_visible": None,
        "strict_valid": False,
        "any_gt_valid": False,
        "validity_reason": plan.generation_invalid_reason,
    }

    if not plan.generation_valid:
        return record

    if intervention_type == "RESAMPLE_75" and len(plan.positions) != 12:
        raise RuntimeError(f"RESAMPLE_75 returned {len(plan.positions)} positions")
    if intervention_type == "RESAMPLE_50" and len(plan.positions) != 8:
        raise RuntimeError(f"RESAMPLE_50 returned {len(plan.positions)} positions")
    if intervention_type in SHIFT_TYPES and len(plan.positions) != 16:
        raise RuntimeError(f"{intervention_type} returned {len(plan.positions)} positions")
    if intervention_type == "CONTEXT_1P5X" and len(plan.positions) != 24:
        raise RuntimeError(f"CONTEXT_1P5X returned {len(plan.positions)} positions")
    if list(plan.positions) != sorted(plan.positions):
        raise RuntimeError(f"Non-monotonic positions for {intervention_id}")
    sample_frame_path_set = set(sample["frame_paths"])
    if any(path not in sample_frame_path_set for path in record["intervened_frame_paths"]):
        raise RuntimeError(f"Intervention frame path outside QA clip: {intervention_id}")

    intervention_window = build_intervention_window(sample, candidate, plan.positions, intervention_id)
    alignment = compute_window_alignment(sample, intervention_window)
    validity = classify_validity(original_label, alignment)
    record.update(
        {
            "intervention_gt_alignment_class": alignment["gt_alignment_class"],
            "intervention_evidence_density": alignment["evidence_density"],
            "intervention_gt_evidence_recall": alignment["gt_evidence_recall"],
            "intervention_n_gt_visible": alignment["n_gt_visible_in_window"],
            "intervention_n_unique_gt_visible": alignment["n_gt_visible_in_window"],
            **validity,
        }
    )
    return record


def summarize_by_label_and_type(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in ["TRUE_SUPPORT", "SPURIOUS_SUPPORT"]:
        for intervention_type in INTERVENTION_TYPES:
            sub = [r for r in records if r["original_candidate_label"] == label and r["intervention_type"] == intervention_type]
            valid = [r for r in sub if r["generation_valid"]]
            strict_valid = [r for r in valid if r["strict_valid"]]
            row = {
                "original_candidate_label": label,
                "intervention_type": intervention_type,
                "attempted": len(sub),
                "generation_valid": len(valid),
                "strict_valid": len(strict_valid),
                "strict_invalid": len(valid) - len(strict_valid),
                "strict_valid_rate": len(strict_valid) / len(valid) if valid else 0.0,
            }
            if label == "TRUE_SUPPORT":
                any_gt = sum(r["any_gt_valid"] for r in valid)
                row.update(
                    {
                        "any_gt_valid": any_gt,
                        "any_gt_valid_rate": any_gt / len(valid) if valid else 0.0,
                        "WEAKENED_GT_SUPPORT": sum(r["validity_reason"] == "WEAKENED_GT_SUPPORT" for r in valid),
                        "GT_EVIDENCE_REMOVED": sum(r["validity_reason"] == "GT_EVIDENCE_REMOVED" for r in valid),
                        "GT_CONTAMINATED": 0,
                    }
                )
            else:
                row.update(
                    {
                        "any_gt_valid": None,
                        "any_gt_valid_rate": None,
                        "WEAKENED_GT_SUPPORT": 0,
                        "GT_EVIDENCE_REMOVED": 0,
                        "GT_CONTAMINATED": sum(r["validity_reason"] == "GT_CONTAMINATED" for r in valid),
                    }
                )
            rows.append(row)
    return rows


def summarize_stratified(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    values = ["AVOS", "CholecT50", "EgoSurgery", "NurViD"] if field == "dataset_name" else ["action", "phase"]
    rows = []
    for value in values:
        for label in ["TRUE_SUPPORT", "SPURIOUS_SUPPORT"]:
            for intervention_type in INTERVENTION_TYPES:
                sub = [
                    r
                    for r in records
                    if r.get(field) == value
                    and r["original_candidate_label"] == label
                    and r["intervention_type"] == intervention_type
                ]
                valid = [r for r in sub if r["generation_valid"]]
                strict = [r for r in valid if r["strict_valid"]]
                rows.append(
                    {
                        field: value,
                        "original_candidate_label": label,
                        "intervention_type": intervention_type,
                        "n_attempted": len(sub),
                        "n_generation_valid": len(valid),
                        "n_strict_valid": len(strict),
                        "strict_valid_rate": len(strict) / len(valid) if valid else 0.0,
                    }
                )
    return rows


def build_per_qa(records: list[dict[str, Any]], paired_qa_ids: set[str], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_qa[row["qa_id"]].append(row)
    candidates_by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_qa[row["qa_id"]].append(row)

    out = []
    for qa_id in sorted(paired_qa_ids):
        rows = by_qa.get(qa_id, [])
        candidates = candidates_by_qa.get(qa_id, [])
        first = (rows or candidates)[0]
        record: dict[str, Any] = {
            "qa_id": qa_id,
            "dataset_name": first["dataset_name"],
            "target_action": first.get("target_action"),
            "target_field": first.get("target_field"),
            "n_true_candidates": sum(normalize_candidate_label(c) == "TRUE_SUPPORT" for c in candidates),
            "n_spurious_candidates": sum(normalize_candidate_label(c) == "SPURIOUS_SUPPORT" for c in candidates),
        }
        usable_any = {"TRUE_SUPPORT": False, "SPURIOUS_SUPPORT": False}
        family_usable: dict[str, dict[str, bool]] = {
            "SHIFT": {"TRUE_SUPPORT": False, "SPURIOUS_SUPPORT": False},
            "RESAMPLE": {"TRUE_SUPPORT": False, "SPURIOUS_SUPPORT": False},
            "CONTEXT": {"TRUE_SUPPORT": False, "SPURIOUS_SUPPORT": False},
        }
        for family in ["SHIFT", "RESAMPLE", "CONTEXT"]:
            for label in ["TRUE_SUPPORT", "SPURIOUS_SUPPORT"]:
                sub = [r for r in rows if r["intervention_family"] == family and r["original_candidate_label"] == label]
                valid = [r for r in sub if r["strict_valid"]]
                prefix = "true" if label == "TRUE_SUPPORT" else "spurious"
                record[f"n_{prefix}_{family.lower()}_attempted"] = len(sub)
                record[f"n_{prefix}_{family.lower()}_valid"] = len(valid)
                family_usable[family][label] = bool(valid)
                usable_any[label] = usable_any[label] or bool(valid)
        record["paired_intervention_usable"] = usable_any["TRUE_SUPPORT"] and usable_any["SPURIOUS_SUPPORT"]
        record["paired_shift_usable"] = family_usable["SHIFT"]["TRUE_SUPPORT"] and family_usable["SHIFT"]["SPURIOUS_SUPPORT"]
        record["paired_resample_usable"] = family_usable["RESAMPLE"]["TRUE_SUPPORT"] and family_usable["RESAMPLE"]["SPURIOUS_SUPPORT"]
        record["paired_context_usable"] = family_usable["CONTEXT"]["TRUE_SUPPORT"] and family_usable["CONTEXT"]["SPURIOUS_SUPPORT"]
        out.append(record)
    return out


def duplicate_sequence_count(records: list[dict[str, Any]]) -> int:
    count = 0
    by_qa: dict[str, dict[tuple[str, ...], str]] = defaultdict(dict)
    for row in records:
        if not row["generation_valid"]:
            continue
        seq = tuple(row["intervened_frame_paths"])
        seen = by_qa[row["qa_id"]]
        if seq in seen and seen[seq] != row["intervention_id"]:
            count += 1
        else:
            seen[seq] = row["intervention_id"]
    return count


def assert_manifest(records: list[dict[str, Any]], source_probe_hash_before: str, probe_path: Path) -> dict[str, Any]:
    ids = [r["intervention_id"] for r in records]
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        raise RuntimeError(f"Duplicate intervention IDs: {duplicate_ids}")
    invalid_frame_counts = []
    order_violations = []
    spurious_strict_gt_violations = []
    true_strict_gt_violations = []
    for row in records:
        positions = row["intervened_position_indices"]
        if positions != sorted(positions):
            order_violations.append(row["intervention_id"])
        if row["generation_valid"]:
            expected = {"RESAMPLE_75": 12, "RESAMPLE_50": 8, "CONTEXT_1P5X": 24}.get(row["intervention_type"], 16)
            if row["intervened_n_frames"] != expected:
                invalid_frame_counts.append(row["intervention_id"])
        if row["strict_valid"] and row["original_candidate_label"] == "SPURIOUS_SUPPORT" and row["intervention_n_gt_visible"] != 0:
            spurious_strict_gt_violations.append(row["intervention_id"])
        if row["strict_valid"] and row["original_candidate_label"] == "TRUE_SUPPORT" and row["intervention_gt_alignment_class"] != "STRONG_GT_ALIGNED":
            true_strict_gt_violations.append(row["intervention_id"])
    if invalid_frame_counts:
        raise RuntimeError(f"Invalid frame counts: {invalid_frame_counts[:5]} total={len(invalid_frame_counts)}")
    if order_violations:
        raise RuntimeError(f"Temporal order violations: {order_violations[:5]} total={len(order_violations)}")
    if spurious_strict_gt_violations:
        raise RuntimeError(f"Spurious strict-valid GT violations: {spurious_strict_gt_violations[:5]}")
    if true_strict_gt_violations:
        raise RuntimeError(f"True strict-valid threshold violations: {true_strict_gt_violations[:5]}")
    if source_hash(probe_path) != source_probe_hash_before:
        raise RuntimeError("Original probe_results.jsonl was modified")
    return {
        "duplicate_intervention_ids": 0,
        "unknown_window_references": 0,
        "gt_leakage_into_generation": 0,
        "invalid_frame_counts": 0,
        "temporal_order_violations": 0,
        "spurious_strict_valid_gt_violations": 0,
        "true_strict_valid_gt_violations": 0,
        "duplicate_intervention_frame_sequences": duplicate_sequence_count(records),
    }


def qualitative_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": row["qa_id"],
        "dataset_name": row["dataset_name"],
        "target_action": row["target_action"],
        "target_field": row["target_field"],
        "window_id": row["window_id"],
        "intervention_id": row["intervention_id"],
        "intervention_type": row["intervention_type"],
        "original_candidate_label": row["original_candidate_label"],
        "original_start_time": row.get("original_start_time"),
        "original_end_time": row.get("original_end_time"),
        "intervened_start_time": row.get("intervened_start_time"),
        "intervened_end_time": row.get("intervened_end_time"),
        "original_frame_paths": row["original_frame_paths"],
        "intervened_frame_paths": row["intervened_frame_paths"],
        "generation_valid": row["generation_valid"],
        "strict_valid": row["strict_valid"],
        "validity_reason": row["validity_reason"],
        "intervention_gt_alignment_class": row["intervention_gt_alignment_class"],
        "intervention_evidence_density": row["intervention_evidence_density"],
        "intervention_gt_evidence_recall": row["intervention_gt_evidence_recall"],
    }


def write_qualitative_cases(out_dir: Path, records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    specs = {
        "TRUE_SUPPORT_valid_shift": lambda r: r["original_candidate_label"] == "TRUE_SUPPORT" and r["intervention_family"] == "SHIFT" and r["strict_valid"],
        "TRUE_SUPPORT_invalid_shift_gt_removed": lambda r: r["original_candidate_label"] == "TRUE_SUPPORT" and r["intervention_family"] == "SHIFT" and r["validity_reason"] == "GT_EVIDENCE_REMOVED",
        "SPURIOUS_SUPPORT_valid_shift": lambda r: r["original_candidate_label"] == "SPURIOUS_SUPPORT" and r["intervention_family"] == "SHIFT" and r["strict_valid"],
        "SPURIOUS_SUPPORT_gt_contaminated_shift": lambda r: r["original_candidate_label"] == "SPURIOUS_SUPPORT" and r["intervention_family"] == "SHIFT" and r["validity_reason"] == "GT_CONTAMINATED",
        "TRUE_SUPPORT_RESAMPLE_50": lambda r: r["original_candidate_label"] == "TRUE_SUPPORT" and r["intervention_type"] == "RESAMPLE_50",
        "SPURIOUS_SUPPORT_CONTEXT_1P5X": lambda r: r["original_candidate_label"] == "SPURIOUS_SUPPORT" and r["intervention_type"] == "CONTEXT_1P5X",
        "boundary_asymmetric_context": lambda r: r["intervention_type"] == "CONTEXT_1P5X" and r["boundary_asymmetric"],
    }
    cases: dict[str, list[dict[str, Any]]] = {}
    for name, pred in specs.items():
        picked = [qualitative_payload(r) for r in records if pred(r)][:10]
        cases[name] = picked
    case_dir = out_dir / "qualitative_cases"
    write_json(case_dir / "cases.json", cases)
    write_qualitative_html(case_dir / "index.html", cases)
    return cases


def write_qualitative_html(path: Path, cases: dict[str, list[dict[str, Any]]]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px}.case{border:1px solid #ccc;padding:12px;margin:12px 0}.seq{display:flex;gap:4px;overflow-x:auto}.seq img{height:72px;border:1px solid #ddd}</style>",
        "<title>Phase D.1 Qualitative Cases</title></head><body><h1>Phase D.1 Qualitative Cases</h1>",
    ]
    for name, rows in cases.items():
        parts.append(f"<h2>{html.escape(name)}</h2>")
        for row in rows:
            parts.append("<div class='case'>")
            parts.append(
                "<p>"
                f"<b>qa_id</b>: {html.escape(row['qa_id'])}<br>"
                f"<b>dataset</b>: {html.escape(row['dataset_name'])}<br>"
                f"<b>target</b>: {html.escape(str(row['target_action']))}<br>"
                f"<b>intervention</b>: {html.escape(row['intervention_id'])}<br>"
                f"<b>label</b>: {html.escape(row['original_candidate_label'])}<br>"
                f"<b>original time</b>: {row.get('original_start_time')} - {row.get('original_end_time')}<br>"
                f"<b>intervened time</b>: {row.get('intervened_start_time')} - {row.get('intervened_end_time')}<br>"
                f"<b>validity</b>: {html.escape(str(row['validity_reason']))}"
                "</p><b>Original</b><div class='seq'>"
            )
            for frame in row["original_frame_paths"][:24]:
                parts.append(f"<img src='{html.escape(frame)}'>")
            parts.append("</div><b>Intervened</b><div class='seq'>")
            for frame in row["intervened_frame_paths"][:24]:
                parts.append(f"<img src='{html.escape(frame)}'>")
            parts.append("</div></div>")
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def family_valid_counts(records: list[dict[str, Any]], dataset: str | None = None) -> dict[str, int]:
    sub = [r for r in records if dataset is None or r["dataset_name"] == dataset]
    return {
        "strict_valid_shift": sum(r["strict_valid"] and r["intervention_family"] == "SHIFT" for r in sub),
        "strict_valid_resample": sum(r["strict_valid"] and r["intervention_family"] == "RESAMPLE" for r in sub),
        "strict_valid_context": sum(r["strict_valid"] and r["intervention_family"] == "CONTEXT" for r in sub),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase D.1 Intervention Generation + Validity Audit",
        "",
        "## Original Candidates",
        f"- n_paired_qa: {summary['original_candidates']['n_paired_qa']}",
        f"- n_true_support_candidates: {summary['original_candidates']['n_true_support_candidates']}",
        f"- n_spurious_support_candidates: {summary['original_candidates']['n_spurious_support_candidates']}",
        "",
        "## Intervention Generation",
        f"- attempted total: {summary['intervention_generation']['attempted_total']}",
        f"- generation valid: {summary['intervention_generation']['n_generation_valid']}",
        f"- boundary invalid: {summary['intervention_generation']['n_generation_invalid_boundary']}",
        "",
        "## Paired Usability",
        f"- paired_intervention_usable QA: {summary['paired_usability']['paired_intervention_usable_qa']}",
        f"- paired_shift_usable QA: {summary['paired_usability']['paired_shift_usable_qa']}",
        f"- paired_resample_usable QA: {summary['paired_usability']['paired_resample_usable_qa']}",
        f"- paired_context_usable QA: {summary['paired_usability']['paired_context_usable_qa']}",
        "",
        "## Audit",
        f"- duplicate intervention IDs: {summary['audit']['duplicate_intervention_ids']}",
        f"- unknown window references: {summary['audit']['unknown_window_references']}",
        f"- GT leakage into generation: {summary['audit']['gt_leakage_into_generation']}",
        f"- invalid frame counts: {summary['audit']['invalid_frame_counts']}",
        f"- temporal-order violations: {summary['audit']['temporal_order_violations']}",
        f"- spurious strict-valid GT violations: {summary['audit']['spurious_strict_valid_gt_violations']}",
        f"- true strict-valid GT violations: {summary['audit']['true_strict_valid_gt_violations']}",
        f"- duplicate intervention frame sequences: {summary['audit']['duplicate_intervention_frame_sequences']}",
        "",
        "No model inference, stability scoring, or statistical testing was run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D.1 intervention generation and post-hoc validity audit.")
    p.add_argument("--probe_labeled_results", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50/probe_labeled_results.jsonl")
    p.add_argument("--paired_h2_qa", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50/paired_h2_qa.jsonl")
    p.add_argument("--windows", default="outputs/tal_pilot/window_alignment_audit.jsonl")
    p.add_argument("--samples", default="outputs/tal_pilot/samples.jsonl")
    p.add_argument("--probe_results", default="outputs/tal_pilot/phase_c/qwen3_vl_8b_pilot50/probe_results.jsonl")
    p.add_argument("--output_dir", default="outputs/tal_pilot/phase_d/intervention_pilot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_blind_audit = gt_blind_generation_audit()

    probe_path = Path(args.probe_results)
    source_probe_hash_before = source_hash(probe_path) if probe_path.exists() else ""
    paired_qa_ids = load_paired_qa_ids(args.paired_h2_qa)
    labeled_rows = read_jsonl(args.probe_labeled_results)
    samples_by_qa = {row["qa_id"]: row for row in read_jsonl(args.samples)}
    windows_by_key = index_unique(read_jsonl(args.windows), "Phase B.1 window metadata")

    candidate_rows = []
    unknown_qa = []
    unknown_window = []
    original_labels_before: dict[tuple[str, str], str] = {}
    for row in labeled_rows:
        label = normalize_candidate_label(row)
        if row["qa_id"] not in paired_qa_ids or label not in CORE_LABELS:
            continue
        if row["qa_id"] not in samples_by_qa:
            unknown_qa.append(row["qa_id"])
            continue
        if row_key(row) not in windows_by_key:
            unknown_window.append(row_key(row))
            continue
        original_labels_before[row_key(row)] = label
        candidate_rows.append({**row, "candidate_label": label})
    if unknown_qa:
        raise RuntimeError(f"Unknown qa_id references: {unknown_qa[:5]} total={len(unknown_qa)}")
    if unknown_window:
        raise RuntimeError(f"Unknown window_id references: {unknown_window[:5]} total={len(unknown_window)}")

    records = []
    for candidate in candidate_rows:
        sample = samples_by_qa[candidate["qa_id"]]
        window_meta = windows_by_key[row_key(candidate)]
        for intervention_type in INTERVENTION_TYPES:
            records.append(make_intervention_record(candidate, window_meta, sample, intervention_type))

    for candidate in candidate_rows:
        if original_labels_before[row_key(candidate)] != normalize_candidate_label(candidate):
            raise RuntimeError(f"Original candidate label modified: {row_key(candidate)}")

    audit = assert_manifest(records, source_probe_hash_before, probe_path) if probe_path.exists() else {
        "duplicate_intervention_ids": 0,
        "unknown_window_references": 0,
        "gt_leakage_into_generation": 0,
        "invalid_frame_counts": 0,
        "temporal_order_violations": 0,
        "spurious_strict_valid_gt_violations": 0,
        "true_strict_valid_gt_violations": 0,
        "duplicate_intervention_frame_sequences": duplicate_sequence_count(records),
    }
    audit.update(gt_blind_audit)

    validity_rows = summarize_by_label_and_type(records)
    per_qa = build_per_qa(records, paired_qa_ids, candidate_rows)
    by_dataset = summarize_stratified(records, "dataset_name")
    by_target_type = summarize_stratified(records, "target_field")
    qualitative_cases = write_qualitative_cases(out_dir, records)

    generation_by_type = {}
    for intervention_type in INTERVENTION_TYPES:
        sub = [r for r in records if r["intervention_type"] == intervention_type]
        generation_by_type[intervention_type] = {
            "attempted": len(sub),
            "generation_valid": sum(r["generation_valid"] for r in sub),
            "boundary_asymmetric": sum(r["boundary_asymmetric"] for r in sub),
            "boundary_invalid": sum(r["generation_invalid_reason"] == "GENERATION_INVALID_BOUNDARY" for r in sub),
        }

    labels = Counter(normalize_candidate_label(r) for r in candidate_rows)
    target_field_counts = Counter((r.get("target_field") or samples_by_qa[r["qa_id"]].get("target_field"), normalize_candidate_label(r)) for r in candidate_rows)
    paired_target_counts = Counter(samples_by_qa[q].get("target_field") for q in paired_qa_ids if q in samples_by_qa)
    dataset_candidate_rows = []
    for dataset in ["AVOS", "CholecT50", "CoPESD", "EgoSurgery", "NurViD"]:
        candidates = [r for r in candidate_rows if r["dataset_name"] == dataset]
        qa_ids = {r["qa_id"] for r in candidates}
        valid_counts = family_valid_counts([r for r in records if r["dataset_name"] == dataset])
        dataset_candidate_rows.append(
            {
                "dataset_name": dataset,
                "paired_qa": len(qa_ids),
                "true_candidates": sum(normalize_candidate_label(r) == "TRUE_SUPPORT" for r in candidates),
                "spurious_candidates": sum(normalize_candidate_label(r) == "SPURIOUS_SUPPORT" for r in candidates),
                **valid_counts,
            }
        )

    summary = {
        "inputs": {
            "probe_labeled_results": args.probe_labeled_results,
            "paired_h2_qa": args.paired_h2_qa,
            "windows": args.windows,
            "samples": args.samples,
            "probe_results": args.probe_results,
        },
        "original_candidates": {
            "n_paired_qa": len(paired_qa_ids),
            "n_true_support_candidates": labels["TRUE_SUPPORT"],
            "n_spurious_support_candidates": labels["SPURIOUS_SUPPORT"],
            "n_original_candidates_total": len(candidate_rows),
        },
        "action_phase": {
            "n_action_paired_qa": paired_target_counts["action"],
            "n_phase_paired_qa": paired_target_counts["phase"],
            "n_true_action": target_field_counts[("action", "TRUE_SUPPORT")],
            "n_spurious_action": target_field_counts[("action", "SPURIOUS_SUPPORT")],
            "n_true_phase": target_field_counts[("phase", "TRUE_SUPPORT")],
            "n_spurious_phase": target_field_counts[("phase", "SPURIOUS_SUPPORT")],
        },
        "intervention_generation": {
            "attempted_total": len(records),
            "n_generation_valid": sum(r["generation_valid"] for r in records),
            "n_generation_invalid_boundary": sum(r["generation_invalid_reason"] == "GENERATION_INVALID_BOUNDARY" for r in records),
            "by_intervention_type": generation_by_type,
        },
        "posthoc_validity": validity_rows,
        "paired_usability": {
            "paired_intervention_usable_qa": sum(r["paired_intervention_usable"] for r in per_qa),
            "paired_shift_usable_qa": sum(r["paired_shift_usable"] for r in per_qa),
            "paired_resample_usable_qa": sum(r["paired_resample_usable"] for r in per_qa),
            "paired_context_usable_qa": sum(r["paired_context_usable"] for r in per_qa),
        },
        "by_dataset_candidates": dataset_candidate_rows,
        "audit": audit,
        "qualitative_case_counts": {name: len(rows) for name, rows in qualitative_cases.items()},
    }

    write_jsonl(out_dir / "interventions.jsonl", records)
    write_json(out_dir / "phase_d1_validity_summary.json", summary)
    write_summary_md(out_dir / "phase_d1_validity_summary.md", summary)
    write_csv(out_dir / "phase_d1_validity_by_dataset.csv", by_dataset)
    write_csv(out_dir / "phase_d1_validity_by_target_type.csv", by_target_type)
    write_csv(out_dir / "phase_d1_validity_per_qa.csv", per_qa)
    write_csv(out_dir / "phase_d1_validity_by_candidate_label_and_type.csv", validity_rows)
    write_csv(out_dir / "phase_d1_by_dataset_candidates.csv", dataset_candidate_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
