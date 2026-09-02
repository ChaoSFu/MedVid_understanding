#!/usr/bin/env python3
"""Evaluate Qwen3.8-Max trainval predictions with MedVidBench leaderboard code.

The inference run may contain only successful samples, while the sampled
trainval ground-truth file contains all 300 samples. This wrapper therefore
merges predictions with ground truth by original index first, then by
(id, qa_type), before calling the leaderboard evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "qwen38_max_trainval_50_per_qa_type_seed42"
DEFAULT_GT_FILE = (
    REPO_ROOT
    / "data_json"
    / "medvidu_filtered"
    / "medvidu_trainval_50_per_qa_type_seed42.json"
)
DEFAULT_EVALUATOR = (
    REPO_ROOT
    / "MedVidBench-Leaderboard"
    / "evaluation"
    / "evaluate_predictions.py"
)

TASK_CHOICES = [
    "dvc",
    "tal",
    "stg",
    "rc",
    "vs",
    "next_action",
    "skill_assessment",
    "cvs_assessment",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sample_key(sample: dict[str, Any]) -> tuple[str, str]:
    return str(sample.get("id", "")), str(sample.get("qa_type", ""))


def normalize_records(data: Any) -> list[tuple[str | None, dict[str, Any]]]:
    if isinstance(data, dict):
        return [(str(k), v) for k, v in data.items() if isinstance(v, dict)]
    if isinstance(data, list):
        return [(str(i), v) for i, v in enumerate(data) if isinstance(v, dict)]
    raise TypeError(f"Expected predictions to be a JSON object or array, got {type(data).__name__}")


def extract_question_and_gnd(gt_sample: dict[str, Any]) -> tuple[str, str]:
    question = str(gt_sample.get("question", "") or "")
    gnd = str(gt_sample.get("gnd", "") or "")

    for msg in gt_sample.get("conversations", []) or []:
        sender = msg.get("from")
        value = str(msg.get("value", "") or "")
        if sender in {"human", "user"} and not question:
            question = value.replace("<video>\n", "").replace("<video>", "")
        elif sender in {"gpt", "assistant"} and not gnd:
            gnd = value

    return question, gnd


def prediction_text(record: dict[str, Any]) -> str:
    if "answer" in record:
        return str(record.get("answer", "") or "")
    return str(record.get("prediction", "") or "")


def maybe_int(value: Any) -> int | None:
    try:
        text = str(value)
        if re.fullmatch(r"\d+", text):
            return int(text)
    except Exception:
        return None
    return None


def choose_prediction_file(run_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    results_file = run_dir / "results.json"
    if results_file.exists():
        return results_file

    submission_file = run_dir / "submission.json"
    if submission_file.exists():
        return submission_file

    raise FileNotFoundError(
        f"Could not find results.json or submission.json under {run_dir}"
    )


def merge_predictions_with_trainval_gt(
    predictions: Any,
    ground_truth: list[dict[str, Any]],
    strict: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pred_records = normalize_records(predictions)

    gt_by_key: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for idx, sample in enumerate(ground_truth):
        gt_by_key[sample_key(sample)].append(idx)

    merged: dict[str, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    matched_indices: list[int] = []

    for source_key, pred in pred_records:
        pred_key = sample_key(pred)
        source_idx = maybe_int(source_key)
        gt_idx = None

        if source_idx is not None and 0 <= source_idx < len(ground_truth):
            candidate = ground_truth[source_idx]
            if sample_key(candidate) == pred_key:
                gt_idx = source_idx

        if gt_idx is None and gt_by_key[pred_key]:
            gt_idx = gt_by_key[pred_key].popleft()

        if gt_idx is None:
            unmatched.append(
                {
                    "source_key": source_key,
                    "id": pred.get("id"),
                    "qa_type": pred.get("qa_type"),
                }
            )
            continue

        gt_sample = ground_truth[gt_idx]
        question, gnd = extract_question_and_gnd(gt_sample)

        merged[str(len(merged))] = {
            "id": gt_sample.get("id", pred.get("id")),
            "metadata": gt_sample.get("metadata", {}),
            "qa_type": gt_sample.get("qa_type", pred.get("qa_type", "")),
            "struc_info": gt_sample.get("struc_info", []),
            "question": question,
            "gnd": gnd,
            "answer": prediction_text(pred),
            "data_source": gt_sample.get(
                "data_source",
                gt_sample.get("dataset_name", pred.get("data_source", "")),
            ),
            "original_idx": gt_idx,
        }
        matched_indices.append(gt_idx)

    if strict and unmatched:
        raise ValueError(f"Unmatched predictions: {unmatched[:10]}")

    gt_counts = Counter(str(s.get("qa_type", "")) for s in ground_truth)
    merged_counts = Counter(r["qa_type"] for r in merged.values())
    missing_gt_indices = sorted(set(range(len(ground_truth))) - set(matched_indices))

    report = {
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(pred_records),
        "merged_count": len(merged),
        "unmatched_count": len(unmatched),
        "unmatched_examples": unmatched[:20],
        "missing_ground_truth_count": len(missing_gt_indices),
        "missing_ground_truth_indices": missing_gt_indices,
        "ground_truth_qa_type_counts": dict(sorted(gt_counts.items())),
        "merged_qa_type_counts": dict(sorted(merged_counts.items())),
    }
    return merged, report


def stream_command(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)

    return proc.wait(), "".join(lines)


def parse_float_from_line(line: str) -> float | None:
    match = re.search(r":\s*([-+]?\d+(?:\.\d+)?)", line)
    if not match:
        return None
    return float(match.group(1))


def parse_metrics(output: str, skip_llm_judge: bool = False) -> dict[str, float]:
    metrics: dict[str, float] = {}
    current_task = ""

    for raw_line in output.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        if "cvs assessment" in lower:
            current_task = "cvs_assessment"
        elif "next action" in lower:
            current_task = "next_action"
        elif "skill assessment" in lower:
            current_task = "skill_assessment"
        elif "stg" in lower and "overall evaluation" in lower:
            current_task = "stg"
        elif "tal" in lower and "overall evaluation" in lower:
            current_task = "tal"
        elif "dense video captioning" in lower or "dvc" in lower:
            current_task = "dvc"
        elif "rc" in lower and "overall evaluation" in lower:
            current_task = "rc"
        elif "vs" in lower and "overall evaluation" in lower:
            current_task = "vs"

        value = parse_float_from_line(line)
        if value is None:
            continue

        if "component_balanced_accuracy" in lower:
            metrics["cvs_acc"] = value
        elif "aspect_balanced_accuracy" in lower:
            metrics["sa_acc"] = value
        elif current_task == "next_action" and "accuracy" in lower:
            metrics["nap_acc"] = value
        elif current_task == "stg" and "mean_iou" in lower:
            metrics["stg_miou"] = value
        elif "miou@0.3" in lower:
            metrics["tag_miou_03"] = value
        elif "miou@0.5" in lower:
            metrics["tag_miou_05"] = value
        elif "temporal_f1" in lower:
            metrics["dvc_f1"] = value
        elif "caption_score" in lower and not skip_llm_judge:
            metrics["dvc_llm"] = value
        elif current_task == "rc" and lower.startswith("score:"):
            metrics["rc_llm"] = value
        elif current_task == "vs" and lower.startswith("score:"):
            metrics["vs_llm"] = value

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3.8-Max trainval predictions with leaderboard code."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT_FILE)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/leaderboard_eval",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_CHOICES,
        default=None,
        help="Default: auto-detect from merged predictions.",
    )
    parser.add_argument(
        "--grouping",
        choices=["overall", "per-dataset"],
        default="overall",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip RC/VS LLM judge and DVC caption judge; DVC temporal F1 still runs.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze task/dataset structure after merging.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any prediction cannot be matched to trainval ground truth.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    predictions_file = choose_prediction_file(run_dir, args.predictions).resolve()
    ground_truth_file = args.ground_truth.resolve()
    evaluator = args.evaluator.resolve()
    output_dir = (args.output_dir or (run_dir / "leaderboard_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_json(predictions_file)
    ground_truth = load_json(ground_truth_file)
    if not isinstance(ground_truth, list):
        raise TypeError("Ground-truth trainval file must be a JSON array.")

    merged, merge_report = merge_predictions_with_trainval_gt(
        predictions,
        ground_truth,
        strict=args.strict,
    )

    merged_file = output_dir / "merged_for_leaderboard.json"
    merge_report_file = output_dir / "merge_report.json"
    stdout_file = output_dir / "eval_stdout.txt"
    summary_file = output_dir / "eval_summary.json"

    dump_json(merged_file, merged)
    dump_json(merge_report_file, merge_report)

    cmd = [
        sys.executable,
        str(evaluator),
        str(merged_file),
        "--grouping",
        args.grouping,
    ]
    if args.tasks:
        cmd.extend(["--tasks", *args.tasks])
    if args.skip_llm_judge:
        cmd.append("--skip-llm-judge")
    if args.analyze_only:
        cmd.append("--analyze-only")

    print(f"Predictions: {predictions_file}")
    print(f"Ground truth: {ground_truth_file}")
    print(f"Merged file: {merged_file}")
    print(f"Matched {merge_report['merged_count']}/{merge_report['ground_truth_count']} GT samples")
    print(f"Command: {shlex.join(cmd)}")
    print()

    returncode, output = stream_command(cmd, cwd=evaluator.parent)
    stdout_file.write_text(output, encoding="utf-8")

    summary = {
        "command": cmd,
        "returncode": returncode,
        "predictions_file": str(predictions_file),
        "ground_truth_file": str(ground_truth_file),
        "merged_file": str(merged_file),
        "merge_report_file": str(merge_report_file),
        "stdout_file": str(stdout_file),
        "merge_report": merge_report,
        "metrics": parse_metrics(output, skip_llm_judge=args.skip_llm_judge),
    }
    dump_json(summary_file, summary)

    print()
    print(f"Saved stdout: {stdout_file}")
    print(f"Saved summary: {summary_file}")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
