from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import LEADERBOARD_TASK_BY_ADAPTER, REGION_CAPTION_QA_TYPES, REPO_ROOT, RunConfig, TASK_NAME_BY_QA_TYPE
from ..io_utils import read_json, read_jsonl, write_json, write_jsonl


def prediction_rows_for_evaluator(predictions: list[dict[str, Any]], include_rc: bool = False) -> list[dict[str, Any]]:
    rows = []
    for pred in predictions:
        qa_type = str(pred.get("qa_type", ""))
        if qa_type in REGION_CAPTION_QA_TYPES and not include_rc:
            continue
        rows.append(
            {
                "sample_id": pred.get("sample_id"),
                "original_idx": pred.get("original_index"),
                "id": pred.get("id"),
                "qa_type": qa_type,
                "prediction": pred.get("prediction", ""),
                "data_source": pred.get("data_source"),
            }
        )
    return rows


def write_submission_json(prediction_jsonl: Path, output_json: Path, include_rc: bool = False) -> list[dict[str, Any]]:
    rows = prediction_rows_for_evaluator(read_jsonl(prediction_jsonl), include_rc=include_rc)
    write_json(output_json, rows)
    return rows


def tasks_from_predictions(predictions: list[dict[str, Any]], include_rc: bool = False) -> list[str]:
    tasks = []
    for qa_type in sorted({str(row.get("qa_type", "")) for row in predictions}):
        if qa_type in REGION_CAPTION_QA_TYPES and not include_rc:
            continue
        adapter_task = TASK_NAME_BY_QA_TYPE[qa_type]
        task = LEADERBOARD_TASK_BY_ADAPTER[adapter_task]
        if task not in tasks:
            tasks.append(task)
    return tasks


def run_official_evaluation(
    prediction_jsonl: Path,
    ground_truth_path: Path,
    output_dir: Path,
    evaluator: Path,
    include_rc: bool = False,
    skip_llm_judge: bool = True,
    analyze_only: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_json = output_dir / "qwen35_videoitg_submission.json"
    rows = write_submission_json(prediction_jsonl, submission_json, include_rc=include_rc)
    write_jsonl(output_dir / "per_sample_metrics.jsonl", [])

    tasks = tasks_from_predictions(rows, include_rc=include_rc)
    wrapper = REPO_ROOT / "scripts" / "evaluate_qwen38_max_trainval.py"
    wrapper_output = output_dir / "leaderboard_eval"
    cmd = [
        sys.executable,
        str(wrapper),
        "--predictions",
        str(submission_json),
        "--ground-truth",
        str(ground_truth_path),
        "--evaluator",
        str(evaluator),
        "--output-dir",
        str(wrapper_output),
        "--tasks",
        *tasks,
    ]
    if skip_llm_judge:
        cmd.append("--skip-llm-judge")
    if analyze_only:
        cmd.append("--analyze-only")

    env = None
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    stdout = proc.stdout
    (output_dir / "eval_stdout.txt").write_text(stdout, encoding="utf-8")
    wrapper_summary_path = wrapper_output / "eval_summary.json"
    wrapper_summary = read_json(wrapper_summary_path) if wrapper_summary_path.exists() else {}
    metrics = wrapper_summary.get("metrics") or parse_metrics(stdout, skip_llm_judge=skip_llm_judge)
    per_task = {
        "tasks": tasks,
        "metrics": metrics,
        "returncode": proc.returncode,
        "evaluator": str(evaluator),
        "wrapper": str(wrapper),
        "wrapper_summary": str(wrapper_summary_path),
        "command": shlex.join(cmd),
    }
    gt = read_json(ground_truth_path)
    summary = {
        "prediction_count": len(rows),
        "ground_truth_count": len(gt) if isinstance(gt, list) else None,
        "prediction_qa_type_counts": dict(sorted(Counter(row["qa_type"] for row in rows).items())),
        "include_rc": include_rc,
        "skip_llm_judge": skip_llm_judge,
        "merge_report": wrapper_summary.get("merge_report"),
        **per_task,
    }
    write_json(output_dir / "per_task_metrics.json", per_task)
    write_json(output_dir / "overall_summary.json", summary)
    (output_dir / "overall_summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    return summary


def parse_float_from_line(line: str) -> float | None:
    import re

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
            metrics["tal_miou_03"] = value
        elif "miou@0.5" in lower:
            metrics["tal_miou_05"] = value
        elif "temporal_f1" in lower:
            metrics["dvc_f1"] = value
        elif "caption_score" in lower and not skip_llm_judge:
            metrics["dvc_llm"] = value
        elif current_task == "rc" and lower.startswith("score:"):
            metrics["rc_llm"] = value
        elif current_task == "vs" and lower.startswith("score:"):
            metrics["vs_llm"] = value
    return metrics


def render_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# VideoITG-32 + Qwen3.5-4B Evaluation Summary",
        "",
        f"- predictions: {summary['prediction_count']}",
        f"- include_rc: {summary['include_rc']}",
        f"- skip_llm_judge: {summary['skip_llm_judge']}",
        f"- returncode: {summary['returncode']}",
        f"- tasks: {', '.join(summary['tasks'])}",
        "",
        "## Metrics",
    ]
    for key, value in sorted(summary.get("metrics", {}).items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    p = argparse.ArgumentParser(description="Evaluate VideoITG-32 + Qwen3.5 predictions with official MedVidBench evaluator.")
    p.add_argument("--predictions", type=Path, default=cfg.prediction_dir / "qwen35_videoitg_predictions.jsonl")
    p.add_argument("--ground-truth", type=Path, default=cfg.data_path)
    p.add_argument("--output-root", type=Path, default=cfg.output_root)
    p.add_argument("--evaluator", type=Path, default=cfg.evaluator)
    p.add_argument("--include-rc", action="store_true")
    p.add_argument("--run-llm-judge", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(output_root=args.output_root, data_path=args.ground_truth, evaluator=args.evaluator)
    cfg.make_dirs()
    summary = run_official_evaluation(
        prediction_jsonl=args.predictions,
        ground_truth_path=args.ground_truth,
        output_dir=cfg.evaluation_dir,
        evaluator=args.evaluator,
        include_rc=args.include_rc,
        skip_llm_judge=not args.run_llm_judge,
        analyze_only=args.analyze_only,
    )
    print(summary)
    return 0 if summary["returncode"] == 0 else int(summary["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
