from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .backend import EVQABackend
from .config import DEFAULT_OUTPUT_ROOT, OFFICIAL_MAX_PIXELS_PER_FRAME, GenerationConfig, RunConfig
from .io_utils import read_json, read_jsonl, write_json
from .provenance import model_fingerprint


TASK_ORDER = ("stg", "rc", "cvs")
Evaluator = Callable[[Path, Path, Path], dict[str, Any]]


def _progress_payload(
    task_results: dict[str, dict[str, Any]], task_order: tuple[str, ...], current_task: str | None, status: str
) -> dict[str, Any]:
    return {
        "version": "evqa_medvidu_full_runner_v1",
        "task_order": list(task_order),
        "current_task": current_task,
        "status": status,
        "tasks": task_results,
        "gt_available_to_inference": False,
    }


def _write_progress(
    cfg: RunConfig,
    task_results: dict[str, dict[str, Any]],
    task_order: tuple[str, ...],
    current_task: str | None,
    status: str,
) -> None:
    write_json(
        cfg.audit_dir / "full_run_progress.json",
        _progress_payload(task_results, task_order, current_task, status),
    )


def run_task_full(
    cfg: RunConfig,
    backend: Any,
    task: str,
    task_results: dict[str, dict[str, Any]],
    task_order: tuple[str, ...] = TASK_ORDER,
) -> dict[str, Any]:
    """Run one GT-free manifest, continuing after per-sample failures."""
    rows = read_jsonl(cfg.manifest_dir / f"{task}_gt_free.jsonl")
    result: dict[str, Any] = {
        "task": task,
        "requested": len(rows),
        "new": 0,
        "cached": 0,
        "errors": 0,
        "error_preview": [],
        "prediction_path": str(cfg.medvidu_prediction_dir / f"{task}.jsonl"),
        "raw_prediction_path": str(cfg.raw_prediction_dir / f"{task}.jsonl"),
        "error_path": str(cfg.error_dir / f"{task}.jsonl"),
        "status": "RUNNING",
    }
    task_results[task] = result
    _write_progress(cfg, task_results, task_order, task, "RUNNING")

    for position, row in enumerate(rows, start=1):
        try:
            run_status, _, _ = backend.run_row(row)
            result[run_status["status"]] += 1
        except Exception as exc:
            result["errors"] += 1
            if len(result["error_preview"]) < 20:
                result["error_preview"].append({"sample_id": row.get("sample_id"), "error": repr(exc)})
        result["processed"] = position
        result["successful"] = result["new"] + result["cached"]
        result["remaining"] = result["requested"] - position
        _write_progress(cfg, task_results, task_order, task, "RUNNING")

    result["status"] = "COMPLETE" if result["errors"] == 0 else "COMPLETE_WITH_SAMPLE_ERRORS"
    _write_progress(cfg, task_results, task_order, task, "RUNNING")
    return result


def evaluator_for_task(task: str) -> Evaluator:
    if task == "stg":
        from .evaluation.stg import evaluate_predictions
    elif task == "rc":
        from .evaluation.rc import evaluate_predictions
    elif task == "cvs":
        from .evaluation.cvs import evaluate_predictions
    else:
        raise KeyError(task)
    return evaluate_predictions


def evaluate_task(cfg: RunConfig, task: str, evaluator: Evaluator | None = None) -> dict[str, Any]:
    """Run evaluation only after inference; this path may read GT, never model input."""
    prediction_path = cfg.medvidu_prediction_dir / f"{task}.jsonl"
    output_path = cfg.evaluation_dir / f"{task}_official.json"
    result = (evaluator or evaluator_for_task(task))(prediction_path, cfg.data_path, output_path)
    # RC evaluation includes judge-ready GT records in its dedicated artifact.
    # Keep progress GT-free by retaining only a compact post-inference summary.
    summary = {"result_keys": sorted(result)}
    if "join_audit" in result:
        summary["join_audit"] = result["join_audit"]
    if "status" in result:
        summary["official_status"] = result["status"]
    return {"status": "COMPLETE", "output_path": str(output_path), "summary": summary}


def run_full_tasks(
    cfg: RunConfig,
    backend: Any,
    task_order: tuple[str, ...] = TASK_ORDER,
    evaluators: dict[str, Evaluator] | None = None,
) -> dict[str, Any]:
    """Run ordered inference then task-local post-inference evaluation."""
    cfg.make_dirs()
    task_results: dict[str, dict[str, Any]] = {}
    _write_progress(cfg, task_results, task_order, None, "RUNNING")
    for task in task_order:
        result = run_task_full(cfg, backend, task, task_results, task_order)
        try:
            result["evaluation"] = evaluate_task(cfg, task, (evaluators or {}).get(task))
        except Exception as exc:
            result["evaluation"] = {"status": "ERROR", "error": repr(exc)}
        _write_progress(cfg, task_results, task_order, task, "RUNNING")

    if any(result["errors"] > 0 for result in task_results.values()):
        final_status = "COMPLETE_WITH_SAMPLE_ERRORS"
    elif any(result.get("evaluation", {}).get("status") != "COMPLETE" for result in task_results.values()):
        final_status = "COMPLETE_WITH_EVALUATION_ERRORS"
    else:
        final_status = "COMPLETE"
    _write_progress(cfg, task_results, task_order, None, final_status)
    report = _progress_payload(task_results, task_order, None, final_status)
    write_json(cfg.audit_dir / "full_run_report.json", report)
    return report


def assert_full_run_preconditions(cfg: RunConfig, allow_without_official_reproduction: bool) -> None:
    preflight_path = cfg.output_root / "preflight_report.json"
    if not preflight_path.exists():
        raise RuntimeError("STOP: preflight_report.json is missing. Run run_preflight.py for this output root first.")
    report = read_json(preflight_path)
    if report.get("STOP_REASONS"):
        raise RuntimeError(f"STOP: unresolved preflight conditions: {report['STOP_REASONS']}")
    official_path = cfg.official_reproduction_dir / "report.json"
    official = read_json(official_path) if official_path.exists() else {}
    if not official.get("success") and not allow_without_official_reproduction:
        raise RuntimeError("STOP: official reproduction is unavailable. Pass --allow-without-official-reproduction only after documenting this limitation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable E-VQA MedVidU full inference in STG -> RC -> CVS order.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-pixels-per-frame", type=int, default=OFFICIAL_MAX_PIXELS_PER_FRAME)
    parser.add_argument("--task", choices=["all", *TASK_ORDER], default="all", help="Use a single task only to resume or diagnose that task.")
    parser.add_argument("--allow-without-official-reproduction", action="store_true")
    parser.add_argument("--i-reviewed-smoke-and-freeze-adapters", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_reviewed_smoke_and_freeze_adapters:
        print("STOP: pass --i-reviewed-smoke-and-freeze-adapters only after reviewing all v4 smoke and cache-restart artifacts.")
        return 2
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")
    cfg = RunConfig(output_root=args.output_root)
    assert_full_run_preconditions(cfg, args.allow_without_official_reproduction)
    fingerprint_path = cfg.provenance_dir / "model_fingerprint.json"
    fingerprint = read_json(fingerprint_path) if fingerprint_path.exists() else model_fingerprint(args.model_path)
    backend = EVQABackend(
        args.model_path,
        cfg.output_root,
        fingerprint,
        device=args.device,
        dtype=args.dtype,
        generation=GenerationConfig(),
        max_pixels_per_frame=args.max_pixels_per_frame,
    )
    tasks = TASK_ORDER if args.task == "all" else (args.task,)
    write_json(
        cfg.provenance_dir / "full_run_config.json",
        {
            "run_config": cfg.to_jsonable(),
            "generation": asdict(GenerationConfig()),
            "task_order": list(tasks),
            "allow_without_official_reproduction": args.allow_without_official_reproduction,
            "gt_available_to_inference": False,
        },
    )
    report = run_full_tasks(cfg, backend, tasks)
    print(report)
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
