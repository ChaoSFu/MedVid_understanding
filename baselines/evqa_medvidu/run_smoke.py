from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from .backend import EVQABackend
from .config import DEFAULT_OUTPUT_ROOT, OFFICIAL_MAX_PIXELS_PER_FRAME, GenerationConfig, RunConfig
from .frame_adapter import select_model_frames
from .io_utils import read_json, read_jsonl, sha256_json, write_json
from .provenance import model_fingerprint


def cap_model_frames_for_dry_run(row: dict[str, Any], max_model_frames: int | None) -> dict[str, Any]:
    if max_model_frames is None:
        return row
    if max_model_frames <= 0:
        raise ValueError("--max-model-frames must be positive")
    if row["model_sampling"]["n_selected"] <= max_model_frames:
        return row
    capped = deepcopy(row)
    sampling = select_model_frames(
        capped["ordered_frame_paths"],
        capped["local_timestamps"],
        capped["clip_local_duration"],
        fps=1.0,
        max_frames=max_model_frames,
    ).to_dict()
    sampling["policy"] = f"{sampling['policy']}__dry_run_global_frame_cap_{max_model_frames}_not_formal"
    sampling["formal_result_allowed"] = False
    sampling["original_n_selected"] = row["model_sampling"]["n_selected"]
    capped["model_sampling"] = sampling
    capped["selected_frame_hash"] = sha256_json(sampling)
    capped["dry_run_frame_cap"] = max_model_frames
    capped["formal_result_allowed"] = False
    return capped


def run_task_smoke(cfg: RunConfig, backend: EVQABackend, task: str, max_model_frames: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(cfg.manifest_dir / f"{task}_gt_free.smoke.jsonl")
    report = {
        "task": task,
        "requested": len(rows),
        "new": 0,
        "cached": 0,
        "errors": [],
        "max_model_frames": max_model_frames,
        "visual_input_config": backend.visual_input_config,
        "inference_config": backend.inference_config(task),
        "formal_result_allowed": max_model_frames is None,
    }
    if task == "stg":
        report.update(
            {
                "target_boxes_requested": 0,
                "target_boxes_emitted": 0,
                "target_boxes_missing": 0,
            }
        )
    for row in rows:
        row = cap_model_frames_for_dry_run(row, max_model_frames)
        try:
            status, _, medvidu = backend.run_row(row)
            report[status["status"]] += 1
            if task == "stg" and medvidu is not None:
                target_alignment = medvidu.get("stg_target_alignment") or []
                missing = medvidu.get("missing_target_timestamps") or []
                report["target_boxes_requested"] += len(target_alignment)
                report["target_boxes_missing"] += len(missing)
                report["target_boxes_emitted"] += len(target_alignment) - len(missing)
        except Exception as exc:
            report["errors"].append({"sample_id": row.get("sample_id"), "error": repr(exc)})
            break
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E-VQA MedVidU smoke inference after preflight review.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--task", choices=["stg", "rc", "cvs", "all"], default="all")
    parser.add_argument("--allow-without-official-reproduction", action="store_true")
    parser.add_argument(
        "--max-model-frames",
        type=int,
        default=None,
        help="Dry-run only: globally cap sampled model frames to diagnose context/OOM issues. Results are not formal unless this policy is separately frozen.",
    )
    parser.add_argument(
        "--max-pixels-per-frame",
        type=int,
        default=OFFICIAL_MAX_PIXELS_PER_FRAME,
        help=(
            "Maximum pixels for each selected video frame. The E-VQA vision utility applies this value "
            "to every image in a video list; keeping it fixed prevents long clips from exceeding context."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(output_root=args.output_root)
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")
    official_report = cfg.official_reproduction_dir / "report.json"
    if not args.allow_without_official_reproduction:
        if not official_report.exists():
            raise RuntimeError("STOP: official reproduction report is missing. Run run_official_reproduction.py on 2-5 official ST-Evidence samples first.")
        report = read_json(official_report)
        if not report.get("success"):
            raise RuntimeError("STOP: official reproduction has not passed. Do not run MedVidU smoke until upstream E-VQA inference works.")
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
    tasks = ["stg", "rc", "cvs"] if args.task == "all" else [args.task]
    first = {task: run_task_smoke(cfg, backend, task, max_model_frames=args.max_model_frames) for task in tasks}
    second = {task: run_task_smoke(cfg, backend, task, max_model_frames=args.max_model_frames) for task in tasks}
    cache_report = {
        "first_run": first,
        "second_run": second,
        "formal_result_allowed": args.max_model_frames is None,
        "dry_run_frame_cap": args.max_model_frames,
        "visual_input_config": backend.visual_input_config,
        "inference_config": {task: backend.inference_config(task) for task in tasks},
        "cache_restart_pass": all(rep["new"] == 0 and rep["cached"] == rep["requested"] for rep in second.values()),
    }
    write_json(cfg.audit_dir / "cache_restart.json", cache_report)
    print(cache_report)
    return 0 if cache_report["cache_restart_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
