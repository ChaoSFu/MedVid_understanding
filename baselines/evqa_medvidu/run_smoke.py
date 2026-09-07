from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .backend import EVQABackend
from .config import DEFAULT_OUTPUT_ROOT, GenerationConfig, RunConfig
from .io_utils import read_json, read_jsonl, write_json
from .provenance import model_fingerprint


def run_task_smoke(cfg: RunConfig, backend: EVQABackend, task: str) -> dict[str, Any]:
    rows = read_jsonl(cfg.manifest_dir / f"{task}_gt_free.smoke.jsonl")
    report = {"task": task, "requested": len(rows), "new": 0, "cached": 0, "errors": []}
    for row in rows:
        try:
            status, _, _ = backend.run_row(row)
            report[status["status"]] += 1
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
    backend = EVQABackend(args.model_path, cfg.output_root, fingerprint, device=args.device, dtype=args.dtype, generation=GenerationConfig())
    tasks = ["stg", "rc", "cvs"] if args.task == "all" else [args.task]
    first = {task: run_task_smoke(cfg, backend, task) for task in tasks}
    second = {task: run_task_smoke(cfg, backend, task) for task in tasks}
    cache_report = {
        "first_run": first,
        "second_run": second,
        "cache_restart_pass": all(rep["new"] == 0 and rep["cached"] == rep["requested"] for rep in second.values()),
    }
    write_json(cfg.audit_dir / "cache_restart.json", cache_report)
    print(cache_report)
    return 0 if cache_report["cache_restart_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
