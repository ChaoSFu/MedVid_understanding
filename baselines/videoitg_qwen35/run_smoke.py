from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunConfig
from .io_utils import read_jsonl, write_json
from .manifest import build_manifest, write_smoke_manifest


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    p = argparse.ArgumentParser(description="Prepare fixed GT-blind smoke inputs for VideoITG-32 + Qwen3.5-4B.")
    p.add_argument("--data-path", type=Path, default=cfg.data_path)
    p.add_argument("--output-root", type=Path, default=cfg.output_root)
    p.add_argument("--old-data-root", default=cfg.old_data_root)
    p.add_argument("--new-data-root", default=cfg.new_data_root)
    p.add_argument("--seed", type=int, default=cfg.smoke_seed)
    p.add_argument("--per-task", type=int, default=cfg.smoke_per_task)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(
        data_path=args.data_path,
        output_root=args.output_root,
        old_data_root=args.old_data_root,
        new_data_root=args.new_data_root,
        smoke_seed=args.seed,
        smoke_per_task=args.per_task,
    )
    cfg.make_dirs()
    manifest_path = cfg.manifest_dir / "medvidu_videoitg_manifest_gt_free.jsonl"
    smoke_path = cfg.manifest_dir / "medvidu_videoitg_manifest_gt_free.smoke.jsonl"
    manifest_report = build_manifest(
        cfg.data_path,
        manifest_path,
        old_data_root=cfg.old_data_root,
        new_data_root=cfg.new_data_root,
    )
    smoke_report = write_smoke_manifest(read_jsonl(manifest_path), smoke_path, per_task=cfg.smoke_per_task, seed=cfg.smoke_seed)
    write_json(cfg.provenance_dir / "run_config.json", cfg.to_jsonable())
    report = {
        "manifest": manifest_report,
        "smoke": smoke_report,
        "next_commands": {
            "qwen_preflight": (
                "python -m baselines.videoitg_qwen35.run_inference "
                "--preflight-only "
                f"--output-root {cfg.output_root}"
            ),
            "selector": (
                "python -m baselines.videoitg_qwen35.run_selector "
                f"--manifest {smoke_path} "
                f"--output-root {cfg.output_root} "
                "--videoitg-repo-dir /path/to/VideoITG"
            ),
            "qwen_inference": (
                "python -m baselines.videoitg_qwen35.run_inference "
                f"--manifest {smoke_path} "
                f"--selector {cfg.selector_dir / 'videoitg_top32.jsonl'} "
                f"--output-root {cfg.output_root}"
            ),
            "evaluation": (
                "python -m baselines.videoitg_qwen35.run_evaluation "
                f"--predictions {cfg.prediction_dir / 'qwen35_videoitg_predictions.jsonl'} "
                f"--ground-truth {cfg.data_path} "
                f"--output-root {cfg.output_root}"
            ),
        },
    }
    write_json(cfg.logs_dir / "smoke_prepare_report.json", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
