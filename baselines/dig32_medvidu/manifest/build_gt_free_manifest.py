from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from baselines.videoitg_qwen35.manifest import (
    build_gt_free_row,
    build_manifest as _build_manifest,
    choose_smoke_samples,
    write_smoke_manifest as _write_smoke_manifest,
)

from ..config import RunConfig
from ..io_utils import read_jsonl, write_json


def frame_count_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(row["n_medvidu_frames"]) for row in rows]
    if not counts:
        return {"min": None, "median": None, "max": None, "n_lt_32": 0, "n_eq_32": 0, "n_gt_32": 0}
    return {
        "min": min(counts),
        "median": median(counts),
        "max": max(counts),
        "n_lt_32": sum(n < 32 for n in counts),
        "n_eq_32": sum(n == 32 for n in counts),
        "n_gt_32": sum(n > 32 for n in counts),
    }


def rc_compatibility_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples = []
    count = 0
    missing_anchor = 0
    for row in rows:
        if not str(row.get("qa_type", "")).startswith("region_caption"):
            continue
        count += 1
        rc_info = row.get("RC_info") or {}
        start_frame = rc_info.get("start_frame") if isinstance(rc_info, dict) else None
        has_bbox = isinstance(rc_info, dict) and bool(rc_info.get("start_frame_bbox"))
        anchor_in_video = start_frame in set(row.get("video") or []) if start_frame else False
        if not start_frame or not has_bbox or not anchor_in_video:
            missing_anchor += 1
        if len(examples) < 5:
            examples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "qa_type": row.get("qa_type"),
                    "start_frame": start_frame,
                    "has_bbox": has_bbox,
                    "anchor_in_video": anchor_in_video,
                }
            )
    return {
        "status": "RC_SELECTOR_ADAPTATION_REQUIRED" if count else "NO_RC_SAMPLES",
        "rc_samples": count,
        "mandatory_region_frame_schema": count > 0,
        "missing_or_unmatched_anchor_count": missing_anchor,
        "policy": "DIG-32 selector is not applied to RC until reviewed.",
        "review_options": [
            "A. RC direct Qwen, DIG marked N/A",
            "B. force-keep anchor region frame + 31 DIG frames",
            "C. another official-task-compatible adaptation",
        ],
        "examples": examples,
    }


def build_manifest(
    data_path: Path,
    output_path: Path,
    old_data_root: str,
    new_data_root: str | None,
) -> dict[str, Any]:
    report = _build_manifest(data_path, output_path, old_data_root=old_data_root, new_data_root=new_data_root)
    rows = read_jsonl(output_path)
    report["frame_count_audit"] = frame_count_audit(rows)
    report["rc_compatibility"] = rc_compatibility_audit(rows)
    report["samples_by_qa_type"] = dict(sorted(Counter(row["qa_type"] for row in rows).items()))
    write_json(output_path.with_suffix(".summary.json"), report)
    return report


def write_smoke_manifest(rows: list[dict[str, Any]], output_path: Path, per_task: int, seed: int) -> dict[str, Any]:
    smoke = _write_smoke_manifest(rows, output_path, per_task=per_task, seed=seed)
    smoke["frame_count_audit"] = frame_count_audit(read_jsonl(output_path))
    return smoke


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    parser = argparse.ArgumentParser(description="Build GT-free MedVidU manifest for Fixed DIG-32.")
    parser.add_argument("--data-path", type=Path, default=cfg.data_path)
    parser.add_argument("--output-root", type=Path, default=cfg.output_root)
    parser.add_argument("--old-data-root", default=cfg.old_data_root)
    parser.add_argument("--new-data-root", default=cfg.new_data_root)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-task", type=int, default=cfg.smoke_per_task)
    parser.add_argument("--seed", type=int, default=cfg.smoke_seed)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root, old_data_root=args.old_data_root, new_data_root=args.new_data_root)
    cfg.make_dirs()
    manifest_path = cfg.selector_dir / "gt_free_manifest.jsonl"
    report = build_manifest(cfg.data_path, manifest_path, old_data_root=cfg.old_data_root, new_data_root=cfg.new_data_root)
    if args.smoke:
        smoke_path = cfg.selector_dir / "gt_free_manifest.smoke.jsonl"
        report["smoke"] = write_smoke_manifest(read_jsonl(manifest_path), smoke_path, per_task=args.smoke_per_task, seed=args.seed)
        write_json(cfg.selector_dir / "smoke_sample_ids.json", report["smoke"])
    write_json(cfg.provenance_dir / "run_config.json", cfg.to_jsonable())
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
