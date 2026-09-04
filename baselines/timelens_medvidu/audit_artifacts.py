from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_ROOT
from .io_utils import first_gt_leak, read_jsonl, write_json


def frame_path_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    duplicate_logical_pairs = 0
    for row in rows:
        video = list(row.get("video") or [])
        sampled = list(row.get("sampled_video_frames") or [])
        for i, path in enumerate(video):
            if not Path(path).exists():
                missing.append({"sample_id": row.get("sample_id"), "frame_position": i, "path": path})
        for i in range(len(sampled) - 1):
            if sampled[i] == sampled[i + 1] or video[i] == video[i + 1]:
                duplicate_logical_pairs += 1
    return {
        "n_samples": len(rows),
        "n_frame_paths": sum(len(row.get("video") or []) for row in rows),
        "missing_frame_paths": len(missing),
        "missing_examples": missing[:50],
        "duplicate_logical_adjacent_pairs": duplicate_logical_pairs,
        "all_frame_paths_exist": not missing,
    }


def gt_leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    leaks = []
    for row in rows:
        leak = first_gt_leak(row)
        if leak:
            leaks.append({"sample_id": row.get("sample_id"), "leak": leak})
    return {
        "n_samples": len(rows),
        "leak_count": len(leaks),
        "leaks": leaks[:50],
        "gt_information_available_to_model_values": dict(
            sorted(Counter(str(row.get("gt_information_available_to_model")) for row in rows).items())
        ),
    }


def write_audits(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame_path_audit(rows)
    leakage = gt_leakage_audit(rows)
    write_json(output_dir / "frame_path_audit.json", frame)
    write_json(output_dir / "gt_leakage_audit.json", leakage)
    return {"frame_path_audit": frame, "gt_leakage_audit": leakage}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write GT-free manifest audits for TimeLens-MedVidU.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifest" / "medvidu_tal_timelens_manifest_gt_free.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = write_audits(args.manifest, args.output_dir)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

