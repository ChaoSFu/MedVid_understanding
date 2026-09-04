from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_ROOT, MIN_TOKENS, TOTAL_TOKENS
from .dataset import build_qwen3_frame_list_messages, process_qwen3_frame_list_messages
from .io_utils import read_jsonl, write_json


def audit_rows(rows: list[dict[str, Any]], output_path: Path, limit: int = 20) -> dict[str, Any]:
    selected = rows[:limit]
    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for row in selected:
        record: dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "dataset_name": row.get("dataset_name"),
            "n_medvidu_frames": row.get("n_medvidu_frames"),
            "clip_duration": row.get("clip_duration"),
            "effective_fps": row.get("effective_fps"),
            "timestamp_spacing_audit": row.get("timestamp_spacing_audit"),
        }
        try:
            messages = build_qwen3_frame_list_messages(row, MIN_TOKENS, TOTAL_TOKENS)
            _, videos, video_kwargs, video_metadatas = process_qwen3_frame_list_messages(messages)
            record["video_tensor_count"] = len(videos)
            record["video_metadata"] = [repr(meta) for meta in video_metadatas]
            record["video_kwargs"] = {k: repr(v) for k, v in video_kwargs.items()}
            record["status"] = "OK"
        except Exception as exc:
            record["status"] = "ERROR"
            record["error"] = repr(exc)
        status_counts[record["status"]] += 1
        records.append(record)
    payload = {
        "audit_name": "qwen3_frame_list_metadata_compatibility_preflight",
        "limit": limit,
        "records": records,
        "status_counts": dict(sorted(status_counts.items())),
        "stop_if_any_error": True,
        "stop_if_single_fps_incompatible": True,
        "single_fps_incompatible_samples": [
            r.get("sample_id")
            for r in selected
            if not r.get("timestamp_spacing_audit", {}).get("qwen_frame_list_single_fps_compatible")
        ],
    }
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Qwen3-VL frame-list video metadata for GT-blind MedVidU TAL samples.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifest" / "medvidu_tal_timelens_manifest_gt_free.smoke.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "audit" / "timestamp_metadata_audit.json")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit_rows(read_jsonl(args.manifest), args.output, limit=args.limit)
    print(payload)
    if payload["status_counts"].get("ERROR") or payload["single_fps_incompatible_samples"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

