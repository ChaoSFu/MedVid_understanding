from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import (
    ALL_QA_TYPES,
    REGION_CAPTION_QA_TYPES,
    SELECTOR_APPLICABLE_QA_TYPES,
    TASK_NAME_BY_QA_TYPE,
    RunConfig,
)
from .io_utils import assert_no_gt_leak, read_json, write_json, write_jsonl


EVIDENCE_STABILITY_SRC = Path(__file__).resolve().parents[2] / "evidence_stability" / "src"
if str(EVIDENCE_STABILITY_SRC) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_STABILITY_SRC))

from evidence_stability.temporal import FrameObservation, MappingResult, TemporalMapper  # noqa: E402


RC_SELECTOR_REASON = "region-conditioned task already specifies local visual evidence"


def extract_question(sample: dict[str, Any]) -> str:
    for turn in sample.get("conversations") or []:
        if turn.get("from") in {"human", "user"}:
            return str(turn.get("value", "") or "")
    return str(sample.get("question", "") or "")


def make_sample_id(sample: dict[str, Any], original_index: int) -> str:
    return f"{original_index:06d}::{sample.get('id', '')}::{sample.get('qa_type', '')}"


def map_temporal_positions(sample: dict[str, Any]) -> MappingResult:
    frame_paths = list(sample.get("video") or [])
    sampled_frames = [int(x) for x in sample.get("sampled_video_frames") or []]
    metadata = dict(sample.get("metadata") or {})
    dataset_name = str(sample.get("dataset_name") or sample.get("data_source") or "")
    try:
        return TemporalMapper.map(dataset_name, sampled_frames, frame_paths, metadata)
    except ValueError as exc:
        if "Unsupported TAL dataset" not in str(exc):
            raise
        if "fps" not in metadata:
            raise
        return _map_by_metadata_fps(sampled_frames, frame_paths, float(metadata["fps"]))


def _map_by_metadata_fps(sampled_frames: list[int], frame_paths: list[str], fps: float) -> MappingResult:
    if len(sampled_frames) != len(frame_paths):
        raise ValueError("len(sampled_video_frames) must equal len(video)")
    if not sampled_frames:
        raise ValueError("sampled_video_frames is empty")
    first = sampled_frames[0]
    last = sampled_frames[-1]
    observations = []
    for i, source_frame in enumerate(sampled_frames):
        observations.append(
            FrameObservation(
                frame_position=i,
                source_frame_index=int(source_frame),
                frame_path=frame_paths[i],
                local_time=float(source_frame - first) / fps,
            )
        )
    return MappingResult(
        clip_duration=float(max(0, last - first)) / fps,
        observations=tuple(observations),
        time_mapping_method="metadata_fps_source_frame_rate",
        source_timebase_hz=fps,
        clip_duration_source="sampled_frame_index_range_metadata_fps",
    )


def build_gt_free_row(sample: dict[str, Any], original_index: int = 0) -> dict[str, Any]:
    qa_type = str(sample.get("qa_type", ""))
    if qa_type not in ALL_QA_TYPES:
        raise KeyError(f"Unsupported qa_type: {qa_type}")

    video = list(sample.get("video") or [])
    sampled_frames = [int(x) for x in sample.get("sampled_video_frames") or []]
    if len(video) != len(sampled_frames):
        raise ValueError(
            f"sample {original_index} len(video)={len(video)} "
            f"!= len(sampled_video_frames)={len(sampled_frames)}"
        )

    mapping = map_temporal_positions(sample)
    row: dict[str, Any] = {
        "sample_id": make_sample_id(sample, original_index),
        "original_index": original_index,
        "id": sample.get("id"),
        "qa_type": qa_type,
        "dataset_name": sample.get("dataset_name"),
        "data_source": sample.get("data_source"),
        "question": extract_question(sample),
        "video": video,
        "sampled_video_frames": sampled_frames,
        "metadata": dict(sample.get("metadata") or {}),
        "selector_applicable": qa_type in SELECTOR_APPLICABLE_QA_TYPES,
        "selector_reason": None,
        "n_medvidu_frames": len(video),
        "clip_duration": mapping.clip_duration,
        "time_mapping_method": mapping.time_mapping_method,
        "source_timebase_hz": mapping.source_timebase_hz,
        "clip_duration_source": mapping.clip_duration_source,
        "frame_observations": [obs.to_dict() for obs in mapping.observations],
    }
    if qa_type in REGION_CAPTION_QA_TYPES:
        row["selector_applicable"] = False
        row["selector_reason"] = RC_SELECTOR_REASON
        row["is_RC"] = bool(sample.get("is_RC"))
        row["RC_info"] = sample.get("RC_info")

    assert_no_gt_leak(row)
    return row


def build_manifest(data_path: Path, output_path: Path) -> dict[str, Any]:
    data = read_json(data_path)
    if not isinstance(data, list):
        raise TypeError("MedVidU input JSON must be a list")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i, sample in enumerate(data):
        try:
            rows.append(build_gt_free_row(sample, i))
        except Exception as exc:
            failures.append(
                {
                    "original_index": i,
                    "id": sample.get("id") if isinstance(sample, dict) else None,
                    "qa_type": sample.get("qa_type") if isinstance(sample, dict) else None,
                    "error": repr(exc),
                }
            )
    write_jsonl(output_path, rows)
    report = {
        "data_path": str(data_path),
        "output_path": str(output_path),
        "total_input_samples": len(data),
        "manifest_samples": len(rows),
        "failures": failures,
        "samples_by_qa_type": dict(sorted(Counter(row["qa_type"] for row in rows).items())),
        "selector_applicable_samples": sum(1 for row in rows if row["selector_applicable"]),
        "rc_samples": sum(1 for row in rows if not row["selector_applicable"]),
    }
    write_json(output_path.with_suffix(".summary.json"), report)
    return report


def choose_smoke_samples(manifest_rows: list[dict[str, Any]], per_task: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        if row.get("selector_applicable"):
            by_task[TASK_NAME_BY_QA_TYPE[str(row["qa_type"])]].append(row)

    selected: list[dict[str, Any]] = []
    for task_name in sorted(by_task):
        rows = list(by_task[task_name])
        rows.sort(key=lambda r: (str(r.get("qa_type") or ""), str(r.get("dataset_name") or ""), int(r.get("n_medvidu_frames") or 0), r["sample_id"]))
        rng.shuffle(rows)
        rows.sort(key=lambda r: (len([x for x in selected if x.get("dataset_name") == r.get("dataset_name")]), r["sample_id"]))
        selected.extend(rows[:per_task])
    selected.sort(key=lambda r: r["sample_id"])
    return selected


def write_smoke_manifest(manifest_rows: list[dict[str, Any]], output_path: Path, per_task: int, seed: int) -> dict[str, Any]:
    rows = choose_smoke_samples(manifest_rows, per_task=per_task, seed=seed)
    write_jsonl(output_path, rows)
    ids_path = output_path.parent / "smoke_sample_ids.json"
    payload = {
        "seed": seed,
        "per_task": per_task,
        "sample_ids": [row["sample_id"] for row in rows],
        "samples": [
            {
                "sample_id": row["sample_id"],
                "qa_type": row["qa_type"],
                "task_name": TASK_NAME_BY_QA_TYPE[str(row["qa_type"])],
                "dataset_name": row.get("dataset_name"),
                "n_medvidu_frames": row.get("n_medvidu_frames"),
            }
            for row in rows
        ],
    }
    write_json(ids_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    p = argparse.ArgumentParser(description="Build GT-free MedVidU manifest for VideoITG-32 + Qwen3.5 baseline.")
    p.add_argument("--data-path", type=Path, default=cfg.data_path)
    p.add_argument("--output-root", type=Path, default=cfg.output_root)
    p.add_argument("--smoke", action="store_true", help="Also write fixed smoke manifest and sample ids.")
    p.add_argument("--smoke-per-task", type=int, default=cfg.smoke_per_task)
    p.add_argument("--seed", type=int, default=cfg.smoke_seed)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root)
    cfg.make_dirs()
    manifest_path = cfg.manifest_dir / "medvidu_videoitg_manifest_gt_free.jsonl"
    report = build_manifest(args.data_path, manifest_path)
    if args.smoke:
        from .io_utils import read_jsonl

        rows = read_jsonl(manifest_path)
        smoke = write_smoke_manifest(
            rows,
            cfg.manifest_dir / "medvidu_videoitg_manifest_gt_free.smoke.jsonl",
            per_task=args.smoke_per_task,
            seed=args.seed,
        )
        report["smoke_samples"] = smoke
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
