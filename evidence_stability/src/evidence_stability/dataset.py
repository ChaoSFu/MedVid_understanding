from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .temporal import (
    GTSpan,
    TemporalMapper,
    build_temporal_cells,
    frame_quality_stats,
    validate_and_clip_gt_spans,
    visible_cells_for_spans,
)


_SPAN_RE = re.compile(
    r"(?P<start>-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<end>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_START_END_RE = re.compile(
    r"start\s*:?\s*(?P<start>-?\d+(?:\.\d+)?)\s*,?\s*"
    r"end\s*:?\s*(?P<end>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def get_question_and_answer(sample: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""
    for turn in sample.get("conversations") or []:
        if turn.get("from") == "human" and not question:
            question = turn.get("value", "")
        elif turn.get("from") == "gpt" and not answer:
            answer = turn.get("value", "")
    return question, answer


def extract_target(sample: dict[str, Any]) -> tuple[str | None, str | None]:
    for item in sample.get("struc_info") or []:
        if not isinstance(item, dict):
            continue
        has_action = "action" in item and item.get("action") is not None
        has_phase = "phase" in item and item.get("phase") is not None
        if has_action:
            return str(item["action"]), "action"
        if has_phase:
            return str(item["phase"]), "phase"
    return None, None


def extract_gt_spans(sample: dict[str, Any], gt_answer: str = "") -> tuple[list[GTSpan], str]:
    spans: list[GTSpan] = []
    for item in sample.get("struc_info") or []:
        if not isinstance(item, dict):
            continue
        for span in item.get("spans") or []:
            if isinstance(span, dict) and "start" in span and "end" in span:
                spans.append(GTSpan(float(span["start"]), float(span["end"])))
    if spans:
        return spans, "struc_info"

    seen: set[tuple[float, float]] = set()
    for regex in (_START_END_RE, _SPAN_RE):
        for match in regex.finditer(gt_answer or ""):
            start = float(match.group("start"))
            end = float(match.group("end"))
            key = (start, end)
            if key not in seen:
                seen.add(key)
                spans.append(GTSpan(start, end))
    return spans, "gt_answer" if spans else "none"


def remap_frame_paths(frame_paths: list[str], frame_root: str | None, old_root: str = "/root/data") -> list[str]:
    if not frame_root:
        return list(frame_paths)
    root = str(frame_root).rstrip("/")
    old = old_root.rstrip("/")
    out: list[str] = []
    for path in frame_paths:
        if path.startswith(old + "/"):
            out.append(root + path[len(old):])
        else:
            out.append(path)
    return out


def normalize_tal_sample(
    sample: dict[str, Any],
    original_index: int,
    frame_root: str | None = None,
    old_frame_root: str = "/root/data",
    verify_paths: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    clip_id = str(sample.get("id", ""))
    qa_id = f"{original_index:04d}::{clip_id}"
    if sample.get("qa_type") != "tal":
        return None, {"qa_id": qa_id, "clip_id": clip_id, "reason": "NOT_TAL"}

    question, gt_answer = get_question_and_answer(sample)
    target, target_field = extract_target(sample)
    raw_spans, parse_source = extract_gt_spans(sample, gt_answer)
    frame_paths = remap_frame_paths(list(sample.get("video") or []), frame_root, old_frame_root)
    sampled_frames = [int(x) for x in sample.get("sampled_video_frames") or []]
    metadata = dict(sample.get("metadata") or {})
    dataset_name = sample.get("dataset_name") or sample.get("data_source")

    failures: list[str] = []
    if not clip_id:
        failures.append("MISSING_CLIP_ID")
    if not question:
        failures.append("MISSING_QUESTION")
    if not gt_answer:
        failures.append("MISSING_GT_ANSWER")
    if target is None:
        failures.append("MISSING_TARGET")
    if not raw_spans:
        failures.append("MISSING_GT_SPANS")
    if not frame_paths or not sampled_frames:
        failures.append("MISSING_FRAMES")
    if len(frame_paths) != len(sampled_frames):
        failures.append("LEN_VIDEO_NE_SAMPLED_FRAMES")
    if "fps" not in metadata:
        failures.append("MISSING_METADATA_FPS")
    if not dataset_name:
        failures.append("MISSING_DATASET_NAME")

    if failures:
        return None, {
            "qa_id": qa_id,
            "clip_id": clip_id,
            "original_index": original_index,
            "reason": ";".join(failures),
        }

    mapping = TemporalMapper.map(str(dataset_name), sampled_frames, frame_paths, metadata)
    clip_duration = mapping.clip_duration
    observations = list(mapping.observations)
    quality = frame_quality_stats(sampled_frames)
    validation = validate_and_clip_gt_spans(raw_spans, clip_duration, float(metadata["fps"]))
    cells = build_temporal_cells(observations, clip_duration)
    visible_total = visible_cells_for_spans(cells, validation.processed_spans)

    missing_paths: list[str] = []
    if verify_paths:
        for frame_path in frame_paths:
            if not Path(frame_path).exists():
                missing_paths.append(frame_path)
                if len(missing_paths) >= 5:
                    break

    temporal_status = validation.temporal_status
    if temporal_status == "OK" and not visible_total:
        temporal_status = "NO_VISIBLE_GT_FRAME"
    analysis_eligible = temporal_status == "OK" and bool(visible_total)

    normalized = {
        "qa_id": qa_id,
        "clip_id": clip_id,
        "sample_id": qa_id,
        "original_index": original_index,
        "question": question,
        "target_action": target,
        "target_field": target_field,
        "gt_answer": gt_answer,
        "gt_spans": [span.to_dict() for span in validation.processed_spans],
        "processed_gt_spans": [span.to_dict() for span in validation.processed_spans],
        "raw_gt_spans": [span.to_dict() for span in raw_spans],
        "invalid_gt_spans": list(validation.invalid_gt_spans),
        "gt_duration_total": sum(span.duration for span in validation.processed_spans),
        "gt_num_spans": len(validation.processed_spans),
        "zero_duration_span_count": sum(1 for span in validation.processed_spans if span.duration == 0),
        "fps": float(metadata["fps"]),
        "metadata_fps": float(metadata["fps"]),
        "metadata": metadata,
        "input_start": metadata.get("input_video_start_time"),
        "input_end": metadata.get("input_video_end_time"),
        "clip_duration": clip_duration,
        "time_mapping_method": mapping.time_mapping_method,
        "source_timebase_hz": mapping.source_timebase_hz,
        "clip_duration_source": mapping.clip_duration_source,
        "frame_paths": frame_paths,
        "sampled_frame_indices": sampled_frames,
        "frame_observations": [obs.to_dict() for obs in observations],
        "temporal_cells": [cell.to_dict() for cell in cells],
        "n_gt_visible_total": len(visible_total),
        "gt_visible_source_frame_indices": sorted(visible_total),
        "dataset_name": dataset_name,
        "data_source": sample.get("data_source"),
        "parse_source": parse_source,
        "parse_ok": True,
        "temporal_status": temporal_status,
        "analysis_eligible": analysis_eligible,
        "qa_id_valid": bool(qa_id),
        "gt_boundary_clipped": validation.gt_boundary_clipped,
        "temporal_excluded_reason": validation.excluded_reason,
        "temporal_tolerance": validation.tolerance,
        "missing_frame_count_preview": len(missing_paths),
        "first_missing_frame": missing_paths[0] if missing_paths else None,
        **quality,
    }
    return normalized, None
