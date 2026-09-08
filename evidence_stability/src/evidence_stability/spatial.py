from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from evidence_stability.cache import stable_hash
from evidence_stability.temporal import (
    GTSpan,
    TemporalMapper,
    build_temporal_cells,
    compute_alignment_metrics,
    frame_quality_stats,
    validate_and_clip_gt_spans,
    visible_cells_for_spans,
)


H4_PROTOCOL_VERSION = "h4_spatial_v1"
H4_DISCOVERY_SEED = 42
H4_PREFLIGHT_QA_COUNT = 12
H4_DISCOVERY_QA_COUNT = 50
H4_WINDOW_SIZE = 16
H4_WINDOW_STRIDE = 8
H4_TEMPORAL_ELIGIBILITY_RULE = "h2_strong_temporal_alignment"
H4_TEMPORAL_DENSITY_THRESHOLD = 0.25
H4_TEMPORAL_RECALL_THRESHOLD = 0.50
H4_SPATIAL_TRUE_IOU_THRESHOLD = 0.5
H4_SPURIOUS_IOU_THRESHOLD = 0.0
H4_SPATIAL_LABEL_PROTOCOL = "true_mean_iou_ge_0.5_spurious_iou_eq_0"
H4_SUPPORT_PROMPT_VERSION = "h4_evidence_presence_stg_v1"
SPATIAL_POINTER_PROMPT_VERSION = "spatial_pointer_v1"
SPATIAL_POINTER_COMPAT_PARSER_VERSION = "spatial_pointer_compat_qwen_list_wrapper_v1"
STG_SOURCE_TIMEBASE_HZ = {
    "CholecTrack20": 25.0,
    "CoPESD": 1.0,
    "EgoSurgery": 0.5,
}

H4_SUPPORT_PROMPT_TEMPLATE = """You are examining a short temporal window from a medical or surgical video.

Target claim or question:
"{human_question}"

Does this video window contain visual evidence that directly supports the target claim?

Answer only:
YES
or
NO

Judge only what is visually observable in the provided frames.

Do not answer YES merely because a related instrument,
anatomical structure, scene, or procedure is present.

Answer YES only if the target claim itself is visually supported."""

SPATIAL_POINTER_PROMPT_TEMPLATE = """You are examining a short temporal window from a medical or surgical video.

Target claim or question:
"{human_question}"

The window has already been judged to contain visual evidence supporting the target claim.

Identify the single smallest rectangular spatial region that most directly contains the visual evidence supporting that claim across this short window.

Return ONE bounding box that best covers the relevant evidence region across the window.

Do not select the whole frame unless the evidence genuinely requires the whole scene.

Do not select an instrument, anatomical structure, or object merely because it is present.
Select the region that most directly supports the target claim.

Return JSON only:

{{
  "bbox": [x1, y1, x2, y2]
}}

Coordinates must be integers normalized to 0-1000, where:
(0,0) is the top-left
and
(1000,1000) is the bottom-right.

Requirements:
0 <= x1 < x2 <= 1000
0 <= y1 < y2 <= 1000

No explanation.
No additional text."""


def h4_protocol_freeze() -> dict[str, Any]:
    return {
        "protocol_version": H4_PROTOCOL_VERSION,
        "study_stage": "mechanism_discovery",
        "training": False,
        "primary_task": "MedVidU STG",
        "temporal_eligibility_rule": H4_TEMPORAL_ELIGIBILITY_RULE,
        "temporal_density_threshold": H4_TEMPORAL_DENSITY_THRESHOLD,
        "temporal_recall_threshold": H4_TEMPORAL_RECALL_THRESHOLD,
        "true_spatial_support_rule": "official mean IoU >= 0.5",
        "spurious_spatial_support_rule": "official mean IoU == 0",
        "spatial_true_iou_threshold": H4_SPATIAL_TRUE_IOU_THRESHOLD,
        "spurious_iou_threshold": H4_SPURIOUS_IOU_THRESHOLD,
        "spatial_label_protocol": H4_SPATIAL_LABEL_PROTOCOL,
        "support_prompt_version": H4_SUPPORT_PROMPT_VERSION,
        "spatial_pointer_prompt_version": SPATIAL_POINTER_PROMPT_VERSION,
        "window_size": H4_WINDOW_SIZE,
        "window_stride": H4_WINDOW_STRIDE,
        "drop_last": True,
        "stg_source_timebase_hz": dict(STG_SOURCE_TIMEBASE_HZ),
    }


def build_h4_support_prompt(human_question: str) -> str:
    return H4_SUPPORT_PROMPT_TEMPLATE.format(human_question=human_question)


def build_spatial_pointer_prompt(human_question: str) -> str:
    return SPATIAL_POINTER_PROMPT_TEMPLATE.format(human_question=human_question)


def prompt_sha256(prompt: str) -> str:
    return stable_hash({"prompt": prompt})


def parse_normalized_bbox_json(raw: str | None) -> dict[str, Any]:
    text = (raw or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"bbox_valid": False, "bbox": None, "reason": f"JSON_DECODE_ERROR: {exc.msg}"}
    if not isinstance(parsed, dict) or set(parsed.keys()) != {"bbox"}:
        return {"bbox_valid": False, "bbox": None, "reason": "EXPECTED_OBJECT_WITH_ONLY_BBOX_KEY"}
    box = parsed.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return {"bbox_valid": False, "bbox": None, "reason": "BBOX_MUST_BE_LIST_OF_FOUR_VALUES"}
    values: list[int] = []
    for value in box:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return {"bbox_valid": False, "bbox": None, "reason": "BBOX_VALUES_MUST_BE_FINITE_NUMBERS"}
        rounded = int(value)
        if float(value) != float(rounded):
            return {"bbox_valid": False, "bbox": None, "reason": "BBOX_VALUES_MUST_BE_INTEGER_OR_INTEGER_FLOAT"}
        values.append(rounded)
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return {"bbox_valid": False, "bbox": None, "reason": "BBOX_COORDINATES_OUT_OF_RANGE_OR_INVERTED"}
    return {
        "bbox_valid": True,
        "bbox": values,
        "reason": "OK",
        "bbox_area_fraction": bbox_area_fraction(values),
    }


def parse_qwen_list_wrapped_bbox(raw: str | None) -> dict[str, Any]:
    """Parse Qwen's known list-wrapped JSON variant without guessing coordinates.

    This is intentionally separate from the frozen strict parser. It accepts only a
    single JSON coordinate quadruple wrapped in Markdown, singleton lists, or a
    singleton JSON string; it never extracts numbers from arbitrary prose.
    """
    strict = parse_normalized_bbox_json(raw)
    if strict["bbox_valid"]:
        return {**strict, "parse_method": "strict_json_object"}

    text = (raw or "").strip()
    text = re.sub(r"^`{1,3}json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*`{1,3}$", "", text).strip()
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"bbox_valid": False, "bbox": None, "reason": f"JSON_DECODE_ERROR: {exc.msg}", "parse_method": None}

    for _ in range(3):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {"bbox_valid": False, "bbox": None, "reason": "SINGLETON_STRING_IS_NOT_JSON", "parse_method": None}
        elif isinstance(value, list) and len(value) == 1:
            value = value[0]
        else:
            break

    if not isinstance(value, list) or len(value) != 4:
        return {"bbox_valid": False, "bbox": None, "reason": "EXPECTED_SINGLE_LIST_WRAPPED_BBOX", "parse_method": None}

    validated = parse_normalized_bbox_json(json.dumps({"bbox": value}))
    if not validated["bbox_valid"]:
        return {**validated, "parse_method": None}
    return {
        **validated,
        "reason": "OK_COMPAT_QWEN_LIST_WRAPPER",
        "parse_method": "markdown_json_singleton_list_or_string",
    }


def bbox_area_fraction(box: list[int | float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1_000_000.0


def normalized_to_pixel_bbox(box: list[int | float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        max(0, min(width, math.floor(x1 / 1000.0 * width))),
        max(0, min(height, math.floor(y1 / 1000.0 * height))),
        max(0, min(width, math.ceil(x2 / 1000.0 * width))),
        max(0, min(height, math.ceil(y2 / 1000.0 * height))),
    ]


def control_bbox_norm(box: list[int | float]) -> tuple[list[int], bool, float]:
    x1, y1, x2, y2 = [int(v) for v in box]
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0 or width > 1000 or height > 1000:
        return [x1, y1, x2, y2], False, 1.0
    candidates = [
        [0, 0, width, height],
        [1000 - width, 0, 1000, height],
        [0, 1000 - height, width, 1000],
        [1000 - width, 1000 - height, 1000, 1000],
    ]
    scored = [(box_iou(box, candidate), idx, candidate) for idx, candidate in enumerate(candidates)]
    scored.sort(key=lambda item: (item[0], item[1]))
    best_iou, _, best = scored[0]
    control_valid = best_iou < 0.95 and best != [x1, y1, x2, y2]
    return best, control_valid, float(best_iou)


def box_iou(a: list[int | float], b: list[int | float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_human_question(sample: dict[str, Any]) -> str:
    for turn in sample.get("conversations") or []:
        if turn.get("from") in {"human", "user"}:
            return str(turn.get("value", "") or "")
    return str(sample.get("question") or "")


def stg_struct(sample: dict[str, Any]) -> dict[str, Any] | None:
    info = sample.get("struc_info")
    if isinstance(info, list) and info and isinstance(info[0], dict):
        return info[0]
    if isinstance(info, dict):
        return info
    return None


def stg_dataset_name(sample: dict[str, Any]) -> str:
    return str(sample.get("dataset_name") or sample.get("data_source") or "Unknown")


def stg_gt_spans(sample: dict[str, Any]) -> list[GTSpan]:
    struct = stg_struct(sample)
    if not struct:
        return []
    if "start" not in struct or "end" not in struct:
        return []
    return [GTSpan(float(struct["start"]), float(struct["end"]))]


def map_stg_frames(
    sample: dict[str, Any],
    frame_paths: list[str],
    sampled_frames: list[int],
) -> Any:
    dataset_name = stg_dataset_name(sample)
    if dataset_name not in STG_SOURCE_TIMEBASE_HZ:
        raise ValueError(f"Unsupported STG dataset for H4 temporal mapping: {dataset_name}")
    return TemporalMapper._map_source_timebase(
        sampled_frames,
        frame_paths,
        source_timebase_hz=STG_SOURCE_TIMEBASE_HZ[dataset_name],
    )


def normalize_stg_sample(
    sample: dict[str, Any],
    original_index: int,
    frame_root: str | None = None,
    source_frame_prefix: str | list[str] | tuple[str, ...] | None = None,
    verify_paths: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from evidence_stability.phase_d2 import runtime_frame_path

    clip_id = str(sample.get("id", ""))
    qa_id = f"{original_index:04d}::{clip_id}"
    if str(sample.get("qa_type", "")).lower() != "stg":
        return None, {"qa_id": qa_id, "clip_id": clip_id, "reason": "NOT_STG"}

    question = extract_human_question(sample)
    struct = stg_struct(sample)
    metadata = dict(sample.get("metadata") or {})
    dataset_name = stg_dataset_name(sample)
    raw_frame_paths = [str(path) for path in sample.get("video") or []]
    try:
        frame_paths = [
            runtime_frame_path(path, frame_root, source_frame_prefix)
            if frame_root else path
            for path in raw_frame_paths
        ]
    except ValueError as exc:
        return None, {"qa_id": qa_id, "clip_id": clip_id, "reason": f"FRAME_PATH_REMAP_FAILED: {exc}"}

    failures: list[str] = []
    if not clip_id:
        failures.append("MISSING_CLIP_ID")
    if not question:
        failures.append("MISSING_QUESTION")
    if not struct:
        failures.append("MISSING_STRUC_INFO")
    if "fps" not in metadata:
        failures.append("MISSING_METADATA_FPS")
    if not raw_frame_paths:
        failures.append("MISSING_VIDEO_FRAMES")
    sampled_frames: list[int] = []
    try:
        sampled_frames = [int(x) for x in sample.get("sampled_video_frames") or []]
    except Exception:
        failures.append("MALFORMED_SAMPLED_VIDEO_FRAMES")
    if not sampled_frames:
        failures.append("MISSING_SAMPLED_VIDEO_FRAMES")
    if len(raw_frame_paths) != len(sampled_frames):
        failures.append("LEN_VIDEO_NE_SAMPLED_FRAMES")
    if dataset_name not in STG_SOURCE_TIMEBASE_HZ:
        failures.append("UNSUPPORTED_STG_DATASET_TIMEBASE")
    bbox_dict = (struct or {}).get("bbox_dict") or {}
    if not isinstance(bbox_dict, dict) or not bbox_dict:
        failures.append("MISSING_BBOX_DICT")

    if failures:
        return None, {
            "qa_id": qa_id,
            "clip_id": clip_id,
            "original_index": original_index,
            "dataset_name": dataset_name,
            "reason": ";".join(failures),
        }

    mapping = map_stg_frames(sample, frame_paths, sampled_frames)
    raw_spans = stg_gt_spans(sample)
    validation = validate_and_clip_gt_spans(raw_spans, mapping.clip_duration, float(metadata["fps"]))
    cells = build_temporal_cells(mapping.observations, mapping.clip_duration)
    visible_total = visible_cells_for_spans(cells, validation.processed_spans)
    temporal_status = validation.temporal_status
    if temporal_status == "OK" and not visible_total:
        temporal_status = "NO_VISIBLE_GT_FRAME"
    analysis_eligible = temporal_status == "OK" and bool(visible_total)

    missing_paths: list[str] = []
    if verify_paths:
        for frame_path in frame_paths:
            if not Path(frame_path).exists():
                missing_paths.append(frame_path)
                if len(missing_paths) >= 5:
                    break

    normalized = {
        "qa_id": qa_id,
        "clip_id": clip_id,
        "sample_id": qa_id,
        "original_index": original_index,
        "question": question,
        "human_question": question,
        "dataset_name": dataset_name,
        "data_source": sample.get("data_source"),
        "metadata": metadata,
        "metadata_fps": float(metadata["fps"]),
        "fps": float(metadata["fps"]),
        "frame_paths": frame_paths,
        "source_frame_paths": raw_frame_paths,
        "sampled_frame_indices": sampled_frames,
        "frame_observations": [obs.to_dict() for obs in mapping.observations],
        "temporal_cells": [cell.to_dict() for cell in cells],
        "clip_duration": mapping.clip_duration,
        "time_mapping_method": mapping.time_mapping_method,
        "source_timebase_hz": mapping.source_timebase_hz,
        "clip_duration_source": mapping.clip_duration_source,
        "raw_gt_spans": [span.to_dict() for span in raw_spans],
        "processed_gt_spans": [span.to_dict() for span in validation.processed_spans],
        "gt_spans": [span.to_dict() for span in validation.processed_spans],
        "invalid_gt_spans": list(validation.invalid_gt_spans),
        "gt_boundary_clipped": validation.gt_boundary_clipped,
        "temporal_status": temporal_status,
        "temporal_excluded_reason": validation.excluded_reason,
        "temporal_tolerance": validation.tolerance,
        "analysis_eligible": analysis_eligible,
        "n_gt_visible_total": len(visible_total),
        "gt_visible_source_frame_indices": sorted(visible_total),
        "stg_object": str(struct.get("object")),
        "stg_bbox_dict": bbox_dict,
        "stg_start": float(struct["start"]),
        "stg_end": float(struct["end"]),
        "stg_stride": float(struct.get("stride", 0)),
        "qa_id_valid": bool(qa_id),
        "missing_frame_count_preview": len(missing_paths),
        "first_missing_frame": missing_paths[0] if missing_paths else None,
        **frame_quality_stats(sampled_frames),
    }
    return normalized, None


def is_h4_temporally_eligible(metrics: dict[str, Any], sample: dict[str, Any]) -> bool:
    if not sample.get("analysis_eligible"):
        return False
    if int(metrics.get("n_gt_visible_in_window") or 0) <= 0:
        return False
    density = metrics.get("evidence_density")
    recall = metrics.get("gt_evidence_recall")
    return (
        density is not None and float(density) >= H4_TEMPORAL_DENSITY_THRESHOLD
    ) or (
        recall is not None and float(recall) >= H4_TEMPORAL_RECALL_THRESHOLD
    )


def h4_generate_stg_windows(
    sample: dict[str, Any],
    window_size: int = H4_WINDOW_SIZE,
    stride: int = H4_WINDOW_STRIDE,
    drop_last: bool = True,
) -> list[dict[str, Any]]:
    from evidence_stability.windows import generate_position_windows

    windows = generate_position_windows(sample, window_size=window_size, stride=stride, drop_last=drop_last)
    for window in windows:
        window["candidate_id"] = window["window_id"]
        window["human_question"] = sample["human_question"]
    return windows


def h4_window_temporal_alignment(sample: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    cells = [
        cell
        for cell in sample.get("temporal_cells", [])
    ]
    cell_objects = []
    from evidence_stability.temporal import TemporalCell

    for cell in cells:
        cell_objects.append(
            TemporalCell(
                source_frame_index=int(cell["source_frame_index"]),
                local_time=float(cell["local_time"]),
                cell_start=float(cell["cell_start"]),
                cell_end=float(cell["cell_end"]),
                frame_positions=tuple(int(x) for x in cell.get("frame_positions", [])),
            )
        )
    spans = [GTSpan(float(span["start"]), float(span["end"])) for span in sample.get("processed_gt_spans", [])]
    metrics = compute_alignment_metrics(cell_objects, window["frame_indices"], spans).to_dict()
    metrics["h4_temporally_eligible"] = is_h4_temporally_eligible(metrics, sample)
    metrics["temporal_eligibility_rule"] = H4_TEMPORAL_ELIGIBILITY_RULE
    return metrics


def h4_model_window_manifest_row(sample: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    support_prompt = build_h4_support_prompt(sample["human_question"])
    pointer_prompt = build_spatial_pointer_prompt(sample["human_question"])
    return {
        "qa_id": sample["qa_id"],
        "clip_id": sample["clip_id"],
        "candidate_id": window["candidate_id"],
        "window_id": window["window_id"],
        "dataset": sample["dataset_name"],
        "dataset_name": sample["dataset_name"],
        "human_question": sample["human_question"],
        "frame_ids": list(window["frame_indices"]),
        "frame_paths": list(window["frame_paths"]),
        "local_timestamps": list(window["frame_times"]),
        "n_frames": int(window["n_frames"]),
        "n_unique_frames": int(window["n_unique_frames"]),
        "support_prompt_version": H4_SUPPORT_PROMPT_VERSION,
        "support_prompt_hash": prompt_sha256(support_prompt),
        "spatial_pointer_prompt_version": SPATIAL_POINTER_PROMPT_VERSION,
        "spatial_pointer_prompt_hash": prompt_sha256(pointer_prompt),
    }


def h4_gt_leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_fields = {
        "gt",
        "gt_spans",
        "raw_gt_spans",
        "processed_gt_spans",
        "invalid_gt_spans",
        "gt_visible_source_frame_indices",
        "stg_bbox_dict",
        "spatial_match_score",
        "candidate_spatial_type",
        "true_spatial_support",
        "spurious_spatial_support",
        "iou",
        "mean_iou",
        "h4_temporally_eligible",
        "evidence_density",
        "gt_evidence_recall",
        "n_gt_visible_in_window",
        "n_gt_visible_total",
    }
    forbidden_terms = [
        "ground_truth",
        "ground truth",
        "true_spatial_support",
        "spurious_spatial_support",
        "candidate_spatial_type",
        "spatial_match_score",
        "mean_iou",
        "iou",
        "bbox_dict",
        "gt_spans",
    ]
    field_hits = []
    term_hits = []
    for row in rows:
        fields = set(row)
        hits = sorted(fields & forbidden_fields)
        if hits:
            field_hits.append({"candidate_id": row.get("candidate_id"), "fields": hits})
        serialized = json.dumps(row, ensure_ascii=False).lower()
        hits_terms = [term for term in forbidden_terms if term in serialized]
        if hits_terms:
            term_hits.append({"candidate_id": row.get("candidate_id"), "terms": hits_terms})
    return {
        "gt_information_used_in_model_inference": False,
        "n_model_manifest_rows": len(rows),
        "n_rows_with_forbidden_fields": len(field_hits),
        "n_rows_with_forbidden_terms": len(term_hits),
        "forbidden_field_examples": field_hits[:20],
        "forbidden_term_examples": term_hits[:20],
        "gt_leakage": len(field_hits) + len(term_hits),
    }


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(clean),
        "median": float(median(clean)),
        "mean": float(mean(clean)),
        "max": max(clean),
    }


def audit_stg_schema(samples: list[dict[str, Any]]) -> dict[str, Any]:
    stg = [row for row in samples if str(row.get("qa_type", "")).lower() == "stg"]
    datasets = Counter(str(row.get("dataset_name") or row.get("data_source") or "Unknown") for row in stg)
    fps = Counter(
        f"{row.get('dataset_name') or row.get('data_source') or 'Unknown'}::{row.get('metadata', {}).get('fps')}"
        for row in stg
    )
    struc_lengths = Counter(len(row.get("struc_info") or []) if isinstance(row.get("struc_info"), list) else "dict" for row in stg)
    objects = Counter()
    starts: list[float] = []
    ends: list[float] = []
    strides: list[float] = []
    bbox_counts: list[int] = []
    coords: list[float] = []
    key_patterns = Counter()
    examples = []
    missing_struct = 0
    malformed_bbox = 0
    for row in stg:
        struct = stg_struct(row)
        if not struct:
            missing_struct += 1
            continue
        objects[str(struct.get("object"))] += 1
        starts.append(float(struct.get("start", 0)))
        ends.append(float(struct.get("end", 0)))
        if "stride" in struct:
            strides.append(float(struct["stride"]))
        bbox_dict = struct.get("bbox_dict") or {}
        bbox_counts.append(len(bbox_dict))
        for key, box in bbox_dict.items():
            key_patterns["float_like" if re.fullmatch(r"\d+(?:\.\d+)?", str(key)) else "other"] += 1
            if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
                malformed_bbox += 1
                continue
            coords.extend(float(v) for v in box)
        if len(examples) < 8:
            examples.append(
                {
                    "id": row.get("id"),
                    "dataset_name": row.get("dataset_name"),
                    "metadata": row.get("metadata"),
                    "question": extract_human_question(row),
                    "struc_info": struct,
                    "n_video_frames": len(row.get("video") or []),
                    "n_sampled_video_frames": len(row.get("sampled_video_frames") or []),
                }
            )
    return {
        "protocol_version": H4_PROTOCOL_VERSION,
        "n_stg": len(stg),
        "datasets": dict(sorted(datasets.items())),
        "fps_by_dataset": dict(sorted(fps.items())),
        "sample_keys": sorted(stg[0].keys()) if stg else [],
        "struc_info_length_distribution": dict(struc_lengths),
        "missing_struct_count": missing_struct,
        "malformed_bbox_count": malformed_bbox,
        "object_top20": objects.most_common(20),
        "temporal_gt": {
            "fields": ["start", "end", "stride"],
            "start_seconds": _numeric_summary(starts),
            "end_seconds": _numeric_summary(ends),
            "stride_seconds_counts": dict(Counter(str(v) for v in strides)),
            "interpretation": "clip-local seconds from struc_info; STG asks for boxes at bbox_dict timestamp keys.",
        },
        "spatial_gt": {
            "type": "per-requested-timestamp xyxy bbox_dict",
            "bbox_count_distribution": dict(Counter(bbox_counts)),
            "coordinate_summary": _numeric_summary(coords),
            "n_coordinates_lt_0": sum(v < 0 for v in coords),
            "n_coordinates_gt_1000": sum(v > 1000 for v in coords),
            "key_pattern_counts": dict(key_patterns),
            "coordinate_convention": "absolute pixel xyxy coordinates in original/evaluator frame scale, not normalized 0-1000.",
        },
        "official_stg_metric": {
            "source": "MedVidBench-Leaderboard/evaluation/eval_stg.py",
            "prediction_format": "'<seconds> seconds: [x1, y1, x2, y2]' for each GT bbox_dict key",
            "timestamp_matching": "prediction key formatted as f'{float(key):.1f}' must match each GT bbox_dict key",
            "record_score": "mean IoU over valid predicted boxes and GT boxes for bbox_dict keys",
            "reported_metrics": ["mIoU", "iou@0.3", "iou@0.5", "iou@0.7"],
            "single_canonical_positive_threshold": None,
            "threshold_freeze_status": "NEEDS_HUMAN_FREEZE_BEFORE_TRUE_SPATIAL_SUPPORT_LABELS",
        },
        "examples": examples,
    }


def write_stg_schema_md(path: str, audit: dict[str, Any]) -> None:
    lines = [
        "# MedVidU STG Schema Audit for H4 Spatial v1",
        "",
        "## Dataset",
        f"- STG samples: {audit['n_stg']}",
        f"- datasets: {audit['datasets']}",
        f"- fps by dataset: {audit['fps_by_dataset']}",
        f"- sample keys: {audit['sample_keys']}",
        "",
        "## Input Fields",
        "- `video`: ordered benchmark frame paths.",
        "- `sampled_video_frames`: source/pre-extracted frame indices aligned 1:1 with `video`.",
        "- `conversations[human].value`: human STG question. The assistant answer is GT and must be stripped from model-side manifests.",
        "- `metadata`: video_id, fps, input_video_start_frame, input_video_end_frame.",
        "",
        "## Temporal GT",
        f"- schema: {audit['temporal_gt']['fields']}",
        f"- start seconds: {audit['temporal_gt']['start_seconds']}",
        f"- end seconds: {audit['temporal_gt']['end_seconds']}",
        f"- stride counts: {audit['temporal_gt']['stride_seconds_counts']}",
        f"- interpretation: {audit['temporal_gt']['interpretation']}",
        "",
        "## Spatial GT",
        f"- type: {audit['spatial_gt']['type']}",
        f"- bbox count distribution: {audit['spatial_gt']['bbox_count_distribution']}",
        f"- coordinate summary: {audit['spatial_gt']['coordinate_summary']}",
        f"- coordinates < 0: {audit['spatial_gt']['n_coordinates_lt_0']}",
        f"- coordinates > 1000: {audit['spatial_gt']['n_coordinates_gt_1000']}",
        f"- convention: {audit['spatial_gt']['coordinate_convention']}",
        "",
        "## Official Evaluator",
        f"- source: {audit['official_stg_metric']['source']}",
        f"- prediction format: {audit['official_stg_metric']['prediction_format']}",
        f"- timestamp matching: {audit['official_stg_metric']['timestamp_matching']}",
        f"- record score: {audit['official_stg_metric']['record_score']}",
        f"- reported metrics: {audit['official_stg_metric']['reported_metrics']}",
        f"- single canonical positive threshold: {audit['official_stg_metric']['single_canonical_positive_threshold']}",
        "",
        "## H4 Stop Condition",
        "STG spatial GT and official IoU computation are readable, but the official evaluator reports multiple thresholds (`iou@0.3`, `iou@0.5`, `iou@0.7`) and does not designate one canonical positive criterion for TRUE_SPATIAL_SUPPORT. Per H4-v1 protocol, freeze this threshold before post-hoc TRUE/SPURIOUS spatial labeling.",
    ]
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_temporal_protocol_review_md(path: str) -> None:
    lines = [
        "# H4 Temporal Eligibility Protocol Review",
        "",
        "## H2 Frozen Rule",
        "- D_E = |F_window intersect F_GT| / |F_window|.",
        "- R_E = |F_window intersect F_GT| / |F_GT|.",
        "- strong temporal alignment if D_E >= 0.25 or R_E >= 0.50.",
        "",
        "## STG Official Evaluator Rule",
        "- The STG evaluator does not evaluate temporal localization as a separate prediction.",
        "- It expects predicted boxes at the timestamp keys already specified by GT `bbox_dict`.",
        "- It computes per-record mean IoU over these timestamp-aligned boxes.",
        "",
        "## H4 Freeze Point",
        "For H4-v1, temporal eligibility is frozen to the H2 strong temporal alignment rule before preflight/discovery: D_E >= 0.25 or R_E >= 0.50.",
    ]
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_spatial_threshold_review_md(path: str, audit: dict[str, Any]) -> None:
    lines = [
        "# H4 Spatial Label Threshold Review",
        "",
        "## Official Metric",
        f"- source: {audit['official_stg_metric']['source']}",
        f"- record score: {audit['official_stg_metric']['record_score']}",
        f"- reported thresholds: {audit['official_stg_metric']['reported_metrics']}",
        "",
        "## Freeze",
        "H4-v1 uses the official mean IoU computation and freezes `TRUE_SPATIAL_SUPPORT` as mean IoU >= 0.5. `SPURIOUS_SPATIAL_SUPPORT` is frozen as mean IoU == 0.",
        "",
        "## Note",
        "The official evaluator still reports several thresholds and does not designate a single canonical H4 label threshold. The H4-v1 threshold above is a pre-inference protocol choice, not a post-hoc result-dependent choice.",
        "",
        "## Current Status",
        "READY_FOR_H4_PREFLIGHT.",
    ]
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
