from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from evidence_stability.cache import stable_hash


H4_PROTOCOL_VERSION = "h4_spatial_v1"
H4_DISCOVERY_SEED = 42
SPATIAL_POINTER_PROMPT_VERSION = "spatial_pointer_v1"
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
        "For H4 16-frame candidate windows, temporal eligibility is needed to avoid mixing wrong-time and wrong-space errors. The H2 frozen frame-overlap rule is compatible with the H4 window framework, but it is not an official STG temporal match criterion. Freeze whether H4-v1 should use the H2 temporal eligibility rule before post-hoc spatial labeling.",
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
        "## Issue",
        "H4 requires a frozen positive criterion for `TRUE_SPATIAL_SUPPORT` and a zero-overlap criterion for `SPURIOUS_SPATIAL_SUPPORT`. The official evaluator reports multiple thresholded metrics and mIoU, but this repository does not contain an explicit single canonical positive threshold for H4 labels.",
        "",
        "## Required Human Freeze",
        "Choose and freeze one positive criterion before H4 post-hoc labels are computed, for example `mean IoU >= 0.5` if that is scientifically intended. Until this is frozen, H4 must not generate TRUE_SPATIAL_SUPPORT / SPURIOUS_SPATIAL_SUPPORT labels or discovery conclusions.",
        "",
        "## Current Status",
        "STOP_NEEDS_SPATIAL_LABEL_THRESHOLD_FREEZE.",
    ]
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
