"""Post-hoc GT evaluation. Runtime modules never import this module."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from relive.spatial import validate_region

from .common import SpatialReliabilityError, sha256_path, write_once


EVALUATION_FORMAT = "relive-spatial-posthoc-gt-evaluation-v1"


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialReliabilityError(f"INVALID_JSONL:{path}") from exc


def iou(a: list[float], b: list[float]) -> float:
    a, b = validate_region(a), validate_region(b)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union else 0.0


def _stats(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "mean_iou": mean(values) if values else None,
            "median_iou": median(values) if values else None,
            "recall_at_0_3": sum(value >= 0.3 for value in values) / len(values) if values else None,
            "recall_at_0_5": sum(value >= 0.5 for value in values) / len(values) if values else None}


def evaluate(*, runtime_results: str | Path, oracle_manifest: str | Path,
             runtime_manifest: str | Path, output_dir: str | Path) -> dict[str, Any]:
    results, oracle, runtime = _jsonl(runtime_results), _jsonl(oracle_manifest), _jsonl(runtime_manifest)
    oracle_by_id, runtime_by_id = ({row.get("sample_id"): row for row in rows} for rows in (oracle, runtime))
    evaluated = []
    for result in results:
        sample_id = result.get("sample_id")
        hidden, public = oracle_by_id.get(sample_id), runtime_by_id.get(sample_id)
        if not hidden or not public or hidden.get("source_record_sha256") != public.get("source_record_sha256"):
            raise SpatialReliabilityError("POSTHOC_BINDING_MISMATCH")
        with Image.open(public["frame_path"]) as image:
            width, height = image.size
        raw = hidden.get("gt_bbox")
        gt = list(validate_region([raw[0] / width, raw[1] / height, raw[2] / width, raw[3] / height]))
        regions = result.get("regions", [])
        if not regions:
            continue
        r0_rows = [item for item in result.get("trace", []) if item.get("stage") == "R0_PROPOSAL"]
        raw_r0 = r0_rows[0].get("result", {}).get("bbox_normalized_0_1000") if len(r0_rows) == 1 else None
        if not isinstance(raw_r0, list) or len(raw_r0) != 4:
            raise SpatialReliabilityError("R0_PROPOSAL_BINDING_MISSING")
        r0 = [value / 1000 for value in raw_r0]
        row = {"sample_id": sample_id, "status": result.get("status"), "r0_iou": iou(r0, gt),
               "r1_iou": None, "evidence_region_count": len(regions), "gt_region": gt}
        repair = any(item.get("stage") == "R1_PROPOSAL" and item.get("result", {}).get("status") == "PROPOSED"
                     for item in result.get("trace", []))
        if repair:
            row["r1_iou"] = iou(regions[0], gt)
        if len(regions) == 2:
            row["evidence_set_member_ious"] = [iou(region, gt) for region in regions]
        evaluated.append(row)
    all_iou = [row["r1_iou"] if row["r1_iou"] is not None else row["r0_iou"] for row in evaluated]
    admitted_statuses = {"DEPENDENCY_VERIFIED", "DEPENDENCY_SUPPORTED_NO_CONTROL", "EVIDENCE_SET_VERIFIED"}
    admitted = [value for row, value in zip(evaluated, all_iou) if row["status"] in admitted_statuses]
    supported = [value for row, value in zip(evaluated, all_iou) if row["status"] in {"SUPPORTED_EVIDENCE", *admitted_statuses}]
    repaired_before = [row["r0_iou"] for row in evaluated if row["r1_iou"] is not None]
    repaired_after = [row["r1_iou"] for row in evaluated if row["r1_iou"] is not None]
    summary = {
        "format": EVALUATION_FORMAT, "status": "PASS", "runtime_results_sha256": sha256_path(runtime_results),
        "runtime_manifest_sha256": sha256_path(runtime_manifest), "oracle_manifest_sha256": sha256_path(oracle_manifest),
        "proposal_quality_all": _stats(all_iou), "proposal_quality_supported_evidence": _stats(supported),
        "proposal_quality_dependency_admitted": _stats(admitted),
        "certificate_conditioned_recall_at_0_5": _stats(admitted)["recall_at_0_5"],
        "adaptation": {"repair_count": len(repaired_after), "mean_iou_before_repair": mean(repaired_before) if repaired_before else None,
                       "mean_iou_after_repair": mean(repaired_after) if repaired_after else None},
        "gt_used": True, "model_calls_made": 0, "cache_writes": 0, "certificate_created": False,
        "new_verified_count": 0,
    }
    output = Path(output_dir)
    write_once(output / "gt_alignment_rows.jsonl", evaluated, jsonl=True)
    write_once(output / "gt_alignment_summary.json", summary)
    return summary
