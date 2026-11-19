"""Read-only, status-stratified post-hoc GT evaluation v2.

This module never participates in runtime inference or certificate admission.
It reads already-frozen results, creates descriptive GT alignment metrics and
visualizations, and writes into a separate immutable output directory.
"""
from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, median
import tempfile
from typing import Any

from PIL import Image, ImageDraw

from relive.spatial import validate_region

from .common import SpatialReliabilityError, sha256_path, write_once
from .posthoc import iou


EVALUATION_V2_FORMAT = "relive-spatial-posthoc-gt-evaluation-v2"
STRICT_DEPENDENCY_STATUSES = frozenset({"DEPENDENCY_VERIFIED"})
SUPPORTED_WITHOUT_CONTROL_STATUSES = frozenset({"DEPENDENCY_SUPPORTED_NO_CONTROL"})
DEPENDENCY_ADMITTED_DESCRIPTIVE_STATUSES = frozenset({
    "DEPENDENCY_VERIFIED", "DEPENDENCY_SUPPORTED_NO_CONTROL", "EVIDENCE_SET_VERIFIED",
})
NAMED_STATUS_GROUPS = {
    "strict_dependency_verified": STRICT_DEPENDENCY_STATUSES,
    "supported_without_control": SUPPORTED_WITHOUT_CONTROL_STATUSES,
    "dependency_unresolved": frozenset({"DEPENDENCY_UNRESOLVED"}),
    "base_unsupported": frozenset({"BASE_UNSUPPORTED"}),
    "intervention_unstable": frozenset({"INTERVENTION_UNSTABLE"}),
    "refinement_no_op": frozenset({"REFINEMENT_NO_OP"}),
}


def _strict_jsonl(path: str | Path, code: str) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(value)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    try:
        lines = Path(path).read_bytes().decode("utf-8").splitlines()
        rows = [json.loads(line, parse_constant=reject_constant,
                           object_pairs_hook=reject_duplicates)
                for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SpatialReliabilityError(code) from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise SpatialReliabilityError(code)
    return rows


def _index(rows: list[dict[str, Any]], code: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if (not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]+", sample_id)
                or sample_id in result):
            raise SpatialReliabilityError(code)
        result[sample_id] = row
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["effective_iou"] for row in rows if row.get("effective_iou") is not None]
    return {
        "cohort_count": len(rows),
        "localized_count": len(values),
        "mean_iou": mean(values) if values else None,
        "median_iou": median(values) if values else None,
        "recall_at_0_3": sum(value >= 0.3 for value in values) / len(values) if values else None,
        "recall_at_0_5": sum(value >= 0.5 for value in values) / len(values) if values else None,
    }


def _write_bytes_once(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise SpatialReliabilityError(f"IMMUTABLE_ARTIFACT_EXISTS:{path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def _pixel_box(region: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = validate_region(region)
    return (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))


def _overlay(source: Image.Image, row: dict[str, Any]) -> Image.Image:
    image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line = max(2, min(width, height) // 250)
    draw.rectangle(_pixel_box(row["gt_region"], width, height), outline=(0, 255, 0), width=line)
    if row.get("r0_region") is not None and row.get("selected_region_source") == "R1":
        draw.rectangle(_pixel_box(row["r0_region"], width, height), outline=(255, 165, 0), width=line)
    for index, region in enumerate(row.get("proposal_regions", [])):
        color = (0, 180, 255) if index == 0 else (255, 0, 255)
        draw.rectangle(_pixel_box(region, width, height), outline=color, width=line)
    effective = row.get("effective_iou")
    label = (f"{row['sample_id']} | {row['status']} | "
             f"IoU={'NA' if effective is None else f'{effective:.3f}'}")
    legend = "GT=green proposal=cyan R0-before-repair=orange residual=magenta"
    box_width = min(width, max(360, 7 * max(len(label), len(legend))))
    draw.rectangle((0, 0, box_width, 31), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255))
    draw.text((4, 17), legend, fill=(255, 255, 255))
    return image


def _contact_sheet(overlays: list[tuple[dict[str, Any], Image.Image]]) -> Image.Image:
    columns, tile_width, tile_height, caption_height = 3, 420, 260, 34
    rows = max(1, math.ceil(len(overlays) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (row, image) in enumerate(overlays):
        column, line = index % columns, index // columns
        thumbnail = image.copy()
        thumbnail.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = column * tile_width + (tile_width - thumbnail.width) // 2
        y = line * (tile_height + caption_height) + (tile_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
        effective = row.get("effective_iou")
        caption = f"{row['sample_id']} | {row['status']} | IoU={'NA' if effective is None else f'{effective:.3f}'}"
        draw.text((column * tile_width + 3, line * (tile_height + caption_height) + tile_height + 3),
                  caption, fill=(0, 0, 0))
    return sheet


def _source_binding(result: dict[str, Any], public: dict[str, Any], hidden: dict[str, Any]) -> None:
    if (hidden.get("source_record_sha256") != public.get("source_record_sha256")
            or result.get("source_frame_sha256") != public.get("frame_sha256")
            or not isinstance(result.get("status"), str)
            or result.get("gt_used") is not False
            or result.get("certificate_created") is not False
            or result.get("new_verified_count") != 0):
        raise SpatialReliabilityError("POSTHOC_V2_BINDING_INVALID")


def evaluate_v2(*, runtime_results: str | Path, oracle_manifest: str | Path,
                runtime_manifest: str | Path, output_dir: str | Path) -> dict[str, Any]:
    results = _strict_jsonl(runtime_results, "POSTHOC_V2_RESULTS_INVALID")
    runtime = _strict_jsonl(runtime_manifest, "POSTHOC_V2_RUNTIME_INVALID")
    oracle = _strict_jsonl(oracle_manifest, "POSTHOC_V2_ORACLE_INVALID")
    result_by_id = _index(results, "POSTHOC_V2_RESULT_ID_INVALID")
    runtime_by_id = _index(runtime, "POSTHOC_V2_RUNTIME_ID_INVALID")
    oracle_by_id = _index(oracle, "POSTHOC_V2_ORACLE_ID_INVALID")
    if len(result_by_id) != len(results):
        raise SpatialReliabilityError("POSTHOC_V2_RESULT_ID_INVALID")

    evaluated: list[dict[str, Any]] = []
    source_hashes_before: dict[str, str] = {}
    images: dict[str, Image.Image] = {}
    for result in results:
        sample_id = result["sample_id"]
        public, hidden = runtime_by_id.get(sample_id), oracle_by_id.get(sample_id)
        if public is None or hidden is None:
            raise SpatialReliabilityError("POSTHOC_V2_BINDING_INVALID")
        _source_binding(result, public, hidden)
        frame_path = Path(public.get("frame_path", ""))
        if not frame_path.is_file():
            raise SpatialReliabilityError("POSTHOC_V2_FRAME_MISSING")
        frame_sha = sha256_path(frame_path)
        declared_sha = public.get("frame_sha256")
        if declared_sha is not None and declared_sha != frame_sha:
            raise SpatialReliabilityError("POSTHOC_V2_FRAME_HASH_MISMATCH")
        source_hashes_before[sample_id] = frame_sha
        with Image.open(frame_path) as opened:
            source = opened.convert("RGB")
        width, height = source.size
        raw_gt = hidden.get("gt_bbox")
        if (not isinstance(raw_gt, list) or len(raw_gt) != 4
                or any(type(value) not in {int, float} or not math.isfinite(value) for value in raw_gt)):
            raise SpatialReliabilityError("POSTHOC_V2_GT_BBOX_INVALID")
        gt = list(validate_region([
            raw_gt[0] / width, raw_gt[1] / height, raw_gt[2] / width, raw_gt[3] / height,
        ]))
        raw_regions = result.get("regions")
        if not isinstance(raw_regions, list) or len(raw_regions) > 2:
            raise SpatialReliabilityError("POSTHOC_V2_REGION_CONTRACT_INVALID")
        try:
            regions = [list(validate_region(region)) for region in raw_regions]
        except (TypeError, ValueError) as exc:
            raise SpatialReliabilityError("POSTHOC_V2_REGION_CONTRACT_INVALID") from exc
        r0_entries = [entry for entry in result.get("trace", [])
                      if isinstance(entry, dict) and entry.get("stage") == "R0_PROPOSAL"]
        r0_region = None
        if regions:
            raw_r0 = (r0_entries[0].get("result") or {}).get("bbox_normalized_0_1000") \
                if len(r0_entries) == 1 else None
            if (not isinstance(raw_r0, list) or len(raw_r0) != 4
                    or any(type(value) is not int for value in raw_r0)):
                raise SpatialReliabilityError("POSTHOC_V2_R0_BINDING_INVALID")
            r0_region = list(validate_region([value / 1000 for value in raw_r0]))
        repairs = [entry for entry in result.get("trace", [])
                   if isinstance(entry, dict) and entry.get("stage") == "R1_PROPOSAL"
                   and (entry.get("result") or {}).get("status") == "PROPOSED"]
        if len(repairs) > 1:
            raise SpatialReliabilityError("POSTHOC_V2_REPAIR_BINDING_INVALID")
        selected_source = "R1" if repairs else "R0" if regions else None
        primary = regions[0] if regions else None
        row = {
            "format": EVALUATION_V2_FORMAT + "-row",
            "sample_id": sample_id,
            "status": result.get("status"),
            "frame_sha256": frame_sha,
            "gt_region": gt,
            "r0_region": r0_region,
            "r0_iou": iou(r0_region, gt) if r0_region is not None else None,
            "selected_region_source": selected_source,
            "proposal_regions": regions,
            "evidence_region_count": len(regions),
            "effective_iou": iou(primary, gt) if primary is not None else None,
            "evidence_set_member_ious": [iou(region, gt) for region in regions],
        }
        evaluated.append(row)
        images[sample_id] = source

    output = Path(output_dir)
    visualization_rows: list[dict[str, Any]] = []
    overlay_images: list[tuple[dict[str, Any], Image.Image]] = []
    for row in evaluated:
        overlay = _overlay(images[row["sample_id"]], row)
        body = _png_bytes(overlay)
        relative = Path("overlays") / f"{row['sample_id']}.png"
        _write_bytes_once(output / relative, body)
        visualization_rows.append({
            "format": EVALUATION_V2_FORMAT + "-visualization",
            "sample_id": row["sample_id"],
            "overlay_path": relative.as_posix(),
            "overlay_sha256": hashlib.sha256(body).hexdigest(),
            "source_frame_sha256": row["frame_sha256"],
        })
        overlay_images.append((row, overlay))
    contact_body = _png_bytes(_contact_sheet(overlay_images))
    _write_bytes_once(output / "contact_sheet_v2.png", contact_body)
    write_once(output / "gt_alignment_rows_v2.jsonl", evaluated, jsonl=True)
    write_once(output / "visualization_manifest_v2.jsonl", visualization_rows, jsonl=True)

    status_counts = Counter(row.get("status") for row in evaluated)
    metrics_by_status = {
        status: _metrics([row for row in evaluated if row.get("status") == status])
        for status in sorted(status_counts)
    }
    named_metrics = {
        name: _metrics([row for row in evaluated if row.get("status") in statuses])
        for name, statuses in NAMED_STATUS_GROUPS.items()
    }
    strict_rows = [row for row in evaluated if row.get("status") in STRICT_DEPENDENCY_STATUSES]
    admitted_rows = [row for row in evaluated
                     if row.get("status") in DEPENDENCY_ADMITTED_DESCRIPTIVE_STATUSES]
    repaired_rows = [row for row in evaluated if row.get("selected_region_source") == "R1"]
    summary = {
        "format": EVALUATION_V2_FORMAT,
        "status": "PASS",
        "runtime_results_sha256": sha256_path(runtime_results),
        "runtime_manifest_sha256": sha256_path(runtime_manifest),
        "oracle_manifest_sha256": sha256_path(oracle_manifest),
        "cohort_count": len(evaluated),
        "scientific_status_counts": dict(sorted(status_counts.items())),
        "proposal_quality_all": _metrics(evaluated),
        "metrics_by_scientific_status": metrics_by_status,
        "named_status_metrics": named_metrics,
        "strict_dependency_verified_recall_at_0_5": _metrics(strict_rows)["recall_at_0_5"],
        "dependency_admitted_recall_at_0_5": _metrics(admitted_rows)["recall_at_0_5"],
        "dependency_admitted_is_descriptive_not_certificate": True,
        "certificate_metrics_computed": False,
        "adaptation": {
            "repair_count": len(repaired_rows),
            "mean_iou_before_repair": mean(row["r0_iou"] for row in repaired_rows) if repaired_rows else None,
            "mean_iou_after_repair": mean(row["effective_iou"] for row in repaired_rows) if repaired_rows else None,
        },
        "artifact_sha256": {
            "gt_alignment_rows_v2.jsonl": sha256_path(output / "gt_alignment_rows_v2.jsonl"),
            "visualization_manifest_v2.jsonl": sha256_path(output / "visualization_manifest_v2.jsonl"),
            "contact_sheet_v2.png": hashlib.sha256(contact_body).hexdigest(),
        },
        "source_frames_unchanged": all(
            sha256_path(runtime_by_id[sample_id]["frame_path"]) == value
            for sample_id, value in source_hashes_before.items()
        ),
        "gt_used": True,
        "model_calls_made": 0,
        "cache_writes": 0,
        "certificate_created": False,
        "new_verified_count": 0,
    }
    write_once(output / "gt_alignment_summary_v2.json", summary)
    report = (
        "# ReliVE-Spatial post-hoc GT evaluation v2\n\n"
        f"- Frozen cohort: {len(evaluated)}\n"
        f"- Status counts: `{json.dumps(dict(sorted(status_counts.items())), sort_keys=True)}`\n"
        f"- Strict dependency Recall@0.5: {summary['strict_dependency_verified_recall_at_0_5']}\n"
        f"- Descriptive dependency-admitted Recall@0.5: {summary['dependency_admitted_recall_at_0_5']}\n"
        "- Certificate metrics computed: false\n"
        "- Model calls/cache writes/certificates/new VERIFIED: 0/0/0/0\n\n"
        "The dependency-admitted aggregate is descriptive and includes downgraded no-control "
        "outcomes. It is not a certificate-conditioned metric. GT is used only in this post-hoc "
        "evaluation directory and never enters runtime inference.\n"
    ).encode("utf-8")
    _write_bytes_once(output / "report_v2.md", report)
    return summary
