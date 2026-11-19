"""Read-only human-review preparation for frozen oracle-feasibility results.

This module opens spatial GT only to render review overlays.  It never imports
the inference backend, opens a cache, changes a frozen oracle result, or creates
a certificate.  Human decisions are intentionally left blank.
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
import tempfile
import textwrap
from typing import Any

from PIL import Image, ImageDraw

from relive.spatial import validate_region

from .common import SpatialReliabilityError, sha256_path, write_once
from .engine import oracle_classification


ORACLE_REVIEW_FORMAT = "relive-spatial-oracle-feasibility-human-review-v1"
SEMANTIC_STATUSES = frozenset({"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"})
CLAIM_SCOPE_VALUES = [
    "SINGLE_FRAME_LOCAL",
    "SINGLE_FRAME_RELATIONAL",
    "TEMPORAL_OR_SEQUENCE",
    "NEGATIVE_VISIBILITY",
    "UNBOUND_ENTITY",
    "OTHER_UNSUITABLE",
]
FRAME_SUPPORT_VALUES = ["SUPPORTED", "NOT_SUPPORTED", "AMBIGUOUS"]
OBSERVABILITY_VALUES = [
    "CLEAR", "LOW_CONTRAST", "PARTIALLY_OCCLUDED", "TOO_SMALL", "NOT_VISIBLE", "AMBIGUOUS",
]
GT_RELATION_VALUES = [
    "ALIGNED_SINGLE_REGION_EVIDENCE",
    "PARTIAL_EVIDENCE_REGION",
    "OBJECT_ONLY_NOT_FULL_EVIDENCE",
    "TOO_BROAD",
    "WRONG_REGION",
    "NO_LOCAL_BOX_APPLICABLE",
    "AMBIGUOUS",
]


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
        if (not isinstance(sample_id, str)
                or not re.fullmatch(r"[A-Za-z0-9._:-]+", sample_id)
                or sample_id in result):
            raise SpatialReliabilityError(code)
        result[sample_id] = row
    return result


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


def _normalized_gt(box: Any, size: tuple[int, int]) -> list[float]:
    if (not isinstance(box, list) or len(box) != 4
            or any(type(value) not in {int, float} or not math.isfinite(value) for value in box)):
        raise SpatialReliabilityError("ORACLE_REVIEW_GT_BBOX_INVALID")
    width, height = size
    try:
        return list(validate_region([
            box[0] / width, box[1] / height, box[2] / width, box[3] / height,
        ]))
    except (TypeError, ValueError) as exc:
        raise SpatialReliabilityError("ORACLE_REVIEW_GT_BBOX_INVALID") from exc


def _pixel_box(region: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = validate_region(region)
    return round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)


def _overlay(source: Image.Image, item: dict[str, Any]) -> Image.Image:
    image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line = max(2, min(width, height) // 200)
    draw.rectangle(_pixel_box(item["gt_region_normalized_0_1_xyxy"], width, height),
                   outline=(0, 255, 0), width=line)
    title = f"{item['sample_id']} | dataset GT=green"
    title_width = min(width, max(300, 7 * len(title)))
    draw.rectangle((0, 0, title_width, 18), fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 255))
    return image


def _contact_sheet(items: list[dict[str, Any]], images: dict[str, Image.Image]) -> Image.Image:
    columns, tile_width, image_height, caption_height = 2, 620, 350, 126
    rows = max(1, math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * (image_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(items):
        column, line = index % columns, index // columns
        image = images[item["sample_id"]].copy()
        image.thumbnail((tile_width, image_height), Image.Resampling.LANCZOS)
        x = column * tile_width + (tile_width - image.width) // 2
        y = line * (image_height + caption_height) + (image_height - image.height) // 2
        sheet.paste(image, (x, y))
        base_x = column * tile_width + 5
        base_y = line * (image_height + caption_height) + image_height + 3
        semantic = item["semantic"]
        header = (f"{item['sample_id']} | {item['classification']} | "
                  f"O/K/D={semantic['ORIGINAL']}/{semantic['KEEP_GT']}/{semantic['DROP_GT']}")
        draw.text((base_x, base_y), header, fill=(0, 0, 0))
        for offset, claim_line in enumerate(textwrap.wrap(item["claim_text"], width=88)[:5], start=1):
            draw.text((base_x, base_y + offset * 16), claim_line, fill=(0, 0, 0))
    return sheet


def prepare_oracle_review(*, runtime_manifest: str | Path, oracle_manifest: str | Path,
                          oracle_results: str | Path, output_dir: str | Path) -> dict[str, Any]:
    runtime = _strict_jsonl(runtime_manifest, "ORACLE_REVIEW_RUNTIME_INVALID")
    oracle = _strict_jsonl(oracle_manifest, "ORACLE_REVIEW_ORACLE_INVALID")
    results = _strict_jsonl(oracle_results, "ORACLE_REVIEW_RESULTS_INVALID")
    runtime_by_id = _index(runtime, "ORACLE_REVIEW_RUNTIME_ID_INVALID")
    oracle_by_id = _index(oracle, "ORACLE_REVIEW_ORACLE_ID_INVALID")
    _index(results, "ORACLE_REVIEW_RESULT_ID_INVALID")
    runtime_sha = sha256_path(runtime_manifest)
    oracle_sha = sha256_path(oracle_manifest)

    items: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    overlay_images: dict[str, Image.Image] = {}
    review_rows: list[dict[str, Any]] = []
    visualization_rows: list[dict[str, Any]] = []
    output = Path(output_dir)
    for result in results:
        sample_id = result["sample_id"]
        public, hidden = runtime_by_id.get(sample_id), oracle_by_id.get(sample_id)
        if public is None or hidden is None:
            raise SpatialReliabilityError("ORACLE_REVIEW_BINDING_INVALID")
        bindings = result.get("bindings")
        semantic = result.get("semantic")
        if (hidden.get("source_record_sha256") != public.get("source_record_sha256")
                or result.get("claim_id") != public.get("claim_id")
                or not isinstance(bindings, dict)
                or bindings.get("runtime_manifest_sha256") != runtime_sha
                or bindings.get("oracle_manifest_sha256") != oracle_sha
                or not isinstance(semantic, dict)
                or set(semantic) != {"ORIGINAL", "KEEP_GT", "DROP_GT"}
                or any(value not in SEMANTIC_STATUSES for value in semantic.values())
                or result.get("classification") != oracle_classification(
                    semantic["ORIGINAL"], semantic["KEEP_GT"], semantic["DROP_GT"]
                )):
            raise SpatialReliabilityError("ORACLE_REVIEW_BINDING_INVALID")
        frame_path = Path(public.get("frame_path", ""))
        if not frame_path.is_file():
            raise SpatialReliabilityError("ORACLE_REVIEW_FRAME_MISSING")
        frame_sha = sha256_path(frame_path)
        if public.get("frame_sha256") not in {None, frame_sha} or bindings.get("source_frame_sha256") != public.get("frame_sha256"):
            raise SpatialReliabilityError("ORACLE_REVIEW_FRAME_HASH_MISMATCH")
        source_hashes[sample_id] = frame_sha
        with Image.open(frame_path) as opened:
            source = opened.convert("RGB")
        gt_region = _normalized_gt(hidden.get("gt_bbox"), source.size)
        if result.get("gt_region_normalized_0_1_xyxy") != gt_region:
            raise SpatialReliabilityError("ORACLE_REVIEW_GT_BINDING_INVALID")
        item = {
            "format": ORACLE_REVIEW_FORMAT + "-item",
            "sample_id": sample_id,
            "claim_id": public["claim_id"],
            "claim_text": public["claim_text"],
            "dataset_name": public.get("dataset_name"),
            "video_id": public.get("video_id"),
            "frame_path": str(frame_path),
            "frame_sha256": frame_sha,
            "gt_region_normalized_0_1_xyxy": gt_region,
            "gt_bbox_pixel_xyxy": hidden["gt_bbox"],
            "classification": result["classification"],
            "semantic": semantic,
        }
        overlay = _overlay(source, item)
        overlay_body = _png_bytes(overlay)
        relative = Path("overlays") / f"{sample_id}.png"
        _write_bytes_once(output / relative, overlay_body)
        item["overlay_path"] = relative.as_posix()
        item["overlay_sha256"] = hashlib.sha256(overlay_body).hexdigest()
        items.append(item)
        overlay_images[sample_id] = overlay
        visualization_rows.append({
            "format": ORACLE_REVIEW_FORMAT + "-visualization",
            "sample_id": sample_id,
            "overlay_path": relative.as_posix(),
            "overlay_sha256": item["overlay_sha256"],
            "source_frame_sha256": frame_sha,
        })
        review_rows.append({
            "format": ORACLE_REVIEW_FORMAT,
            "sample_id": sample_id,
            "claim_id": public["claim_id"],
            "claim_text": public["claim_text"],
            "classification": result["classification"],
            "semantic": semantic,
            "overlay_path": relative.as_posix(),
            "allowed_values": {
                "claim_scope": CLAIM_SCOPE_VALUES,
                "frame_claim_support": FRAME_SUPPORT_VALUES,
                "observability": OBSERVABILITY_VALUES,
                "gt_box_evidence_relation": GT_RELATION_VALUES,
            },
            "human_review": {
                "claim_scope": "",
                "frame_claim_support": "",
                "observability": "",
                "gt_box_evidence_relation": "",
                "reviewer_id": "",
                "rationale": "",
            },
        })

    contact_body = _png_bytes(_contact_sheet(items, overlay_images))
    _write_bytes_once(output / "oracle_review_contact_sheet_v1.png", contact_body)
    write_once(output / "oracle_review_items_v1.jsonl", items, jsonl=True)
    write_once(output / "oracle_review_visualization_manifest_v1.jsonl", visualization_rows, jsonl=True)
    write_once(output / "oracle_human_review_template_v1.jsonl", review_rows, jsonl=True)
    source_unchanged = all(
        sha256_path(runtime_by_id[sample_id]["frame_path"]) == digest
        for sample_id, digest in source_hashes.items()
    )
    manifest = {
        "format": ORACLE_REVIEW_FORMAT,
        "status": "PASS",
        "cohort_count": len(items),
        "classification_counts": dict(sorted(Counter(item["classification"] for item in items).items())),
        "runtime_manifest_sha256": runtime_sha,
        "oracle_manifest_sha256": oracle_sha,
        "oracle_results_sha256": sha256_path(oracle_results),
        "artifact_sha256": {
            "oracle_review_items_v1.jsonl": sha256_path(output / "oracle_review_items_v1.jsonl"),
            "oracle_review_visualization_manifest_v1.jsonl": sha256_path(output / "oracle_review_visualization_manifest_v1.jsonl"),
            "oracle_human_review_template_v1.jsonl": sha256_path(output / "oracle_human_review_template_v1.jsonl"),
            "oracle_review_contact_sheet_v1.png": hashlib.sha256(contact_body).hexdigest(),
        },
        "human_decisions_filled": 0,
        "source_frames_unchanged": source_unchanged,
        "gt_used": True,
        "model_calls_made": 0,
        "cache_opened": False,
        "cache_writes": 0,
        "certificate_created": False,
        "new_verified_count": 0,
    }
    write_once(output / "oracle_review_manifest_v1.json", manifest)
    report = (
        "# Oracle feasibility human review\n\n"
        "The green rectangle is the dataset GT region. Review the original frame and claim; do not "
        "reinterpret the frozen Oracle verdict. Fill only `human_review` in "
        "`oracle_human_review_template_v1.jsonl`.\n\n"
        "This is a GT-only, read-only review artifact. Model calls/cache writes/certificates/new "
        "VERIFIED: 0/0/0/0.\n"
    ).encode("utf-8")
    _write_bytes_once(output / "README.md", report)
    return manifest
