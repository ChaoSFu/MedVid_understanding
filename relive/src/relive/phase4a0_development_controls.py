"""Zero-model preparation of new Phase 4A-0 development controls.

This is deliberately separate from ``fresh_public_runtime``: a development
control cohort may contain more than one claim from the same public source
record, whereas a prospective runtime must contain unique source records.
It binds the human-selected public frame and claim to a SHA-256 before any
ROI, intervention image, reviewer packet, backend, verifier, or certificate
is involved.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .data.medvidu import load_public_records, audit_frame_mapping
from .phase4a0 import claim_sha256
from .spatial import validate_region
from .storage.artifacts import canonical_json


SELECTION_FORMAT = "relive-phase4a0-development-control-selection-v1"
ROI_TEMPLATE_FORMAT = "relive-phase4a0-development-roi-freeze-template-v1"
TARGET_OVERRIDE_FORMAT = "relive-phase4a0-development-target-roi-overrides-v1"
FORMAT = "relive-phase4a0-development-control-preparation-v1"
FORBIDDEN = {"reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt",
             "struc_info", "RC_info", "evaluation_artifacts", "roi", "bbox", "mask"}


class DevelopmentControlError(ValueError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_selection(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentControlError(f"unreadable development-control selection JSONL: {path}") from exc
    expected = {"format", "development_case_id", "source_record_index", "public_record_sha256",
                "claim_id", "claim_text", "frozen_frame_orders", "development_control", "gt_used"}
    if not rows or any(not isinstance(row, dict) or set(row) != expected for row in rows):
        raise DevelopmentControlError("DEVELOPMENT_CONTROL_SELECTION_SCHEMA_INVALID")
    ids: set[str] = set(); claims: set[str] = set(); locations: set[tuple[int, int]] = set()
    for row in rows:
        if row["format"] != SELECTION_FORMAT or row["development_control"] is not True or row["gt_used"] is not False:
            raise DevelopmentControlError("DEVELOPMENT_CONTROL_SELECTION_PROVENANCE_INVALID")
        if (not isinstance(row["development_case_id"], str) or not row["development_case_id"].strip()
                or row["development_case_id"] in ids):
            raise DevelopmentControlError("DEVELOPMENT_CASE_ID_INVALID_OR_DUPLICATE")
        ids.add(row["development_case_id"])
        if not isinstance(row["claim_id"], str) or not row["claim_id"].strip() or row["claim_id"] in claims:
            raise DevelopmentControlError("DEVELOPMENT_CLAIM_ID_INVALID_OR_DUPLICATE")
        claims.add(row["claim_id"])
        if not isinstance(row["claim_text"], str) or not row["claim_text"].strip():
            raise DevelopmentControlError("DEVELOPMENT_CLAIM_TEXT_INVALID")
        if type(row["source_record_index"]) is not int or row["source_record_index"] < 0:
            raise DevelopmentControlError("SOURCE_RECORD_INDEX_INVALID")
        if not isinstance(row["public_record_sha256"], str) or len(row["public_record_sha256"]) != 64:
            raise DevelopmentControlError("PUBLIC_RECORD_SHA256_INVALID")
        orders = row["frozen_frame_orders"]
        if (not isinstance(orders, list) or len(orders) != 1 or type(orders[0]) is not int or orders[0] < 0):
            raise DevelopmentControlError("DEVELOPMENT_CONTROL_REQUIRES_EXACTLY_ONE_PUBLIC_FRAME_ORDER")
        location = (row["source_record_index"], orders[0])
        if location in locations:
            raise DevelopmentControlError("DUPLICATE_SOURCE_FRAME_SELECTION")
        locations.add(location)
    return rows


def _write_new(path: Path, body: bytes) -> None:
    if path.exists():
        raise DevelopmentControlError(f"refusing to overwrite frozen artifact: {path}")
    path.write_bytes(body)


def _render_selected_frame(*, image_path: Path, destination: Path, case_id: str, order: int) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    try:
        title_height = 34
        canvas = Image.new("RGB", (image.width, image.height + title_height), "white")
        canvas.paste(image, (0, title_height))
        ImageDraw.Draw(canvas).text((4, 8), f"{case_id} | public frame order={order} | ROI not yet frozen",
                                    fill="black", font=ImageFont.load_default())
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="PNG")
    finally:
        image.close()


def prepare_development_controls(*, selection_path: Path, source_json: Path, frame_root: Path,
                                 source_prefix: str, output_dir: Path) -> dict[str, Any]:
    """Bind a user-authored selection to public frames and emit an ROI worksheet.

    No model or certificate components are imported or called.  Repeated source
    records are valid because every development case has an independent claim
    and single frozen frame.
    """
    if output_dir.exists():
        raise DevelopmentControlError("OUTPUT_DIRECTORY_MUST_NOT_EXIST")
    selections = _read_selection(selection_path)
    records, source_json_sha256 = load_public_records(source_json)
    by_index = {record.source_record_index: record for record in records}
    selected_records = []
    for row in selections:
        record = by_index.get(row["source_record_index"])
        if record is None or record.public_record_sha256 != row["public_record_sha256"]:
            raise DevelopmentControlError("SELECTION_DOES_NOT_MATCH_PUBLIC_SOURCE_PROJECTION")
        if row["frozen_frame_orders"][0] >= len(record.video_paths):
            raise DevelopmentControlError("FROZEN_FRAME_ORDER_OUT_OF_RANGE")
        selected_records.append(record)
    mapping_audit, mapper = audit_frame_mapping(selected_records, source_prefix, frame_root)
    if mapping_audit["status"] != "PASS":
        raise DevelopmentControlError("PUBLIC_FRAME_MAPPING_AUDIT_FAILED")

    output_dir.mkdir(parents=True)
    prepared: list[dict[str, Any]] = []
    for selection, record in zip(selections, selected_records):
        order = selection["frozen_frame_orders"][0]
        mapped = mapper.map(record.video_paths[order])
        if mapped.status != "PASS" or mapped.resolved_path is None:
            raise DevelopmentControlError("FROZEN_PUBLIC_FRAME_UNAVAILABLE")
        source_frame = Path(mapped.resolved_path)
        selected_frame = output_dir / "selected_public_frames" / f"{selection['development_case_id']}.png"
        _render_selected_frame(image_path=source_frame, destination=selected_frame,
                               case_id=selection["development_case_id"], order=order)
        prepared.append({
            "format": ROI_TEMPLATE_FORMAT,
            "development_case_id": selection["development_case_id"],
            "source_claim_id": selection["claim_id"], "claim_text": selection["claim_text"],
            "claim_sha256": claim_sha256(selection["claim_text"]),
            "source_record_index": selection["source_record_index"],
            "public_record_sha256": selection["public_record_sha256"], "sample_id": record.sample_id,
            "dataset_name": record.dataset_name, "frame_orders": [order],
            "frame_ids": [f"{record.sample_id}:frame:{order:06d}"],
            "frame_paths": [str(source_frame)], "frame_sha256": [_sha_file(source_frame)],
            "selected_public_frame_preview": str(selected_frame),
            "target_roi_normalized_0_1_xyxy": None,
            "matched_control_roi_normalized_0_1_xyxy": None,
            "intervention": {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1",
                               "parameters": {"fill_rgb": [127, 127, 127]}},
            "selection_status": "AWAITING_HUMAN_ROI_AND_MATCHED_CONTROL",
            "development_control": True, "gt_used": False,
        })
    body = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in prepared)
    _write_new(output_dir / "phase4a0_development_roi_freeze.template.jsonl", body)
    selection_bytes = selection_path.read_bytes()
    report = {
        "format": FORMAT, "status": "PASS", "mode": "prepare_development_controls",
        "model_calls_made": 0, "backend_loaded": False, "spatial_proposer_called": False,
        "verifier_called": False, "certificate_created": False, "new_verified_count": 0,
        "gt_used": False, "candidate_count": len(prepared),
        "selection_input": str(selection_path), "selection_input_sha256": _sha_bytes(selection_bytes),
        "source_json": str(source_json), "source_json_sha256": source_json_sha256,
        "roi_freeze_template": str(output_dir / "phase4a0_development_roi_freeze.template.jsonl"),
        "roi_freeze_template_sha256": _sha_bytes(body), "selected_frame_directory": str(output_dir / "selected_public_frames"),
        "public_frame_mapping_audit": mapping_audit,
        "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt",
                                          "struc_info", "RC_info", "evaluation_artifacts"],
    }
    _write_new(output_dir / "phase4a0_development_control_preparation_report.json",
               (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentControlError(f"unreadable {label} JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise DevelopmentControlError(f"{label.upper().replace(' ', '_')}_JSONL_INVALID")
    return rows


def apply_target_roi_overrides(*, roi_template_path: Path, target_roi_overrides_path: Path,
                               output_path: Path) -> dict[str, Any]:
    """Make a new ROI worksheet with user-authored targets, never mutating input.

    Matched controls deliberately remain unset.  Selecting them is a separate
    human decision because a convenient automatic placement could itself carry
    salient evidence for the claim.
    """
    templates = _read_jsonl_objects(roi_template_path, "ROI template")
    overrides = _read_jsonl_objects(target_roi_overrides_path, "target ROI override")
    template_ids: set[str] = set()
    for row in templates:
        if row.get("format") != ROI_TEMPLATE_FORMAT or row.get("selection_status") != "AWAITING_HUMAN_ROI_AND_MATCHED_CONTROL":
            raise DevelopmentControlError("ROI_TEMPLATE_NOT_AWAITING_HUMAN_SELECTION")
        case_id = row.get("development_case_id")
        if not isinstance(case_id, str) or case_id in template_ids:
            raise DevelopmentControlError("ROI_TEMPLATE_CASE_ID_INVALID_OR_DUPLICATE")
        if row.get("target_roi_normalized_0_1_xyxy") is not None or row.get("matched_control_roi_normalized_0_1_xyxy") is not None:
            raise DevelopmentControlError("ROI_TEMPLATE_ALREADY_CONTAINS_HUMAN_REGIONS")
        template_ids.add(case_id)
    by_id: dict[str, list[float]] = {}
    expected = {"format", "development_case_id", "target_roi_normalized_0_1_xyxy"}
    for row in overrides:
        if set(row) != expected or row.get("format") != TARGET_OVERRIDE_FORMAT:
            raise DevelopmentControlError("TARGET_ROI_OVERRIDE_SCHEMA_INVALID")
        case_id = row["development_case_id"]
        if not isinstance(case_id, str) or case_id not in template_ids or case_id in by_id:
            raise DevelopmentControlError("TARGET_ROI_OVERRIDE_CASE_ID_INVALID_OR_DUPLICATE")
        by_id[case_id] = list(validate_region(row["target_roi_normalized_0_1_xyxy"]))
    if set(by_id) != template_ids:
        raise DevelopmentControlError("TARGET_ROI_OVERRIDE_MUST_COVER_EVERY_TEMPLATE_CASE")
    if output_path.exists():
        raise DevelopmentControlError("REFUSING_TO_OVERWRITE_TARGET_ROI_WORKSHEET")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = []
    for row in templates:
        updated = dict(row)
        updated["target_roi_normalized_0_1_xyxy"] = by_id[row["development_case_id"]]
        updated["selection_status"] = "AWAITING_HUMAN_MATCHED_CONTROL"
        merged.append(updated)
    body = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in merged)
    output_path.write_bytes(body)
    return {"format": FORMAT, "status": "PASS", "mode": "apply_target_roi_overrides",
            "model_calls_made": 0, "backend_loaded": False, "certificate_created": False,
            "new_verified_count": 0, "gt_used": False, "case_count": len(merged),
            "roi_template": str(roi_template_path), "roi_template_sha256": _sha_file(roi_template_path),
            "target_roi_overrides": str(target_roi_overrides_path),
            "target_roi_overrides_sha256": _sha_file(target_roi_overrides_path),
            "output": str(output_path), "output_sha256": _sha_bytes(body),
            "matched_control_status": "AWAITING_HUMAN_SELECTION"}
