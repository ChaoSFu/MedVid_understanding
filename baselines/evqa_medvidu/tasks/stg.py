from __future__ import annotations

from typing import Any

from baselines.evqa_medvidu.raw_output import parse_temporal_segments
from baselines.evqa_medvidu.spatial.mask_frame_alignment import build_mask_frame_alignment
from baselines.evqa_medvidu.spatial.mask_to_bbox import mask_to_bbox_audit, masklet_to_bboxes

from .base import MedVidUTaskAdapter, SchemaMismatch


class STGAdapter(MedVidUTaskAdapter):
    task = "stg"

    def build_model_input(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": "st_evidence_direct",
            "question": sample["human_question"],
            "ordered_frame_paths": sample["ordered_frame_paths"],
            "selected_frame_paths": sample["model_sampling"]["selected_frame_paths"],
        }

    def parse_evqa_output(self, raw: dict[str, Any]) -> dict[str, Any]:
        masklets = raw.get("raw_masklets") or []
        if len(masklets) > 1:
            raise SchemaMismatch("MedVidU STG official evaluator accepts one bbox per timestamp; multiple E-VQA masklets require a frozen GT-free merge/selection rule.")
        bboxes: list[list[float] | None] = []
        alignment: list[dict[str, Any]] = []
        bbox_audit: dict[str, Any] | None = None
        if len(masklets) == 1:
            bboxes = masklet_to_bboxes(masklets[0]["mask"])
            alignment = build_mask_frame_alignment(raw["manifest_row"], len(bboxes))
            bbox_audit = mask_to_bbox_audit(raw["sample_id"], bboxes)
        return {
            "semantic_answer": raw.get("semantic_answer"),
            "temporal_segments": parse_temporal_segments(raw.get("raw_textual_response")),
            "masklet_count": len(masklets),
            "bboxes_by_mask_frame": bboxes,
            "mask_frame_alignment": alignment,
            "mask_to_bbox_audit": bbox_audit,
            "parse_valid": bool(bboxes),
        }

    def to_medvidu_prediction(self, sample: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        parts = []
        bboxes = parsed.get("bboxes_by_mask_frame") or []
        target_alignment = sample.get("stg_target_alignment") or []
        missing_target_timestamps: list[float] = []
        for target in target_alignment:
            logical_index = int(target["logical_medvidu_frame_index"])
            bbox = bboxes[logical_index] if logical_index < len(bboxes) else None
            if bbox is None:
                missing_target_timestamps.append(float(target["target_timestamp"]))
                continue
            t = float(target["target_timestamp"])
            parts.append(f"{t:.1f} seconds: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        return {
            "sample_id": sample["sample_id"],
            "qa_id": sample["qa_id"],
            "task": "stg",
            "qa_type": "stg",
            "dataset_name": sample.get("dataset_name"),
            "data_source": sample.get("data_source"),
            "metadata": sample.get("clip_metadata_gt_free", {}),
            "answer": " ".join(parts),
            "stg_target_schedule": sample.get("stg_target_schedule"),
            "stg_target_alignment": target_alignment,
            "missing_target_timestamps": missing_target_timestamps,
            "raw_temporal_segments": parsed.get("temporal_segments", []),
            "gt_information_available_to_model": False,
        }

    def validate_prediction(self, prediction: dict[str, Any]) -> None:
        if prediction.get("task") != "stg":
            raise ValueError("not an STG prediction")
        if "answer" not in prediction:
            raise ValueError("missing answer")
