from __future__ import annotations

from typing import Any

from baselines.evqa_medvidu.raw_output import format_cvs_components, parse_cvs_components, parse_temporal_segments

from .base import MedVidUTaskAdapter


class CVSAdapter(MedVidUTaskAdapter):
    task = "cvs"

    def build_model_input(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": "semantic_answer_with_optional_evidence",
            "question": sample["human_question"],
            "ordered_frame_paths": sample["ordered_frame_paths"],
            "selected_frame_paths": sample["model_sampling"]["selected_frame_paths"],
        }

    def parse_evqa_output(self, raw: dict[str, Any]) -> dict[str, Any]:
        components = parse_cvs_components(raw.get("raw_textual_response"))
        valid = set(components) == {"two_structures", "cystic_plate", "hepatocystic_triangle"}
        return {
            "components": components,
            "answer": format_cvs_components(components) if valid else "",
            "parse_valid": valid,
            "parse_status": "OK" if valid else "PARSE_INVALID",
            "temporal_segments": parse_temporal_segments(raw.get("raw_textual_response")),
        }

    def to_medvidu_prediction(self, sample: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": sample["sample_id"],
            "qa_id": sample["qa_id"],
            "task": "cvs",
            "qa_type": "cvs_assessment",
            "dataset_name": sample.get("dataset_name"),
            "data_source": sample.get("data_source"),
            "metadata": sample.get("clip_metadata_gt_free", {}),
            "answer": parsed.get("answer", ""),
            "parse_valid": parsed.get("parse_valid", False),
            "parse_status": parsed.get("parse_status"),
            "raw_temporal_segments": parsed.get("temporal_segments", []),
            "gt_information_available_to_model": False,
        }

    def validate_prediction(self, prediction: dict[str, Any]) -> None:
        if prediction.get("task") != "cvs":
            raise ValueError("not a CVS prediction")
        if prediction.get("parse_status") == "PARSE_INVALID":
            return
        parse_cvs_components(prediction.get("answer"))

