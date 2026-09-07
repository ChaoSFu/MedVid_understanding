from __future__ import annotations

from typing import Any

from .base import MedVidUTaskAdapter


class RCAdapter(MedVidUTaskAdapter):
    task = "rc"

    def build_model_input(self, sample: dict[str, Any]) -> dict[str, Any]:
        region = sample["provided_region"]
        return {
            "mode": "native_ref_box_conditioned",
            "question": sample["human_question"],
            "ordered_frame_paths": sample["ordered_frame_paths"],
            "selected_frame_paths": sample["model_sampling"]["selected_frame_paths"],
            "provided_region": region,
            "provided_region_is_task_input": True,
        }

    def parse_evqa_output(self, raw: dict[str, Any]) -> dict[str, Any]:
        text = str(raw.get("raw_textual_response") or "").strip()
        parse_valid = bool(text) and "<|" not in text
        return {
            "caption": text if parse_valid else "",
            "parse_valid": parse_valid,
            "parse_status": "OK" if parse_valid else "PARSE_INVALID",
            "provided_region": raw["manifest_row"].get("provided_region"),
        }

    def to_medvidu_prediction(self, sample: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": sample["sample_id"],
            "qa_id": sample["qa_id"],
            "task": "rc",
            "qa_type": sample.get("qa_type"),
            "dataset_name": sample.get("dataset_name"),
            "data_source": sample.get("data_source"),
            "metadata": sample.get("clip_metadata_gt_free", {}),
            "answer": parsed.get("caption", ""),
            "parse_valid": parsed.get("parse_valid", False),
            "parse_status": parsed.get("parse_status"),
            "provided_region": sample.get("provided_region"),
            "provided_region_is_task_input": True,
            "gt_information_available_to_model": False,
        }

    def validate_prediction(self, prediction: dict[str, Any]) -> None:
        if prediction.get("task") != "rc":
            raise ValueError("not an RC prediction")
        if not prediction.get("provided_region_is_task_input"):
            raise ValueError("RC prediction must preserve provided region provenance")
