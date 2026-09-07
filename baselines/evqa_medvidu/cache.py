from __future__ import annotations

from typing import Any

from .config import CACHE_VERSION
from .io_utils import sha256_json


def build_cache_key(
    row: dict[str, Any],
    model_fingerprint: dict[str, Any],
    prompt: str,
    adapter_payload: dict[str, Any],
    visual_input_config: dict[str, Any] | None = None,
    inference_config: dict[str, Any] | None = None,
) -> str:
    return sha256_json(
        {
            "cache_version": CACHE_VERSION,
            "sample_id": row["sample_id"],
            "task": row["task"],
            "qa_id": row.get("qa_id"),
            "question_hash": row["human_question_sha256"],
            "ordered_benchmark_frame_hash": row["ordered_benchmark_frame_hash"],
            "selected_frame_hash": row["selected_frame_hash"],
            "timestamp_hash": row["timestamp_hash"],
            "stg_target_schedule_hash": row.get("stg_target_schedule_hash"),
            "provided_region_hash": row.get("provided_region_hash"),
            "model_fingerprint": model_fingerprint,
            "prompt": prompt,
            "prompt_hash": sha256_json(prompt),
            "sampling_policy": row["model_sampling"]["policy"],
            "adapter_payload": adapter_payload,
            "visual_input_config": visual_input_config or {},
            "inference_config": inference_config or {},
        }
    )
