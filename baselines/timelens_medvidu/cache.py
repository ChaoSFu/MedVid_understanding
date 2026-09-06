from __future__ import annotations

from typing import Any

from .config import PROMPT_VERSION, TIMESTAMP_ADAPTER_VERSION
from .io_utils import sha256_json, sha256_text


def build_cache_key(
    row: dict[str, Any],
    model_fingerprint: dict[str, Any],
    prompt: str,
    pixel_config: dict[str, Any],
    decoding_config: dict[str, Any],
    timestamp_adapter: str = TIMESTAMP_ADAPTER_VERSION,
) -> str:
    frame_identity = [
        {
            "position": obs["frame_position"],
            "source_frame_index": obs["source_frame_index"],
            "frame_path": obs["frame_path"],
        }
        for obs in row["frame_observations"]
    ]
    timestamp_sequence = [obs["local_time"] for obs in row["frame_observations"]]
    payload = {
        "sample_id": row["sample_id"],
        "ordered_logical_frame_identities": frame_identity,
        "gt_free_timestamp_sequence_hash": sha256_json(timestamp_sequence),
        "question_hash": sha256_text(row["human_question"]),
        "model_fingerprint": model_fingerprint,
        "prompt_hash": sha256_text(prompt),
        "prompt_version": PROMPT_VERSION,
        "pixel_config": pixel_config,
        "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
        "timestamp_adapter": timestamp_adapter,
        "decoding_config": decoding_config,
    }
    return sha256_json(payload)
