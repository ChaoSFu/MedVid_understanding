from __future__ import annotations

from typing import Any

from .config import CACHE_VERSION, FRAME_LOADER_VERSION, TIMESTAMP_ADAPTER_VERSION
from .io_utils import sha256_json, sha256_text


def build_cache_key(
    row: dict[str, Any],
    checkpoint_sha256: str,
    prompt_sha256: str,
    generation_config: dict[str, Any],
    model_fingerprint: dict[str, Any] | None = None,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "sample_id": row["sample_id"],
        "ordered_frame_identity_hash": sha256_json(row["logical_frame_identities"]),
        "selected_indices": row["selected_logical_indices"],
        "selected_timestamp_hash": sha256_json(row["selected_local_timestamps"]),
        "question_hash": sha256_text(row["human_question"]),
        "checkpoint_sha256": checkpoint_sha256,
        "prompt_sha256": prompt_sha256,
        "generation_config": generation_config,
        "timestamp_mapper_version": "baselines.timelens_medvidu.temporal_mapper.TemporalMapper",
        "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
        "frame_loader_version": FRAME_LOADER_VERSION,
        "model_fingerprint": model_fingerprint or {},
    }
    return sha256_json(payload)

