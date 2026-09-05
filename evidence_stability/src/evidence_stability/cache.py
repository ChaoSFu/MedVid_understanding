from __future__ import annotations

import hashlib
import json


def stable_hash(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def make_probe_cache_key(
    model_name: str,
    model_revision: str | None,
    prompt_version: str,
    qa_id: str,
    window_id: str,
    frame_paths: list[str],
    prompt: str,
    model_fingerprint: dict | None = None,
    decoding_config: dict | None = None,
) -> str:
    return stable_hash(
        {
            "model_name": model_name,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "qa_id": qa_id,
            "window_id": window_id,
            "frame_paths": list(frame_paths),
            "prompt_hash": stable_hash({"prompt": prompt}),
            "model_fingerprint": model_fingerprint or {},
            "decoding_config": decoding_config or {},
        }
    )


def make_intervention_probe_cache_key(
    model_identity_hash: str | None,
    prompt_hash: str,
    qa_id: str,
    window_id: str,
    intervention_id: str,
    ordered_frame_paths: list[str],
    frame_count: int,
    decoding_config: dict | None = None,
) -> str:
    return stable_hash(
        {
            "model_identity_hash": model_identity_hash,
            "prompt_hash": prompt_hash,
            "qa_id": qa_id,
            "window_id": window_id,
            "intervention_id": intervention_id,
            "ordered_frame_paths": list(ordered_frame_paths),
            "frame_count": frame_count,
            "decoding_config": decoding_config or {},
        }
    )


def make_h3_dynamic_probe_cache_key(
    model_identity_hash: str | None,
    prompt_hash: str,
    qa_id: str,
    candidate_id: str,
    intervention_id: str,
    intervention_type: str,
    ordered_frame_paths: list[str],
    ordered_frame_hash: str,
    target_action: str,
    decoding_config: dict | None = None,
    intervention_protocol: dict | None = None,
) -> str:
    return stable_hash(
        {
            "model_identity_hash": model_identity_hash,
            "prompt_hash": prompt_hash,
            "qa_id": qa_id,
            "candidate_id": candidate_id,
            "intervention_id": intervention_id,
            "intervention_type": intervention_type,
            "ordered_frame_paths": list(ordered_frame_paths),
            "ordered_frame_hash": ordered_frame_hash,
            "target_action": target_action,
            "decoding_config": decoding_config or {},
            "intervention_protocol": intervention_protocol or {},
        }
    )
