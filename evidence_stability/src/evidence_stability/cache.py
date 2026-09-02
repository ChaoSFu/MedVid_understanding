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
        }
    )
