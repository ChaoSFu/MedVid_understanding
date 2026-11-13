from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from relive.backends import make_backend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference
from relive.verification import parse_verification

from .protocol import VERIFIER_MASK_NOTICE, VERIFIER_PROMPT_VERSION


class ModelSession:
    def __init__(self, *, config_path: str | Path, cache_dir: str | Path, max_calls: int):
        self.config = load_config(config_path)
        self.backend = make_backend(self.config["backend"])
        self.inference = CachedInference(self.backend, ArtifactStore(cache_dir))
        self.budget = Budget(max_calls)

    def raw(self, *, stage: str, prompt: str, prompt_version: str,
            image_path: str, frame_id: str,
            identity_fields: dict[str, Any] | None = None) -> dict[str, Any]:
        request = {
            "stage": stage, "prompt": prompt, "prompt_version": prompt_version,
            "image_paths": [image_path], "frame_ids": [frame_id],
        }
        if identity_fields:
            request["spatial_protocol_identity"] = deepcopy(identity_fields)
        return self.inference.call(request, self.budget)

    def verify(self, *, claim: str, variant: str, image_path: str, frame_id: str,
               identity_fields: dict[str, Any] | None = None) -> dict[str, Any]:
        prompt = (
            "Judge whether the supplied image supports this claim: " + claim + "\n" +
            VERIFIER_MASK_NOTICE + "\nReturn only strict JSON with fields status, observation, and "
            "frame_references. status must be SUPPORTED, CONTRADICTED, or INSUFFICIENT. "
            f"The input variant is {variant}; use identical semantics for every variant."
        )
        result = self.raw(stage="relive_spatial_verify", prompt=prompt,
                          prompt_version=VERIFIER_PROMPT_VERSION, image_path=image_path, frame_id=frame_id,
                          identity_fields={"verification_mode": variant, **(identity_fields or {})})
        if result["execution_status"] != "OK":
            return {"semantic_status": None, "execution_status": result["execution_status"],
                    "failure_reason": result["failure_reason"], "cache_hit": result["cache_hit"],
                    "cache_key": result["cache_key"], "raw_response_ref": result["raw_response_ref"]}
        parsed = parse_verification(result["raw_text"], result["raw_response_ref"], VERIFIER_PROMPT_VERSION,
                                    {"frame_ids": [frame_id], "image_paths": [image_path], "variant": variant})
        return {"semantic_status": parsed.semantic_status.value if parsed.semantic_status else None,
                "execution_status": parsed.execution_status.value, "failure_reason": parsed.failure_reason,
                "cache_hit": result["cache_hit"], "cache_key": result["cache_key"],
                "raw_response_ref": result["raw_response_ref"]}

    def snapshot(self) -> dict[str, Any]:
        return self.budget.snapshot()
