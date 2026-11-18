from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from relive.backends import make_backend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference
from relive.verification import parse_single_frame_verification

from .proposal_constraint import TokenLevelProposalConstraint, proposal_constraint_identity
from .protocol import VERIFIER_PROMPT_VERSION, verifier_prompt


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

    def propose(self, *, stage: str, prompt: str, prompt_version: str,
                image_path: str, frame_id: str,
                identity_fields: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate one localization proposal under the versioned token grammar.

        Synthetic fixtures retain their normal deterministic backend path and
        are marked as parser-only.  Every non-synthetic backend must implement
        the real logits-mask interface; there is no silent unconstrained
        production fallback.
        """
        contract = proposal_constraint_identity()
        request = {
            "stage": stage, "prompt": prompt, "prompt_version": prompt_version,
            "image_paths": [image_path], "frame_ids": [frame_id],
            "token_constraint_identity": contract,
        }
        if identity_fields:
            request["spatial_protocol_identity"] = deepcopy(identity_fields)
        if self.backend.synthetic:
            result = self.inference.call(request, self.budget)
            result["constraint_enforcement"] = "SYNTHETIC_FIXTURE_STRICT_PARSE_ONLY"
            result["generation_metadata"] = None
            result["constraint_metadata"] = None
            result["constraint_failure"] = None
            return result
        if not callable(getattr(self.backend, "infer_with_token_constraint", None)):
            raise RuntimeError("TOKEN_CONSTRAINT_INTERFACE_UNAVAILABLE")
        result = self.inference.call_with_token_constraint(
            request, self.budget, TokenLevelProposalConstraint,
        )
        execution = (result.get("constraint_metadata") or {}).get("execution") or {}
        result["constraint_enforcement"] = contract["enforcement"]
        result["constraint_failure"] = execution.get("constraint_failure")
        return result

    def verify(self, *, claim: str, variant: str, image_path: str, frame_id: str,
               identity_fields: dict[str, Any] | None = None) -> dict[str, Any]:
        prompt = verifier_prompt(claim=claim, variant=variant)
        result = self.raw(stage="relive_spatial_verify", prompt=prompt,
                          prompt_version=VERIFIER_PROMPT_VERSION, image_path=image_path, frame_id=frame_id,
                          identity_fields={"verification_mode": variant, **(identity_fields or {})})
        if result["execution_status"] != "OK":
            return {"semantic_status": None, "execution_status": result["execution_status"],
                    "failure_reason": result["failure_reason"], "cache_hit": result["cache_hit"],
                    "cache_key": result["cache_key"], "raw_response_ref": result["raw_response_ref"]}
        parsed = parse_single_frame_verification(
            result["raw_text"], result["raw_response_ref"], VERIFIER_PROMPT_VERSION,
            {"frame_ids": [frame_id], "image_paths": [image_path], "variant": variant},
        )
        return {"semantic_status": parsed.semantic_status.value if parsed.semantic_status else None,
                "execution_status": parsed.execution_status.value, "failure_reason": parsed.failure_reason,
                "cache_hit": result["cache_hit"], "cache_key": result["cache_key"],
                "raw_response_ref": result["raw_response_ref"]}

    def snapshot(self) -> dict[str, Any]:
        return self.budget.snapshot()
