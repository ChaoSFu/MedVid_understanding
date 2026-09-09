"""Deterministic synthetic fixtures. This backend provides no scientific evidence."""
from __future__ import annotations

import json
from typing import Any

from .base import Backend, BackendError
from relive.config import validate_backend


class MockBackend(Backend):
    synthetic = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(validate_backend(config or {"kind": "mock"}))
        if self.config["kind"] != "mock":
            raise ValueError("MockBackend requires kind=mock")

    def infer(self, request: dict[str, Any]) -> str:
        self.calls += 1
        context = dict(request.get("context", {}))
        context["stage"] = request["stage"]
        for rule in self.config["rules"]:
            if all(context.get(key) == value for key, value in rule["match"].items()):
                if "error" in rule:
                    # Fixture diagnostics are codes, never arbitrary secret-bearing text.
                    raise BackendError("MOCK_FIXTURE_ERROR", retryable=rule.get("retryable", False))
                response = rule["response"]
                return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False, sort_keys=True)
        stage = request["stage"]
        if stage in {"semantic", "verify", "verification"}:
            status = "INSUFFICIENT" if context.get("variant") == "DROP_TARGET" else "SUPPORTED"
            response = {"status": status, "observation": "Synthetic preset; not a medical observation.",
                        "frame_references": list(request.get("frame_ids", []))}
        elif stage in {"spatial", "spatial_proposal", "proposal"}:
            response = {"support_region": [0.1, 0.1, 0.4, 0.4],
                        "coordinate_system": "normalized_0_1_xyxy"}
        elif stage in {"claims", "claim_generation"}:
            response = {"claims": [{"text": "The instrument moves toward the visible tissue."}]}
        elif stage in {"contrasts", "contrast_generation"}:
            response = {"alternatives": [{"text": "The instrument moves away from the visible tissue."}],
                        "comparison_dimension": "direction of visible instrument movement"}
        elif stage in {"answer", "reasoning"}:
            response = {"answer": "The verified synthetic claim describes movement toward the visible tissue."}
        else:
            raise BackendError("UNSUPPORTED_MOCK_STAGE")
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
