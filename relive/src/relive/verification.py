"""Strict three-state semantic parsing; technical errors never become verdicts."""
from __future__ import annotations

import json
from typing import Any

from .types import ExecutionStatus, SemanticStatus, VerificationResult


def strict_json(raw: str) -> Any:
    """Reject non-JSON constants and duplicate keys as ambiguous model output."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"DUPLICATE_JSON_KEY: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"NONFINITE_JSON_CONSTANT: {value}")

    if not isinstance(raw, str):
        raise ValueError("RESPONSE_MUST_BE_TEXT")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def execution_failure(
    status: ExecutionStatus,
    reason: str,
    raw_response_ref: str | None = None,
    prompt_version: str = "relive-semantic-v1",
    input_references: dict[str, Any] | None = None,
) -> VerificationResult:
    if status == ExecutionStatus.OK:
        raise ValueError("A technical failure cannot have execution_status=OK")
    return VerificationResult(None, status, raw_response_ref, prompt_version,
                              dict(input_references or {}), reason)


def parse_verification(
    raw: str,
    raw_response_ref: str | None = None,
    prompt_version: str = "relive-semantic-v1",
    input_references: dict[str, Any] | None = None,
) -> VerificationResult:
    references = dict(input_references or {})
    try:
        parsed = strict_json(raw)
        if not isinstance(parsed, dict) or "status" not in parsed:
            raise ValueError("EXPECTED_OBJECT_WITH_STATUS")
        if set(parsed) - {"status", "observation", "frame_references"}:
            raise ValueError("UNEXPECTED_VERIFICATION_FIELDS")
        if not isinstance(parsed["status"], str):
            raise ValueError("STATUS_MUST_BE_STRING")
        status = SemanticStatus(parsed["status"])
        observation = parsed.get("observation")
        if observation is not None and not isinstance(observation, str):
            raise ValueError("OBSERVATION_MUST_BE_TEXT_OR_NULL")
        frame_refs = parsed.get("frame_references", [])
        if not isinstance(frame_refs, list) or any(not isinstance(x, str) for x in frame_refs):
            raise ValueError("FRAME_REFERENCES_MUST_BE_STRING_LIST")
        if len(set(frame_refs)) != len(frame_refs):
            raise ValueError("DUPLICATE_FRAME_REFERENCE")
        supplied_frame_ids = references.get("frame_ids", [])
        if not isinstance(supplied_frame_ids, (list, tuple)):
            raise ValueError("INPUT_FRAME_IDS_MUST_BE_SEQUENCE")
        if not set(frame_refs).issubset(set(supplied_frame_ids)):
            raise ValueError("FRAME_REFERENCE_NOT_IN_INPUT")
    except (ValueError, TypeError) as exc:
        return execution_failure(ExecutionStatus.PARSE_ERROR, str(exc), raw_response_ref,
                                 prompt_version, references)
    return VerificationResult(status, ExecutionStatus.OK, raw_response_ref, prompt_version,
                              references, observation=observation, frame_references=tuple(frame_refs))
