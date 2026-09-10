"""Strict three-state semantic parsing; technical errors never become verdicts."""
from __future__ import annotations

from typing import Any

from .json_protocol import strict_json
from .types import ExecutionStatus, SemanticStatus, VerificationResult


def execution_failure(
    status: ExecutionStatus,
    reason: str,
    raw_response_ref: str | None = None,
    prompt_version: str = "relive-semantic-v4",
    input_references: dict[str, Any] | None = None,
) -> VerificationResult:
    if status == ExecutionStatus.OK:
        raise ValueError("A technical failure cannot have execution_status=OK")
    return VerificationResult(None, status, raw_response_ref, prompt_version,
                              dict(input_references or {}), reason)


def parse_verification(
    raw: str,
    raw_response_ref: str | None = None,
    prompt_version: str = "relive-semantic-v4",
    input_references: dict[str, Any] | None = None,
) -> VerificationResult:
    references = dict(input_references or {})
    try:
        parsed = strict_json(raw)
        if not isinstance(parsed, dict) or "status" not in parsed:
            raise ValueError("EXPECTED_OBJECT_WITH_STATUS")
        if set(parsed) - {"status", "observation", "frame_references", "frame_index"}:
            raise ValueError("UNEXPECTED_VERIFICATION_FIELDS")
        if not isinstance(parsed["status"], str):
            raise ValueError("STATUS_MUST_BE_STRING")
        status = SemanticStatus(parsed["status"])
        observation = parsed.get("observation")
        if observation is not None and not isinstance(observation, str):
            raise ValueError("OBSERVATION_MUST_BE_TEXT_OR_NULL")
        supplied_frame_ids = references.get("frame_ids", [])
        if not isinstance(supplied_frame_ids, (list, tuple)):
            raise ValueError("INPUT_FRAME_IDS_MUST_BE_SEQUENCE")
        has_ids = "frame_references" in parsed
        has_index = "frame_index" in parsed
        if has_ids and has_index:
            raise ValueError("AMBIGUOUS_FRAME_REFERENCE")
        if has_index:
            index = parsed["frame_index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("FRAME_INDEX_MUST_BE_INTEGER")
            if not 0 <= index < len(supplied_frame_ids):
                raise ValueError("FRAME_INDEX_OUT_OF_INPUT_RANGE")
            frame_refs = [supplied_frame_ids[index]]
        else:
            frame_refs = parsed.get("frame_references", [])
            if not isinstance(frame_refs, list) or any(not isinstance(x, str) for x in frame_refs):
                raise ValueError("FRAME_REFERENCES_MUST_BE_STRING_LIST")
            if len(set(frame_refs)) != len(frame_refs):
                raise ValueError("DUPLICATE_FRAME_REFERENCE")
            if not set(frame_refs).issubset(set(supplied_frame_ids)):
                raise ValueError("FRAME_REFERENCE_NOT_IN_INPUT")
        if status in {SemanticStatus.SUPPORTED, SemanticStatus.CONTRADICTED} and not frame_refs:
            raise ValueError("EVIDENCE_STATUS_REQUIRES_FRAME_REFERENCE")
    except (ValueError, TypeError) as exc:
        return execution_failure(ExecutionStatus.PARSE_ERROR, str(exc), raw_response_ref,
                                 prompt_version, references)
    return VerificationResult(status, ExecutionStatus.OK, raw_response_ref, prompt_version,
                              references, observation=observation, frame_references=tuple(frame_refs))
