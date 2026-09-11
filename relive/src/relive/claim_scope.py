"""GT-free applicability routing for single-ROI spatial certificates.

The router answers a narrow prospective question: whether the *form* of a
claim can be tested by the current one-ROI KEEP/DROP/matched-control protocol.
It is deliberately independent of a spatial proposal, model response, and
certificate outcome. Routing is not a reliability judgment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .types import Claim, RuntimeSample


CLAIM_SCOPE_ROUTER_VERSION = "relive-claim-scope-router-v1"
LOCAL_ATOMIC = "LOCAL_ATOMIC"
MULTI_SUPPORT_POSSIBLE = "MULTI_SUPPORT_POSSIBLE"
GLOBAL_DISTRIBUTED = "GLOBAL_DISTRIBUTED"
UNRESOLVED_SCOPE = "UNRESOLVED_SCOPE"
CLAIM_SCOPES = frozenset({LOCAL_ATOMIC, MULTI_SUPPORT_POSSIBLE, GLOBAL_DISTRIBUTED, UNRESOLVED_SCOPE})

# These are the only runtime metadata fields admitted by this component. The
# current rules classify on claim text; metadata is only provenance and cannot
# smuggle annotations or historical model evidence into the router.
_ALLOWED_METADATA = frozenset({"dataset_name", "source_qa_type", "question_scope",
                               "runtime_adapter", "nonofficial_protocol"})
_FORBIDDEN_INPUT_TOKENS = frozenset({"reference_answer", "assistant_answer", "temporal_gt", "bbox", "mask",
                                     "struc_info", "rc_info", "true_support", "spurious_support", "gt_iou",
                                     "certificate", "semantic_status", "failure_reason", "keep", "drop", "r0", "r1"})


class ClaimScopeError(ValueError):
    """Raised when an input violates the prospective routing boundary."""


@dataclass(frozen=True)
class ClaimScopeDecision:
    claim_scope: str
    single_roi_certificate_applicable: bool
    reason_code: str
    router_version: str = CLAIM_SCOPE_ROUTER_VERSION
    gt_used: bool = False
    diagnostic_rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_scope": self.claim_scope,
            "single_roi_certificate_applicable": self.single_roi_certificate_applicable,
            "reason_code": self.reason_code,
            "router_version": self.router_version,
            "gt_used": self.gt_used,
            "diagnostic_rationale": self.diagnostic_rationale,
        }


def claim_text_sha256(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ClaimScopeError("claim text must be nonempty")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ClaimScopeError("task metadata must be an object")
    keys = set(metadata)
    forbidden = sorted(key for key in keys if key.casefold() in _FORBIDDEN_INPUT_TOKENS)
    if forbidden:
        raise ClaimScopeError(f"claim-scope router rejects prohibited input fields: {forbidden}")
    unknown = sorted(keys - _ALLOWED_METADATA)
    if unknown:
        raise ClaimScopeError(f"claim-scope router received unsupported task metadata: {unknown}")
    return {key: metadata[key] for key in sorted(keys)}


def _normalize_claim(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ClaimScopeError("claim text must be nonempty")
    return " ".join(text.casefold().split())


def _decision(scope: str, code: str, rationale: str) -> ClaimScopeDecision:
    if scope not in CLAIM_SCOPES:
        raise AssertionError("unknown claim scope")
    return ClaimScopeDecision(scope, scope == LOCAL_ATOMIC, code,
                              diagnostic_rationale=rationale)


def route_claim_scope(claim_text: str, *, qa_type: str | None = None,
                      task_metadata: dict[str, Any] | None = None) -> ClaimScopeDecision:
    """Classify claim logic using only text and declared public task metadata.

    ``qa_type`` is validated but does not override explicit logical form. Thus
    the same public claim text gets the same routing under every task label.
    """
    if qa_type is not None and (not isinstance(qa_type, str) or not qa_type.strip()):
        raise ClaimScopeError("qa_type must be a nonempty string when supplied")
    _validate_metadata(task_metadata)
    text = _normalize_claim(claim_text)

    # An existential statement about an interchangeable object class can have
    # another independent witness after a single witness ROI is removed.
    multi_patterns = (
        r"\bat least one\s+(?:[a-z-]+\s+){0,4}(?:instrument|tool|device|object)s?\b",
        r"\b(?:one or more|any|some)\s+(?:[a-z-]+\s+){0,4}(?:instrument|tool|device|object)s?\b",
        r"\bthere (?:is|are)\s+(?:an?|at least one|some)\s+",
        r"\b(?:an?|the)\s+(?:[a-z-]+\s+){0,3}(?:instrument|tool|device)\s+(?:is|are)\s+(?:visibly\s+)?(?:present|visible)\b",
    )
    if any(re.search(pattern, text) for pattern in multi_patterns):
        return _decision(MULTI_SUPPORT_POSSIBLE, "EXISTENTIAL_INTERCHANGEABLE_WITNESS",
                         "The claim permits independently located witnesses; one witness ROI need not be necessary.")

    # These assertions concern a view, environment, or the overall procedure,
    # rather than one bounded visual fact.
    global_patterns = (
        r"\b(?:egocentric|first-person)\s+view(?:point)?\b",
        r"\b(?:overall|global|entire)\s+(?:scene|view|state|procedure)\b",
        r"\b(?:scene|environment)\s+(?:is|shows|depicts|contains)\b",
        r"\b(?:supplied|provided)\s+frames?\s+(?:visibly\s+)?depict\b.*\b(?:procedure|scene|viewpoint)\b",
        r"\b(?:an?|the)\s+(?:open\s+)?surgical procedure\s+from\s+an?\s+(?:egocentric|first-person)\b",
    )
    if any(re.search(pattern, text) for pattern in global_patterns):
        return _decision(GLOBAL_DISTRIBUTED, "GLOBAL_SCENE_OR_VIEWPOINT",
                         "The claim describes a distributed scene, procedure, or viewpoint rather than one local visual fact.")

    # A named object/state/relation with a localized predicate is a valid input
    # to the present protocol. This does not say that the claim is true.
    local_patterns = (
        r"\b(?:contact(?:s|ing)?|touch(?:es|ing)?|press(?:es|ing)?|grasp(?:s|ing)?|cut(?:s|ting)?|enter(?:s|ing)?)\b",
        r"\b(?:inside|within|centered|attached|secured|open)\b",
        r"\b(?:forceps|jaw|baseplate|stoma|specimen-retrieval bag|retrieval bag|catheter|tubular structure)\b.*\b(?:visible|identifiable|contact(?:s|ing)?|inside|enter(?:s|ing)|open)\b",
    )
    if any(re.search(pattern, text) for pattern in local_patterns):
        return _decision(LOCAL_ATOMIC, "LOCAL_OBJECT_STATE_OR_RELATION",
                         "The claim names one localized object, state, or relation suitable for a one-ROI protocol.")

    return _decision(UNRESOLVED_SCOPE, "SCOPE_NOT_DETERMINABLE_FROM_PUBLIC_CLAIM_TEXT",
                     "The claim text does not establish whether a single ROI can be individually necessary.")


def route_runtime_claim(sample: RuntimeSample) -> ClaimScopeDecision:
    """Route one frozen runtime target claim without inspecting any outcomes."""
    claim: Claim | None = sample.target_claim
    if claim is None:
        raise ClaimScopeError("claim-scope routing requires one runtime target_claim")
    metadata = {key: value for key, value in sample.metadata.items() if key in _ALLOWED_METADATA}
    return route_claim_scope(claim.text, qa_type=sample.metadata.get("source_qa_type"), task_metadata=metadata)


def canonical_scope_record(sample: RuntimeSample, *, candidate_id: str | None = None,
                           candidate_rank: int | None = None,
                           development_only: bool) -> dict[str, Any]:
    """Create an audit row. No spatial or certificate field is accepted here."""
    if sample.target_claim is None:
        raise ClaimScopeError("claim-scope audit requires target_claim")
    decision = route_runtime_claim(sample)
    row = {
        "format": "relive-claim-scope-audit-v1",
        "sample_id": sample.sample_id,
        "dataset_name": sample.metadata.get("dataset_name"),
        "source_qa_type": sample.metadata.get("source_qa_type"),
        "claim_id": sample.target_claim.claim_id,
        "claim_text_sha256": claim_text_sha256(sample.target_claim.text),
        "development_diagnostic_only": development_only,
        **decision.as_dict(),
    }
    if candidate_id is not None:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ClaimScopeError("candidate_id must be nonempty when supplied")
        row["candidate_id"] = candidate_id
        row["candidate_rank"] = candidate_rank
    return row


def stable_scope_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
