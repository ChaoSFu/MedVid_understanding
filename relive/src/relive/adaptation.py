"""Deterministic evidence actions. No parameter updates or result-conditioned retries."""
from __future__ import annotations

from relive.claims import stable_id
from relive.types import AdaptationEvent, EvidenceCertificate, FinalStatus

IMPLEMENTED_ACTIONS = frozenset({"NEXT_CANDIDATE", "EXPAND_TEMPORAL_CONTEXT",
                               "TRY_ALTERNATE_SUPPORT_REGION", "ACQUIRE_MISSING_CLAIM", "STOP"})


def action_status(action: str) -> str:
    return "SUPPORTED_ACTION" if action in IMPLEMENTED_ACTIONS else "UNSUPPORTED_ACTION"


def choose_action(certificate: EvidenceCertificate, *, enabled: bool, allowed: list[str],
                  can_expand: bool, can_alternate: bool, can_next: bool) -> tuple[str, str]:
    if certificate.final_status == FinalStatus.VERIFIED:
        return "STOP", "CLAIM_VERIFIED"
    if certificate.final_status == FinalStatus.REJECTED:
        if can_next and "NEXT_CANDIDATE" in allowed:
            return "NEXT_CANDIDATE", "PAIR_REJECTED_TRY_NEXT_CANDIDATE"
        return "STOP", "CANDIDATES_EXHAUSTED_AFTER_PAIR_REJECTION"
    if not enabled:
        return "STOP", "ADAPTATION_DISABLED"
    reasons = " ".join(certificate.failure_reasons)
    if "KEEP_SUPPORT_LOST" in reasons:
        if can_alternate and "TRY_ALTERNATE_SUPPORT_REGION" in allowed:
            return "TRY_ALTERNATE_SUPPORT_REGION", "SYNTHETIC_PRECONFIGURED_REGION"
        return "STOP", "RE_GROUNDING_NOT_IMPLEMENTED"
    # Neither a non-specific response nor missing controls admits the claim.
    if can_expand and "EXPAND_TEMPORAL_CONTEXT" in allowed:
        return "EXPAND_TEMPORAL_CONTEXT", "FIXED_NEIGHBOR_CONTEXT"
    if can_next and "NEXT_CANDIDATE" in allowed:
        return "NEXT_CANDIDATE", "ACQUISITION_ORDER"
    return "STOP", "CANDIDATES_EXHAUSTED"


def event(candidate, claim, certificate, action, parameters, next_candidate, usage, reason=None):
    fields = [candidate.candidate_id if candidate else None, claim.claim_id if claim else None,
              certificate.certificate_id if certificate else None, action, parameters,
              next_candidate.candidate_id if next_candidate else None, reason]
    return AdaptationEvent(stable_id("adapt", fields), fields[0], fields[1], fields[2], action,
                           parameters, fields[5], usage, reason)
