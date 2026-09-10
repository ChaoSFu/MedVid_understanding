"""Conservative claim-level aggregation over pair-level certificates.

Certificates remain statements about one Evidence–Claim pair.  This module
does not turn a local failed candidate into a video-global contradiction.
"""
from __future__ import annotations

from typing import Sequence

from .types import Claim, EvidenceCertificate, FinalStatus


def _scope_key(scope: dict) -> tuple[str, ...] | None:
    frame_ids = scope.get("frame_ids") if isinstance(scope, dict) else None
    if not isinstance(frame_ids, (list, tuple)) or not frame_ids:
        return None
    if any(not isinstance(frame_id, str) or not frame_id for frame_id in frame_ids):
        return None
    return tuple(frame_ids)


def aggregate_claim(claim: Claim, certificates: Sequence[EvidenceCertificate]) -> dict:
    """Return a conservative claim status and certificate IDs.

    A contradiction is admitted only for a claim that itself declares an exact
    frame scope and has a rejected certificate bound to precisely that scope.
    A VERIFIED/REJECTED conflict in the same evidence scope stays unresolved.
    """
    relevant = [certificate for certificate in certificates if certificate.claim_id == claim.claim_id]
    by_scope: dict[tuple[str, ...], set[FinalStatus]] = {}
    for certificate in relevant:
        key = _scope_key(certificate.time_scope)
        if key is not None:
            by_scope.setdefault(key, set()).add(certificate.final_status)
    conflict_scopes = tuple(sorted(
        (scope for scope, statuses in by_scope.items()
         if FinalStatus.VERIFIED in statuses and FinalStatus.REJECTED in statuses),
        key=lambda scope: (len(scope), scope),
    ))
    verified = tuple(c.certificate_id for c in relevant if c.final_status == FinalStatus.VERIFIED)
    rejected = tuple(c.certificate_id for c in relevant if c.final_status == FinalStatus.REJECTED)
    exact_claim_scope = _scope_key(claim.time_scope)
    if conflict_scopes:
        status, reason = "CONFLICT", "VERIFIED_AND_REJECTED_IN_SAME_SCOPE"
    elif verified:
        status, reason = "SUPPORTED", "UNCONFLICTED_PAIR_VERIFIED"
    elif exact_claim_scope is not None and FinalStatus.REJECTED in by_scope.get(exact_claim_scope, set()):
        status, reason = "CONTRADICTED", "EXACT_CLAIM_SCOPE_REJECTED"
    else:
        status, reason = "INSUFFICIENT", "NO_UNCONFLICTED_VERIFIED_OR_EXACT_SCOPE_CONTRADICTION"
    return {
        "claim_id": claim.claim_id,
        "status": status,
        "reason": reason,
        "claim_explicit_frame_scope": list(exact_claim_scope) if exact_claim_scope is not None else None,
        "conflict_scopes": [list(scope) for scope in conflict_scopes],
        "verified_certificate_ids": list(verified),
        "rejected_certificate_ids": list(rejected),
    }
