"""Explicit required-claim coverage, independently evaluated after admission."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .types import Claim, CoverageResult, EvidenceCertificate, FinalStatus


def assess_coverage(required_claims: Sequence[Claim | str], certificates: Sequence[EvidenceCertificate],
                    unresolved_relations: Sequence[str] = ()) -> CoverageResult:
    required = tuple(dict.fromkeys(claim.claim_id if isinstance(claim, Claim) else claim for claim in required_claims))
    if any(not isinstance(claim_id, str) or not claim_id for claim_id in required):
        raise ValueError("REQUIRED_CLAIM_IDS_MUST_BE_NONEMPTY_STRINGS")
    relations = list(dict.fromkeys(unresolved_relations))
    by_pair = defaultdict(set)
    for certificate in certificates:
        by_pair[(certificate.candidate_id, certificate.claim_id)].add(certificate.final_status)
    conflicted = {pair for pair, statuses in by_pair.items() if FinalStatus.VERIFIED in statuses and FinalStatus.REJECTED in statuses}
    for candidate_id, claim_id in sorted(conflicted):
        if claim_id in required:
            relations.append(f"CONFLICTING_CERTIFICATES:{candidate_id}:{claim_id}")
    verified = {certificate.claim_id for certificate in certificates
                if certificate.final_status == FinalStatus.VERIFIED
                and (certificate.candidate_id, certificate.claim_id) not in conflicted}
    supported = tuple(claim_id for claim_id in required if claim_id in verified)
    missing = tuple(claim_id for claim_id in required if claim_id not in verified)
    # Empty/open requirements cannot be silently treated as a fully answered task.
    status = "COMPLETE" if required and not missing and not relations else "INCOMPLETE"
    return CoverageResult(required, supported, missing, tuple(dict.fromkeys(relations)), status)
