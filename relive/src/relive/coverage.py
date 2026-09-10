"""Explicit required-claim coverage, independently evaluated after admission."""
from __future__ import annotations

from typing import Sequence

from .claim_aggregation import aggregate_claim
from .types import Claim, CoverageResult, EvidenceCertificate, FinalStatus


def assess_coverage(required_claims: Sequence[Claim | str], certificates: Sequence[EvidenceCertificate],
                    unresolved_relations: Sequence[str] = ()) -> CoverageResult:
    required_rows = []
    for item in required_claims:
        if isinstance(item, Claim):
            required_rows.append(item)
        elif isinstance(item, str):
            required_rows.append(Claim(item, item))
        else:
            raise ValueError("REQUIRED_CLAIMS_MUST_BE_CLAIMS_OR_STRINGS")
    by_id = {claim.claim_id: claim for claim in required_rows}
    required = tuple(by_id)
    if any(not isinstance(claim_id, str) or not claim_id for claim_id in required):
        raise ValueError("REQUIRED_CLAIM_IDS_MUST_BE_NONEMPTY_STRINGS")
    relations = list(dict.fromkeys(unresolved_relations))
    aggregations = {claim_id: aggregate_claim(claim, certificates) for claim_id, claim in by_id.items()}
    for claim_id, aggregation in aggregations.items():
        if aggregation["status"] == "CONFLICT":
            relations.append(f"CONFLICTING_CERTIFICATES:{claim_id}")
    supported = tuple(claim_id for claim_id in required if aggregations[claim_id]["status"] == "SUPPORTED")
    missing = tuple(claim_id for claim_id in required if claim_id not in supported)
    # Empty/open requirements cannot be silently treated as a fully answered task.
    status = "COMPLETE" if required and not missing and not relations else "INCOMPLETE"
    return CoverageResult(required, supported, missing, tuple(dict.fromkeys(relations)), status, aggregations)
