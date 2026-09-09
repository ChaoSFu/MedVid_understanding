"""Conservative contrast evaluation; exclusivity is declared, never inferred here."""
from __future__ import annotations

from typing import Any, Mapping

from .types import ContrastGroup, ExecutionStatus, ExclusivityStatus, SemanticStatus, VerificationResult, to_dict


def evaluate_contrast(group: ContrastGroup, results: Mapping[str, VerificationResult],
                      target_claim_id: str, strict_alternatives: bool = True) -> dict[str, Any]:
    ids = group.claim_ids
    reasons = []
    valid_group = len(ids) >= 2 and len(set(ids)) == len(ids) and target_claim_id in ids
    if not valid_group:
        reasons.append("INVALID_CONTRAST_GROUP")
    complete = all(claim_id in results and results[claim_id].execution_status == ExecutionStatus.OK for claim_id in ids)
    if not complete:
        reasons.append("CONTRAST_TECHNICAL_FAILURE_OR_MISSING_RESULT")
    supported = [claim_id for claim_id in ids if claim_id in results
                 and results[claim_id].execution_status == ExecutionStatus.OK
                 and results[claim_id].semantic_status == SemanticStatus.SUPPORTED]
    alternatives = [claim_id for claim_id in ids if claim_id != target_claim_id]
    unique_supported = complete and len(supported) == 1
    alternatives_contradicted = bool(alternatives) and all(
        claim_id in results and results[claim_id].execution_status == ExecutionStatus.OK
        and results[claim_id].semantic_status == SemanticStatus.CONTRADICTED for claim_id in alternatives)
    target_supported = target_claim_id in supported
    status = "CONTRAST_UNRESOLVED"
    passed = False
    if group.exclusivity_status == ExclusivityStatus.UNRESOLVED:
        reasons.append("EXCLUSIVITY_UNRESOLVED")
    elif group.exclusivity_status == ExclusivityStatus.NONEXCLUSIVE:
        # These claims may describe simultaneous actions: no exactly-one rule.
        passed = valid_group and complete and target_supported
        status = "NONEXCLUSIVE_TARGET_SUPPORTED" if passed else "NONEXCLUSIVE_TARGET_UNRESOLVED"
    elif group.exclusivity_status == ExclusivityStatus.DECLARED_EXCLUSIVE:
        if not supported:
            reasons.append("NO_SUPPORTED_CLAIM")
        elif len(supported) > 1:
            reasons.append("MULTIPLE_EXCLUSIVE_CLAIMS_SUPPORTED")
        if not target_supported:
            reasons.append("TARGET_NOT_SUPPORTED")
        if strict_alternatives and not alternatives_contradicted:
            reasons.append("ALTERNATIVES_NOT_CONTRADICTED")
        passed = valid_group and complete and target_supported and unique_supported and (alternatives_contradicted or not strict_alternatives)
        status = "EXCLUSIVE_CONTRAST_PASS" if passed else "EXCLUSIVE_CONTRAST_UNRESOLVED"
    else:
        reasons.append("INVALID_EXCLUSIVITY_STATUS")
    return {"pass": passed, "status": status, "reasons": list(dict.fromkeys(reasons)),
            "target_claim_id": target_claim_id,
            "group_id": group.group_id, "comparison_dimension": group.comparison_dimension,
            "exclusivity_status": to_dict(group.exclusivity_status),
            "exclusivity_provenance": to_dict(group.provenance),
            "exactly_one_rule_applied": group.exclusivity_status == ExclusivityStatus.DECLARED_EXCLUSIVE,
            "strict_alternatives": strict_alternatives, "supported_claims": supported,
            "unique_supported": unique_supported, "alternatives_contradicted": alternatives_contradicted,
            "references": {claim_id: to_dict(results.get(claim_id)) for claim_id in ids}}
