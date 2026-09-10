"""Strict answer construction uses only certificate-bound claim text."""
from __future__ import annotations

from relive.types import FinalStatus


def strict_answer(sample, claims, certificates, coverage):
    by_id = {c.claim_id: c for c in claims}
    verified_by_claim = {}
    for claim_id in coverage.supported_claims:
        certificate = next((c for c in certificates
                            if c.claim_id == claim_id and c.final_status == FinalStatus.VERIFIED), None)
        if certificate is not None:
            verified_by_claim[claim_id] = certificate
    admitted_claim_ids = set(verified_by_claim)
    covered = (coverage.status == "COMPLETE"
               and set(coverage.required_claims).issubset(admitted_claim_ids))
    if not covered:
        target_id = sample.target_claim.claim_id if sample.target_claim else None
        aggregation = coverage.claim_aggregations.get(target_id, {}) if target_id else {}
        judgment = "CONTRADICTED" if aggregation.get("status") == "CONTRADICTED" else "UNCERTAIN"
        return {"mode": "strict_reliability", "status": "ABSTAIN", "answer": None,
                "judgment": judgment,
                "claim_ids": [], "certificate_ids": [], "input_references": [],
                "reason": "COVERAGE_INCOMPLETE", "fallback_used": False}
    claim_ids = list(coverage.required_claims)
    admitted = [verified_by_claim[claim_id] for claim_id in claim_ids]
    answer = "SUPPORTED" if sample.task == "claim_verification" else " ".join(by_id[i].text for i in claim_ids)
    return {"mode": "strict_reliability", "status": "ANSWERED", "answer": answer,
            "judgment": "VERIFIED", "claim_ids": claim_ids,
            "certificate_ids": [c.certificate_id for c in admitted],
            "input_references": [{"candidate_id": c.candidate_id, "claim_id": c.claim_id,
                                  "certificate_id": c.certificate_id,
                                  "frame_ids": list(c.provenance.get("frame_ids", []))}
                                 for c in admitted],
            "fallback_used": False, "composition": "verbatim_verified_claims",
            "unverified_generation_risk": "Model-proposed claims may be wrong despite protocol passage; no additional answer model call."}


def forced_answer(sample, claims, candidates, strict):
    if strict["status"] == "ANSWERED":
        return {**strict, "mode": "benchmark_forced", "fallback_used": False}
    # A forced baseline is explicitly separate and makes no reliability claim.
    return {"mode": "benchmark_forced", "status": "FALLBACK" if claims else "UNAVAILABLE",
            "answer": claims[0].text if claims else None, "fallback_used": True,
            "certificate_ids": [], "claim_ids": [claims[0].claim_id] if claims else [],
            "unverified_input_sources": [{"candidate_id": c.candidate_id,
                                           "frame_ids": list(c.frame_ids)} for c in candidates],
            "risk": "Unverified proposed statement; excluded from strict metrics."}
