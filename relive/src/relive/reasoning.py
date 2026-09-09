"""Strict answer construction uses only certificate-bound claim text."""
from __future__ import annotations

from relive.types import FinalStatus, to_dict


def strict_answer(sample, claims, certificates, coverage):
    by_id = {c.claim_id: c for c in claims}
    rejected_pairs = {(c.candidate_id, c.claim_id) for c in certificates
                      if c.final_status == FinalStatus.REJECTED}
    verified = [c for c in certificates if c.final_status == FinalStatus.VERIFIED
                and c.claim_id in coverage.supported_claims
                and (c.candidate_id, c.claim_id) not in rejected_pairs]
    admitted = [c for c in verified if c.claim_id in by_id]
    admitted_claim_ids = {c.claim_id for c in admitted}
    covered = (coverage.status == "COMPLETE"
               and set(coverage.required_claims).issubset(admitted_claim_ids))
    if not covered:
        rejected = any(c.final_status == FinalStatus.REJECTED for c in certificates)
        return {"mode": "strict_reliability", "status": "ABSTAIN", "answer": None,
                "judgment": "REJECTED" if rejected and sample.task == "claim_verification" else "UNCERTAIN",
                "claim_ids": [], "certificate_ids": [], "input_references": [],
                "reason": "COVERAGE_INCOMPLETE", "fallback_used": False}
    claim_ids = list(dict.fromkeys(c.claim_id for c in admitted))
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
