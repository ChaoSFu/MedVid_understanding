"""Frozen admission policies for Evidence–Claim pairs.

VERIFIED describes protocol completion, not medical factual truth. Requirements
are selected before examining results, so failed checks cannot become inapplicable.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .types import Claim, EvidenceCandidate, EvidenceCertificate, ExecutionStatus, FinalStatus, SemanticStatus, VerificationResult, to_dict

POLICY_VERSION = "relive-v1-policy-1"
POLICIES = {
    "acquisition_only": {"semantic": False, "spatial": False, "contrast": False, "controls": False},
    "semantic_only": {"semantic": True, "spatial": False, "contrast": False, "controls": False},
    "semantic_keep_drop": {"semantic": True, "spatial": True, "contrast": False, "controls": False},
    "semantic_spatial": {"semantic": True, "spatial": True, "contrast": False, "controls": True},
    "semantic_contrast_spatial": {"semantic": True, "spatial": True, "contrast": True, "controls": True},
}


def _check_result(result: dict[str, Any] | None, required: bool) -> dict[str, Any]:
    observed = dict(result or {})
    # Caller-controlled applicability never overrides the frozen policy.
    observed.update({"required": required,
                     "applicability": "REQUIRED" if required else "NOT_APPLICABLE_FIXED_POLICY"})
    if required and result is None:
        observed.update({"pass": False, "status": "REQUIRED_CHECK_NOT_RUN", "reasons": ["REQUIRED_CHECK_NOT_RUN"]})
    return observed


def _bound_to_pair(result: dict[str, Any] | None, candidate: EvidenceCandidate,
                   claim_id: str) -> bool:
    """A reused result must explicitly identify its evidence and claim."""
    references = result.get("input_references", {}) if isinstance(result, dict) else {}
    if not isinstance(references, dict):
        return False
    if references.get("candidate_id") != candidate.candidate_id or references.get("claim_id") != claim_id:
        return False
    if "frame_ids" in references:
        ids = references["frame_ids"]
        if not isinstance(ids, (tuple, list)) or tuple(ids) != candidate.frame_ids:
            return False
    return True


def build_certificate(candidate: EvidenceCandidate, claim: Claim, original: VerificationResult,
                      policy_name: str, spatial_check: dict[str, Any] | None = None,
                      contrast_check: dict[str, Any] | None = None,
                      provenance: dict[str, Any] | None = None) -> EvidenceCertificate:
    if policy_name not in POLICIES:
        raise ValueError(f"UNKNOWN_CERTIFICATE_POLICY: {policy_name}")
    required = POLICIES[policy_name]
    semantic_ok = original.execution_status == ExecutionStatus.OK and original.semantic_status == SemanticStatus.SUPPORTED
    semantic = {"pass": semantic_ok, "status": original.semantic_status.value if original.semantic_status else original.execution_status.value,
                "references": to_dict(original), "reasons": [] if semantic_ok else [original.failure_reason or "ORIGINAL_" + (original.semantic_status.value if original.semantic_status else original.execution_status.value)]}
    checks = {"semantic": _check_result(semantic, required["semantic"]),
              "spatial": _check_result(spatial_check, required["spatial"]),
              "contrast": _check_result(contrast_check, required["contrast"])}
    reasons = []
    if required["controls"] and spatial_check is not None and spatial_check.get("require_controls") is not True:
        checks["spatial"].update({"pass": False, "status": "MATCHED_CONTROLS_REQUIRED",
                                  "reasons": list(spatial_check.get("reasons", [])) + ["MATCHED_CONTROLS_REQUIRED"]})
    binding_conflict = not _bound_to_pair(to_dict(original), candidate, claim.claim_id)
    if binding_conflict:
        reasons.append("ORIGINAL_REFERENCE_BINDING_MISMATCH")
    if required["spatial"] and spatial_check is not None and "references" in spatial_check:
        references = spatial_check.get("references", {})
        references = references if isinstance(references, dict) else {}
        spatial_results = [references.get(name) for name in ("original", "keep", "drop")]
        if required["controls"]:
            controls = references.get("controls", [])
            if not isinstance(controls, (tuple, list)) or not controls:
                spatial_results.append(None)
            else:
                spatial_results.extend(controls)
        if not all(_bound_to_pair(result, candidate, claim.claim_id) for result in spatial_results):
            checks["spatial"].update({"pass": False, "status": "SPATIAL_REFERENCE_BINDING_MISMATCH",
                                      "reasons": list(checks["spatial"].get("reasons", [])) + ["SPATIAL_REFERENCE_BINDING_MISMATCH"]})
    if required["contrast"] and contrast_check is not None:
        references = contrast_check.get("references", {})
        bound = (isinstance(references, dict) and claim.claim_id in references
                 and contrast_check.get("target_claim_id") == claim.claim_id
                 and all(_bound_to_pair(result, candidate, claim_id) for claim_id, result in references.items()))
        if not bound:
            checks["contrast"].update({"pass": False, "status": "CONTRAST_REFERENCE_BINDING_MISMATCH",
                                       "reasons": list(checks["contrast"].get("reasons", [])) + ["CONTRAST_REFERENCE_BINDING_MISMATCH"]})
    for name, check in checks.items():
        if check["required"] and check.get("pass") is not True:
            reasons.extend(check.get("reasons") or [f"{name.upper()}_UNRESOLVED"])
    if original.execution_status == ExecutionStatus.OK and original.semantic_status == SemanticStatus.CONTRADICTED and not binding_conflict:
        status = FinalStatus.REJECTED
        if "ORIGINAL_CONTRADICTED" not in reasons:
            reasons.insert(0, "ORIGINAL_CONTRADICTED")
    elif policy_name == "acquisition_only":
        status = FinalStatus.UNCERTAIN
        reasons.append("ACQUISITION_ONLY_NO_VERIFICATION")
    elif reasons:
        status = FinalStatus.UNCERTAIN
    else:
        status = FinalStatus.VERIFIED
    scope = dict(claim.time_scope) if claim.time_scope else {"frame_ids": list(candidate.frame_ids), "timestamps": list(candidate.timestamps)}
    certificate_provenance = {**dict(provenance or {}), "requirements_frozen": dict(required),
                              "interpretation": "Passed the named verification protocol; not a medical truth guarantee."}
    payload = {"candidate_id": candidate.candidate_id, "claim_id": claim.claim_id, "time_scope": scope,
               "checks": checks, "final_status": status.value, "failure_reasons": list(dict.fromkeys(reasons)),
               "policy_version": POLICY_VERSION, "policy_name": policy_name, "provenance": certificate_provenance}
    digest = hashlib.sha256(json.dumps(to_dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return EvidenceCertificate("certificate-" + digest[:24], candidate.candidate_id, claim.claim_id,
                               scope, checks, status, tuple(dict.fromkeys(reasons)), POLICY_VERSION,
                               policy_name, certificate_provenance)
