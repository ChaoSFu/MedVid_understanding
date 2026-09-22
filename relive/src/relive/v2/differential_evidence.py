"""Calibrated differential-evidence policy, with no model authority.

This module computes predeclared score contrasts. It deliberately does not
select evidence, call a verifier, read a cache, or create a core certificate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import TALSelectionError, strict_json_loads

FORMAT = "relive-v2-differential-evidence-policy-v1"
STRICT_LOCAL_DEPENDENCE = "STRICT_LOCAL_DEPENDENCE"
TAINTS = frozenset({"HUMAN_ORACLE", "MANUAL_EVIDENCE", "MANUAL_CONTROL", "TEST_GT", "POST_HOC_EDIT"})
ADAPT_ROUTES = frozenset({"TEMPORAL_REACQUIRE", "RESIDUAL_LOCALIZE", "SPATIAL_RECOMPOSE"})
CONTROL_TIERS = frozenset({"TIER_1_EXACT", "TIER_2_MATCHED"})


class DifferentialEvidenceError(ValueError):
    pass


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object(path: str | Path, code: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(Path(path).read_bytes().decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise DifferentialEvidenceError(f"{code}_INVALID") from exc
    if not isinstance(value, dict):
        raise DifferentialEvidenceError(f"{code}_OBJECT_REQUIRED")
    return value


def load_policy(path: str | Path) -> tuple[dict[str, Any], str]:
    policy = _object(path, "DIFFERENTIAL_POLICY")
    required = {"format", "policy_version", "supported_claim_types", "diagnostic_only_interventions", "automatic_control_tiers", "threshold_symbols", "adaptation_budget", "human_oracle_taints", "certificate_bindings", "postcondition_requirements"}
    if set(policy) != required or policy.get("format") != FORMAT:
        raise DifferentialEvidenceError("DIFFERENTIAL_POLICY_SCHEMA_INVALID")
    threshold_symbols = policy.get("threshold_symbols")
    if not isinstance(threshold_symbols, dict) or set(threshold_symbols) != {"tau_original", "tau_keep_floor", "tau_keep", "tau_drop", "tau_evidence", "tau_operator"} or any(value is not None for value in threshold_symbols.values()):
        raise DifferentialEvidenceError("DIFFERENTIAL_POLICY_THRESHOLDS_MUST_BE_SYMBOLIC")
    if (set(policy.get("automatic_control_tiers", [])) != CONTROL_TIERS or set(policy.get("human_oracle_taints", [])) != TAINTS
            or set(policy.get("postcondition_requirements", [])) != {"INTERACTION_END_LOCALIZED", "POST_WINDOW_AVAILABLE", "OPERATOR_SUPPORT_ABSENT"}):
        raise DifferentialEvidenceError("DIFFERENTIAL_POLICY_ENUM_INVALID")
    return policy, sha256_path(path)


@dataclass(frozen=True)
class InterventionVariant:
    variant: str
    score: float | None
    semantic_status: str | None
    renderer_sha256: str

    def __post_init__(self) -> None:
        if self.variant not in {"ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "KEEP_MATCHED_CONTROL", "DROP_MATCHED_CONTROL"}:
            raise DifferentialEvidenceError("INTERVENTION_VARIANT_INVALID")
        if self.score is not None and (not isinstance(self.score, (int, float)) or not float("-inf") < float(self.score) < float("inf")):
            raise DifferentialEvidenceError("CONTINUOUS_SUPPORT_SCORE_INVALID")
        if not isinstance(self.renderer_sha256, str) or len(self.renderer_sha256) != 64:
            raise DifferentialEvidenceError("INTERVENTION_RENDERER_HASH_INVALID")


@dataclass(frozen=True)
class MatchedControlSet:
    tier: str
    keep_controls: tuple[InterventionVariant, ...]
    drop_controls: tuple[InterventionVariant, ...]
    required_evidence_clean: bool
    same_operator: bool
    same_modified_frames: bool
    same_modified_pixels: bool
    geometry_valid: bool
    area_curve_matched: bool
    connectivity_matched: bool
    foreground_fraction_matched: bool
    salience_matched: bool
    occlusion_matched: bool
    tolerance_source_sha256: str | None

    def __post_init__(self) -> None:
        if self.tier not in CONTROL_TIERS:
            raise DifferentialEvidenceError("AUTOMATIC_CONTROL_TIER_INVALID")
        minimum = 1 if self.tier == "TIER_1_EXACT" else 2
        if len(self.keep_controls) < minimum or len(self.drop_controls) < minimum:
            raise DifferentialEvidenceError("MATCHED_CONTROL_CARDINALITY_INSUFFICIENT")
        if any(item.variant != "KEEP_MATCHED_CONTROL" for item in self.keep_controls) or any(item.variant != "DROP_MATCHED_CONTROL" for item in self.drop_controls):
            raise DifferentialEvidenceError("MATCHED_CONTROL_VARIANT_INVALID")
        if not all((self.required_evidence_clean, self.same_operator, self.same_modified_frames, self.same_modified_pixels, self.geometry_valid)):
            raise DifferentialEvidenceError("MATCHED_CONTROL_QUALITY_INVALID")
        if self.tier == "TIER_2_MATCHED":
            if not all((self.area_curve_matched, self.connectivity_matched, self.foreground_fraction_matched,
                        self.salience_matched, self.occlusion_matched)):
                raise DifferentialEvidenceError("TIER_2_MATCHING_QUALITY_INVALID")
            if not isinstance(self.tolerance_source_sha256, str) or len(self.tolerance_source_sha256) != 64:
                raise DifferentialEvidenceError("TIER_2_TOLERANCE_BINDING_INVALID")

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class DifferentialEvidenceMetrics:
    original_score: float
    keep_target_score: float
    drop_target_score: float
    keep_control_worst: float
    drop_control_worst: float
    g_keep: float
    g_drop: float
    strict_local_dependence: bool


@dataclass(frozen=True)
class CalibrationArtifact:
    policy_sha256: str
    thresholds: Mapping[str, float]
    calibration_id: str

    def __post_init__(self) -> None:
        required = {"tau_original", "tau_keep_floor", "tau_keep", "tau_drop", "tau_evidence", "tau_operator"}
        if set(self.thresholds) != required or any(not isinstance(value, (int, float)) for value in self.thresholds.values()):
            raise DifferentialEvidenceError("CALIBRATION_THRESHOLDS_INVALID")
        if not isinstance(self.policy_sha256, str) or len(self.policy_sha256) != 64 or not isinstance(self.calibration_id, str) or not self.calibration_id:
            raise DifferentialEvidenceError("CALIBRATION_BINDING_INVALID")

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class DifferentialCertificate:
    status: str
    diagnostic_labels: tuple[str, ...]
    failure_reason: str | None
    bindings: Mapping[str, str]


def metrics(*, original: InterventionVariant, keep_target: InterventionVariant, drop_target: InterventionVariant,
            controls: MatchedControlSet) -> DifferentialEvidenceMetrics:
    values = (original.score, keep_target.score, drop_target.score, *(item.score for item in controls.keep_controls), *(item.score for item in controls.drop_controls))
    if any(value is None for value in values):
        raise DifferentialEvidenceError("CONTINUOUS_SUPPORT_SCORE_UNAVAILABLE")
    keep_worst = max(float(item.score) for item in controls.keep_controls)
    drop_worst = min(float(item.score) for item in controls.drop_controls)
    strict = (original.semantic_status == "SUPPORTED" and keep_target.semantic_status == "SUPPORTED"
              and drop_target.semantic_status == "INSUFFICIENT"
              and all(item.semantic_status == "SUPPORTED" for item in controls.drop_controls))
    return DifferentialEvidenceMetrics(float(original.score), float(keep_target.score), float(drop_target.score),
        keep_worst, drop_worst, float(keep_target.score) - keep_worst, drop_worst - float(drop_target.score), strict)


def admit(*, policy_sha256: str, calibration: CalibrationArtifact | None, value: DifferentialEvidenceMetrics,
          evidence_quality: float, operator_quality: float, model_revision: str, prompt_hash: str,
          evidence_program_hash: str, control_set_hash: str, intervention_renderer_hash: str,
          provenance: set[str] = frozenset()) -> DifferentialCertificate:
    bindings = {"policy_sha256": policy_sha256, "calibration_artifact_sha256": calibration.sha256 if calibration else "UNAVAILABLE",
                "model_revision": model_revision, "prompt_hash": prompt_hash, "evidence_program_hash": evidence_program_hash,
                "control_set_hash": control_set_hash, "intervention_renderer_hash": intervention_renderer_hash}
    if provenance & TAINTS:
        return DifferentialCertificate("ORACLE_DIAGNOSTIC_ONLY", (), "HUMAN_ORACLE_CONTAMINATION", bindings)
    labels = (STRICT_LOCAL_DEPENDENCE,) if value.strict_local_dependence else ()
    if calibration is None:
        return DifferentialCertificate("DIAGNOSTIC_ONLY", labels, "CALIBRATION_ARTIFACT_REQUIRED", bindings)
    if calibration.policy_sha256 != policy_sha256:
        return DifferentialCertificate("DIAGNOSTIC_ONLY", labels, "CALIBRATION_POLICY_HASH_MISMATCH", bindings)
    tau = calibration.thresholds
    passed = (value.original_score >= tau["tau_original"] and value.keep_target_score >= tau["tau_keep_floor"]
              and value.g_keep >= tau["tau_keep"] and value.g_drop >= tau["tau_drop"]
              and evidence_quality >= tau["tau_evidence"] and operator_quality >= tau["tau_operator"])
    return DifferentialCertificate("CALIBRATED_DIFFERENTIAL_DEPENDENCE" if passed else "DIAGNOSTIC_ONLY", labels,
                                   None if passed else "DIFFERENTIAL_THRESHOLDS_NOT_MET", bindings)


def choose_adapt_route(routes: tuple[str, ...], *, completed_rounds: int) -> str:
    if completed_rounds >= 2:
        raise DifferentialEvidenceError("R1_FORBIDS_R2")
    if len(routes) != 1:
        raise DifferentialEvidenceError("ADAPT_ROUTE_AMBIGUOUS")
    if routes[0] not in ADAPT_ROUTES:
        raise DifferentialEvidenceError("ADAPT_ROUTE_UNSUPPORTED")
    return routes[0]


def differential_cache_key(*, policy_sha256: str, control_set_hash: str,
                           evidence_program_hash: str, prompt_hash: str,
                           model_revision: str) -> str:
    """Identity for a future inference cache; no cache is opened here."""
    return stable_hash({"policy_sha256": policy_sha256, "control_set_hash": control_set_hash,
                        "evidence_program_hash": evidence_program_hash, "prompt_hash": prompt_hash,
                        "model_revision": model_revision})


def verify_closed_pilot(path: str | Path) -> dict[str, Any]:
    closure = _object(path, "PILOT_CLOSURE")
    expected = closure.get("closure_manifest_sha256"); payload = dict(closure); payload.pop("closure_manifest_sha256", None)
    if closure.get("status") != "PILOT_CLOSED" or stable_hash(payload) != expected or closure.get("source_artifacts_unchanged") is not True or closure.get("label_resolution_applied") is not True:
        raise DifferentialEvidenceError("PILOT_CLOSURE_INVALID")
    return closure
