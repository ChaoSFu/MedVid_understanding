"""Pure, diagnostic-only ReliVE-v2 failure routing and cycle guards."""
from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Collection
from relive.storage.artifacts import stable_hash
from .failure_taxonomy import DiagnosisOutcome, VerificationDiagnosis, VerificationFailure

class AdaptationControllerError(ValueError):
    pass

class AdaptationAction(str, Enum):
    STOP_DIAGNOSTIC = "STOP_DIAGNOSTIC"
    STOP_ABSTAIN = "STOP_ABSTAIN"
    TEMPORAL_REACQUIRE = "TEMPORAL_REACQUIRE"
    TEMPORAL_EXPAND_CONTEXT = "TEMPORAL_EXPAND_CONTEXT"
    TEMPORAL_DISCRIMINATIVE_RETRIEVAL = "TEMPORAL_DISCRIMINATIVE_RETRIEVAL"
    SPATIAL_RECOMPOSE = "SPATIAL_RECOMPOSE"
    SPATIAL_RECOMPOSE_CONTROL_GEOMETRY = "SPATIAL_RECOMPOSE_CONTROL_GEOMETRY"
    RESIDUAL_LOCALIZE = "RESIDUAL_LOCALIZE"
    VERIFIER_SWITCH = "VERIFIER_SWITCH"
    INTERVENTION_DIAGNOSTIC = "INTERVENTION_DIAGNOSTIC"
    TECHNICAL_DIAGNOSTIC = "TECHNICAL_DIAGNOSTIC"

_CONSUMES = {
    AdaptationAction.TEMPORAL_REACQUIRE: "temporal_jumps_remaining",
    AdaptationAction.TEMPORAL_EXPAND_CONTEXT: "boundary_refinements_remaining",
    AdaptationAction.TEMPORAL_DISCRIMINATIVE_RETRIEVAL: "temporal_jumps_remaining",
    AdaptationAction.SPATIAL_RECOMPOSE: "spatial_regrounds_remaining",
    AdaptationAction.SPATIAL_RECOMPOSE_CONTROL_GEOMETRY: "spatial_regrounds_remaining",
    AdaptationAction.RESIDUAL_LOCALIZE: "residual_components_remaining",
    AdaptationAction.VERIFIER_SWITCH: "verifier_switches_remaining",
}

@dataclass(frozen=True)
class ClaimBudget:
    temporal_jumps_remaining: int
    boundary_refinements_remaining: int
    spatial_regrounds_remaining: int
    residual_components_remaining: int
    verifier_switches_remaining: int
    def __post_init__(self) -> None:
        for value in (self.temporal_jumps_remaining, self.boundary_refinements_remaining, self.spatial_regrounds_remaining, self.residual_components_remaining, self.verifier_switches_remaining):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AdaptationControllerError("BUDGET_NONNEGATIVE_INTEGER_REQUIRED")
    def to_canonical_dict(self) -> dict:
        return {"temporal_jumps_remaining": self.temporal_jumps_remaining, "boundary_refinements_remaining": self.boundary_refinements_remaining,
                "spatial_regrounds_remaining": self.spatial_regrounds_remaining, "residual_components_remaining": self.residual_components_remaining,
                "verifier_switches_remaining": self.verifier_switches_remaining}

@dataclass(frozen=True)
class ReliVEState:
    requirement_sha256: str
    claim_graph_sha256: str
    active_claim_id: str
    temporal_window_sha256: str | None
    evidence_geometry_sha256: str | None
    verifier_id: str
    budget: ClaimBudget
    round_index: int
    def __post_init__(self) -> None:
        for name, value in (("requirement_sha256", self.requirement_sha256), ("claim_graph_sha256", self.claim_graph_sha256), ("active_claim_id", self.active_claim_id), ("verifier_id", self.verifier_id)):
            if not isinstance(value, str) or not value:
                raise AdaptationControllerError(f"{name.upper()}_REQUIRED")
        if self.temporal_window_sha256 is not None and not isinstance(self.temporal_window_sha256, str):
            raise AdaptationControllerError("TEMPORAL_WINDOW_HASH_INVALID")
        if self.evidence_geometry_sha256 is not None and not isinstance(self.evidence_geometry_sha256, str):
            raise AdaptationControllerError("EVIDENCE_GEOMETRY_HASH_INVALID")
        if not isinstance(self.budget, ClaimBudget) or not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index < 0:
            raise AdaptationControllerError("STATE_FIELDS_INVALID")

@dataclass(frozen=True)
class FrozenAdaptationPolicy:
    policy_version: str
    allowed_actions: tuple[AdaptationAction, ...]
    verifier_switch_enabled: bool
    def __post_init__(self) -> None:
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise AdaptationControllerError("POLICY_VERSION_REQUIRED")
        if not isinstance(self.allowed_actions, tuple) or any(not isinstance(action, AdaptationAction) for action in self.allowed_actions) or len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise AdaptationControllerError("CLOSED_ALLOWED_ACTIONS_REQUIRED")
        if not isinstance(self.verifier_switch_enabled, bool):
            raise AdaptationControllerError("VERIFIER_SWITCH_FLAG_REQUIRED")

@dataclass(frozen=True)
class TransitionDecision:
    action: AdaptationAction
    decision_reason: str
    consumed_budget_dimension: str | None
    budget_before: ClaimBudget
    budget_after: ClaimBudget
    controller_policy_version: str
    source_content_state_sha256: str
    diagnosis_sha256: str
    termination_reason: str | None
    diagnostic_only: bool = True
    certificate_authority: bool = False
    def to_canonical_dict(self) -> dict:
        return {"action": self.action.value, "decision_reason": self.decision_reason, "consumed_budget_dimension": self.consumed_budget_dimension,
                "budget_before": self.budget_before.to_canonical_dict(), "budget_after": self.budget_after.to_canonical_dict(),
                "controller_policy_version": self.controller_policy_version, "source_content_state_sha256": self.source_content_state_sha256,
                "diagnosis_sha256": self.diagnosis_sha256, "termination_reason": self.termination_reason,
                "diagnostic_only": True, "certificate_authority": False}

@dataclass(frozen=True)
class CycleCheck:
    cycle_detected: bool
    action: AdaptationAction | None
    termination_reason: str | None
    proposed_content_state_sha256: str


def content_state_sha256(state: ReliVEState) -> str:
    return stable_hash({"requirement_sha256": state.requirement_sha256, "claim_graph_sha256": state.claim_graph_sha256,
                        "active_claim_id": state.active_claim_id, "temporal_window_sha256": state.temporal_window_sha256,
                        "evidence_geometry_sha256": state.evidence_geometry_sha256, "verifier_id": state.verifier_id})

def event_state_sha256(state: ReliVEState) -> str:
    return stable_hash({"content_state_sha256": content_state_sha256(state), "budget": state.budget.to_canonical_dict(), "round_index": state.round_index})

def consume_budget(budget: ClaimBudget, action: AdaptationAction) -> ClaimBudget:
    dimension = _CONSUMES.get(action)
    if dimension is None:
        return budget
    if getattr(budget, dimension) <= 0:
        raise AdaptationControllerError(f"BUDGET_{dimension.removesuffix('_remaining').upper()}_EXHAUSTED")
    return replace(budget, **{dimension: getattr(budget, dimension) - 1})

def _route(diagnosis: VerificationDiagnosis) -> tuple[AdaptationAction, str]:
    if diagnosis.outcome is DiagnosisOutcome.DIAGNOSTIC_COMPLETE:
        return AdaptationAction.STOP_DIAGNOSTIC, "DIAGNOSTIC_COMPLETE"
    if diagnosis.outcome is DiagnosisOutcome.TECHNICAL_FAILURE:
        return AdaptationAction.TECHNICAL_DIAGNOSTIC, "TECHNICAL_FAILURE"
    mapping = {VerificationFailure.ORIGINAL_INSUFFICIENT: AdaptationAction.TEMPORAL_REACQUIRE,
               VerificationFailure.TEMPORAL_ORDER_UNRESOLVED: AdaptationAction.TEMPORAL_EXPAND_CONTEXT,
               VerificationFailure.KEEP_SUPPORT_LOST: AdaptationAction.SPATIAL_RECOMPOSE,
               VerificationFailure.DROP_SUPPORT_PERSISTS: AdaptationAction.RESIDUAL_LOCALIZE,
               VerificationFailure.DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT: AdaptationAction.VERIFIER_SWITCH,
               VerificationFailure.CONTROL_UNAVAILABLE: AdaptationAction.SPATIAL_RECOMPOSE_CONTROL_GEOMETRY,
               VerificationFailure.CONTROL_SUPPORT_LOST: AdaptationAction.INTERVENTION_DIAGNOSTIC,
               VerificationFailure.HYPOTHESES_NOT_DISCRIMINATED: AdaptationAction.TEMPORAL_DISCRIMINATIVE_RETRIEVAL,
               VerificationFailure.TECHNICAL_EXECUTION_FAILURE: AdaptationAction.TECHNICAL_DIAGNOSTIC}
    if diagnosis.failure not in mapping:
        raise AdaptationControllerError("UNROUTABLE_DIAGNOSIS")
    return mapping[diagnosis.failure], diagnosis.failure.value

def _decision(state: ReliVEState, diagnosis: VerificationDiagnosis, policy: FrozenAdaptationPolicy, action: AdaptationAction, reason: str, termination: str | None = None) -> TransitionDecision:
    dimension = _CONSUMES.get(action)
    before = state.budget
    return TransitionDecision(action, reason, dimension, before, before if dimension is None else consume_budget(before, action), policy.policy_version, content_state_sha256(state), diagnosis.content_sha256(), termination)

def decide_transition(state: ReliVEState, diagnosis: VerificationDiagnosis, policy: FrozenAdaptationPolicy) -> TransitionDecision:
    action, reason = _route(diagnosis)
    if action is AdaptationAction.VERIFIER_SWITCH and not policy.verifier_switch_enabled:
        return _decision(state, diagnosis, policy, AdaptationAction.STOP_ABSTAIN, reason, "VERIFIER_SWITCH_NOT_PREDECLARED")
    if action not in policy.allowed_actions:
        return _decision(state, diagnosis, policy, AdaptationAction.STOP_ABSTAIN, reason, "POLICY_ACTION_NOT_ALLOWED")
    dimension = _CONSUMES.get(action)
    if dimension is not None and getattr(state.budget, dimension) == 0:
        return _decision(state, diagnosis, policy, AdaptationAction.STOP_ABSTAIN, reason, f"BUDGET_{dimension.removesuffix('_remaining').upper()}_EXHAUSTED")
    return _decision(state, diagnosis, policy, action, reason)

def guard_proposed_state(proposed_state: ReliVEState, visited_content_state_hashes: Collection[str]) -> CycleCheck:
    digest = content_state_sha256(proposed_state)
    if digest in visited_content_state_hashes:
        return CycleCheck(True, AdaptationAction.STOP_ABSTAIN, "STATE_CYCLE_DETECTED", digest)
    return CycleCheck(False, None, None, digest)

def default_policy() -> FrozenAdaptationPolicy:
    return FrozenAdaptationPolicy("relive-v2-failure-routing-policy-v1", tuple(AdaptationAction), True)
