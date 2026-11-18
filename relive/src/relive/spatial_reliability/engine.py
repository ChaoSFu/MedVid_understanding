from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .dataset import parse_proposal
from .operators import regions_equal
from .protocol import (
    MAX_EVIDENCE_REGIONS,
    ScientificStatus,
    classify_formal_result,
    controller_action,
)


def _overlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


@dataclass
class SpatialState:
    sample_id: str
    regions: list[list[float]] = field(default_factory=list)
    repair_attempts: int = 0
    residual_attempts: int = 0
    status: str = "PENDING"
    trace: list[dict[str, Any]] = field(default_factory=list)

    def add_region(self, region: list[float]) -> None:
        if len(self.regions) >= MAX_EVIDENCE_REGIONS:
            raise ValueError("EVIDENCE_SET_REGION_LIMIT_EXCEEDED")
        self.regions.append(region)


def oracle_classification(original: str | None, keep: str | None, drop: str | None) -> str:
    semantic_statuses = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}
    if any(value not in semantic_statuses for value in (original, keep, drop)):
        return "ORACLE_TECHNICAL_FAILURE"
    if original != "SUPPORTED":
        return "BASE_UNSUPPORTED"
    if keep != "SUPPORTED":
        return "ORACLE_SPATIAL_INSUFFICIENT"
    if drop == "SUPPORTED":
        return "ORACLE_RESIDUAL_SUPPORT"
    return "ORACLE_SINGLE_REGION_FEASIBLE"


def run_state_machine(*, sample_id: str, initial_proposal_raw: str,
                      verify_region: Callable[[list[float]], dict[str, Any]],
                      refine: Callable[[list[float]], str] | None = None,
                      residual: Callable[[list[float]], str] | None = None,
                      verify_residual: Callable[[list[float], list[float]], str] | None = None,
                      verify_union: Callable[[list[list[float]]], dict[str, str]] | None = None,
                      mode: str = "relive_spatial") -> dict[str, Any]:
    """Pure bounded controller. Callbacks own inference/cache and intervention I/O."""
    if mode not in {"one_shot", "verify_only", "generic_retry", "relive_spatial"}:
        raise ValueError("UNKNOWN_SPATIAL_BASELINE_MODE")
    state = SpatialState(sample_id)
    proposal = parse_proposal(initial_proposal_raw)
    state.trace.append({"stage": "R0_PROPOSAL", "result": proposal})
    if proposal["status"] != "PROPOSED":
        state.status = ScientificStatus.PROPOSAL_UNRESOLVED.value
        return state.__dict__
    r0 = [value / 1000 for value in proposal["bbox_normalized_0_1000"]]
    state.add_region(r0)
    if mode == "one_shot":
        state.status = "PROPOSAL_ONLY"
        return state.__dict__
    current = r0
    formal = verify_region(current)
    status = classify_formal_result(**formal)
    state.trace.append({"stage": "R0_FORMAL", "result": formal, "status": status.value})
    if mode == "verify_only":
        state.status = status.value
        return state.__dict__
    action = controller_action(status, repair_attempts=state.repair_attempts,
                               residual_attempts=state.residual_attempts)
    if mode == "generic_retry" and action != "STOP":
        action = "GENERIC_RETRY"
    state.trace.append({"stage": "CONTROLLER", "action": action, "from_status": status.value})
    if action in {"SUFFICIENCY_REPAIR", "GENERIC_RETRY"}:
        state.repair_attempts += 1
        if refine is None:
            state.status = ScientificStatus.SPATIAL_SUFFICIENCY_UNRESOLVED.value
            return state.__dict__
        repaired = parse_proposal(refine(current))
        state.trace.append({"stage": "R1_PROPOSAL", "result": repaired, "reason_specific": action == "SUFFICIENCY_REPAIR"})
        if repaired["status"] != "PROPOSED":
            state.status = ScientificStatus.SPATIAL_SUFFICIENCY_UNRESOLVED.value
            return state.__dict__
        r1 = [value / 1000 for value in repaired["bbox_normalized_0_1000"]]
        if regions_equal(current, r1):
            state.status = ScientificStatus.REFINEMENT_NO_OP.value
            return state.__dict__
        state.regions[0], current = r1, r1
        formal = verify_region(current)
        status = classify_formal_result(**formal)
        state.trace.append({"stage": "R1_FORMAL", "result": formal, "status": status.value})
        if status == ScientificStatus.SPATIAL_INSUFFICIENT:
            state.status = ScientificStatus.SPATIAL_SUFFICIENCY_UNRESOLVED.value
            return state.__dict__
        if mode == "generic_retry":
            state.status = status.value
            return state.__dict__
        action = controller_action(status, repair_attempts=state.repair_attempts,
                                   residual_attempts=state.residual_attempts)
        state.trace.append({"stage": "CONTROLLER", "action": action, "from_status": status.value})
    if action == "RESIDUAL_SUPPORT_SEARCH":
        state.residual_attempts += 1
        if residual is None:
            state.status = ScientificStatus.DEPENDENCY_UNRESOLVED.value
            return state.__dict__
        result = parse_proposal(residual(current))
        state.trace.append({"stage": "R2_RESIDUAL_PROPOSAL", "result": result})
        if result["status"] != "PROPOSED":
            state.status = ScientificStatus.DEPENDENCY_UNRESOLVED.value
            return state.__dict__
        r2 = [value / 1000 for value in result["bbox_normalized_0_1000"]]
        if _overlap(current, r2) > 0:
            state.trace.append({"stage": "R2_GEOMETRY", "status": "RESIDUAL_PROPOSAL_OVERLAPS_PRIMARY"})
            state.status = ScientificStatus.DEPENDENCY_UNRESOLVED.value
            return state.__dict__
        residual_status = verify_residual(current, r2) if verify_residual else None
        state.trace.append({"stage": "KEEP_RESIDUAL", "status": residual_status})
        if residual_status != "SUPPORTED":
            state.status = ScientificStatus.RESIDUAL_REGION_NOT_SUFFICIENT.value
            return state.__dict__
        state.add_region(r2)
        union = verify_union(state.regions) if verify_union else {}
        state.trace.append({"stage": "EVIDENCE_SET_FORMAL", "result": union})
        if union.get("keep") == "SUPPORTED" and union.get("drop") != "SUPPORTED":
            state.status = ScientificStatus.EVIDENCE_SET_VERIFIED.value
        else:
            state.status = ScientificStatus.DEPENDENCY_UNRESOLVED.value
        return state.__dict__
    state.status = status.value
    return state.__dict__
