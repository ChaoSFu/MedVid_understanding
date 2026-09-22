"""Protocol-first, model-free ReliVE-v2 contracts.

The contracts in this module describe evidence programs before any model,
segmenter, tracker, or verifier is invoked.  They are intentionally generic:
no dataset, video, action, anchor, or host path is part of the schema.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from relive.storage.artifacts import stable_hash


PROTOCOL_VERSION = "relive-v2-protocol-first-v1"


class ProtocolContractError(ValueError):
    pass


class ClaimType(str, Enum):
    ENTITY_STATE = "ENTITY_STATE"
    SPATIAL_RELATION = "SPATIAL_RELATION"
    CONTACT_ACTION = "CONTACT_ACTION"
    STATE_CHANGE_OR_PERSISTENCE = "STATE_CHANGE_OR_PERSISTENCE"


class OcclusionState(str, Enum):
    VISIBLE = "VISIBLE"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    OCCLUDED = "OCCLUDED"
    AMBIGUOUS = "AMBIGUOUS"


class CertificateStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    ENGINEERING_FAILURE = "ENGINEERING_FAILURE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


CLAIM_ROLE_TEMPLATES: dict[ClaimType, tuple[str, ...]] = {
    ClaimType.ENTITY_STATE: ("SUBJECT", "STATE_INDICATOR"),
    ClaimType.SPATIAL_RELATION: ("SUBJECT", "REFERENCE_ENTITY", "RELATION_INTERFACE"),
    ClaimType.CONTACT_ACTION: ("ACTOR", "ACTION_TARGET", "CONTACT_INTERFACE"),
    ClaimType.STATE_CHANGE_OR_PERSISTENCE: ("SUBJECT", "PRE_STATE", "POST_STATE", "CHANGE_INTERFACE"),
}


def _required_text(value: str, code: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolContractError(code)


def _hash(value: str, code: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProtocolContractError(code)


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    role: str
    frame_sha256: str
    mask_sha256: str | None
    occlusion_state: OcclusionState

    def __post_init__(self) -> None:
        _required_text(self.node_id, "EVIDENCE_NODE_ID_REQUIRED")
        _required_text(self.role, "EVIDENCE_NODE_ROLE_REQUIRED")
        _hash(self.frame_sha256, "EVIDENCE_NODE_FRAME_HASH_INVALID")
        if self.mask_sha256 is not None:
            _hash(self.mask_sha256, "EVIDENCE_NODE_MASK_HASH_INVALID")


@dataclass(frozen=True)
class EvidenceTube:
    tube_id: str
    node_ids: tuple[str, ...]
    temporal_window: tuple[float, float]
    tracker_version: str | None

    def __post_init__(self) -> None:
        _required_text(self.tube_id, "EVIDENCE_TUBE_ID_REQUIRED")
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ProtocolContractError("EVIDENCE_TUBE_NODE_IDS_INVALID")
        if (len(self.temporal_window) != 2 or not all(isinstance(item, (int, float)) for item in self.temporal_window)
                or self.temporal_window[0] > self.temporal_window[1]):
            raise ProtocolContractError("EVIDENCE_TUBE_WINDOW_INVALID")


@dataclass(frozen=True)
class RelationEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation: str
    interface_node_id: str | None

    def __post_init__(self) -> None:
        for value, code in ((self.edge_id, "RELATION_EDGE_ID_REQUIRED"), (self.source_node_id, "RELATION_EDGE_SOURCE_REQUIRED"),
                            (self.target_node_id, "RELATION_EDGE_TARGET_REQUIRED"), (self.relation, "RELATION_EDGE_RELATION_REQUIRED")):
            _required_text(value, code)


@dataclass(frozen=True)
class InterventionPlan:
    plan_id: str
    family: str
    target_node_ids: tuple[str, ...]
    matched_control_tier: str
    operator_version: str
    budget_cost: int

    def __post_init__(self) -> None:
        for value, code in ((self.plan_id, "INTERVENTION_PLAN_ID_REQUIRED"), (self.family, "INTERVENTION_FAMILY_REQUIRED"),
                            (self.matched_control_tier, "MATCHED_CONTROL_TIER_REQUIRED"), (self.operator_version, "OPERATOR_VERSION_REQUIRED")):
            _required_text(value, code)
        if not self.target_node_ids or len(set(self.target_node_ids)) != len(self.target_node_ids):
            raise ProtocolContractError("INTERVENTION_TARGETS_INVALID")
        if type(self.budget_cost) is not int or self.budget_cost < 0:
            raise ProtocolContractError("INTERVENTION_BUDGET_INVALID")


@dataclass(frozen=True)
class AdaptationDecision:
    decision_id: str
    route: str
    maximum_rounds: int
    cycle_key: str
    abstention_reason: str | None

    def __post_init__(self) -> None:
        _required_text(self.decision_id, "ADAPTATION_DECISION_ID_REQUIRED")
        _required_text(self.route, "ADAPTATION_ROUTE_REQUIRED")
        _required_text(self.cycle_key, "ADAPTATION_CYCLE_KEY_REQUIRED")
        if type(self.maximum_rounds) is not int or self.maximum_rounds < 0:
            raise ProtocolContractError("ADAPTATION_MAXIMUM_ROUNDS_INVALID")


@dataclass(frozen=True)
class Certificate:
    certificate_id: str
    status: CertificateStatus
    evidence_program_sha256: str
    necessary_conditions: tuple[str, ...]
    abstention_reason: str | None

    def __post_init__(self) -> None:
        _required_text(self.certificate_id, "CERTIFICATE_ID_REQUIRED")
        _hash(self.evidence_program_sha256, "CERTIFICATE_PROGRAM_HASH_INVALID")
        if self.status is CertificateStatus.VERIFIED and not self.necessary_conditions:
            raise ProtocolContractError("VERIFIED_CERTIFICATE_CONDITIONS_REQUIRED")
        if self.status is not CertificateStatus.VERIFIED and self.abstention_reason is None:
            raise ProtocolContractError("NONVERIFIED_CERTIFICATE_ABSTENTION_REQUIRED")


@dataclass(frozen=True)
class EvidenceProgram:
    program_id: str
    claim_type: ClaimType
    required_roles: tuple[str, ...]
    nodes: tuple[EvidenceNode, ...]
    tubes: tuple[EvidenceTube, ...]
    edges: tuple[RelationEdge, ...]
    intervention_plans: tuple[InterventionPlan, ...]
    adaptation: AdaptationDecision

    def __post_init__(self) -> None:
        _required_text(self.program_id, "EVIDENCE_PROGRAM_ID_REQUIRED")
        if self.required_roles != CLAIM_ROLE_TEMPLATES[self.claim_type]:
            raise ProtocolContractError("CLAIM_TYPE_REQUIRED_ROLE_CONTRACT_INVALID")
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ProtocolContractError("EVIDENCE_NODE_IDS_DUPLICATE")
        if any(node.role not in self.required_roles for node in self.nodes):
            raise ProtocolContractError("EVIDENCE_NODE_ROLE_UNDECLARED")
        if any(not set(tube.node_ids).issubset(ids) for tube in self.tubes):
            raise ProtocolContractError("EVIDENCE_TUBE_NODE_REFERENCE_INVALID")
        if any(edge.source_node_id not in ids or edge.target_node_id not in ids or (edge.interface_node_id is not None and edge.interface_node_id not in ids) for edge in self.edges):
            raise ProtocolContractError("RELATION_EDGE_NODE_REFERENCE_INVALID")
        if any(not set(plan.target_node_ids).issubset(ids) for plan in self.intervention_plans):
            raise ProtocolContractError("INTERVENTION_PLAN_NODE_REFERENCE_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceProgram":
        """Recreate the closed contract and validate every nested reference."""
        expected = {"program_id", "claim_type", "required_roles", "nodes", "tubes", "edges", "intervention_plans", "adaptation"}
        if set(value) != expected:
            raise ProtocolContractError("EVIDENCE_PROGRAM_SCHEMA_INVALID")
        try:
            return cls(
                program_id=value["program_id"], claim_type=ClaimType(value["claim_type"]),
                required_roles=tuple(value["required_roles"]),
                nodes=tuple(EvidenceNode(node_id=item["node_id"], role=item["role"], frame_sha256=item["frame_sha256"],
                    mask_sha256=item["mask_sha256"], occlusion_state=OcclusionState(item["occlusion_state"])) for item in value["nodes"]),
                tubes=tuple(EvidenceTube(tube_id=item["tube_id"], node_ids=tuple(item["node_ids"]),
                    temporal_window=tuple(item["temporal_window"]), tracker_version=item["tracker_version"]) for item in value["tubes"]),
                edges=tuple(RelationEdge(**item) for item in value["edges"]),
                intervention_plans=tuple(InterventionPlan(plan_id=item["plan_id"], family=item["family"],
                    target_node_ids=tuple(item["target_node_ids"]), matched_control_tier=item["matched_control_tier"],
                    operator_version=item["operator_version"], budget_cost=item["budget_cost"]) for item in value["intervention_plans"]),
                adaptation=AdaptationDecision(**value["adaptation"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolContractError("EVIDENCE_PROGRAM_SCHEMA_INVALID") from exc

    @property
    def sha256(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class ProtocolManifest:
    protocol_version: str
    program_sha256: str
    split: str
    source_video_sha256: str
    frame_sha256s: tuple[str, ...]
    mode: str

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolContractError("PROTOCOL_VERSION_UNSUPPORTED")
        _hash(self.program_sha256, "PROTOCOL_MANIFEST_PROGRAM_HASH_INVALID")
        _hash(self.source_video_sha256, "PROTOCOL_MANIFEST_VIDEO_HASH_INVALID")
        if self.split not in {"protocol_dev", "calibration", "blind_test"}:
            raise ProtocolContractError("PROTOCOL_MANIFEST_SPLIT_INVALID")
        if self.mode not in {"automatic", "human_oracle"}:
            raise ProtocolContractError("PROTOCOL_MANIFEST_MODE_INVALID")
        if not self.frame_sha256s or len(set(self.frame_sha256s)) != len(self.frame_sha256s):
            raise ProtocolContractError("PROTOCOL_MANIFEST_FRAME_HASHES_INVALID")
        for frame_hash in self.frame_sha256s:
            _hash(frame_hash, "PROTOCOL_MANIFEST_FRAME_HASH_INVALID")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_sha256"] = stable_hash(payload)
        return payload
