"""ReliVE-v1 protocol records. Semantic and execution outcomes are distinct."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class SemanticStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class ExecutionStatus(str, Enum):
    OK = "OK"
    PARSE_ERROR = "PARSE_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    INVALID_INPUT = "INVALID_INPUT"


class FinalStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"


class ExclusivityStatus(str, Enum):
    DECLARED_EXCLUSIVE = "DECLARED_EXCLUSIVE"
    NONEXCLUSIVE = "NONEXCLUSIVE"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Frame:
    frame_id: str
    path: str
    order: int
    timestamp: float | None = None
    timestamp_source: str | None = None
    source_reference: str | None = None


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    entity: str | None = None
    action: str | None = None
    target: str | None = None
    time_scope: dict[str, Any] = field(default_factory=dict)
    source: str = "runtime_target"
    required_for_question: bool = False


@dataclass(frozen=True)
class RuntimeSample:
    sample_id: str
    task: str
    question: str
    frames: tuple[Frame, ...]
    target_claim: Claim | None = None
    required_claims: tuple[Claim, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceCandidate:
    candidate_id: str
    sample_id: str
    frame_ids: tuple[str, ...]
    timestamps: tuple[float | None, ...]
    acquisition_rank: int
    rank_source: str
    retrieval_score: float | None = None
    parent_id: str | None = None
    round_index: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContrastGroup:
    group_id: str
    claim_ids: tuple[str, ...]
    comparison_dimension: str
    exclusivity_status: ExclusivityStatus
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialProposal:
    proposal_id: str
    candidate_id: str
    claim_id: str
    support_region: tuple[float, float, float, float] | None
    coordinate_system: str = "normalized_0_1_xyxy"
    target_bbox: tuple[float, float, float, float] | None = None
    frame_mapping: dict[str, Any] = field(default_factory=dict)
    parser_status: ExecutionStatus = ExecutionStatus.OK
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    semantic_status: SemanticStatus | None
    execution_status: ExecutionStatus
    raw_response_ref: str | None
    prompt_version: str
    input_references: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    observation: str | None = None
    frame_references: tuple[str, ...] = ()

    def __post_init__(self):
        if (self.execution_status == ExecutionStatus.OK) != (self.semantic_status is not None):
            raise ValueError("Semantic result exists exactly when execution status is OK")


@dataclass(frozen=True)
class EvidenceCertificate:
    certificate_id: str
    candidate_id: str
    claim_id: str
    time_scope: dict[str, Any]
    checks: dict[str, Any]
    final_status: FinalStatus
    failure_reasons: tuple[str, ...]
    policy_version: str
    policy_name: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoverageResult:
    required_claims: tuple[str, ...]
    supported_claims: tuple[str, ...]
    missing_claims: tuple[str, ...]
    unresolved_relations: tuple[str, ...]
    status: str
    claim_aggregations: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptationEvent:
    event_id: str
    previous_candidate_id: str | None
    previous_claim_id: str | None
    previous_certificate_id: str | None
    action: str
    parameters: dict[str, Any]
    next_candidate_id: str | None
    budget_usage: dict[str, Any]
    termination_reason: str | None = None


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return to_dict(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(v) for v in value]
    return value
