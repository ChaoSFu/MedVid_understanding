"""Immutable, CPU-only ReliVE-v2 protocol contracts.

This module deliberately has no dependency on a backend, cache, runtime frame,
or certificate implementation.  Its records describe future work; they do not
admit evidence or make an answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Any, Mapping, Sequence, TypeVar

from relive.storage.artifacts import stable_hash


class ContractError(ValueError):
    pass


class TaskType(str, Enum):
    TEMPORAL_LOCALIZATION = "TEMPORAL_LOCALIZATION"


class AnswerSchema(str, Enum):
    EVENT_INTERVAL = "EVENT_INTERVAL"
    CLOSED_LABEL = "CLOSED_LABEL"


class TemporalRequirement(str, Enum):
    LOCALIZE_INTERVAL = "LOCALIZE_INTERVAL"
    ORDERED_CONTEXT = "ORDERED_CONTEXT"


class SpatialRequirement(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    SINGLE_REGION = "SINGLE_REGION"
    RELATIONAL_COMPOSITE = "RELATIONAL_COMPOSITE"


class ClaimRole(str, Enum):
    TARGET_HYPOTHESIS = "TARGET_HYPOTHESIS"
    NULL_HYPOTHESIS = "NULL_HYPOTHESIS"
    OBSERVATION = "OBSERVATION"


class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class TemporalQuantifier(str, Enum):
    AT_FRAME = "AT_FRAME"
    EXISTS_IN_WINDOW = "EXISTS_IN_WINDOW"
    THROUGHOUT_WINDOW = "THROUGHOUT_WINDOW"
    EVENT_INTERVAL = "EVENT_INTERVAL"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    PERSISTS_AFTER = "PERSISTS_AFTER"
    UNRESOLVED = "UNRESOLVED"


class TemporalScope(str, Enum):
    INSTANT = "INSTANT"
    WINDOW = "WINDOW"
    INTERVAL = "INTERVAL"
    RELATIVE = "RELATIVE"
    UNRESOLVED = "UNRESOLVED"


class LogicalScope(str, Enum):
    LOCAL = "LOCAL"
    EXISTENTIAL = "EXISTENTIAL"
    GLOBAL = "GLOBAL"
    UNRESOLVED = "UNRESOLVED"


class Observability(str, Enum):
    DIRECTLY_VISIBLE = "DIRECTLY_VISIBLE"
    APPARENT_2D = "APPARENT_2D"
    INFERENTIAL_NOT_ADMISSIBLE = "INFERENTIAL_NOT_ADMISSIBLE"
    UNRESOLVED = "UNRESOLVED"


class EvidenceGeometryType(str, Enum):
    SINGLE_REGION = "SINGLE_REGION"
    RELATIONAL_COMPOSITE = "RELATIONAL_COMPOSITE"
    DYNAMIC_SUPPORT_TUBE = "DYNAMIC_SUPPORT_TUBE"
    EVIDENCE_SET = "EVIDENCE_SET"
    GLOBAL_CONTEXT = "GLOBAL_CONTEXT"
    UNRESOLVED = "UNRESOLVED"


class GenerationReason(str, Enum):
    USER_AUTHORED = "USER_AUTHORED"
    HYPOTHESIS_ENUMERATION = "HYPOTHESIS_ENUMERATION"
    OBSERVATION_DECOMPOSITION = "OBSERVATION_DECOMPOSITION"
    FAILURE_ROUTED_PROPOSAL = "FAILURE_ROUTED_PROPOSAL"


@dataclass(frozen=True)
class FrozenJSONObject:
    items: tuple[tuple[str, "FrozenJSON"], ...]

    def __post_init__(self) -> None:
        if tuple(key for key, _ in self.items) != tuple(sorted(key for key, _ in self.items)):
            raise ContractError("FROZEN_JSON_KEYS_MUST_BE_SORTED")
        if len({key for key, _ in self.items}) != len(self.items):
            raise ContractError("FROZEN_JSON_DUPLICATE_KEY")

    def to_dict(self) -> dict[str, Any]:
        return {key: thaw_json(value) for key, value in self.items}

    def __getitem__(self, key: str) -> "FrozenJSON":
        for current, value in self.items:
            if current == key:
                return value
        raise KeyError(key)


FrozenJSON = type(None) | bool | int | float | str | tuple["FrozenJSON", ...] | FrozenJSONObject


def freeze_json(value: Any) -> FrozenJSON:
    """Defensively copy JSON-compatible values and reject non-finite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("FROZEN_JSON_FINITE_NUMBER_REQUIRED")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ContractError("FROZEN_JSON_OBJECT_KEYS_MUST_BE_NONEMPTY_STRINGS")
        return FrozenJSONObject(tuple((key, freeze_json(value[key])) for key in sorted(value)))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise ContractError("FROZEN_JSON_VALUE_REQUIRED")


def thaw_json(value: FrozenJSON) -> Any:
    if isinstance(value, FrozenJSONObject):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_string(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(code)
    return value


def _enum(value: Any, kind: type[Enum], code: str) -> None:
    if not isinstance(value, kind):
        raise ContractError(code)


def _prohibited_provenance(value: FrozenJSON) -> bool:
    forbidden = ("frame", "roi", "bbox", "mask", "model_result", "certificate", "reference_answer", "temporal_gt", "ground_truth", "hypothesis_interval")
    if isinstance(value, FrozenJSONObject):
        for key, nested in value.items:
            normalized = key.casefold()
            if any(token in normalized for token in forbidden):
                return True
            if _prohibited_provenance(nested):
                return True
    elif isinstance(value, tuple):
        return any(_prohibited_provenance(item) for item in value)
    return False


def _content_id(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}_{stable_hash(payload)[:24]}"


@dataclass(frozen=True)
class RequirementSpec:
    requirement_id: str
    task: TaskType
    question_text: str
    question_sha256: str
    target_event: str
    answer_schema: AnswerSchema
    required_evidence: tuple[str, ...]
    temporal_requirement: TemporalRequirement
    spatial_requirement: SpatialRequirement
    planner_version: str
    provenance: FrozenJSON

    def __post_init__(self) -> None:
        _enum(self.task, TaskType, "TASK_TYPE_REQUIRED")
        _enum(self.answer_schema, AnswerSchema, "ANSWER_SCHEMA_REQUIRED")
        _enum(self.temporal_requirement, TemporalRequirement, "TEMPORAL_REQUIREMENT_REQUIRED")
        _enum(self.spatial_requirement, SpatialRequirement, "SPATIAL_REQUIREMENT_REQUIRED")
        _required_string(self.question_text, "QUESTION_TEXT_REQUIRED")
        _required_string(self.target_event, "TARGET_EVENT_REQUIRED")
        _required_string(self.planner_version, "PLANNER_VERSION_REQUIRED")
        if self.question_sha256 != _sha256_text(self.question_text):
            raise ContractError("QUESTION_SHA256_MISMATCH")
        if not isinstance(self.required_evidence, tuple) or not self.required_evidence or any(not isinstance(x, str) or not x.strip() for x in self.required_evidence):
            raise ContractError("REQUIRED_EVIDENCE_NONEMPTY_TUPLE_REQUIRED")
        if len(set(self.required_evidence)) != len(self.required_evidence):
            raise ContractError("REQUIRED_EVIDENCE_MUST_BE_UNIQUE")
        if not isinstance(self.provenance, (FrozenJSONObject, tuple, str, int, float, bool, type(None))):
            raise ContractError("FROZEN_PROVENANCE_REQUIRED")
        if _prohibited_provenance(self.provenance):
            raise ContractError("REQUIREMENT_PROHIBITED_PROTOCOL_INPUT")
        if self.requirement_id != _content_id("requirement", self._semantic_payload()):
            raise ContractError("REQUIREMENT_ID_CONTENT_MISMATCH")

    def _semantic_payload(self) -> dict[str, Any]:
        return {"task": self.task.value, "question_text": self.question_text, "question_sha256": self.question_sha256,
                "target_event": self.target_event, "answer_schema": self.answer_schema.value,
                "required_evidence": list(self.required_evidence), "temporal_requirement": self.temporal_requirement.value,
                "spatial_requirement": self.spatial_requirement.value, "planner_version": self.planner_version,
                "provenance": thaw_json(self.provenance)}

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"requirement_id": self.requirement_id, **self._semantic_payload()}

    def content_sha256(self) -> str:
        return stable_hash(self.to_canonical_dict())


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    requirement_id: str
    parent_claim_id: str | None
    claim_role: ClaimRole
    surface_text: str
    subject: str
    predicate: str
    object: str | None
    polarity: Polarity
    temporal_quantifier: TemporalQuantifier
    temporal_scope: TemporalScope
    logical_scope: LogicalScope
    observability: Observability
    evidence_geometry: EvidenceGeometryType
    required_components: tuple[str, ...]
    generation_reason: GenerationReason
    provenance: FrozenJSON

    def __post_init__(self) -> None:
        _required_string(self.requirement_id, "CLAIM_REQUIREMENT_ID_REQUIRED")
        if self.parent_claim_id is not None:
            _required_string(self.parent_claim_id, "CLAIM_PARENT_ID_INVALID")
        for value, kind, code in ((self.claim_role, ClaimRole, "CLAIM_ROLE_REQUIRED"), (self.polarity, Polarity, "POLARITY_REQUIRED"),
                                  (self.temporal_quantifier, TemporalQuantifier, "TEMPORAL_QUANTIFIER_REQUIRED"),
                                  (self.temporal_scope, TemporalScope, "TEMPORAL_SCOPE_REQUIRED"), (self.logical_scope, LogicalScope, "LOGICAL_SCOPE_REQUIRED"),
                                  (self.observability, Observability, "OBSERVABILITY_REQUIRED"), (self.evidence_geometry, EvidenceGeometryType, "EVIDENCE_GEOMETRY_REQUIRED"),
                                  (self.generation_reason, GenerationReason, "GENERATION_REASON_REQUIRED")):
            _enum(value, kind, code)
        for value, code in ((self.surface_text, "SURFACE_TEXT_REQUIRED"), (self.subject, "SUBJECT_REQUIRED"), (self.predicate, "PREDICATE_REQUIRED")):
            _required_string(value, code)
        if self.object is not None:
            _required_string(self.object, "OBJECT_INVALID")
        if not isinstance(self.required_components, tuple) or any(not isinstance(x, str) or not x.strip() for x in self.required_components):
            raise ContractError("CLAIM_COMPONENTS_TUPLE_REQUIRED")
        if len(set(self.required_components)) != len(self.required_components):
            raise ContractError("CLAIM_COMPONENTS_MUST_BE_UNIQUE")
        if not isinstance(self.provenance, (FrozenJSONObject, tuple, str, int, float, bool, type(None))):
            raise ContractError("FROZEN_PROVENANCE_REQUIRED")
        if _prohibited_provenance(self.provenance):
            raise ContractError("CLAIM_PROHIBITED_PROTOCOL_INPUT")
        if self.claim_id != _content_id("claim", self._semantic_payload()):
            raise ContractError("CLAIM_ID_CONTENT_MISMATCH")

    def _semantic_payload(self) -> dict[str, Any]:
        return {"requirement_id": self.requirement_id, "parent_claim_id": self.parent_claim_id,
                "claim_role": self.claim_role.value, "surface_text": self.surface_text, "subject": self.subject,
                "predicate": self.predicate, "object": self.object, "polarity": self.polarity.value,
                "temporal_quantifier": self.temporal_quantifier.value, "temporal_scope": self.temporal_scope.value,
                "logical_scope": self.logical_scope.value, "observability": self.observability.value,
                "evidence_geometry": self.evidence_geometry.value, "required_components": list(self.required_components),
                "generation_reason": self.generation_reason.value, "provenance": thaw_json(self.provenance)}

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, **self._semantic_payload()}

    def content_sha256(self) -> str:
        return stable_hash(self.to_canonical_dict())


def make_requirement_spec(*, task: TaskType, question_text: str, target_event: str, answer_schema: AnswerSchema,
                          required_evidence: Sequence[str], temporal_requirement: TemporalRequirement,
                          spatial_requirement: SpatialRequirement, planner_version: str, provenance: Any) -> RequirementSpec:
    frozen = freeze_json(provenance)
    question_sha256 = _sha256_text(question_text)
    payload = {"task": task.value, "question_text": question_text, "question_sha256": question_sha256,
               "target_event": target_event, "answer_schema": answer_schema.value, "required_evidence": list(tuple(required_evidence)),
               "temporal_requirement": temporal_requirement.value, "spatial_requirement": spatial_requirement.value,
               "planner_version": planner_version, "provenance": thaw_json(frozen)}
    return RequirementSpec(_content_id("requirement", payload), task, question_text, question_sha256, target_event, answer_schema,
                           tuple(required_evidence), temporal_requirement, spatial_requirement, planner_version, frozen)


def make_claim_spec(*, requirement_id: str, parent_claim_id: str | None, claim_role: ClaimRole, surface_text: str,
                    subject: str, predicate: str, object: str | None, polarity: Polarity,
                    temporal_quantifier: TemporalQuantifier, temporal_scope: TemporalScope, logical_scope: LogicalScope,
                    observability: Observability, evidence_geometry: EvidenceGeometryType,
                    required_components: Sequence[str], generation_reason: GenerationReason, provenance: Any) -> ClaimSpec:
    frozen = freeze_json(provenance)
    payload = {"requirement_id": requirement_id, "parent_claim_id": parent_claim_id, "claim_role": claim_role.value,
               "surface_text": surface_text, "subject": subject, "predicate": predicate, "object": object,
               "polarity": polarity.value, "temporal_quantifier": temporal_quantifier.value,
               "temporal_scope": temporal_scope.value, "logical_scope": logical_scope.value,
               "observability": observability.value, "evidence_geometry": evidence_geometry.value,
               "required_components": list(tuple(required_components)), "generation_reason": generation_reason.value,
               "provenance": thaw_json(frozen)}
    return ClaimSpec(_content_id("claim", payload), requirement_id, parent_claim_id, claim_role, surface_text, subject,
                     predicate, object, polarity, temporal_quantifier, temporal_scope, logical_scope, observability,
                     evidence_geometry, tuple(required_components), generation_reason, frozen)
