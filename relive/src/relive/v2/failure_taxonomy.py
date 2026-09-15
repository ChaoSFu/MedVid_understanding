"""Closed diagnostic-only failure vocabulary for ReliVE-v2."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from .contracts import FrozenJSON, freeze_json, thaw_json
from relive.storage.artifacts import stable_hash

class FailureTaxonomyError(ValueError):
    pass

class DiagnosisOutcome(str, Enum):
    DIAGNOSTIC_COMPLETE = "DIAGNOSTIC_COMPLETE"
    UNRESOLVED = "UNRESOLVED"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"

class VerificationFailure(str, Enum):
    ORIGINAL_INSUFFICIENT = "ORIGINAL_INSUFFICIENT"
    TEMPORAL_ORDER_UNRESOLVED = "TEMPORAL_ORDER_UNRESOLVED"
    KEEP_SUPPORT_LOST = "KEEP_SUPPORT_LOST"
    DROP_SUPPORT_PERSISTS = "DROP_SUPPORT_PERSISTS"
    DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT = "DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT"
    CONTROL_UNAVAILABLE = "CONTROL_UNAVAILABLE"
    CONTROL_SUPPORT_LOST = "CONTROL_SUPPORT_LOST"
    HYPOTHESES_NOT_DISCRIMINATED = "HYPOTHESES_NOT_DISCRIMINATED"
    TECHNICAL_EXECUTION_FAILURE = "TECHNICAL_EXECUTION_FAILURE"

@dataclass(frozen=True)
class VerificationDiagnosis:
    outcome: DiagnosisOutcome
    failure: VerificationFailure | None
    details: FrozenJSON
    diagnosis_version: str = "relive-v2-diagnosis-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, DiagnosisOutcome):
            raise FailureTaxonomyError("DIAGNOSIS_OUTCOME_REQUIRED")
        if self.outcome is DiagnosisOutcome.DIAGNOSTIC_COMPLETE:
            if self.failure is not None:
                raise FailureTaxonomyError("COMPLETE_DIAGNOSIS_CANNOT_HAVE_FAILURE")
        elif not isinstance(self.failure, VerificationFailure):
            raise FailureTaxonomyError("NONCOMPLETE_DIAGNOSIS_REQUIRES_FAILURE")
        if self.outcome is DiagnosisOutcome.TECHNICAL_FAILURE and self.failure is not VerificationFailure.TECHNICAL_EXECUTION_FAILURE:
            raise FailureTaxonomyError("TECHNICAL_OUTCOME_REQUIRES_EXECUTION_FAILURE")
        if self.outcome is DiagnosisOutcome.UNRESOLVED and self.failure is VerificationFailure.TECHNICAL_EXECUTION_FAILURE:
            raise FailureTaxonomyError("EXECUTION_FAILURE_REQUIRES_TECHNICAL_OUTCOME")
        if not isinstance(self.diagnosis_version, str) or not self.diagnosis_version:
            raise FailureTaxonomyError("DIAGNOSIS_VERSION_REQUIRED")

    def to_canonical_dict(self) -> dict:
        return {"outcome": self.outcome.value, "failure": self.failure.value if self.failure else None,
                "details": thaw_json(self.details), "diagnosis_version": self.diagnosis_version}

    def content_sha256(self) -> str:
        return stable_hash(self.to_canonical_dict())


def diagnosis_complete(details: object | None = None) -> VerificationDiagnosis:
    return VerificationDiagnosis(DiagnosisOutcome.DIAGNOSTIC_COMPLETE, None, freeze_json({} if details is None else details))
