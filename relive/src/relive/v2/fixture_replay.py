"""Strict, zero-model replay of frozen Phase 4A future-controller fixtures."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .adaptation_controller import (AdaptationAction, ClaimBudget, FrozenAdaptationPolicy,
                                    ReliVEState, TransitionDecision, decide_transition)
from .contracts import freeze_json
from .failure_taxonomy import (DiagnosisOutcome, VerificationDiagnosis, VerificationFailure,
                                diagnosis_complete)

FIXTURE_FORMAT_VERSION = "relive-v2-phase4a-failure-routing-fixture-v1"

class FixtureReplayError(ValueError):
    pass

_REQUIRED = frozenset({"case_id", "source_trace_sha256", "observed_diagnostic_class", "auxiliary_diagnostic",
    "proposed_v2_failure_code", "allowed_action_family", "forbidden_action_family", "analysis_rule_version",
    "diagnostic_only", "certificate_created", "certificate_unchanged", "new_verified_count", "gt_used",
    "model_calls_made", "backend_loaded"})
_OPTIONAL = frozenset({"legacy_posthoc_gate", "legacy_status"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")

# Old Phase 4A action labels are intentionally normalized here, once, at the
# artifact boundary. They are not v2 action identifiers.
_ACTION_FAMILIES: dict[str, frozenset[AdaptationAction]] = {
    "STOP_DIAGNOSTIC": frozenset({AdaptationAction.STOP_DIAGNOSTIC}),
    "TEMPORAL_REACQUIRE": frozenset({AdaptationAction.TEMPORAL_REACQUIRE}),
    "TEMPORAL_EXPAND_CONTEXT": frozenset({AdaptationAction.TEMPORAL_EXPAND_CONTEXT}),
    "TEMPORAL_DISCRIMINATIVE_RETRIEVAL": frozenset({AdaptationAction.TEMPORAL_DISCRIMINATIVE_RETRIEVAL}),
    "RECOMPOSE_SPATIAL_EVIDENCE": frozenset({AdaptationAction.SPATIAL_RECOMPOSE}),
    "SPATIAL_RECOMPOSE": frozenset({AdaptationAction.SPATIAL_RECOMPOSE}),
    "SPATIAL_RECOMPOSE_CONTROL_GEOMETRY": frozenset({AdaptationAction.SPATIAL_RECOMPOSE_CONTROL_GEOMETRY}),
    "LOCALIZE_RESIDUAL_COMPONENT": frozenset({AdaptationAction.RESIDUAL_LOCALIZE}),
    "RESIDUAL_LOCALIZE": frozenset({AdaptationAction.RESIDUAL_LOCALIZE}),
    "INTERVENTION_DIAGNOSTIC": frozenset({AdaptationAction.INTERVENTION_DIAGNOSTIC}),
    "TECHNICAL_DIAGNOSTIC": frozenset({AdaptationAction.TECHNICAL_DIAGNOSTIC}),
    "STOP_OR_PREDECLARED_VERIFIER_SWITCH": frozenset({AdaptationAction.STOP_ABSTAIN, AdaptationAction.VERIFIER_SWITCH}),
    "VERIFIER_SWITCH": frozenset({AdaptationAction.VERIFIER_SWITCH}),
}
_FORBIDDEN_FAMILIES = frozenset({"CREATE_VERIFIED", "LOWER_KEEP_STANDARD", "SPATIAL_REGROUND_ON_SAME_WINDOW", "EXPAND_ROI_BLINDLY", "UNBOUNDED_R2_R3"}) | frozenset(_ACTION_FAMILIES)
_FAILURE_CODES: dict[str, tuple[DiagnosisOutcome, VerificationFailure | None]] = {
    "NO_FAILURE_DIAGNOSTIC_COMPLETE": (DiagnosisOutcome.DIAGNOSTIC_COMPLETE, None),
    "ORIGINAL_INSUFFICIENT": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.ORIGINAL_INSUFFICIENT),
    "TEMPORAL_ORDER_UNRESOLVED": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.TEMPORAL_ORDER_UNRESOLVED),
    "KEEP_SUPPORT_LOST": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.KEEP_SUPPORT_LOST),
    "DROP_SUPPORT_PERSISTS": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.DROP_SUPPORT_PERSISTS),
    "VERIFIER_INSENSITIVITY_UNRESOLVED": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT),
    "DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT),
    "CONTROL_UNAVAILABLE": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.CONTROL_UNAVAILABLE),
    "CONTROL_SUPPORT_LOST": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.CONTROL_SUPPORT_LOST),
    "HYPOTHESES_NOT_DISCRIMINATED": (DiagnosisOutcome.UNRESOLVED, VerificationFailure.HYPOTHESES_NOT_DISCRIMINATED),
    "TECHNICAL_EXECUTION_FAILURE": (DiagnosisOutcome.TECHNICAL_FAILURE, VerificationFailure.TECHNICAL_EXECUTION_FAILURE),
}

@dataclass(frozen=True)
class FailureRoutingFixture:
    case_id: str
    source_trace_sha256: str
    observed_diagnostic_class: str
    auxiliary_diagnostic: str | None
    proposed_v2_failure_code: str
    allowed_action_family: str
    forbidden_action_family: str
    analysis_rule_version: str
    legacy_posthoc_gate: bool
    legacy_status: str | None

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "source_trace_sha256": self.source_trace_sha256,
                "observed_diagnostic_class": self.observed_diagnostic_class, "auxiliary_diagnostic": self.auxiliary_diagnostic,
                "proposed_v2_failure_code": self.proposed_v2_failure_code, "allowed_action_family": self.allowed_action_family,
                "forbidden_action_family": self.forbidden_action_family, "analysis_rule_version": self.analysis_rule_version,
                "legacy_posthoc_gate": self.legacy_posthoc_gate, "legacy_status": self.legacy_status}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureReplayError("JSON_DUPLICATE_KEY")
        result[key] = value
    return result

def _read_strict_json(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line, object_pairs_hook=_pairs_no_duplicates, parse_constant=lambda _: (_ for _ in ()).throw(FixtureReplayError("NONFINITE_JSON_FORBIDDEN")))
    except (json.JSONDecodeError, TypeError) as exc:
        raise FixtureReplayError("INVALID_FIXTURE_JSON") from exc
    if not isinstance(value, dict):
        raise FixtureReplayError("FIXTURE_OBJECT_REQUIRED")
    return value

def _forbidden_field_present(row: dict[str, Any]) -> bool:
    forbidden = ("reference_answer", "assistant_answer", "temporal_gt", "bbox", "mask", "struc_info", "rc_info", "evaluation", "frame_path", "roi")
    return any(any(token in key.casefold() for token in forbidden) for key in row)

def _fixture(row: dict[str, Any]) -> FailureRoutingFixture:
    unknown = set(row) - _REQUIRED - _OPTIONAL
    if unknown:
        raise FixtureReplayError("UNKNOWN_FIXTURE_FIELD")
    if set(row) < _REQUIRED:
        raise FixtureReplayError("MISSING_FIXTURE_FIELD")
    if _forbidden_field_present(row):
        raise FixtureReplayError("PROHIBITED_FIXTURE_FIELD")
    for name in ("case_id", "observed_diagnostic_class", "proposed_v2_failure_code", "allowed_action_family", "forbidden_action_family", "analysis_rule_version"):
        if not isinstance(row[name], str) or not row[name]:
            raise FixtureReplayError("FIXTURE_NONEMPTY_STRING_REQUIRED")
    if not isinstance(row["auxiliary_diagnostic"], (str, type(None))):
        raise FixtureReplayError("AUXILIARY_DIAGNOSTIC_INVALID")
    if not isinstance(row["source_trace_sha256"], str) or not _HASH.fullmatch(row["source_trace_sha256"]):
        raise FixtureReplayError("SOURCE_TRACE_SHA256_INVALID")
    if row["proposed_v2_failure_code"] not in _FAILURE_CODES:
        raise FixtureReplayError("UNKNOWN_FAILURE_CODE")
    if row["allowed_action_family"] not in _ACTION_FAMILIES or row["forbidden_action_family"] not in _FORBIDDEN_FAMILIES:
        raise FixtureReplayError("UNKNOWN_ACTION_FAMILY")
    if row["allowed_action_family"] == row["forbidden_action_family"]:
        raise FixtureReplayError("ACTION_FAMILY_CONTRADICTION")
    exact = {"diagnostic_only": True, "certificate_created": False, "certificate_unchanged": True, "new_verified_count": 0, "gt_used": False, "model_calls_made": 0, "backend_loaded": False}
    if any(row[key] != value for key, value in exact.items()):
        raise FixtureReplayError("DIAGNOSTIC_ONLY_FIXTURE_REQUIRED")
    legacy = row.get("legacy_posthoc_gate", False)
    if not isinstance(legacy, bool):
        raise FixtureReplayError("LEGACY_POSTHOC_GATE_INVALID")
    status = row.get("legacy_status")
    if status is not None and status != "POSTHOC_VERIFIED_FOR_DIAGNOSTIC_FINALIZATION_ONLY":
        raise FixtureReplayError("UNKNOWN_LEGACY_STATUS")
    return FailureRoutingFixture(row["case_id"], row["source_trace_sha256"], row["observed_diagnostic_class"], row["auxiliary_diagnostic"],
                                 row["proposed_v2_failure_code"], row["allowed_action_family"], row["forbidden_action_family"], row["analysis_rule_version"], legacy, status)

def load_phase4a_failure_fixtures(path: Path, *, expected_sha256: str | None = None) -> tuple[FailureRoutingFixture, ...]:
    if not path.is_file():
        raise FixtureReplayError("FIXTURE_FILE_REQUIRED")
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise FixtureReplayError("FIXTURE_FILE_SHA256_MISMATCH")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FixtureReplayError("FIXTURE_FILE_UNREADABLE") from exc
    if not lines:
        raise FixtureReplayError("NONEMPTY_FIXTURE_JSONL_REQUIRED")
    fixtures = tuple(_fixture(_read_strict_json(line)) for line in lines if line.strip())
    if not fixtures or len({item.case_id for item in fixtures}) != len(fixtures):
        raise FixtureReplayError("FIXTURE_CASE_ID_MUST_BE_UNIQUE")
    return fixtures

def diagnosis_from_fixture(fixture: FailureRoutingFixture) -> VerificationDiagnosis:
    outcome, failure = _FAILURE_CODES[fixture.proposed_v2_failure_code]
    details = freeze_json({"fixture_code": fixture.proposed_v2_failure_code, "observed_diagnostic_class": fixture.observed_diagnostic_class})
    return VerificationDiagnosis(outcome, failure, details)

def _fixture_state(fixture: FailureRoutingFixture) -> ReliVEState:
    # Case id is an audit subject only; routing is driven solely by diagnosis.
    return ReliVEState(fixture.source_trace_sha256, stable_hash({"source": fixture.source_trace_sha256}), fixture.case_id,
                       None, None, "predeclared-v2-verifier", ClaimBudget(1, 1, 1, 1, 1), 0)

def replay_fixtures(fixtures: tuple[FailureRoutingFixture, ...], policy: FrozenAdaptationPolicy) -> tuple[dict[str, Any], ...]:
    rows = []
    for fixture in fixtures:
        diagnosis = diagnosis_from_fixture(fixture)
        decision = decide_transition(_fixture_state(fixture), diagnosis, policy)
        allowed = _ACTION_FAMILIES[fixture.allowed_action_family]
        forbidden = _ACTION_FAMILIES.get(fixture.forbidden_action_family, frozenset())
        if decision.action not in allowed or decision.action in forbidden:
            raise FixtureReplayError("CONTROLLER_ACTION_VIOLATES_FIXTURE")
        legacy_seen = fixture.legacy_posthoc_gate or fixture.legacy_status == "POSTHOC_VERIFIED_FOR_DIAGNOSTIC_FINALIZATION_ONLY"
        rows.append({"case_id": fixture.case_id, "source_trace_sha256": fixture.source_trace_sha256,
                     "proposed_v2_failure_code": fixture.proposed_v2_failure_code,
                     "decision": decision.to_canonical_dict(), "allowed_action_family": fixture.allowed_action_family,
                     "legacy_status_seen": legacy_seen,
                     "legacy_status_interpretation": "NO_CERTIFICATE_VERIFICATION" if legacy_seen else None,
                     "legacy_status_normalized": "POSTHOC_VALIDATED_FOR_DIAGNOSTIC_FINALIZATION_ONLY" if legacy_seen else None,
                     "certificate_status": "NOT_APPLICABLE", "diagnostic_only": True, "certificate_created": False,
                     "new_verified_count": 0, "gt_used": False, "model_calls_made": 0, "backend_loaded": False,
                     "cache_opened": False})
    return tuple(rows)

def write_replay_artifacts(*, fixtures_path: Path, output_dir: Path, policy: FrozenAdaptationPolicy) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FixtureReplayError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    fixtures = load_phase4a_failure_fixtures(fixtures_path)
    trace = replay_fixtures(fixtures, policy)
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "v2_failure_routing_trace.jsonl"
    trace_path.write_text("".join(canonical_json(row) + "\n" for row in trace), encoding="utf-8")
    trace_hash = sha256_file(trace_path)
    legacy_seen = any(row["legacy_status_seen"] for row in trace)
    counts: dict[str, int] = {}
    for row in trace:
        action = row["decision"]["action"]
        counts[action] = counts.get(action, 0) + 1
    shared = {"format": "relive-v2-failure-routing-validation-v1", "status": "PASS", "fixture_count": len(fixtures),
              "action_counts": dict(sorted(counts.items())), "diagnostic_only": True, "model_calls_made": 0,
              "backend_loaded": False, "cache_opened": False, "gt_used": False, "certificate_created": False,
              "new_verified_count": 0, "controller_policy_version": policy.policy_version,
              "fixture_file_sha256": sha256_file(fixtures_path), "routing_trace_sha256": trace_hash,
              "legacy_status_seen": legacy_seen,
              "legacy_status_interpretation": "NO_CERTIFICATE_VERIFICATION" if legacy_seen else None,
              "legacy_status_normalized": "POSTHOC_VALIDATED_FOR_DIAGNOSTIC_FINALIZATION_ONLY" if legacy_seen else None,
              "certificate_status": "NOT_APPLICABLE"}
    summary_path = output_dir / "v2_failure_routing_summary.json"
    summary_path.write_text(canonical_json(shared) + "\n", encoding="utf-8")
    audit = {**shared, "fixture_format": FIXTURE_FORMAT_VERSION, "strict_jsonl": True,
             "no_backend_or_cache_or_certificate_authority": True}
    (output_dir / "v2_failure_routing_audit.json").write_text(canonical_json(audit) + "\n", encoding="utf-8")
    return audit
