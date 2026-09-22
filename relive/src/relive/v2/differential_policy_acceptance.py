"""No-model acceptance audit for the differential-evidence contracts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .differential_evidence import (
    DifferentialEvidenceError, admit, load_policy, variant_set_from_mapping,
    verify_closed_pilot,
)
from .task_selection import TALSelectionError, strict_json_loads

FORMAT = "relive-v2-differential-policy-acceptance-v1"


class DifferentialAcceptanceError(ValueError):
    pass


def _read_json(path: str | Path, code: str) -> dict[str, Any]:
    try:
        result = strict_json_loads(Path(path).read_bytes().decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise DifferentialAcceptanceError(f"{code}_INVALID") from exc
    if not isinstance(result, dict):
        raise DifferentialAcceptanceError(f"{code}_OBJECT_REQUIRED")
    return result


def _write_once(path: Path, value: dict[str, Any] | str) -> None:
    if path.exists():
        raise DifferentialAcceptanceError("ACCEPTANCE_OUTPUT_IMMUTABLE_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = value if isinstance(value, str) else canonical_json(value) + "\n"
    path.write_text(body, encoding="utf-8")


def run_acceptance_audit(*, policy_path: str | Path, fixture_path: str | Path,
                         output_dir: str | Path, pilot_closure: str | Path | None = None) -> dict[str, Any]:
    """Assess parser and aggregation contracts using a fixed synthetic fixture.

    This function has no backend/cache/certificate imports. It cannot replay an
    inference run; a supplied pilot closure is read-only evidence that the
    historical run already replayed with zero new calls.
    """
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise DifferentialAcceptanceError("ACCEPTANCE_OUTPUT_DIRECTORY_NOT_EMPTY")
    policy, policy_sha = load_policy(policy_path)
    fixture = _read_json(fixture_path, "ACCEPTANCE_FIXTURE")
    fixture_required = {"variant_set", "evidence_quality", "operator_quality", "model_revision", "prompt_hash", "evidence_program_hash", "intervention_renderer_hash", "provenance"}
    if set(fixture) != fixture_required:
        raise DifferentialAcceptanceError("ACCEPTANCE_FIXTURE_SCHEMA_INVALID")
    source_fixture_sha = hashlib.sha256(Path(fixture_path).read_bytes()).hexdigest()
    try:
        variants = variant_set_from_mapping(fixture["variant_set"])
        result = variants.compute_metrics()
    except DifferentialEvidenceError as exc:
        raise DifferentialAcceptanceError(str(exc)) from exc
    # Deliberately no calibration artifact: acceptance confirms the policy
    # refuses automatic verification before calibration exists.
    certificate = admit(policy_sha256=policy_sha, calibration=None, value=result,
        evidence_quality=float(fixture["evidence_quality"]), operator_quality=float(fixture["operator_quality"]),
        model_revision=str(fixture["model_revision"]), prompt_hash=str(fixture["prompt_hash"]),
        evidence_program_hash=str(fixture["evidence_program_hash"]), control_set_hash=variants.controls.sha256,
        intervention_renderer_hash=str(fixture["intervention_renderer_hash"]), provenance=set(fixture["provenance"]))
    if certificate.status == "VERIFIED":
        raise DifferentialAcceptanceError("ACCEPTANCE_MUST_NOT_CREATE_VERIFIED")
    pilot = None
    if pilot_closure is not None:
        try:
            pilot = verify_closed_pilot(pilot_closure)
        except DifferentialEvidenceError as exc:
            raise DifferentialAcceptanceError("PILOT_REGRESSION_CHECK_FAILED") from exc
        if (pilot.get("r0_replay_new_model_calls") != 0 or pilot.get("r1_replay_new_model_calls") != 0
                or pilot.get("cache_opened") is not False or pilot.get("certificate_created") is not False
                or pilot.get("new_verified_count") != 0):
            raise DifferentialAcceptanceError("PILOT_REPLAY_OR_WRITE_CONTRACT_INVALID")
    manifest = {"format": FORMAT, "status": "PASS", "fixture_sha256": source_fixture_sha,
                "policy_sha256": policy_sha, "variant_families": ["ORIGINAL", "KEEP_TARGET", "KEEP_MATCHED_CONTROL[k]", "DROP_TARGET", "DROP_MATCHED_CONTROL[k]"],
                "control_tier": variants.controls.tier, "control_set_sha256": variants.controls.sha256,
                "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
                "cache_writes": 0, "certificate_created": False, "certificate_writes": 0,
                "new_verified_count": 0, "gt_used": False}
    manifest["manifest_sha256"] = stable_hash(manifest)
    report = {"format": FORMAT, "status": "PASS", "policy_sha256": policy_sha,
              "metrics": {"original_score": result.original_score, "keep_target_score": result.keep_target_score,
                  "drop_target_score": result.drop_target_score, "keep_control_worst": result.keep_control_worst,
                  "drop_control_worst": result.drop_control_worst, "g_keep": result.g_keep, "g_drop": result.g_drop,
                  "strict_local_dependence": result.strict_local_dependence},
              "certificate_status": certificate.status, "certificate_failure_reason": certificate.failure_reason,
              "strict_diagnostic_labels": list(certificate.diagnostic_labels),
              "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "cache_writes": 0,
              "certificate_created": False, "certificate_writes": 0, "new_verified_count": 0, "gt_used": False}
    report["report_sha256"] = stable_hash(report)
    regression = {"format": FORMAT, "status": "PASS", "pilot_checked": pilot is not None,
                  "pilot_closure_manifest_sha256": None if pilot is None else pilot["closure_manifest_sha256"],
                  "r0_replay_new_model_calls": None if pilot is None else pilot["r0_replay_new_model_calls"],
                  "r1_replay_new_model_calls": None if pilot is None else pilot["r1_replay_new_model_calls"],
                  "historical_replay_rerun": False, "new_model_calls": 0, "cache_writes": 0,
                  "certificate_writes": 0, "certificate_created": False, "new_verified_count": 0,
                  "gt_used": False}
    regression["regression_sha256"] = stable_hash(regression)
    readme = "# Differential policy acceptance audit\n\nThis directory is a fixture-only, no-model acceptance audit. It does not replay, modify, or overwrite historical pilot artifacts. `pilot_regression_report.json` records the closure's existing zero-call replay evidence.\n"
    _write_once(out / "differential_policy_test_manifest.json", manifest)
    _write_once(out / "differential_policy_acceptance_report.json", report)
    _write_once(out / "pilot_regression_report.json", regression)
    _write_once(out / "README.md", readme)
    return {"status": "PASS", "output_dir": str(out), "policy_sha256": policy_sha,
            "report": str(out / "differential_policy_acceptance_report.json"),
            "manifest": str(out / "differential_policy_test_manifest.json"),
            "pilot_regression": str(out / "pilot_regression_report.json"),
            "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0,
            "new_verified_count": 0, "gt_used": False}
