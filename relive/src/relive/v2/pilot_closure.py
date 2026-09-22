"""Read-only closure binding for a completed protocol-development pilot.

The closure manifest records immutable input artifact bytes.  It has no model
or certificate authority and refuses to write into an existing directory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .reviewed_anchor_formal_intervention import _object, _rows, sha256_path, validate_label_resolution_manifest

FORMAT = "relive-v2-protocol-pilot-closure-v1"


class PilotClosureError(ValueError):
    pass


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        return _object(path, code)
    except Exception as exc:
        raise PilotClosureError(f"{code}_INVALID") from exc


def _trace(path: Path, code: str) -> list[dict[str, Any]]:
    try:
        return _rows(path, code)
    except Exception as exc:
        raise PilotClosureError(f"{code}_INVALID") from exc


def _ensure_new_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise PilotClosureError("PILOT_CLOSURE_IMMUTABLE_OUTPUT_EXISTS")
    path.mkdir(parents=True, exist_ok=True)


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise PilotClosureError("PILOT_CLOSURE_IMMUTABLE_OUTPUT_EXISTS")
    path.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))


def freeze_pilot_closure(*, r0_output_dir: str | Path, r1_output_dir: str | Path,
                          label_resolution_manifest: str | Path, output_dir: str | Path,
                          expected_formal_cohort: int = 5) -> dict[str, Any]:
    """Freeze a completed R0/R1 pilot without changing any source artifact."""
    r0, r1, resolution, out = map(Path, (r0_output_dir, r1_output_dir, label_resolution_manifest, output_dir))
    _ensure_new_dir(out)
    files = {
        "r0_plan": r0 / "reviewed_anchor_intervention_plan.json",
        "r0_run_trace": r0 / "run" / "reviewed_anchor_intervention_trace.jsonl",
        "r0_replay_trace": r0 / "replay" / "reviewed_anchor_intervention_trace.jsonl",
        "r0_run_summary": r0 / "run" / "reviewed_anchor_summary.json",
        "r0_replay_summary": r0 / "replay" / "reviewed_anchor_summary.json",
        "r1_plan": r1 / "reviewed_anchor_r1_recomposition_plan.json",
        "r1_run_trace": r1 / "run" / "reviewed_anchor_r1_trace.jsonl",
        "r1_replay_trace": r1 / "replay" / "reviewed_anchor_r1_trace.jsonl",
        "r1_run_summary": r1 / "run" / "reviewed_anchor_r1_summary.json",
        "r1_replay_summary": r1 / "replay" / "reviewed_anchor_r1_summary.json",
        "label_resolution": resolution,
    }
    if any(not path.is_file() for path in files.values()):
        raise PilotClosureError("PILOT_CLOSURE_REQUIRED_ARTIFACT_MISSING")
    r0_plan = _read(files["r0_plan"], "R0_PLAN")
    r0_run, r0_replay = _read(files["r0_run_summary"], "R0_RUN_SUMMARY"), _read(files["r0_replay_summary"], "R0_REPLAY_SUMMARY")
    r1_plan = _read(files["r1_plan"], "R1_PLAN")
    r1_run, r1_replay = _read(files["r1_run_summary"], "R1_RUN_SUMMARY"), _read(files["r1_replay_summary"], "R1_REPLAY_SUMMARY")
    r0_traces, replay_traces = _trace(files["r0_run_trace"], "R0_RUN_TRACE"), _trace(files["r0_replay_trace"], "R0_REPLAY_TRACE")
    r1_traces, r1_replay_traces = _trace(files["r1_run_trace"], "R1_RUN_TRACE"), _trace(files["r1_replay_trace"], "R1_REPLAY_TRACE")
    if (r0_run.get("candidate_count") != expected_formal_cohort or r0_run.get("full_formal_cohort_complete") is not True
            or r0_replay.get("candidate_count") != expected_formal_cohort or r0_replay.get("new_model_calls") != 0
            or len(r0_traces) != expected_formal_cohort or len(replay_traces) != expected_formal_cohort):
        raise PilotClosureError("R0_COHORT_OR_REPLAY_INCOMPLETE")
    if r1_run.get("new_verified_count") != 0 or r1_replay.get("new_model_calls") != 0 or len(r1_traces) != len(r1_replay_traces):
        raise PilotClosureError("R1_REPLAY_OR_CERTIFICATE_CONTRACT_INVALID")
    bindings = r0_plan.get("bindings")
    if not isinstance(bindings, dict):
        raise PilotClosureError("R0_PLAN_BINDINGS_INVALID")
    # Resolution manifest validates that explicit canonical labels made a
    # derived review and recomputed route/eligible files, rather than merely
    # opening a warning gate.
    resolution_data = _read(resolution, "LABEL_RESOLUTION")
    artifact_hashes = resolution_data.get("recomputed_artifact_sha256", {})
    if not isinstance(artifact_hashes, dict):
        raise PilotClosureError("LABEL_RESOLUTION_ARTIFACT_BINDING_INVALID")
    try:
        validate_label_resolution_manifest(
            resolution,
            raw_grounding_sha256=str(resolution_data["raw_grounding_sha256"]),
            derived_human_review_sha256=str(resolution_data["derived_human_review_sha256"]),
            validation_report_sha256=str(artifact_hashes["human_review_validation_report.json"]),
            observation_decisions_sha256=str(artifact_hashes["observation_anchor_decisions.jsonl"]),
            eligible_manifest_sha256=str(artifact_hashes["eligible_anchor_manifest.jsonl"]),
            warning_queue_sha256=str(resolution_data["source_warning_queue_sha256"]),
            warning_adjudication_sha256=str(resolution_data["source_warning_adjudication_sha256"]),
        )
    except Exception as exc:
        raise PilotClosureError("LABEL_RESOLUTION_NOT_APPLIED") from exc
    if bindings.get("label_resolution_manifest") != sha256_path(resolution):
        raise PilotClosureError("R0_PLAN_DOES_NOT_BIND_APPLIED_LABEL_RESOLUTION")
    if r1_plan.get("bindings", {}).get("r0_plan_sha256") != sha256_path(files["r0_plan"]):
        raise PilotClosureError("R1_PLAN_R0_BINDING_INVALID")
    source_hashes_before = {name: sha256_path(path) for name, path in sorted(files.items())}
    payload = {"format": FORMAT, "status": "PILOT_CLOSED", "r0_candidate_count": expected_formal_cohort,
               "r1_candidate_count": len(r1_traces), "r0_replay_new_model_calls": r0_replay["new_model_calls"],
               "r1_replay_new_model_calls": r1_replay["new_model_calls"],
               "label_resolution_applied": True, "source_artifact_sha256": source_hashes_before,
               "source_artifacts_unchanged": True, "gt_used": False, "model_calls_made": 0,
               "backend_loaded": False, "cache_opened": False, "certificate_created": False,
               "new_verified_count": 0}
    payload["closure_manifest_sha256"] = stable_hash(payload)
    _write_new(out / "relive_v2_pilot_closure_manifest.json", payload)
    # Verify inputs after writing closure to detect accidental source mutation.
    if source_hashes_before != {name: sha256_path(path) for name, path in sorted(files.items())}:
        raise PilotClosureError("PILOT_SOURCE_ARTIFACT_MUTATED_DURING_CLOSURE")
    return payload
