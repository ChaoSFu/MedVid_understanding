"""Separated, dry-run-only oracle feasibility harness.

Runtime preparation deliberately takes no ground-truth object. Evaluation is a
separate post-output join by case ID, so truth labels cannot affect runtime
planning, prompts, cache keys, evidence acquisition, or adaptation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .differential_evidence import DifferentialEvidenceError, load_policy
from .task_selection import TALSelectionError, strict_jsonl

RUNTIME_FIELDS = {"case_id", "source_video_id", "claim", "claim_type", "evidence_references", "intervention_plan", "provenance", "control_tier"}
GT_FIELDS = {"case_id", "truth_label", "adjudication_metadata"}


class OracleFeasibilityError(ValueError):
    pass


def _rows(path: str | Path, code: str) -> list[dict[str, Any]]:
    try:
        return list(strict_jsonl(Path(path), error_code=code))
    except (OSError, TALSelectionError) as exc:
        raise OracleFeasibilityError(f"{code}_INVALID") from exc


def prepare_runtime(*, runtime_manifest: str | Path, policy_path: str | Path,
                    calibration_artifact: str | Path | None, output_root: str | Path,
                    cache_root: str | Path, model_path: str | Path | None, dry_run: bool) -> dict[str, Any]:
    """Prepare only runtime data; no truth manifest argument exists by design."""
    policy, policy_sha = load_policy(policy_path)
    rows = _rows(runtime_manifest, "ORACLE_RUNTIME")
    if not rows or any(set(row) != RUNTIME_FIELDS for row in rows):
        raise OracleFeasibilityError("ORACLE_RUNTIME_SCHEMA_INVALID")
    if len({row["case_id"] for row in rows}) != len(rows) or any(not isinstance(row["case_id"], str) or not row["case_id"] for row in rows):
        raise OracleFeasibilityError("ORACLE_RUNTIME_CASE_ID_INVALID")
    # The required containment keeps oracle artifacts/cache outside automatic
    # namespaces without relying on a host-specific path.
    out = Path(output_root) / "oracle_feasibility"
    cache = Path(cache_root) / "oracle_feasibility"
    if out.exists() and any(out.iterdir()):
        raise OracleFeasibilityError("ORACLE_OUTPUT_IMMUTABLE_EXISTS")
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    result = {"format": "relive-v2-oracle-feasibility-runtime-v1", "status": "ORACLE_DIAGNOSTIC_ONLY",
              "runtime_case_count": len(rows), "runtime_manifest_sha256": hashlib.sha256(Path(runtime_manifest).read_bytes()).hexdigest(),
              "policy_sha256": policy_sha, "calibration_artifact": None if calibration_artifact is None else hashlib.sha256(Path(calibration_artifact).read_bytes()).hexdigest(),
              "output_namespace": "oracle_feasibility", "cache_namespace": "oracle_feasibility", "dry_run": dry_run,
              "model_path_supplied": model_path is not None, "model_calls_made": 0, "backend_loaded": False,
              "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    result["runtime_plan_sha256"] = stable_hash(result)
    (out / "oracle_runtime_plan.json").write_bytes((canonical_json(result) + "\n").encode("utf-8"))
    return result


def evaluate_truth_after_runtime(*, runtime_plan: str | Path, ground_truth_manifest: str | Path) -> dict[str, Any]:
    """Evaluator-only post-output join; it has no certificate authority."""
    import json
    plan = json.loads(Path(runtime_plan).read_text(encoding="utf-8"))
    truths = _rows(ground_truth_manifest, "ORACLE_GROUND_TRUTH")
    if any(set(row) != GT_FIELDS for row in truths):
        raise OracleFeasibilityError("ORACLE_GROUND_TRUTH_SCHEMA_INVALID")
    return {"format": "relive-v2-oracle-feasibility-evaluation-v1", "status": "ORACLE_DIAGNOSTIC_ONLY",
            "runtime_plan_sha256": plan["runtime_plan_sha256"], "ground_truth_case_count": len(truths),
            "gt_used": True, "certificate_created": False, "new_verified_count": 0}
