#!/usr/bin/env python3
"""Preflight, execute, and replay the bounded Phase 2 engineering smoke.

The script never selects claims, frames, or parameters.  It accepts only an
already-frozen public runtime and an explicit Phase 2 config, then validates
that the fixed candidate-pool contract is present before delegating every
candidate attempt to the core runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from relive.audits import audit_run, audit_runtime_imports
from relive.config import load_config
from relive.data.schemas import load_runtime
from relive.runner import run
from relive.storage.artifacts import stable_hash
from relive.traversal import (TRAVERSAL_MODE, audit_candidate_frames,
                              audit_fixed_candidate_pool, build_fixed_candidate_pool)


class Phase2Error(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase2Error(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase2Error(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_phase2_config(config: dict[str, Any]) -> None:
    acquisition = config["acquisition"]
    if config["traversal"]["mode"] != TRAVERSAL_MODE:
        raise Phase2Error("requires traversal.mode=fixed_sliding_window_pool")
    expected = {"method": "sliding_windows", "window_size": 3, "stride": 2, "max_candidates": 4}
    if {key: acquisition[key] for key in expected} != expected:
        raise Phase2Error("Phase 2 engineering smoke requires frozen acquisition 3/2/4 sliding windows")
    if config["adaptation"]["enabled"] is not False:
        raise Phase2Error("Phase 2 fixed traversal requires adaptation.enabled=false")
    if config["budget"]["max_candidates"] < acquisition["max_candidates"]:
        raise Phase2Error("Phase 2 budget.max_candidates must cover the frozen four-candidate pool")
    if config["budget"]["max_spatial_proposals"] < acquisition["max_candidates"]:
        raise Phase2Error("Phase 2 needs one existing spatial proposal per frozen candidate; max_spatial_proposals must be >= 4")
    intervention = config["spatial"]["intervention"]
    if intervention.get("operator") != "opaque_gray" or intervention.get("operator_version") != "relive-opaque-gray-hard-mask-v1":
        raise Phase2Error("Phase 2 requires the frozen opaque_gray / relive-opaque-gray-hard-mask-v1 operator")


def _gt_audit(runtime: Path, samples) -> dict[str, Any]:
    sidecar = Path(str(runtime) + ".gt_isolation_audit.json")
    audit = _json(sidecar)
    expected_sha = samples[0].provenance["runtime_sha256"] if samples else None
    if audit.get("status") != "PASS" or audit.get("runtime_sha256") != expected_sha:
        raise Phase2Error("runtime GT-isolation audit is missing, failed, or not bound to this runtime")
    return {"status": "PASS", "path": str(sidecar), "runtime_sha256": expected_sha,
            "adapter": audit.get("adapter"), "classification": audit.get("classification")}


def _comparison(baseline_run: Path | None, phase2_run: Path) -> dict[str, Any] | None:
    """Compare runtime artifacts only; this never opens evaluation/GT files."""
    if baseline_run is None:
        return None
    if not baseline_run.is_dir() or not (baseline_run / "samples").is_dir():
        raise Phase2Error("baseline run must contain completed runtime sample artifacts")

    def summarize(root: Path) -> dict[str, Any]:
        rows = [_json(path) for path in sorted((root / "samples").glob("*.json"))]
        return {"samples": len(rows),
                "verified_qa_count": sum(row.get("strict", {}).get("status") == "ANSWERED" for row in rows),
                "abstain_qa_count": sum(row.get("strict", {}).get("status") == "ABSTAIN" for row in rows),
                "candidate_attempts": sum(len(row.get("candidate_traversal", {}).get("trace", row.get("candidates", []))) for row in rows),
                "logical_model_calls": sum(row.get("usage", {}).get("calls", 0) for row in rows),
                "new_model_calls": sum(row.get("usage", {}).get("new_calls", 0) for row in rows),
                "cache_hits": sum(row.get("usage", {}).get("cache_hits", 0) for row in rows),
                "latency_seconds": sum(row.get("usage", {}).get("latency_seconds", 0.0) for row in rows)}
    return {"scope": "runtime-artifact engineering comparison only; no benchmark-performance claim",
            "existing_one_shot": summarize(baseline_run), "fixed_sliding_window_traversal": summarize(phase2_run)}


def _preflight(root: Path, config_path: Path, runtime_path: Path, max_samples: int) -> dict[str, Any]:
    if root.exists():
        raise Phase2Error("output directory must not exist before zero-call preflight")
    config = load_config(config_path)
    _verify_phase2_config(config)
    samples = load_runtime(runtime_path)[:max_samples]
    if not samples or len(samples) != max_samples:
        raise Phase2Error("runtime does not contain the requested fixed smoke cohort")
    if config["backend"]["kind"] != "local_hf":
        raise Phase2Error("Phase 2 real engineering smoke requires the reviewed local_hf backend")
    pools, rows, pool_hash = build_fixed_candidate_pool(samples, config)
    pool_audit, frame_audit = audit_fixed_candidate_pool(rows, pool_hash), audit_candidate_frames(samples, rows)
    if pool_audit["status"] != "PASS" or frame_audit["status"] != "PASS":
        raise Phase2Error("candidate-pool or frame audit failed")
    isolation = _gt_audit(runtime_path, samples)
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise Phase2Error("runtime import audit failed")
    root.mkdir(parents=True)
    preflight = {
        "status": "PASS", "mode": "preflight", "model_calls_made": 0, "cache_mutated": False,
        "phase": "ReliVE Phase 2 fixed sliding-window candidate-pool traversal engineering smoke",
        "config": str(config_path), "config_sha256": _sha256(config_path), "runtime": str(runtime_path),
        "runtime_sha256": samples[0].provenance["runtime_sha256"], "max_samples": max_samples,
        "candidate_pool_manifest_hash": pool_hash, "candidate_pool_size_by_qa": {sample_id: len(pool) for sample_id, pool in pools.items()},
        "candidate_manifest_audit": pool_audit, "frame_audit": frame_audit, "gt_isolation_audit": isolation,
        "runtime_import_audit": source_audit, "spatial_intervention": config["spatial"]["intervention"],
        "model_fingerprint": config["backend"], "planned_max_model_calls": sum(len(pool) * 5 for pool in pools.values()),
        "planned_call_derivation": "at most ORIGINAL + one spatial proposal + KEEP + DROP + matched control for each frozen candidate; later stages are conditional",
        "candidate_protocol": {"generator": "relive.acquisition.acquire", "method": "sliding_windows", "window_size": 3,
                               "stride": 2, "max_candidates": 4, "ordering": "temporal start ascending",
                               "tail_behavior": "existing short final window retained"},
        "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "TRUE_SUPPORT", "SPURIOUS_SUPPORT", "GT_IoU", "evaluation_artifacts"],
    }
    (root / "phase2_preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return preflight


def _execute(root: Path, config_path: Path, runtime_path: Path, max_samples: int, mode: str,
             baseline_run: Path | None) -> dict[str, Any]:
    preflight = _json(root / "phase2_preflight.json")
    if preflight.get("config_sha256") != _sha256(config_path) or preflight.get("runtime_sha256") != load_runtime(runtime_path)[0].provenance["runtime_sha256"]:
        raise Phase2Error("missing or incompatible zero-call preflight")
    output = root / mode
    if output.exists():
        raise Phase2Error(f"{mode} output already exists")
    config = load_config(config_path)
    _verify_phase2_config(config)
    summary = run(config, runtime_path, output, cache_dir=root / "cache", max_samples=max_samples, phase="smoke")
    if mode == "run" and summary["new_model_calls"] <= 0:
        raise Phase2Error("fresh Phase 2 run made zero model calls")
    if mode == "replay" and summary["new_model_calls"] != 0:
        raise Phase2Error("Phase 2 replay made unexpected new model calls")
    report = {"status": "PASS", "mode": mode, "summary": summary,
              "candidate_pool_manifest_hash": preflight["candidate_pool_manifest_hash"],
              "gt_isolation_audit": preflight["gt_isolation_audit"],
              "strict_audit": audit_run(output),
              "run_audit": _json(output / "analysis" / "phase2_traversal_audit.json"),
              "failure_reason_distribution": _json(output / "analysis" / "failure_reason_distribution.json")}
    if mode == "run":
        report["comparison_with_existing_one_shot"] = _comparison(baseline_run, output)
    (root / f"phase2_{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 2 fixed sliding-window candidate-pool smoke")
    parser.add_argument("--mode", choices=("preflight", "run", "replay"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--baseline-run", help="Optional existing five-sample runtime run for a GT-free artifact comparison")
    args = parser.parse_args()
    try:
        if args.max_samples != 5:
            raise Phase2Error("Phase 2 engineering smoke is frozen to exactly five samples")
        root, config, runtime = Path(args.output_dir).resolve(), Path(args.config).resolve(), Path(args.runtime).resolve()
        baseline = Path(args.baseline_run).resolve() if args.baseline_run else None
        report = (_preflight(root, config, runtime, args.max_samples) if args.mode == "preflight"
                  else _execute(root, config, runtime, args.max_samples, args.mode, baseline))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, Phase2Error) as exc:
        print(f"ReliVE Phase 2 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
