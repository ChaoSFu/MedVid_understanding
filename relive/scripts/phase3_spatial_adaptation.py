#!/usr/bin/env python3
"""Frozen-cohort Phase 3 spatial adaptation preflight, run, and replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from relive.audits import audit_runtime_imports
from relive.backends import make_backend
from relive.certificate import POLICY_VERSION, build_certificate
from relive.config import load_config
from relive.data.schemas import load_runtime
from relive.failure_diagnostics import PHASE25_FORMAT
from relive.interventions import OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION
from relive.runner import SampleRunner
from relive.spatial_adaptation import (MAX_SPATIAL_REFINEMENT_ROUNDS, PHASE3_FORMAT,
                                       SpatialAdaptationError, SpatialEvidenceAdaptationController, candidate_from_manifest,
                                       geometry_change, historical_pattern)
from relive.storage.artifacts import ArtifactStore, canonical_json, stable_hash
from relive.storage.cache import CachedInference
from relive.types import ExecutionStatus, SpatialProposal, to_dict

EXPECTED_ELIGIBLE_COHORT = 9


class Phase3Error(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3Error(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase3Error(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3Error(f"unreadable JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase3Error(f"nonempty JSONL object rows required: {path}")
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    body = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def _gt_audit(runtime: Path, samples) -> dict[str, Any]:
    sidecar = Path(str(runtime) + ".gt_isolation_audit.json")
    audit = _json(sidecar)
    if audit.get("status") != "PASS" or audit.get("runtime_sha256") != samples[0].provenance["runtime_sha256"]:
        raise Phase3Error("runtime GT-isolation audit is missing, failed, or not bound to this runtime")
    return {"status": "PASS", "path": str(sidecar), "runtime_sha256": samples[0].provenance["runtime_sha256"],
            "adapter": audit.get("adapter"), "classification": audit.get("classification")}


def _phase2_inputs(phase2_run: Path, diagnostics: Path, runtime: Path) -> tuple[list[dict], list[dict], dict, dict]:
    manifest_path = phase2_run / "fixed_candidate_pool.jsonl"
    manifest = _jsonl(manifest_path)
    manifest_hashes = {row.get("manifest_hash") for row in manifest}
    if len(manifest_hashes) != 1 or not isinstance(next(iter(manifest_hashes)), str):
        raise Phase3Error("frozen Phase 2 candidate manifest has no single hash binding")
    diagnoses = _jsonl(diagnostics)
    if any(row.get("format") != PHASE25_FORMAT for row in diagnoses):
        raise Phase3Error("Phase 3 requires a Phase 2.5 diagnostics artifact")
    eligible = [row for row in diagnoses if row.get("refinement_eligible") is True]
    if len(eligible) != EXPECTED_ELIGIBLE_COHORT:
        raise Phase3Error(f"Phase 3-v1 requires exactly {EXPECTED_ELIGIBLE_COHORT} frozen eligible candidates; found {len(eligible)}")
    if len({(row.get("qa_id"), row.get("candidate_id")) for row in eligible}) != len(eligible):
        raise Phase3Error("Phase 2.5 eligible cohort has duplicate candidate identities")
    runtime_samples = {sample.sample_id: sample for sample in load_runtime(runtime)}
    phase2_samples = {_json(path).get("sample_id"): _json(path) for path in sorted((phase2_run / "samples").glob("*.json"))}
    if not phase2_samples:
        raise Phase3Error("Phase 2 run has no completed sample artifacts")
    by_candidate = {(row.get("sample_id"), row.get("candidate_id")): row for row in manifest}
    cohort = []
    for diagnosis in eligible:
        key = (diagnosis.get("qa_id"), diagnosis.get("candidate_id"))
        row = by_candidate.get(key)
        sample = runtime_samples.get(key[0])
        phase2_sample = phase2_samples.get(key[0])
        if not row or not sample or not phase2_sample:
            raise Phase3Error("eligible diagnostic cannot bind to frozen Phase 2 candidate and runtime")
        certificate = next((item for item in phase2_sample.get("certificates", [])
                            if item.get("candidate_id") == key[1] and item.get("certificate_id") == diagnosis.get("certificate_id")), None)
        if not certificate:
            raise Phase3Error("eligible diagnostic cannot bind to historical certificate")
        cohort.append({"diagnosis": diagnosis, "candidate_row": row, "sample": sample, "certificate": certificate})
    return manifest, diagnoses, runtime_samples, {"cohort": cohort, "manifest_hash": next(iter(manifest_hashes)),
                                                    "manifest_path": manifest_path}


def _verify_config(config: dict[str, Any]) -> None:
    intervention = config["spatial"]["intervention"]
    if config["backend"].get("kind") != "local_hf":
        raise Phase3Error("Phase 3 requires the frozen local_hf backend")
    if config["policy"]["name"] != "semantic_spatial":
        raise Phase3Error("Phase 3 requires the existing formal semantic_spatial policy")
    if intervention.get("operator") != OPAQUE_GRAY_OPERATOR or intervention.get("operator_version") != OPAQUE_GRAY_VERSION:
        raise Phase3Error("Phase 3 requires the frozen opaque_gray operator")


def _preflight(root: Path, config_path: Path, runtime: Path, phase2_run: Path, diagnostics: Path) -> dict[str, Any]:
    if root.exists():
        raise Phase3Error("Phase 3 output directory must not exist before zero-call preflight")
    config = load_config(config_path)
    _verify_config(config)
    samples = load_runtime(runtime)
    gt = _gt_audit(runtime, samples)
    manifest, diagnoses, _, inputs = _phase2_inputs(phase2_run, diagnostics, runtime)
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise Phase3Error("runtime import audit failed")
    controller = SpatialEvidenceAdaptationController()
    routes = []
    for item in inputs["cohort"]:
        route = controller.route(item["diagnosis"], current_round=0)
        if not route["allowed"]:
            raise Phase3Error("frozen eligible candidate is not allowed by reason-specific routing")
        pattern = historical_pattern(item["certificate"])
        if not isinstance(pattern["bbox"], list) or len(pattern["bbox"]) != 4:
            raise Phase3Error("eligible candidate lacks a historical R0 proposal bbox")
        routes.append({"qa_id": item["diagnosis"]["qa_id"], "candidate_id": item["diagnosis"]["candidate_id"],
                       "routing_reason": route["routing_reason"], "prompt_version": route["prompt_version"]})
    root.mkdir(parents=True)
    report = {"status": "PASS", "mode": "preflight", "phase": "ReliVE Phase 3 Failure-Aware Spatial Evidence Adaptation",
              "model_calls_made": 0, "cache_mutated": False, "config": str(config_path), "config_sha256": _sha(config_path),
              "runtime": str(runtime), "runtime_sha256": samples[0].provenance["runtime_sha256"],
              "phase2_run": str(phase2_run), "phase2_candidate_manifest": str(inputs["manifest_path"]),
              "phase2_candidate_manifest_sha256": _sha(inputs["manifest_path"]), "candidate_manifest_hash": inputs["manifest_hash"],
              "phase25_diagnostics": str(diagnostics), "phase25_diagnostics_sha256": _sha(diagnostics),
              "eligible_candidate_count": len(inputs["cohort"]), "excluded_candidate_count": len(diagnoses) - len(inputs["cohort"]),
              "frozen_refinement_rounds": MAX_SPATIAL_REFINEMENT_ROUNDS,
              "planned_max_model_calls": len(inputs["cohort"]) * 5,
              "planned_call_derivation": "per frozen candidate: ORIGINAL + one reason-specific R1 proposal + KEEP + DROP + one matched CONTROL; later calls are conditional on formal protocol progress",
              "routes": routes,
              "model_fingerprint": config["backend"], "spatial_intervention": intervention,
              "semantic_prompt_version": "relive-semantic-v4", "certificate_policy_version": POLICY_VERSION,
              "certificate_builder_source_sha256": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest(),
              "runtime_gt_isolation_audit": gt, "runtime_import_audit": source_audit,
              "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "TRUE_SUPPORT", "SPURIOUS_SUPPORT", "GT_IoU", "evaluation_artifacts"]}
    (root / "phase3_preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _semantics(certificate: dict[str, Any]) -> dict[str, Any]:
    spatial = certificate.get("checks", {}).get("spatial", {})
    refs = spatial.get("references", {}) if isinstance(spatial, dict) and isinstance(spatial.get("references"), dict) else {}
    controls = refs.get("controls", []) if isinstance(refs.get("controls"), list) else []
    return {"original": certificate.get("checks", {}).get("semantic", {}).get("status"),
            "keep": refs.get("keep", {}).get("semantic_status") if isinstance(refs.get("keep"), dict) else None,
            "drop": refs.get("drop", {}).get("semantic_status") if isinstance(refs.get("drop"), dict) else None,
            "control": [value.get("semantic_status") for value in controls if isinstance(value, dict)],
            "control_available": spatial.get("control_available") if isinstance(spatial, dict) else None,
            "pixel_audit_refs": spatial.get("pixel_audit_refs", []) if isinstance(spatial, dict) else []}


def _pixel_status(store: ArtifactStore, refs: list[str]) -> str:
    if not refs:
        return "NOT_RUN"
    try:
        audits = [store.get_json(ref) for ref in refs]
    except Exception:
        return "UNREADABLE"
    return "PASS" if all(item.get("pixel_audit_pass") for item in audits) else "FAIL"


def _run_one(item: dict[str, Any], config: dict[str, Any], store: ArtifactStore, inference: CachedInference,
             manifest_hash: str, git_commit: str | None) -> dict[str, Any]:
    diagnosis, sample, historical_cert = item["diagnosis"], item["sample"], item["certificate"]
    candidate = candidate_from_manifest(item["candidate_row"], sample)
    controller = SpatialEvidenceAdaptationController()
    route = controller.route(diagnosis, current_round=0)
    if not route["allowed"]:
        raise SpatialAdaptationError("refinement routing is forbidden for a frozen cohort member")
    r0 = historical_pattern(historical_cert)
    previous = r0["bbox"]
    parent = SpatialProposal(r0["proposal_id"], candidate.candidate_id, sample.target_claim.claim_id, tuple(previous),
                             provenance={"source": "phase2_historical_artifact"})
    engine = SampleRunner(config, store, inference)
    before = engine.budget.snapshot()
    original = engine.infer(sample, candidate, sample.target_claim, "semantic")
    stage_data = {"previous_bbox_normalized_0_1_xyxy": previous,
                  "previous_failure_reason": route["routing_reason"],
                  "previous_original_semantic_status": r0["original_status"],
                  "previous_keep_semantic_status": r0["keep_status"],
                  "previous_drop_semantic_status": r0["drop_status"],
                  "refinement_round": 1}
    response = engine.infer(sample, candidate, sample.target_claim, route["stage"], stage_data=stage_data)
    refined = controller.parse(response.get("raw_text") or "", candidate, sample.target_claim,
                                parent=parent, route=route, raw_response_ref=response.get("raw_response_ref"))
    spatial = engine.spatial_with_proposal(sample, candidate, sample.target_claim, original, refined, proposal_index=1)
    certificate = engine.certificate(sample, candidate, sample.target_claim, original, spatial, None)
    after = engine.budget.snapshot()
    r1 = _semantics(to_dict(certificate))
    trace = {"format": PHASE3_FORMAT, "qa_id": sample.sample_id, "candidate_id": candidate.candidate_id,
             "temporal_candidate_rank": candidate.acquisition_rank, "spatial_round": 1,
             "proposal_id": refined.proposal_id, "parent_proposal_id": parent.proposal_id,
             "previous_bbox": previous, "previous_bbox_area_fraction": r0["bbox_area_fraction"],
             "historical_failure_reason": diagnosis["original_failure_reasons"],
             "phase2_5_root_cause": diagnosis["root_cause_class"], "adaptation_routing_reason": route["routing_reason"],
             "refinement_prompt_version": route["prompt_version"], "refinement_action": route["action"],
             "raw_model_response": response.get("raw_text"), "raw_response_ref": response.get("raw_response_ref"),
             "refinement_execution_status": response.get("execution_status"),
             "parsed_new_bbox": list(refined.support_region) if refined.support_region else None,
             "new_bbox_area_fraction": refined.provenance.get("support_area_fraction"),
             "bbox_geometry_change": geometry_change(previous, refined.support_region) if refined.support_region else None,
             "round0": r0, "round1": {**r1, "certificate_status": certificate.final_status.value,
                                         "failure_reasons": list(certificate.failure_reasons),
                                         "pixel_audit_status": _pixel_status(store, r1["pixel_audit_refs"])},
             "formal_certificate": to_dict(certificate), "new_model_calls": after["new_calls"] - before["new_calls"],
             "cache_hits": after["cache_hits"] - before["cache_hits"],
             "logical_model_calls": after["calls"] - before["calls"], "model_fingerprint": inference.backend.fingerprint(),
             "operator_version": config["spatial"]["intervention"], "git_commit": git_commit,
             "candidate_manifest_hash": manifest_hash, "gt_used": False,
             "termination_reason": "FORMAL_CERTIFICATE_RETURNED" if refined.parser_status == ExecutionStatus.OK else "SPATIAL_UNRESOLVED"}
    store.append_event("spatial_adaptation", trace)
    return trace


def _execute(root: Path, config_path: Path, runtime: Path, phase2_run: Path, diagnostics: Path, mode: str) -> dict[str, Any]:
    preflight = _json(root / "phase3_preflight.json")
    bindings = {"config_sha256": _sha(config_path), "runtime_sha256": load_runtime(runtime)[0].provenance["runtime_sha256"],
                "phase25_diagnostics_sha256": _sha(diagnostics),
                "phase2_candidate_manifest_sha256": _sha(phase2_run / "fixed_candidate_pool.jsonl")}
    if any(preflight.get(key) != value for key, value in bindings.items()):
        raise Phase3Error("incompatible or missing zero-call Phase 3 preflight")
    output = root / mode
    if output.exists():
        raise Phase3Error(f"Phase 3 {mode} output already exists")
    config = load_config(config_path)
    _verify_config(config)
    _, diagnoses, _, inputs = _phase2_inputs(phase2_run, diagnostics, runtime)
    backend = make_backend(config["backend"])
    if backend.synthetic:
        raise Phase3Error("Phase 3 real engineering run requires the frozen local_hf backend")
    store, cache = ArtifactStore(output), ArtifactStore(root / "cache")
    inference = CachedInference(backend, cache)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    started = time.monotonic()
    traces = []
    with store.run_lock():
        for item in inputs["cohort"]:
            traces.append(_run_one(item, config, store, inference, inputs["manifest_hash"], commit))
        _write_jsonl(output / "phase3_spatial_adaptation_trace.jsonl", traces)
    new_calls, hits = sum(row["new_model_calls"] for row in traces), sum(row["cache_hits"] for row in traces)
    if mode == "run" and new_calls <= 0:
        raise Phase3Error("fresh Phase 3 run made zero model calls")
    if mode == "replay" and new_calls != 0:
        raise Phase3Error("Phase 3 replay made unexpected new model calls")
    transitions = Counter(f"{row['round0']['certificate_status']}->{row['round1']['certificate_status']}" for row in traces)
    by_reason: dict[str, Counter] = defaultdict(Counter)
    for row in traces:
        by_reason[row["adaptation_routing_reason"]][f"{row['round0']['certificate_status']}->{row['round1']['certificate_status']}"] += 1
    summary = {"format": PHASE3_FORMAT, "status": "PASS", "mode": mode, "eligible_candidate_count": len(traces),
               "excluded_candidate_count": len(diagnoses) - len(traces), "transition_matrix": dict(sorted(transitions.items())),
               "by_failure_reason": {key: dict(value) for key, value in sorted(by_reason.items())},
               "uncertain_to_verified": transitions.get("UNCERTAIN->VERIFIED", 0), "new_model_calls": new_calls,
               "cache_hits": hits, "logical_model_calls": sum(row["logical_model_calls"] for row in traces),
               "latency_seconds": time.monotonic() - started, "cache_dir": str(cache.root), "output_dir": str(output),
               "candidate_manifest_hash": inputs["manifest_hash"], "gt_used": False,
               "operator_unchanged": config["spatial"]["intervention"] == preflight["spatial_intervention"],
               "certificate_builder_unchanged": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest() == preflight["certificate_builder_source_sha256"],
               "semantic_prompt_version": "relive-semantic-v4", "refinement_prompt_versions": sorted({row["refinement_prompt_version"] for row in traces})}
    (output / "phase3_candidate_transition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failure_lines = ["# ReliVE Phase 3 failure-reason summary", "", "| Routing reason | Transition | Count |", "|---|---|---:|"]
    for reason, values in sorted(by_reason.items()):
        for transition, count in sorted(values.items()):
            failure_lines.append(f"| {reason} | {transition} | {count} |")
    (output / "phase3_failure_reason_summary.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    audit = {"status": "PASS", "mode": mode, "phase": "Phase 3-v1", "candidate_manifest_hash": inputs["manifest_hash"],
             "frozen_temporal_candidates_unchanged": all(row["candidate_manifest_hash"] == inputs["manifest_hash"] for row in traces),
             "max_spatial_refinement_rounds": MAX_SPATIAL_REFINEMENT_ROUNDS,
             "all_rounds_are_one": all(row["spatial_round"] == 1 for row in traces),
             "gt_isolation": preflight["runtime_gt_isolation_audit"], "operator_unchanged": summary["operator_unchanged"],
             "certificate_builder_unchanged": summary["certificate_builder_unchanged"],
             "pixel_audits": Counter(row["round1"]["pixel_audit_status"] for row in traces),
             "prohibited_inputs_not_opened": preflight["prohibited_inputs_not_opened"]}
    (output / "phase3_audit.json").write_text(json.dumps(to_dict(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 3 frozen-cohort spatial adaptation")
    parser.add_argument("--mode", choices=("preflight", "run", "replay"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--phase2-run-dir", required=True)
    parser.add_argument("--phase25-diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        root, config, runtime = Path(args.output_dir).resolve(), Path(args.config).resolve(), Path(args.runtime).resolve()
        phase2, diagnostics = Path(args.phase2_run_dir).resolve(), Path(args.phase25_diagnostics).resolve()
        report = (_preflight(root, config, runtime, phase2, diagnostics) if args.mode == "preflight"
                  else _execute(root, config, runtime, phase2, diagnostics, args.mode))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, Phase3Error, SpatialAdaptationError) as exc:
        print(f"ReliVE Phase 3 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
