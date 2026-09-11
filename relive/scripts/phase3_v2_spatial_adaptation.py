#!/usr/bin/env python3
"""Phase 3-v2 contract preflight and frozen-cohort spatial re-proposal."""
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
from relive.spatial_adaptation import (candidate_from_manifest, geometry_change,
                                       historical_pattern)
from relive.spatial_adaptation_v2 import (PHASE3_V2_FORMAT,
                                           SpatialEvidenceAdaptationControllerV2,
                                           canonicalize_region_1000)
from relive.storage.artifacts import ArtifactStore, canonical_json
from relive.storage.cache import CachedInference
from relive.types import ExecutionStatus, SpatialProposal, to_dict

EXPECTED_ELIGIBLE = 9
EXPECTED_CONTRACT_CASES = 5


class Phase3V2Error(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3V2Error(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase3V2Error(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3V2Error(f"unreadable JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase3V2Error(f"nonempty JSONL object rows required: {path}")
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _verify_config(config: dict[str, Any]) -> None:
    if config["backend"].get("kind") != "local_hf":
        raise Phase3V2Error("Phase 3-v2 requires frozen local_hf backend")
    if config["policy"]["name"] != "semantic_spatial":
        raise Phase3V2Error("Phase 3-v2 requires frozen semantic_spatial policy")
    spec = config["spatial"]["intervention"]
    if spec.get("operator") != OPAQUE_GRAY_OPERATOR or spec.get("operator_version") != OPAQUE_GRAY_VERSION:
        raise Phase3V2Error("Phase 3-v2 requires frozen opaque_gray operator")


def _gt_audit(runtime: Path, samples) -> dict[str, Any]:
    audit = _json(Path(str(runtime) + ".gt_isolation_audit.json"))
    sha = samples[0].provenance["runtime_sha256"]
    if audit.get("status") != "PASS" or audit.get("runtime_sha256") != sha:
        raise Phase3V2Error("runtime GT-isolation audit is missing, failed, or unbound")
    return {"status": "PASS", "runtime_sha256": sha, "path": str(Path(str(runtime) + ".gt_isolation_audit.json")),
            "adapter": audit.get("adapter"), "classification": audit.get("classification")}


def _inputs(phase2_run: Path, diagnostics: Path, runtime: Path) -> tuple[list[dict], dict]:
    manifest_path = phase2_run / "fixed_candidate_pool.jsonl"
    manifest = _jsonl(manifest_path)
    hashes = {row.get("manifest_hash") for row in manifest}
    if len(hashes) != 1 or not isinstance(next(iter(hashes)), str):
        raise Phase3V2Error("Phase 2 candidate manifest lacks one immutable hash")
    diagnoses = _jsonl(diagnostics)
    if any(row.get("format") != PHASE25_FORMAT for row in diagnoses):
        raise Phase3V2Error("Phase 3-v2 requires Phase 2.5 diagnostics")
    eligible = [row for row in diagnoses if row.get("refinement_eligible") is True]
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise Phase3V2Error(f"expected {EXPECTED_ELIGIBLE} frozen eligible candidates, found {len(eligible)}")
    samples = {sample.sample_id: sample for sample in load_runtime(runtime)}
    phase2_samples = {_json(path).get("sample_id"): _json(path) for path in (phase2_run / "samples").glob("*.json")}
    by_candidate = {(row.get("sample_id"), row.get("candidate_id")): row for row in manifest}
    cohort = []
    for diagnosis in eligible:
        key = diagnosis.get("qa_id"), diagnosis.get("candidate_id")
        row, sample, prior = by_candidate.get(key), samples.get(key[0]), phase2_samples.get(key[0])
        if not row or not sample or not prior:
            raise Phase3V2Error("eligible candidate cannot bind frozen manifest/runtime/Phase 2 artifact")
        certificate = next((cert for cert in prior.get("certificates", [])
                            if cert.get("candidate_id") == key[1] and cert.get("certificate_id") == diagnosis.get("certificate_id")), None)
        if not certificate:
            raise Phase3V2Error("eligible candidate cannot bind historical certificate")
        cohort.append({"diagnosis": diagnosis, "candidate_row": row, "sample": sample, "certificate": certificate})
    return diagnoses, {"cohort": cohort, "manifest_path": manifest_path, "manifest_hash": next(iter(hashes))}


def _v1_contract_failures(v1_run: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    trace_path = v1_run / "phase3_spatial_adaptation_trace.jsonl"
    rows = _jsonl(trace_path)
    by_key = {(row["sample"].sample_id, row["diagnosis"]["candidate_id"]): row for row in inputs["cohort"]}
    selected = []
    for row in rows:
        r1 = row.get("round1", {})
        if row.get("parsed_new_bbox") is None or r1.get("pixel_audit_status") == "NOT_RUN":
            key = row.get("qa_id"), row.get("candidate_id")
            item = by_key.get(key)
            if not item:
                raise Phase3V2Error("v1 contract-failure trace is outside frozen v2 cohort")
            selected.append(item)
    if len(selected) != EXPECTED_CONTRACT_CASES or len({item["diagnosis"]["candidate_id"] for item in selected}) != EXPECTED_CONTRACT_CASES:
        raise Phase3V2Error(f"expected exactly {EXPECTED_CONTRACT_CASES} v1 proposal-contract failures, found {len(selected)}")
    return selected


def _parent(item: dict[str, Any], candidate) -> tuple[dict[str, Any], SpatialProposal]:
    r0 = historical_pattern(item["certificate"])
    if not isinstance(r0["bbox"], list) or len(r0["bbox"]) != 4:
        raise Phase3V2Error("historical R0 bbox is missing")
    return r0, SpatialProposal(r0["proposal_id"], candidate.candidate_id, item["sample"].target_claim.claim_id,
                               tuple(r0["bbox"]), provenance={"source": "phase2_historical_artifact"})


def _refine(item: dict[str, Any], config: dict[str, Any], store: ArtifactStore, inference: CachedInference) -> tuple[Any, dict, dict, Any]:
    candidate = candidate_from_manifest(item["candidate_row"], item["sample"])
    controller = SpatialEvidenceAdaptationControllerV2()
    route = controller.route(item["diagnosis"])
    if not route["allowed"]:
        raise Phase3V2Error("frozen v2 cohort member is forbidden by routing")
    r0, parent = _parent(item, candidate)
    engine = SampleRunner(config, store, inference)
    prior_1000 = canonicalize_region_1000(parent.support_region)
    response = engine.infer(item["sample"], candidate, item["sample"].target_claim, route["stage"], stage_data={
        "previous_bbox_normalized_0_1000": prior_1000,
        "diagnosed_failure_reason": route["routing_reason"],
        "previous_original_semantic_status": r0["original_status"],
        "previous_keep_semantic_status": r0["keep_status"],
        "previous_drop_semantic_status": r0["drop_status"],
        "refinement_round": 1,
    })
    outcome = controller.parse(response.get("raw_text") or "", candidate, item["sample"].target_claim,
                               parent=parent, route=route, raw_response_ref=response.get("raw_response_ref"))
    return candidate, route, outcome, engine


def _contract_row(item, candidate, route, outcome, engine) -> dict[str, Any]:
    return {"format": PHASE3_V2_FORMAT, "qa_id": item["sample"].sample_id, "candidate_id": candidate.candidate_id,
            "temporal_candidate_rank": candidate.acquisition_rank, "route": route["routing_reason"],
            "prompt_version": route["prompt_version"], "previous_bbox_normalized_0_1_xyxy": historical_pattern(item["certificate"])["bbox"],
            "previous_bbox_normalized_0_1000": outcome["previous_bbox_normalized_0_1000"],
            "outcome": outcome["outcome"], "failure_code": outcome["failure_code"],
            "bbox_normalized_0_1000": outcome["bbox_normalized_0_1000"], "model_reason": outcome["model_reason"],
            "raw_model_response": outcome["raw_model_response"], "raw_response_ref": outcome["raw_response_ref"],
            "usage": engine.budget.snapshot(), "model_fingerprint": engine.inference.backend.fingerprint(), "gt_used": False}


def _contract_preflight(root: Path, config_path: Path, runtime: Path, phase2: Path, diagnostics: Path, v1_run: Path) -> dict[str, Any]:
    if root.exists():
        raise Phase3V2Error("v2 output directory must not exist")
    config = load_config(config_path)
    _verify_config(config)
    samples = load_runtime(runtime)
    gt = _gt_audit(runtime, samples)
    diagnoses, inputs = _inputs(phase2, diagnostics, runtime)
    failures = _v1_contract_failures(v1_run, inputs)
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise Phase3V2Error("runtime import audit failed")
    root.mkdir(parents=True)
    store, cache = ArtifactStore(root / "contract_preflight"), ArtifactStore(root / "cache")
    backend = make_backend(config["backend"])
    if backend.synthetic:
        raise Phase3V2Error("v2 requires local_hf backend")
    inference = CachedInference(backend, cache)
    traces = []
    with store.run_lock():
        for item in failures:
            candidate, route, outcome, engine = _refine(item, config, store, inference)
            traces.append(_contract_row(item, candidate, route, outcome, engine))
        _write_jsonl(store.root / "phase3_v2_contract_trace.jsonl", traces)
    counts = Counter(row["outcome"] for row in traces)
    coordinate = sum(row["failure_code"] == "REFINEMENT_COORDINATE_VIOLATION" for row in traces)
    parse = sum(row["outcome"] == "PARSE_FAILURE" for row in traces)
    no_op = sum(row["outcome"] == "REFINEMENT_NO_OP" for row in traces)
    gate = parse == 0 and coordinate == 0 and no_op == 0
    report = {"format": PHASE3_V2_FORMAT, "status": "PASS" if gate else "FAIL", "mode": "contract-preflight",
              "engineering_gate_pass": gate, "model_calls_made": sum(row["usage"]["new_calls"] for row in traces),
              "cache_hits": sum(row["usage"]["cache_hits"] for row in traces), "contract_case_count": len(traces),
              "parse_failure_count": parse, "coordinate_violation_count": coordinate, "silent_R0_copy_count": no_op,
              "proposed_count": counts["PROPOSED"], "unresolved_count": counts["UNRESOLVED"],
              "config_sha256": _sha(config_path), "runtime_sha256": samples[0].provenance["runtime_sha256"],
              "phase2_manifest_sha256": _sha(inputs["manifest_path"]), "phase25_sha256": _sha(diagnostics),
              "phase3_v1_trace_sha256": _sha(v1_run / "phase3_spatial_adaptation_trace.jsonl"),
              "candidate_manifest_hash": inputs["manifest_hash"], "gt_isolation": gt, "runtime_import_audit": source_audit,
              "spatial_intervention": config["spatial"]["intervention"],
              "certificate_builder_source_sha256": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest(),
              "contract_trace": str(store.root / "phase3_v2_contract_trace.jsonl"),
              "cache_dir": str(cache.root), "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"]}
    (root / "phase3_v2_contract_preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _semantics(certificate: dict[str, Any]) -> dict[str, Any]:
    spatial = certificate.get("checks", {}).get("spatial", {})
    refs = spatial.get("references", {}) if isinstance(spatial, dict) else {}
    controls = refs.get("controls", []) if isinstance(refs, dict) and isinstance(refs.get("controls"), list) else []
    return {"original": certificate.get("checks", {}).get("semantic", {}).get("status"),
            "keep": refs.get("keep", {}).get("semantic_status") if isinstance(refs.get("keep"), dict) else None,
            "drop": refs.get("drop", {}).get("semantic_status") if isinstance(refs.get("drop"), dict) else None,
            "control": [row.get("semantic_status") for row in controls if isinstance(row, dict)],
            "control_available": spatial.get("control_available") if isinstance(spatial, dict) else None,
            "pixel_audit_refs": spatial.get("pixel_audit_refs", []) if isinstance(spatial, dict) else []}


def _pixel_status(store: ArtifactStore, refs: list[str]) -> str:
    if not refs:
        return "NOT_RUN"
    try:
        return "PASS" if all(store.get_json(ref).get("pixel_audit_pass") for ref in refs) else "FAIL"
    except Exception:
        return "UNREADABLE"


def _run_one(item, config, store, inference, manifest_hash, commit) -> dict[str, Any]:
    candidate, route, outcome, engine = _refine(item, config, store, inference)
    r0, parent = _parent(item, candidate)
    original = engine.infer(item["sample"], candidate, item["sample"].target_claim, "semantic")
    certificate, spatial, r1 = None, None, {"original": None, "keep": None, "drop": None, "control": [],
                                              "control_available": None, "pixel_audit_status": "NOT_RUN"}
    if outcome["outcome"] == "PROPOSED":
        spatial = engine.spatial_with_proposal(item["sample"], candidate, item["sample"].target_claim, original,
                                               outcome["proposal"], proposal_index=1)
        certificate = engine.certificate(item["sample"], candidate, item["sample"].target_claim, original, spatial, None)
        r1 = _semantics(to_dict(certificate))
        r1["pixel_audit_status"] = _pixel_status(store, r1["pixel_audit_refs"])
        r1["certificate_status"] = certificate.final_status.value
        r1["failure_reasons"] = list(certificate.failure_reasons)
        if r1["original"] == "SUPPORTED" and r1["drop"] == "SUPPORTED":
            r1["diagnostic_label"] = "RESIDUAL_SUPPORT_OR_SEMANTIC_INSENSITIVITY"
    else:
        r1.update({"certificate_status": "UNRESOLVED", "failure_reasons": [outcome["failure_code"] or "REFINEMENT_UNRESOLVED"],
                   "diagnostic_label": None})
    after = engine.budget.snapshot()
    return {"format": PHASE3_V2_FORMAT, "qa_id": item["sample"].sample_id, "candidate_id": candidate.candidate_id,
            "temporal_candidate_rank": candidate.acquisition_rank, "spatial_round": 1, "parent_proposal_id": parent.proposal_id,
            "r0_bbox_normalized_0_1_xyxy": list(parent.support_region),
            "r0_bbox_normalized_0_1000": outcome["previous_bbox_normalized_0_1000"], "r0_certificate_status": r0["certificate_status"],
            "original_failure_reasons": list(item["diagnosis"]["original_failure_reasons"]),
            "routing_reason": route["routing_reason"], "refinement_prompt_version": route["prompt_version"],
            "refinement_outcome": outcome["outcome"], "refinement_failure_code": outcome["failure_code"],
            "raw_model_response": outcome["raw_model_response"], "raw_response_ref": outcome["raw_response_ref"], "model_reason": outcome["model_reason"],
            "r1_bbox_normalized_0_1000": outcome["bbox_normalized_0_1000"],
            "r1_bbox_normalized_0_1_xyxy": list(outcome["proposal"].support_region) if outcome["proposal"] else None,
            "bbox_geometry_change": geometry_change(parent.support_region, outcome["proposal"].support_region) if outcome["proposal"] else None,
            "round1": r1, "formal_certificate": to_dict(certificate) if certificate else None,
            "new_model_calls": after["new_calls"], "cache_hits": after["cache_hits"],
            "logical_model_calls": after["calls"], "candidate_manifest_hash": manifest_hash,
            "model_fingerprint": inference.backend.fingerprint(), "operator": config["spatial"]["intervention"],
            "git_commit": commit, "gt_used": False}


def _execute(root: Path, mode: str, config_path: Path, runtime: Path, phase2: Path, diagnostics: Path, v1_run: Path) -> dict[str, Any]:
    contract = _json(root / "phase3_v2_contract_preflight.json")
    if contract.get("engineering_gate_pass") is not True:
        raise Phase3V2Error("v2 contract preflight gate did not pass; full cohort is forbidden")
    bindings = {"config_sha256": _sha(config_path), "runtime_sha256": load_runtime(runtime)[0].provenance["runtime_sha256"],
                "phase2_manifest_sha256": _sha(phase2 / "fixed_candidate_pool.jsonl"), "phase25_sha256": _sha(diagnostics),
                "phase3_v1_trace_sha256": _sha(v1_run / "phase3_spatial_adaptation_trace.jsonl")}
    if any(contract.get(key) != value for key, value in bindings.items()):
        raise Phase3V2Error("v2 contract preflight bindings do not match requested run")
    output = root / mode
    if output.exists():
        raise Phase3V2Error(f"v2 {mode} output already exists")
    config = load_config(config_path)
    _verify_config(config)
    diagnoses, inputs = _inputs(phase2, diagnostics, runtime)
    backend = make_backend(config["backend"])
    if backend.synthetic:
        raise Phase3V2Error("v2 requires local_hf backend")
    store, cache = ArtifactStore(output), ArtifactStore(root / "cache")
    inference = CachedInference(backend, cache)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    started, traces = time.monotonic(), []
    with store.run_lock():
        for item in inputs["cohort"]:
            traces.append(_run_one(item, config, store, inference, inputs["manifest_hash"], commit))
        _write_jsonl(output / "phase3_v2_spatial_adaptation_trace.jsonl", traces)
    new, hits = sum(row["new_model_calls"] for row in traces), sum(row["cache_hits"] for row in traces)
    if mode == "run" and new <= 0:
        raise Phase3V2Error("v2 first cohort run made zero new model calls")
    if mode == "replay" and new != 0:
        raise Phase3V2Error("v2 replay made new model calls")
    transitions = Counter(f"{row['r0_certificate_status']}->{row['round1']['certificate_status']}" for row in traces)
    by_reason: dict[str, Counter] = defaultdict(Counter)
    for row in traces:
        transition = f"{row['r0_certificate_status']}->{row['round1']['certificate_status']}"
        for reason in row["original_failure_reasons"]:
            by_reason[reason][transition] += 1
    proposal_outcomes = Counter(row["refinement_outcome"] for row in traces)
    residual = sum(row["round1"].get("diagnostic_label") == "RESIDUAL_SUPPORT_OR_SEMANTIC_INSENSITIVITY" for row in traces)
    summary = {"format": PHASE3_V2_FORMAT, "status": "PASS", "mode": mode, "eligible_candidate_count": len(traces),
               "excluded_candidate_count": len(diagnoses) - len(traces), "proposal_contract_outcomes": dict(proposal_outcomes),
               "formal_transition_matrix": dict(transitions), "by_original_failure_reason": {k: dict(v) for k, v in by_reason.items()},
               "formal_verified_count": transitions.get("UNCERTAIN->VERIFIED", 0), "unresolved_count": transitions.get("UNCERTAIN->UNRESOLVED", 0),
               "residual_support_or_semantic_insensitivity_count": residual, "new_model_calls": new, "cache_hits": hits,
               "logical_model_calls": sum(row["logical_model_calls"] for row in traces), "latency_seconds": time.monotonic() - started,
               "cache_dir": str(cache.root), "output_dir": str(output), "candidate_manifest_hash": inputs["manifest_hash"], "gt_used": False,
               "operator_unchanged": config["spatial"]["intervention"] == contract["spatial_intervention"],
               "certificate_builder_source_sha256": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest(),
               "certificate_builder_unchanged": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest() == contract["certificate_builder_source_sha256"],
               "semantic_prompt_version": "relive-semantic-v4", "refinement_prompt_versions": sorted({row["refinement_prompt_version"] for row in traces})}
    (output / "phase3_v2_candidate_transition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"status": "PASS", "mode": mode, "phase": "Phase 3-v2", "contract_preflight_gate": True,
             "frozen_temporal_candidates_unchanged": all(row["candidate_manifest_hash"] == inputs["manifest_hash"] for row in traces),
             "one_refinement_round": all(row["spatial_round"] == 1 for row in traces), "gt_used": False,
             "operator_unchanged": summary["operator_unchanged"], "certificate_builder_unchanged": summary["certificate_builder_unchanged"],
             "unresolved_never_enters_certificate_builder": all((row["refinement_outcome"] == "PROPOSED") == (row["formal_certificate"] is not None) for row in traces),
             "pixel_audits_for_proposed": Counter(row["round1"]["pixel_audit_status"] for row in traces if row["refinement_outcome"] == "PROPOSED")}
    (output / "phase3_v2_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 3-v2 spatial refiner contract repair")
    parser.add_argument("--mode", required=True, choices=("contract-preflight", "run", "replay"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--phase2-run-dir", required=True)
    parser.add_argument("--phase25-diagnostics", required=True)
    parser.add_argument("--phase3-v1-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        root, config, runtime = Path(args.output_dir).resolve(), Path(args.config).resolve(), Path(args.runtime).resolve()
        phase2, diag, v1 = Path(args.phase2_run_dir).resolve(), Path(args.phase25_diagnostics).resolve(), Path(args.phase3_v1_run_dir).resolve()
        report = (_contract_preflight(root, config, runtime, phase2, diag, v1) if args.mode == "contract-preflight"
                  else _execute(root, args.mode, config, runtime, phase2, diag, v1))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, Phase3V2Error) as exc:
        print(f"ReliVE Phase 3-v2 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
