"""Phase 3.6: one-round, failure-conditioned automatic ROI re-grounding."""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
import time
from typing import Any

from .audits import audit_run, audit_runtime_imports
from .backends import make_backend
from .certificate import build_certificate
from .config import load_config
from .interventions import OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION
from .phase35_v3 import (PHASE35_V3_FORMAT, Phase35V3Error, candidate_for, runtime_gt_audit,
                         sha256_path, validate_config, validate_prospective_manifest)
from .runner import SampleRunner
from .spatial_adaptation import geometry_change
from .spatial_adaptation_v2 import canonicalize_region_1000, parse_refinement_v2
from .storage.artifacts import ArtifactStore, canonical_json, stable_hash
from .storage.cache import CachedInference
from .types import ExecutionStatus, SpatialProposal, to_dict

PHASE36_FORMAT = "relive-phase36-failure-conditioned-regrounding-v1"
MAX_REGROUNDING_ROUNDS = 1
ROUTES = {
    "phase35-local-001": {"reason": "OVERBROAD_OR_MISLOCALIZED_INTERACTION_ROI",
                            "stage": "reground_local_interaction_overbroad",
                            "action": "REGROUND_LOCAL_INTERACTION_OVERBROAD"},
    "phase35-local-002": {"reason": "INCOMPLETE_INTERACTION_EVIDENCE",
                            "stage": "reground_local_interaction_incomplete",
                            "action": "REGROUND_LOCAL_INTERACTION_INCOMPLETE"},
}
EXCLUDED_CLAIM = "phase35-local-003"


class Phase36Error(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase36Error(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase36Error(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase36Error(f"unreadable JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase36Error(f"nonempty JSONL object rows required: {path}")
    return rows


def _semantics(certificate: dict[str, Any] | None) -> dict[str, Any]:
    if not certificate:
        return {"original": None, "keep": None, "drop": None, "control": [], "pixel_audit_refs": []}
    spatial = certificate.get("checks", {}).get("spatial", {})
    refs = spatial.get("references", {}) if isinstance(spatial, dict) else {}
    controls = refs.get("controls", []) if isinstance(refs.get("controls"), list) else []
    return {"original": certificate.get("checks", {}).get("semantic", {}).get("status"),
            "keep": refs.get("keep", {}).get("semantic_status") if isinstance(refs.get("keep"), dict) else None,
            "drop": refs.get("drop", {}).get("semantic_status") if isinstance(refs.get("drop"), dict) else None,
            "control": [row.get("semantic_status") for row in controls if isinstance(row, dict)],
            "spatial_status": spatial.get("status") if isinstance(spatial, dict) else None,
            "pixel_audit_refs": spatial.get("pixel_audit_refs", []) if isinstance(spatial, dict) else []}


def _pixel(store: ArtifactStore, refs: list[str]) -> str:
    if not refs:
        return "NOT_RUN"
    try:
        audits = [store.get_json(ref) for ref in refs]
    except Exception:
        return "UNREADABLE"
    return "PASS" if all(audit.get("pixel_audit_pass") for audit in audits) else "FAIL"


def _phase35_inputs(v3_run_dir: Path, samples: list[Any], config_path: Path, prospective_hash: str) -> dict[str, Any]:
    trace_path = v3_run_dir / "phase35_v3_trace.jsonl"
    traces = _jsonl(trace_path)
    by_claim = {row.get("claim_id"): row for row in traces}
    if len(by_claim) != 3 or set(by_claim) != {*ROUTES, EXCLUDED_CLAIM}:
        raise Phase36Error("Phase 3.6 requires exactly the frozen three-claim Phase 3-v3 trace")
    parent_root = v3_run_dir.parent
    prior_preflight = _json(parent_root / "phase35_v3_preflight.json")
    if prior_preflight.get("format") != PHASE35_V3_FORMAT or prior_preflight.get("prospective_manifest_sha256") != prospective_hash:
        raise Phase36Error("Phase 3-v3 preflight is not bound to this prospective manifest")
    if prior_preflight.get("config_sha256") != sha256_path(config_path):
        raise Phase36Error("Phase 3-v3 config differs from requested Phase 3.6 config")
    by_sample_result = {}
    for path in sorted((v3_run_dir / "samples").glob("*.json")):
        value = _json(path)
        by_sample_result[value.get("sample_id")] = value
    if len(by_sample_result) != 3:
        raise Phase36Error("Phase 3-v3 run requires exactly three completed sample artifacts")
    sample_by_claim = {sample.target_claim.claim_id: sample for sample in samples}
    if set(sample_by_claim) != set(by_claim):
        raise Phase36Error("fresh runtime claim IDs differ from Phase 3-v3 trace")
    records: dict[str, dict[str, Any]] = {}
    for claim_id, trace in by_claim.items():
        sample = sample_by_claim[claim_id]
        if trace.get("sample_id") != sample.sample_id or trace.get("candidate_id") != candidate_for(sample).candidate_id:
            raise Phase36Error("Phase 3-v3 trace candidate lineage mismatch")
        certs = by_sample_result[sample.sample_id].get("certificates", [])
        if len(certs) != 1:
            raise Phase36Error("Phase 3-v3 sample must have exactly one formal certificate")
        certificate = certs[0]
        spatial = certificate.get("checks", {}).get("spatial", {})
        proposal = spatial.get("proposal") if isinstance(spatial, dict) else None
        records[claim_id] = {"sample": sample, "trace": trace, "certificate": certificate, "proposal": proposal}
    # Frozen selection from the observed Phase 3-v3 artifact. These predicates are
    # diagnostic lineage checks, not new scientific admission criteria.
    first, second, excluded = records["phase35-local-001"], records["phase35-local-002"], records[EXCLUDED_CLAIM]
    if not (first["trace"].get("semantic_variants", {}).get("original") == "SUPPORTED"
            and first["proposal"] and first["trace"].get("semantic_variants", {}).get("drop") == "SUPPORTED"
            and first["trace"].get("automatic_proposal_status") == "CONTROL_UNAVAILABLE"):
        raise Phase36Error("local-001 no longer matches frozen overbroad/mislocalized Phase 3-v3 lineage")
    if not (second["trace"].get("semantic_variants", {}).get("original") == "SUPPORTED"
            and second["proposal"] and second["trace"].get("semantic_variants", {}).get("keep") == "INSUFFICIENT"
            and second["trace"].get("semantic_variants", {}).get("drop") == "SUPPORTED"):
        raise Phase36Error("local-002 no longer matches frozen incomplete-interaction Phase 3-v3 lineage")
    if excluded["trace"].get("semantic_variants", {}).get("original") != "INSUFFICIENT":
        raise Phase36Error("local-003 exclusion lineage mismatch")
    return {"records": records, "prior_preflight": prior_preflight, "trace_path": trace_path}


def _route(claim_id: str) -> dict[str, str]:
    route = ROUTES.get(claim_id)
    if route is None:
        raise Phase36Error("claim is outside the frozen Phase 3.6 cohort")
    from .claims import PROMPT_VERSIONS
    return {"allowed": True, "routing_reason": route["reason"], "stage": route["stage"],
            "action": route["action"], "prompt_version": PROMPT_VERSIONS[route["stage"]]}


def _parent(record: dict[str, Any]) -> SpatialProposal:
    proposal = record["proposal"]
    region = proposal.get("support_region") if isinstance(proposal, dict) else None
    if not isinstance(region, list) or len(region) != 4:
        raise Phase36Error("R0 proposal lacks a valid support region")
    candidate = candidate_for(record["sample"])
    return SpatialProposal(proposal["proposal_id"], candidate.candidate_id, record["sample"].target_claim.claim_id,
                           tuple(region), provenance={"source": "phase35_v3_formal_artifact"})


def _geometry(a, b) -> dict[str, Any]:
    result = geometry_change(a, b)
    return {"r0_area_fraction": (a[2] - a[0]) * (a[3] - a[1]),
            "r1_area_fraction": (b[2] - b[0]) * (b[3] - b[1]), **result}


def preflight(*, config_path: Path, runtime_path: Path, prospective_manifest_path: Path,
              phase35_v3_run_dir: Path, output_dir: Path, require_real: bool = True) -> dict[str, Any]:
    if output_dir.exists():
        raise Phase36Error("Phase 3.6 output directory must not exist before zero-call preflight")
    config = load_config(config_path)
    validate_config(config, require_real=require_real)
    prospective, samples = validate_prospective_manifest(prospective_manifest_path, runtime_path)
    gt = runtime_gt_audit(runtime_path, samples)
    inputs = _phase35_inputs(phase35_v3_run_dir, samples, config_path, prospective["manifest_sha256"])
    source = audit_runtime_imports()
    if source["status"] != "PASS":
        raise Phase36Error("runtime import audit failed")
    cohort = []
    for claim_id in ROUTES:
        record = inputs["records"][claim_id]; parent = _parent(record); route = _route(claim_id)
        cohort.append({"claim_id": claim_id, "sample_id": record["sample"].sample_id,
                       "candidate_id": candidate_for(record["sample"]).candidate_id,
                       "frame_ids": list(candidate_for(record["sample"]).frame_ids),
                       "r0_normalized_0_1_xyxy": list(parent.support_region),
                       "r0_normalized_0_1000_xyxy": canonicalize_region_1000(parent.support_region),
                       "routing_reason": route["routing_reason"], "prompt_version": route["prompt_version"]})
    output_dir.mkdir(parents=True)
    (output_dir / "phase36_frozen_cohort.jsonl").write_text("".join(canonical_json(row) + "\n" for row in cohort), encoding="utf-8")
    plan = {"format": PHASE36_FORMAT, "status": "PASS", "mode": "preflight", "model_calls_made": 0,
            "cache_mutated": False, "config": str(config_path), "config_sha256": sha256_path(config_path),
            "runtime": str(runtime_path), "runtime_sha256": samples[0].provenance["runtime_sha256"],
            "prospective_manifest": str(prospective_manifest_path), "prospective_manifest_sha256": prospective["manifest_sha256"],
            "phase35_v3_run_dir": str(phase35_v3_run_dir), "phase35_v3_trace_sha256": sha256_path(inputs["trace_path"]),
            "cohort": cohort, "cohort_claim_ids": list(ROUTES), "excluded": {"claim_id": EXCLUDED_CLAIM,
              "reason": "ORIGINAL_INSUFFICIENT", "classification": "BASE_SEMANTIC_OR_TEMPORAL_LIMIT", "spatial_refinement_allowed": False},
            "max_regrounding_rounds": MAX_REGROUNDING_ROUNDS, "planned_max_model_calls": 12,
            "planned_call_derivation": "per eligible candidate: one reason-specific R1 refiner + ORIGINAL + automatic-free R1 KEEP + DROP + one matched CONTROL; later formal variants are conditional on the existing core protocol",
            "spatial_intervention": config["spatial"]["intervention"], "certificate_builder_source_sha256": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest(),
            "semantic_prompt_version": "relive-semantic-v4", "operator_unchanged": True, "certificate_builder_unchanged": True,
            "phase35_artifacts_read_only": True, "human_roi_not_in_runtime_or_model_inputs": True,
            "cache_dir": str(output_dir / "cache"), "run_dir": str(output_dir / "run"), "replay_dir": str(output_dir / "replay"),
            "runtime_gt_isolation_audit": gt, "runtime_import_audit": source,
            "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"]}
    (output_dir / "phase36_preflight.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def _run_one(record: dict[str, Any], config: dict[str, Any], store: ArtifactStore, inference: CachedInference,
             bindings: dict[str, Any]) -> dict[str, Any]:
    sample, candidate = record["sample"], candidate_for(record["sample"])
    parent, route = _parent(record), _route(sample.target_claim.claim_id)
    engine = SampleRunner(config, store, inference)
    before = engine.budget.snapshot()
    r0_1000 = canonicalize_region_1000(parent.support_region)
    response = engine.infer(sample, candidate, sample.target_claim, route["stage"], stage_data={
        "previous_bbox_normalized_0_1000": r0_1000, "diagnosed_failure_reason": route["routing_reason"],
        "refinement_round": 1})
    outcome = parse_refinement_v2(response.get("raw_text") or "", candidate, sample.target_claim, parent=parent,
                                  route=route, raw_response_ref=response.get("raw_response_ref"),
                                  protocol_format=PHASE36_FORMAT, proposal_id_prefix="phase36-refinement")
    certificate, semantic = None, {"original": None, "keep": None, "drop": None, "control": [], "pixel_audit_refs": []}
    flags: list[str] = []
    if outcome["outcome"] == "PROPOSED":
        original = engine.infer(sample, candidate, sample.target_claim, "semantic")
        spatial = engine.spatial_with_proposal(sample, candidate, sample.target_claim, original, outcome["proposal"], proposal_index=1)
        certificate = engine.certificate(sample, candidate, sample.target_claim, original, spatial, None)
        engine.certificates.append(certificate)
        semantic = _semantics(to_dict(certificate))
        if semantic["original"] == "SUPPORTED" and semantic["keep"] == "SUPPORTED" and semantic["drop"] == "SUPPORTED":
            flags.append("RESIDUAL_SUPPORT_OR_SEMANTIC_INSENSITIVITY")
        terminal = {"VERIFIED": "VALID_R1_FORMAL_VERIFIED", "REJECTED": "VALID_R1_FORMAL_REJECTED"}.get(certificate.final_status.value, "VALID_R1_FORMAL_UNCERTAIN")
    else:
        terminal = "REGROUNDING_UNRESOLVED" if outcome["outcome"] == "UNRESOLVED" else outcome["outcome"]
    coverage, strict = engine.coverage_and_answer(sample)
    sample_result = to_dict({"sample_id": sample.sample_id, "task": sample.task, "synthetic": inference.backend.synthetic,
        "status": "COMPLETE", "claims": [sample.target_claim], "candidates": [candidate], "spatial_proposals": engine.proposals,
        "certificates": engine.certificates, "coverage": coverage, "strict": strict, "before_adaptation_strict": strict,
        "adaptation_events": [], "termination_reason": terminal, "usage": engine.budget.snapshot(), "provenance": sample.provenance})
    after = engine.budget.snapshot()
    r1 = outcome.get("proposal")
    return {"format": PHASE36_FORMAT, "claim_id": sample.target_claim.claim_id, "sample_id": sample.sample_id,
            "candidate_id": candidate.candidate_id, "frozen_frame_ids": list(candidate.frame_ids),
            "r0_normalized_0_1_xyxy": list(parent.support_region), "r0_normalized_0_1000_xyxy": r0_1000,
            "routing_reason": route["routing_reason"], "prompt_version": route["prompt_version"],
            "raw_refiner_response": outcome["raw_model_response"], "raw_response_ref": outcome["raw_response_ref"],
            "refinement_outcome": outcome["outcome"], "refinement_failure_code": outcome["failure_code"],
            "r1_normalized_0_1000_xyxy": outcome["bbox_normalized_0_1000"],
            "r1_normalized_0_1_xyxy": list(r1.support_region) if r1 else None,
            "geometry": _geometry(parent.support_region, r1.support_region) if r1 else None,
            "semantic": semantic, "pixel_audit_status": _pixel(store, semantic["pixel_audit_refs"]),
            "formal_certificate": to_dict(certificate) if certificate else None,
            "certificate_status": certificate.final_status.value if certificate else "UNCERTAIN",
            "certificate_failure_reasons": list(certificate.failure_reasons) if certificate else [outcome["failure_code"] or "REGROUNDING_UNRESOLVED"],
            "terminal_outcome": terminal, "diagnostic_flags": flags, "new_model_calls": after["new_calls"] - before["new_calls"],
            "cache_hits": after["cache_hits"] - before["cache_hits"], "logical_model_calls": after["calls"] - before["calls"],
            "model_fingerprint": inference.backend.fingerprint(), "git_commit": bindings["git_commit"],
            "bindings": bindings, "operator": config["spatial"]["intervention"], "gt_used": False,
            "_sample_result": sample_result}


def execute(*, config_path: Path, runtime_path: Path, prospective_manifest_path: Path,
            phase35_v3_run_dir: Path, output_dir: Path, mode: str, require_real: bool = True) -> dict[str, Any]:
    if mode not in {"run", "replay"}:
        raise Phase36Error("mode must be run or replay")
    plan = _json(output_dir / "phase36_preflight.json")
    config = load_config(config_path); validate_config(config, require_real=require_real)
    prospective, samples = validate_prospective_manifest(prospective_manifest_path, runtime_path)
    inputs = _phase35_inputs(phase35_v3_run_dir, samples, config_path, prospective["manifest_sha256"])
    expected = {"config_sha256": sha256_path(config_path), "runtime_sha256": samples[0].provenance["runtime_sha256"],
                "prospective_manifest_sha256": prospective["manifest_sha256"], "phase35_v3_trace_sha256": sha256_path(inputs["trace_path"])}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise Phase36Error("missing or incompatible zero-call Phase 3.6 preflight")
    if plan.get("cohort_claim_ids") != list(ROUTES):
        raise Phase36Error("Phase 3.6 cohort changed after preflight")
    target = output_dir / mode
    if target.exists():
        raise Phase36Error(f"Phase 3.6 {mode} output already exists")
    backend = make_backend(config["backend"])
    if require_real and backend.synthetic:
        raise Phase36Error("Phase 3.6 requires the real local_hf backend")
    store, cache = ArtifactStore(target), ArtifactStore(output_dir / "cache")
    inference = CachedInference(backend, cache)
    bindings = {**expected, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    started, traces = time.monotonic(), []
    with store.run_lock():
        for claim_id in ROUTES:
            row = _run_one(inputs["records"][claim_id], config, store, inference, bindings)
            store.put_json("samples", "sample-" + stable_hash(row["sample_id"]), row.pop("_sample_result"))
            traces.append(row)
        (store.root / "phase36_regrounding_trace.jsonl").write_text("".join(canonical_json(row) + "\n" for row in traces), encoding="utf-8")
        (store.root / "phase36_formal_verification.jsonl").write_text("".join(canonical_json({"claim_id": row["claim_id"], "formal_certificate": row["formal_certificate"]}) + "\n" for row in traces if row["formal_certificate"]), encoding="utf-8")
    new, hits = sum(row["new_model_calls"] for row in traces), sum(row["cache_hits"] for row in traces)
    if mode == "run" and new <= 0: raise Phase36Error("Phase 3.6 first run made zero model calls")
    if mode == "replay" and new != 0: raise Phase36Error("Phase 3.6 replay made new model calls")
    summary = {"format": PHASE36_FORMAT, "status": "PASS", "mode": mode, "cohort_count": len(traces),
               "terminal_outcomes": dict(Counter(row["terminal_outcome"] for row in traces)),
               "certificate_distribution": dict(Counter(row["certificate_status"] for row in traces)),
               "formal_verified_count": sum(row["certificate_status"] == "VERIFIED" for row in traces),
               "residual_drop_support_count": sum("RESIDUAL_SUPPORT_OR_SEMANTIC_INSENSITIVITY" in row["diagnostic_flags"] for row in traces),
               "pixel_audit_pass_count": sum(row["pixel_audit_status"] == "PASS" for row in traces), "new_model_calls": new, "cache_hits": hits,
               "logical_model_calls": sum(row["logical_model_calls"] for row in traces), "latency_seconds": time.monotonic() - started,
               "cache_dir": str(cache.root), "output_dir": str(store.root), "operator_unchanged": config["spatial"]["intervention"] == plan["spatial_intervention"],
               "certificate_builder_unchanged": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest() == plan["certificate_builder_source_sha256"],
               "max_regrounding_rounds": MAX_REGROUNDING_ROUNDS,
               "regrounding_attempt_count": len(traces),
               "one_round_only": MAX_REGROUNDING_ROUNDS == 1 and len(traces) == len(ROUTES),
               "reason_specific_refinement_called": True, "gt_used": False}
    strict = audit_run(target)
    audit = {"status": "PASS" if strict["status"] == "PASS" else "FAIL", "mode": mode, "zero_call_preflight_bound": True,
             "cohort_correct": [row["claim_id"] for row in traces] == list(ROUTES), "local003_excluded": True,
             "one_refinement_round": True, "unresolved_never_enters_formal": all((row["refinement_outcome"] == "PROPOSED") == (row["formal_certificate"] is not None) for row in traces),
             "operator_unchanged": summary["operator_unchanged"], "certificate_builder_unchanged": summary["certificate_builder_unchanged"],
             "strict_audit": strict, "gt_isolation": runtime_gt_audit(runtime_path, samples)}
    report = {"status": audit["status"], "summary": summary, "audit": audit,
              "traces": str(store.root / "phase36_regrounding_trace.jsonl")}
    suffix = "" if mode == "run" else f"_{mode}"
    (output_dir / f"phase36{suffix}_certificate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"phase36{suffix}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"phase36_{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if mode == "run":
        lines = ["# ReliVE Phase 3.6 certificate summary", "", "| Claim | R1 outcome | Certificate | Diagnostic flags |", "| --- | --- | --- | --- |"]
        lines.extend(f"| `{row['claim_id']}` | `{row['terminal_outcome']}` | `{row['certificate_status']}` | `{', '.join(row['diagnostic_flags']) or '-'}` |" for row in traces)
        lines.extend(["", "R1 geometry is GT-free diagnostic provenance only. Formal status is emitted solely by the existing certificate builder.", ""])
        (output_dir / "phase36_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report
