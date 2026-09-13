"""Fresh, frozen-window LOCAL_ATOMIC spatial-certificate vertical slice.

This module owns only Phase 3-v3 cohort binding and execution bookkeeping.  It
uses the ordinary :class:`SampleRunner` automatic spatial proposer, registered
intervention, and ``build_certificate`` path.  It deliberately has no access
to human confirmation prose, ROIs, historical certificates, failure labels,
or GT fields.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from PIL import Image

from .audits import audit_run, audit_runtime_imports, audit_strict_result
from .backends import make_backend
from .certificate import POLICY_VERSION, build_certificate
from .claim_scope import (CLAIM_SCOPE_ROUTER_VERSION, LOCAL_ATOMIC, claim_text_sha256,
                          route_runtime_claim, stable_scope_hash)
from .config import load_config
from .data.schemas import load_runtime
from .interventions import OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION
from .runner import SampleRunner
from .storage.artifacts import ArtifactStore, canonical_json, stable_hash
from .storage.cache import CachedInference
from .types import EvidenceCandidate, ExecutionStatus, FinalStatus, SemanticStatus, to_dict

PHASE35_V3_FORMAT = "relive-phase35-v3-fresh-local-atomic-v1"
PROTOCOL_VERSION = "relive-phase35-v3-fixed-public-window-v1"


class Phase35V3Error(ValueError):
    """A pre-inference Phase 3-v3 invariant failed."""


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase35V3Error(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase35V3Error(f"JSON object required: {path}")
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _has_forbidden_shape(value: Any) -> bool:
    forbidden = {"reference_answer", "assistant_answer", "temporal_gt", "bbox", "mask", "struc_info",
                 "rc_info", "true_support", "spurious_support", "gt_iou", "evaluation_artifact",
                 "certificate", "semantic_status", "failure_reason", "keep", "drop", "r0", "r1"}
    if isinstance(value, dict):
        return any(str(key).casefold() in forbidden or _has_forbidden_shape(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_has_forbidden_shape(item) for item in value)
    return False


def validate_prospective_manifest(manifest_path: Path, runtime_path: Path) -> tuple[dict[str, Any], list[Any]]:
    """Bind one Phase 3.5 manifest to exact public runtime claim/windows."""
    manifest = _json(manifest_path)
    if manifest.get("format") != "relive-phase35-prospective-local-atomic-manifest-v1":
        raise Phase35V3Error("requires the frozen Phase 3.5 LOCAL_ATOMIC manifest format")
    if manifest.get("selection_status") != "FROZEN_PRE_SPATIAL_CERTIFICATE_OUTCOMES":
        raise Phase35V3Error("prospective manifest was not frozen before spatial outcomes")
    if manifest.get("router_version") != CLAIM_SCOPE_ROUTER_VERSION:
        raise Phase35V3Error("prospective manifest router version is incompatible")
    supplied_hash = manifest.get("manifest_sha256")
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied_hash != stable_scope_hash([content]):
        raise Phase35V3Error("prospective manifest SHA-256 does not bind its content")
    if manifest.get("not_a_temporal_candidate_pool") is not True or manifest.get("phase3_v3_executed") is not False:
        raise Phase35V3Error("prospective manifest does not authorize its first fixed-window Phase 3-v3 run")
    if manifest.get("gt_used") is not False or _has_forbidden_shape(manifest):
        raise Phase35V3Error("prospective manifest contains GT- or outcome-shaped input")
    samples = load_runtime(runtime_path)
    runtime_sha = samples[0].provenance.get("runtime_sha256") if samples else None
    if manifest.get("prospective_runtime_sha256") != runtime_sha:
        raise Phase35V3Error("prospective manifest does not bind this runtime SHA-256")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 5:
        raise Phase35V3Error("prospective manifest must select 1-5 fresh samples")
    if manifest.get("selected_sample_count") != len(selected):
        raise Phase35V3Error("prospective manifest selected_sample_count mismatch")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise Phase35V3Error("prospective runtime contains duplicate sample IDs")
    selected_ids = [row.get("sample_id") for row in selected if isinstance(row, dict)]
    if len(selected_ids) != len(selected) or len(set(selected_ids)) != len(selected_ids):
        raise Phase35V3Error("prospective manifest has malformed or duplicate sample IDs")
    if set(selected_ids) - set(sample_by_id):
        raise Phase35V3Error("prospective manifest selects a sample absent from its runtime")
    ordered = []
    for row in selected:
        sample = sample_by_id[row["sample_id"]]
        decision = route_runtime_claim(sample)
        if (row.get("claim_scope") != LOCAL_ATOMIC or row.get("single_roi_certificate_applicable") is not True
                or decision.claim_scope != LOCAL_ATOMIC or not decision.single_roi_certificate_applicable):
            raise Phase35V3Error("only prospectively LOCAL_ATOMIC claims may enter Phase 3-v3")
        if row.get("claim_id") != sample.target_claim.claim_id or row.get("claim_text_sha256") != claim_text_sha256(sample.target_claim.text):
            raise Phase35V3Error("prospective manifest claim binding mismatch")
        if not 1 <= len(sample.frames) <= 3:
            raise Phase35V3Error("Phase 3-v3 accepts exactly one frozen public window of 1-3 frames")
        orders = [frame.order for frame in sample.frames]
        if orders != list(range(orders[0], orders[0] + len(orders))):
            raise Phase35V3Error("Phase 3-v3 frozen public frames must be consecutive")
        ordered.append(sample)
    return manifest, ordered


def validate_config(config: dict[str, Any], *, require_real: bool) -> None:
    if require_real and config["backend"].get("kind") != "local_hf":
        raise Phase35V3Error("Phase 3-v3 requires the reviewed local_hf Qwen backend")
    if config["policy"].get("name") != "semantic_spatial" or config["policy"].get("version") != POLICY_VERSION:
        raise Phase35V3Error("Phase 3-v3 requires the existing semantic_spatial certificate policy")
    intervention = config["spatial"].get("intervention", {})
    if (intervention.get("operator") != OPAQUE_GRAY_OPERATOR
            or intervention.get("operator_version") != OPAQUE_GRAY_VERSION):
        raise Phase35V3Error("Phase 3-v3 requires the registered opaque_gray v1 operator")
    if config["adaptation"].get("enabled") is not False:
        raise Phase35V3Error("Phase 3-v3 does not permit temporal or reason-specific adaptation")
    if config["budget"].get("max_calls", 0) < 5 or config["budget"].get("max_spatial_proposals", 0) < 1:
        raise Phase35V3Error("Phase 3-v3 requires budget for one automatic proposal and four semantic variants")
    if config["spatial"].get("control_count") != 1:
        raise Phase35V3Error("Phase 3-v3 requires the existing one matched-control geometry")


def runtime_gt_audit(runtime_path: Path, samples: list[Any]) -> dict[str, Any]:
    sidecar = Path(str(runtime_path) + ".gt_isolation_audit.json")
    audit = _json(sidecar)
    expected = samples[0].provenance.get("runtime_sha256")
    if audit.get("status") != "PASS" or audit.get("runtime_sha256") != expected:
        raise Phase35V3Error("runtime GT-isolation audit is missing, failed, or unbound")
    return {"status": "PASS", "path": str(sidecar), "runtime_sha256": expected,
            "classification": audit.get("classification"), "adapter": audit.get("adapter"),
            "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt",
                                                "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"]}


def candidate_for(sample: Any) -> EvidenceCandidate:
    """Create the only admissible candidate: all and only frozen runtime frames."""
    frame_ids = tuple(frame.frame_id for frame in sample.frames)
    identity = {"protocol": PROTOCOL_VERSION, "sample_id": sample.sample_id, "frame_ids": list(frame_ids),
                "runtime_sha256": sample.provenance.get("runtime_sha256")}
    candidate_id = "ev_" + stable_hash(identity)[:20]
    return EvidenceCandidate(candidate_id, sample.sample_id, frame_ids,
                             tuple(frame.timestamp for frame in sample.frames), 0,
                             "frozen_public_window", provenance={
                                 "acquisition_version": PROTOCOL_VERSION,
                                 "parameters": {"candidate_pool": "exactly_one_frozen_public_window"},
                                 "source_runtime_sha256": sample.provenance.get("runtime_sha256"),
                                 "frame_references": [frame.source_reference for frame in sample.frames],
                                 "semantic_relevance_scored": False,
                             })


def candidate_rows(samples: list[Any]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for sample in samples:
        candidate = candidate_for(sample)
        rows.append({"format": "relive-phase35-v3-fixed-public-window-candidate-v1",
                     "sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
                     "candidate_rank": 0, "claim_id": sample.target_claim.claim_id,
                     "claim_text_sha256": claim_text_sha256(sample.target_claim.text),
                     "frame_ids": list(candidate.frame_ids), "frame_orders": [frame.order for frame in sample.frames],
                     "candidate_pool": "exactly_one_frozen_public_window", "rank_source": candidate.rank_source,
                     "source_runtime_sha256": sample.provenance.get("runtime_sha256"), "gt_used": False})
    digest = stable_hash(rows)
    return [{**row, "candidate_manifest_hash": digest} for row in rows], digest


def audit_candidate_frames(samples: list[Any]) -> dict[str, Any]:
    errors = []
    for sample in samples:
        for frame in sample.frames:
            try:
                with Image.open(frame.path) as image:
                    image.verify()
            except (OSError, ValueError):
                errors.append({"sample_id": sample.sample_id, "frame_id": frame.frame_id, "path": frame.path})
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "frames_checked": sum(len(sample.frames) for sample in samples), "decoder": "Pillow.verify"}


def preflight(*, config_path: Path, runtime_path: Path, prospective_manifest_path: Path,
              output_dir: Path, require_real: bool = True) -> dict[str, Any]:
    if output_dir.exists():
        raise Phase35V3Error("Phase 3-v3 output directory must not exist before zero-call preflight")
    config = load_config(config_path)
    validate_config(config, require_real=require_real)
    prospective, samples = validate_prospective_manifest(prospective_manifest_path, runtime_path)
    gt = runtime_gt_audit(runtime_path, samples)
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise Phase35V3Error("runtime import audit failed")
    frame_audit = audit_candidate_frames(samples)
    if frame_audit["status"] != "PASS":
        raise Phase35V3Error("frozen public frame audit failed")
    rows, candidate_hash = candidate_rows(samples)
    output_dir.mkdir(parents=True)
    (output_dir / "phase35_v3_fixed_candidate_pool.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    plan = {"format": PHASE35_V3_FORMAT, "status": "PASS", "mode": "preflight",
            "phase": "ReliVE Phase 3-v3 fresh LOCAL_ATOMIC automatic spatial-certificate vertical slice",
            "model_calls_made": 0, "cache_mutated": False, "git_commit": _git_commit(),
            "config": str(config_path), "config_sha256": sha256_path(config_path),
            "runtime": str(runtime_path), "runtime_sha256": samples[0].provenance["runtime_sha256"],
            "prospective_manifest": str(prospective_manifest_path),
            "prospective_manifest_sha256": prospective["manifest_sha256"],
            "candidate_manifest": str(output_dir / "phase35_v3_fixed_candidate_pool.jsonl"),
            "candidate_manifest_hash": candidate_hash, "candidate_pool": "one exact frozen public window per sample",
            "sample_count": len(samples), "frame_counts": {sample.sample_id: len(sample.frames) for sample in samples},
            "spatial_intervention": config["spatial"]["intervention"],
            "semantic_prompt_version": "relive-semantic-v4", "certificate_policy_version": POLICY_VERSION,
            "certificate_builder_source_sha256": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest(),
            "planned_max_model_calls": len(samples) * 5,
            "planned_call_derivation": "per frozen sample: ORIGINAL semantic + automatic spatial proposal + KEEP + DROP + one matched CONTROL; latter stages are skipped only if the existing formal protocol stops earlier",
            "automatic_roi_only": True, "human_roi_not_in_runtime_or_model_inputs": True,
            "reason_specific_refinement_called": False, "adaptation_disabled": True,
            "cache_dir": str(output_dir / "cache"), "run_dir": str(output_dir / "run"),
            "replay_dir": str(output_dir / "replay"), "frame_audit": frame_audit,
            "runtime_gt_isolation_audit": gt, "runtime_import_audit": source_audit,
            "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt",
                                               "struc_info", "RC_info", "evaluation_artifacts"]}
    (output_dir / "phase35_v3_preflight.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def _semantic(certificate: dict[str, Any]) -> dict[str, Any]:
    spatial = certificate.get("checks", {}).get("spatial", {})
    refs = spatial.get("references", {}) if isinstance(spatial, dict) else {}
    controls = refs.get("controls", []) if isinstance(refs.get("controls"), list) else []
    return {"original": certificate.get("checks", {}).get("semantic", {}).get("status"),
            "keep": refs.get("keep", {}).get("semantic_status") if isinstance(refs.get("keep"), dict) else None,
            "drop": refs.get("drop", {}).get("semantic_status") if isinstance(refs.get("drop"), dict) else None,
            "control": [row.get("semantic_status") for row in controls if isinstance(row, dict)],
            "spatial_status": spatial.get("status") if isinstance(spatial, dict) else None,
            "pixel_audit_refs": spatial.get("pixel_audit_refs", []) if isinstance(spatial, dict) else []}


def _pixel_audit_status(store: ArtifactStore, refs: list[str]) -> str:
    if not refs:
        return "NOT_RUN"
    try:
        audits = [store.get_json(ref) for ref in refs]
    except Exception:
        return "UNREADABLE"
    return "PASS" if all(item.get("pixel_audit_pass") for item in audits) else "FAIL"


def _run_one(sample: Any, config: dict[str, Any], store: ArtifactStore, inference: CachedInference,
             candidate_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = candidate_for(sample)
    engine = SampleRunner(config, store, inference)
    engine.claims = [sample.target_claim]
    engine.candidates = [candidate]
    engine.seen_candidates.add(candidate.candidate_id)
    engine.record("claims", {"sample_id": sample.sample_id, "claims": engine.claims})
    engine.record("candidates", candidate)
    before = engine.budget.snapshot()
    certificate = engine.pair(sample, candidate, sample.target_claim, proposal_index=0)
    engine.certificates.append(certificate)
    coverage, strict = engine.coverage_and_answer(sample)
    cert = to_dict(certificate)
    final = certificate.final_status
    termination = "FORMAL_FIXED_WINDOW_VERIFIED" if final == FinalStatus.VERIFIED else "FORMAL_FIXED_WINDOW_COMPLETE"
    result = {"sample_id": sample.sample_id, "task": sample.task, "synthetic": inference.backend.synthetic,
              "status": "COMPLETE", "claims": engine.claims, "candidates": engine.candidates,
              "spatial_proposals": engine.proposals, "certificates": engine.certificates, "coverage": coverage,
              "strict": strict, "before_adaptation_strict": strict, "adaptation_events": [],
              "termination_reason": termination, "usage": engine.budget.snapshot(),
              "candidate_protocol": {"version": PROTOCOL_VERSION, "candidate_pool": "exactly_one_frozen_public_window",
                                     "candidate_manifest_hash": candidate_hash, "candidate_id": candidate.candidate_id,
                                     "frame_ids": list(candidate.frame_ids), "automatic_roi_only": True,
                                     "reason_specific_refinement_called": False},
              "provenance": {**sample.provenance, "source_qa_type": sample.metadata.get("source_qa_type"),
                             "dataset_name": sample.metadata.get("dataset_name"),
                             "spatial_intervention": config["spatial"]["intervention"]}}
    result = to_dict(result)
    strict_audit = audit_strict_result(result)
    if strict_audit["status"] != "PASS":
        raise Phase35V3Error(f"strict source audit failed: {strict_audit['issues']}")
    engine.record("strict_audits", strict_audit)
    after = engine.budget.snapshot()
    stages = _semantic(cert)
    spatial = cert.get("checks", {}).get("spatial", {})
    trace = {"format": PHASE35_V3_FORMAT, "sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
             "claim_id": sample.target_claim.claim_id, "frame_ids": list(candidate.frame_ids),
             "automatic_support_region": spatial.get("proposal", {}).get("support_region") if isinstance(spatial.get("proposal"), dict) else None,
             "automatic_proposal_status": spatial.get("status") if isinstance(spatial, dict) else None,
             "semantic_variants": stages, "pixel_audit_status": _pixel_audit_status(store, stages["pixel_audit_refs"]),
             "certificate_status": cert["final_status"], "failure_reasons": cert["failure_reasons"],
             "new_model_calls": after["new_calls"] - before["new_calls"], "cache_hits": after["cache_hits"] - before["cache_hits"],
             "logical_model_calls": after["calls"] - before["calls"], "candidate_manifest_hash": candidate_hash,
             "operator": config["spatial"]["intervention"], "gt_used": False}
    return result, trace


def execute(*, config_path: Path, runtime_path: Path, prospective_manifest_path: Path,
            output_dir: Path, mode: str, require_real: bool = True) -> dict[str, Any]:
    if mode not in {"run", "replay"}:
        raise Phase35V3Error("mode must be run or replay")
    preflight_path = output_dir / "phase35_v3_preflight.json"
    preflight_report = _json(preflight_path)
    config = load_config(config_path)
    validate_config(config, require_real=require_real)
    prospective, samples = validate_prospective_manifest(prospective_manifest_path, runtime_path)
    expected = {"config_sha256": sha256_path(config_path), "runtime_sha256": samples[0].provenance["runtime_sha256"],
                "prospective_manifest_sha256": prospective["manifest_sha256"]}
    if any(preflight_report.get(key) != value for key, value in expected.items()):
        raise Phase35V3Error("missing or incompatible zero-call Phase 3-v3 preflight")
    rows, candidate_hash = candidate_rows(samples)
    if preflight_report.get("candidate_manifest_hash") != candidate_hash:
        raise Phase35V3Error("fixed candidate manifest changed after preflight")
    run_dir = output_dir / mode
    if run_dir.exists():
        raise Phase35V3Error(f"Phase 3-v3 {mode} output already exists")
    backend = make_backend(config["backend"])
    if require_real and backend.synthetic:
        raise Phase35V3Error("Phase 3-v3 requires real local_hf execution")
    store, cache = ArtifactStore(run_dir), ArtifactStore(output_dir / "cache")
    inference = CachedInference(backend, cache)
    started = time.monotonic()
    results, traces = [], []
    with store.run_lock():
        store.put_json("manifest", "run", {"format": PHASE35_V3_FORMAT, "mode": mode, "config": config,
                                              "config_sha256": expected["config_sha256"], "runtime": str(runtime_path),
                                              "runtime_sha256": expected["runtime_sha256"], "candidate_manifest_hash": candidate_hash,
                                              "prospective_manifest_sha256": expected["prospective_manifest_sha256"],
                                              "spatial_intervention": config["spatial"]["intervention"], "git_commit": _git_commit(),
                                              "reason_specific_refinement_called": False, "gt_used": False})
        for sample in samples:
            result, trace = _run_one(sample, config, store, inference, candidate_hash)
            store.put_json("samples", "sample-" + stable_hash(sample.sample_id), result)
            results.append(result); traces.append(trace)
        (store.root / "phase35_v3_trace.jsonl").write_text("".join(canonical_json(row) + "\n" for row in traces), encoding="utf-8")
    new_calls, hits = sum(row["new_model_calls"] for row in traces), sum(row["cache_hits"] for row in traces)
    if mode == "run" and new_calls <= 0:
        raise Phase35V3Error("fresh Phase 3-v3 run made zero model calls")
    if mode == "replay" and new_calls != 0:
        raise Phase35V3Error("Phase 3-v3 replay made unexpected new model calls")
    summary = {"format": PHASE35_V3_FORMAT, "status": "PASS", "mode": mode, "sample_count": len(results),
               "new_model_calls": new_calls, "cache_hits": hits,
               "logical_model_calls": sum(row["logical_model_calls"] for row in traces),
               "certificate_distribution": dict(Counter(row["certificate_status"] for row in traces)),
               "strict_answered": sum(row["strict"]["status"] == "ANSWERED" for row in results),
               "automatic_proposal_count": sum(row["automatic_support_region"] is not None for row in traces),
               "pixel_audit_pass_count": sum(row["pixel_audit_status"] == "PASS" for row in traces),
               "cache_dir": str(cache.root), "output_dir": str(store.root),
               "candidate_manifest_hash": candidate_hash, "prospective_manifest_sha256": prospective["manifest_sha256"],
               "operator_unchanged": config["spatial"]["intervention"] == preflight_report["spatial_intervention"],
               "certificate_builder_unchanged": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest() == preflight_report["certificate_builder_source_sha256"],
               "reason_specific_refinement_called": False, "latency_seconds": time.monotonic() - started}
    strict_audit = audit_run(run_dir)
    report = {"status": "PASS" if strict_audit["status"] == "PASS" else "FAIL", "mode": mode, "summary": summary,
              "traces": str(store.root / "phase35_v3_trace.jsonl"), "strict_audit": strict_audit,
              "runtime_gt_isolation_audit": runtime_gt_audit(runtime_path, samples),
              "automatic_roi_only": True, "human_roi_not_in_runtime_or_model_inputs": True,
              "reason_specific_refinement_called": False}
    (output_dir / f"phase35_v3_{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
