"""Formal, hash-bound semantic-spatial intervention for reviewed anchors.

This module deliberately has a narrow authority: human overlay review only
selects an already-grounded anchor for *eligibility*.  The text sent to the
semantic verifier is the frozen observation claim and never contains a human
label, review reason, or adjudication decision.  All semantic interventions,
pixel audits, controls, cache identities and certificate admission are the
existing ReliVE-v1 implementation in :mod:`relive.runner`.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from relive.backends import make_backend
from relive.claims import PROMPT_VERSIONS, stable_id
from relive.config import load_config
from relive.data.frames import select_frames
from relive.interventions import CONTROL_VERSION, INTERVENTION_VERSION, generate_control_regions
from relive.runner import SampleRunner
from relive.spatial import parse_proposal, validate_region
from relive.storage.artifacts import ArtifactStore, canonical_json, stable_hash
from relive.storage.cache import CachedInference
from relive.types import Claim, EvidenceCandidate, Frame, RuntimeSample, SemanticStatus, to_dict
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl


FORMAT = "reviewed-anchor-formal-intervention-v1"
WARNING_FORMAT = "reviewed-anchor-duplicate-grounding-adjudication-v1"
FORMAL_VARIANTS = ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL")
EXPECTED_RAW_ANCHORS = 75
EXPECTED_REVIEW_ROLES = 321
EXPECTED_ELIGIBLE = 5
EXPECTED_RAW_SHA_PREFIX = "f247e579"
EXPECTED_REVIEW_SHA_PREFIX = "9273c1af"
DECISIONS = frozenset({"UNIFY_LABELS", "CONTEXT_SPECIFIC_ALLOWED", "UNRESOLVED"})


class ReviewedAnchorInterventionError(ValueError):
    pass


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: str | Path, code: str) -> list[dict[str, Any]]:
    try:
        result = list(strict_jsonl(Path(path), error_code=code))
    except (OSError, TALSelectionError) as exc:
        raise ReviewedAnchorInterventionError(f"{code}_INVALID") from exc
    if not all(isinstance(row, dict) for row in result):
        raise ReviewedAnchorInterventionError(f"{code}_ROWS_MUST_BE_OBJECTS")
    return result


def _object(path: str | Path, code: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(Path(path).read_bytes().decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise ReviewedAnchorInterventionError(f"{code}_INVALID") from exc
    if not isinstance(value, dict):
        raise ReviewedAnchorInterventionError(f"{code}_MUST_BE_OBJECT")
    return value


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise ReviewedAnchorInterventionError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        raw = b"".join((canonical_json(item) + "\n").encode("utf-8") for item in value)
    else:
        raw = (canonical_json(value) + "\n").encode("utf-8")
    path.write_bytes(raw)


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise ReviewedAnchorInterventionError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha_map(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_path(path) for name, path in sorted(paths.items())}


def prepare_warning_adjudication_template(schema_issue_candidates: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Freeze a human-only template; it neither alters reviews nor routes claims."""
    source = Path(schema_issue_candidates)
    source_sha = sha256_path(source)
    warnings = [row for row in _rows(source, "WARNING_QUEUE")
                if row.get("warning_code") == "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW"]
    unique: dict[str, dict[str, Any]] = {}
    for row in warnings:
        sig = row.get("grounding_signature")
        if not isinstance(sig, str) or not sig:
            raise ReviewedAnchorInterventionError("WARNING_QUEUE_SIGNATURE_INVALID")
        unique.setdefault(sig, row)
    template = [{"format": WARNING_FORMAT, "grounding_signature": signature,
                 "warning_code": "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW",
                 "decision": "", "reviewer_id": "", "rationale": "",
                 "source_warning_queue_sha256": source_sha}
                for signature in sorted(unique)]
    _write(Path(output_path), template)
    return {"format": WARNING_FORMAT, "status": "AWAITING_HUMAN_ADJUDICATION",
            "warning_count": len(template), "source_warning_queue_sha256": source_sha,
            "template_sha256": sha256_path(output_path), "model_calls_made": 0,
            "certificate_created": False, "new_verified_count": 0, "gt_used": False}


def _adjudications(path: str | Path | None, warning_queue_sha: str, warnings: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {}, None
    path = Path(path)
    source_sha = sha256_path(path)
    result: dict[str, dict[str, Any]] = {}
    for row in _rows(path, "WARNING_ADJUDICATION"):
        required = {"format", "grounding_signature", "warning_code", "decision", "reviewer_id", "rationale", "source_warning_queue_sha256"}
        if set(row) != required or row.get("format") != WARNING_FORMAT or row.get("warning_code") != "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW":
            raise ReviewedAnchorInterventionError("WARNING_ADJUDICATION_SCHEMA_INVALID")
        sig, decision = row.get("grounding_signature"), row.get("decision")
        if sig not in warnings or sig in result or decision not in DECISIONS:
            raise ReviewedAnchorInterventionError("WARNING_ADJUDICATION_BINDING_INVALID")
        if row.get("source_warning_queue_sha256") != warning_queue_sha:
            raise ReviewedAnchorInterventionError("WARNING_ADJUDICATION_SOURCE_HASH_MISMATCH")
        if decision != "UNRESOLVED" and (not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip() or not isinstance(row.get("rationale"), str) or not row["rationale"].strip()):
            raise ReviewedAnchorInterventionError("WARNING_ADJUDICATION_REVIEWER_OR_RATIONALE_MISSING")
        result[sig] = row
    return result, source_sha


def _eligible_candidates(eligible_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    routed = [row for row in decision_rows if row.get("route") == "ELIGIBLE_SPATIAL_ANCHOR"]
    if len(routed) != EXPECTED_ELIGIBLE or len({row.get("observation_claim_id") for row in routed}) != EXPECTED_ELIGIBLE:
        raise ReviewedAnchorInterventionError("ELIGIBLE_ROUTE_COUNT_MUST_EQUAL_FIVE")
    expected = {row["selected_anchor_candidate_id"]: row for row in routed if isinstance(row.get("selected_anchor_candidate_id"), str)}
    if len(expected) != EXPECTED_ELIGIBLE or len(eligible_rows) != EXPECTED_ELIGIBLE:
        raise ReviewedAnchorInterventionError("ELIGIBLE_MANIFEST_COUNT_MUST_EQUAL_FIVE")
    candidates = []
    for row in eligible_rows:
        anchor = row.get("anchor_candidate_id")
        if anchor not in expected or row.get("claim_evidence_route") != "ELIGIBLE_SPATIAL_ANCHOR":
            raise ReviewedAnchorInterventionError("ELIGIBLE_MANIFEST_ROUTE_BINDING_INVALID")
        required = row.get("required_component_roles")
        components = {item.get("role"): item for item in row.get("adjudicated_components", []) if isinstance(item, dict)}
        if not isinstance(required, list) or not required or set(required) - set(components):
            raise ReviewedAnchorInterventionError("ELIGIBLE_REQUIRED_ROLE_BINDING_INVALID")
        # A frozen interface role is the narrowest registered support region for
        # these ACTION observations.  This is a deterministic selection policy,
        # not a model result and never places a review label in the VLM prompt.
        choices = [role for role in required if "INTERFACE" in role and components[role].get("bbox_usable_for_intervention")]
        choices += [role for role in required if components[role].get("bbox_usable_for_intervention") and role not in choices]
        if not choices:
            raise ReviewedAnchorInterventionError("ELIGIBLE_NO_AUDITABLE_TARGET_ROLE")
        target_role = choices[0]
        box = components[target_role].get("adjudicated_bbox")
        try:
            box = list(validate_region(box))
        except ValueError as exc:
            raise ReviewedAnchorInterventionError("ELIGIBLE_TARGET_BBOX_INVALID") from exc
        controls = generate_control_regions(box, 1)
        if not controls.get("available") or len(controls.get("regions", [])) != 1:
            raise ReviewedAnchorInterventionError("ELIGIBLE_TARGET_CONTROL_UNAVAILABLE")
        path = row.get("image_path")
        if not isinstance(path, str) or not Path(path).is_file() or sha256_path(path) != row.get("frame_sha256"):
            raise ReviewedAnchorInterventionError("ELIGIBLE_SOURCE_FRAME_BINDING_INVALID")
        with Image.open(path) as image:
            width, height = image.size
        candidates.append({"candidate_id": "reviewed_anchor_" + stable_hash({"anchor": anchor, "claim": row["observation_claim_id"]})[:24],
                           "anchor_candidate_id": anchor, "observation_claim_id": row["observation_claim_id"],
                           "observation_claim": row["observation_claim"], "target_role": target_role,
                           "required_roles": list(required), "support_region": box,
                           "matched_control_region": controls["regions"][0], "frame_path": path,
                           "frame_sha256": row["frame_sha256"], "image_size": [width, height],
                           "source_frame_reference": row.get("source_frame_reference"),
                           "timestamp_seconds": row.get("timestamp_seconds"),
                           "formal_variants": list(FORMAL_VARIANTS),
                           "target_selection_policy": "required_interface_role_then_required_role_with_registered_control_v1"})
    return sorted(candidates, key=lambda row: (row["observation_claim_id"], row["anchor_candidate_id"]))


def _warning_gate(warning_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], adjudications: dict[str, dict[str, Any]], queue_sha: str, adjudication_sha: str | None) -> dict[str, Any]:
    cohort = {(candidate["anchor_candidate_id"], role) for candidate in candidates for role in candidate["required_roles"]}
    decisions = []
    for row in warning_rows:
        sig = row.get("grounding_signature")
        reviews = row.get("reviews")
        if not isinstance(sig, str) or not isinstance(reviews, list):
            raise ReviewedAnchorInterventionError("WARNING_QUEUE_ROW_INVALID")
        affects = any((item.get("anchor_candidate_id"), item.get("role")) in cohort for item in reviews if isinstance(item, dict))
        adjudication = adjudications.get(sig)
        decision = adjudication.get("decision") if adjudication else "UNRESOLVED"
        decisions.append({"grounding_signature": sig, "warning_code": row.get("warning_code"),
                          "affects_formal_cohort": affects, "decision": decision,
                          "reviewer_id": adjudication.get("reviewer_id") if adjudication else None,
                          "rationale": adjudication.get("rationale") if adjudication else None,
                          "source_reviews": reviews})
    blocking = [item for item in decisions if item["affects_formal_cohort"] and item["decision"] == "UNRESOLVED"]
    return {"format": FORMAT, "status": "FORMAL_RUN_BLOCKED" if blocking else "PASS",
            "warning_queue_sha256": queue_sha, "warning_adjudication_sha256": adjudication_sha,
            "warning_count": len(decisions), "blocking_warning_count": len(blocking),
            "decisions": sorted(decisions, key=lambda item: item["grounding_signature"]),
            "model_calls_made": 0, "cache_opened": False, "certificate_created": False,
            "new_verified_count": 0, "gt_used": False}


def preflight(*, eligible_manifest: str | Path, observation_decisions: str | Path,
              raw_grounding: str | Path, human_review: str | Path, validation_report: str | Path,
              warning_queue: str | Path, warning_adjudication: str | Path | None,
              config_path: str | Path, output_dir: str | Path,
              raw_sha_prefix: str = EXPECTED_RAW_SHA_PREFIX,
              review_sha_prefix: str = EXPECTED_REVIEW_SHA_PREFIX,
              expected_anchor_count: int = EXPECTED_RAW_ANCHORS,
              expected_review_count: int = EXPECTED_REVIEW_ROLES,
              expected_eligible_count: int = EXPECTED_ELIGIBLE) -> dict[str, Any]:
    if expected_eligible_count != EXPECTED_ELIGIBLE:
        raise ReviewedAnchorInterventionError("FORMAL_COHORT_MUST_EQUAL_FIVE")
    paths = {"eligible_manifest": Path(eligible_manifest), "observation_decisions": Path(observation_decisions),
             "raw_grounding": Path(raw_grounding), "human_review": Path(human_review),
             "validation_report": Path(validation_report), "warning_queue": Path(warning_queue),
             "config": Path(config_path)}
    hashes = _sha_map(paths)
    raw_rows, review_rows = _rows(paths["raw_grounding"], "RAW_GROUNDING"), _rows(paths["human_review"], "HUMAN_REVIEW")
    if len(raw_rows) != expected_anchor_count or len({row.get("anchor_candidate_id") for row in raw_rows}) != expected_anchor_count:
        raise ReviewedAnchorInterventionError("RAW_GROUNDING_COUNT_MISMATCH")
    component_count = sum(len(row.get("components", [])) for row in raw_rows)
    if component_count != expected_review_count or len(review_rows) != expected_review_count:
        raise ReviewedAnchorInterventionError("HUMAN_REVIEW_COUNT_MISMATCH")
    if not hashes["raw_grounding"].startswith(raw_sha_prefix) or not hashes["human_review"].startswith(review_sha_prefix):
        raise ReviewedAnchorInterventionError("REVIEWED_ANCHOR_INPUT_HASH_MISMATCH")
    report = _object(paths["validation_report"], "VALIDATION_REPORT")
    if report.get("status") != "PASS" or report.get("raw_anchor_manifest_sha256") != hashes["raw_grounding"] or report.get("review_jsonl_sha256") != hashes["human_review"]:
        raise ReviewedAnchorInterventionError("HUMAN_REVIEW_VALIDATION_REPORT_BINDING_INVALID")
    if report.get("raw_anchor_count") != expected_anchor_count or report.get("review_component_count") != expected_review_count:
        raise ReviewedAnchorInterventionError("HUMAN_REVIEW_VALIDATION_COUNT_INVALID")
    routes = report.get("observation_route_counts")
    if routes != {"ELIGIBLE_SPATIAL_ANCHOR": 5, "RELATIONAL_COMPOSITE_REQUIRED": 5, "TEMPORAL_REACQUIRE": 5}:
        raise ReviewedAnchorInterventionError("HUMAN_REVIEW_ROUTE_COUNTS_INVALID")
    candidates = _eligible_candidates(_rows(paths["eligible_manifest"], "ELIGIBLE_MANIFEST"), _rows(paths["observation_decisions"], "OBSERVATION_DECISIONS"))
    if len(candidates) != expected_eligible_count:
        raise ReviewedAnchorInterventionError("ELIGIBLE_COHORT_COUNT_MISMATCH")
    raw_by_anchor = {row.get("anchor_candidate_id"): row for row in raw_rows}
    for candidate in candidates:
        raw = raw_by_anchor.get(candidate["anchor_candidate_id"])
        if (not isinstance(raw, dict) or raw.get("frame_sha256") != candidate["frame_sha256"]
                or raw.get("observation_claim_id") != candidate["observation_claim_id"]
                or raw.get("image_path") != candidate["frame_path"]):
            raise ReviewedAnchorInterventionError("ELIGIBLE_RAW_GROUNDING_BINDING_INVALID")
    warning_rows = [row for row in _rows(paths["warning_queue"], "WARNING_QUEUE") if row.get("warning_code") == "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW"]
    if len(warning_rows) != 5:
        raise ReviewedAnchorInterventionError("DUPLICATE_WARNING_COUNT_MISMATCH")
    adjudications, adjudication_sha = _adjudications(warning_adjudication, hashes["warning_queue"], {row["grounding_signature"]: row for row in warning_rows})
    gate = _warning_gate(warning_rows, candidates, adjudications, hashes["warning_queue"], adjudication_sha)
    config = load_config(paths["config"])
    if config["policy"]["name"] != "semantic_spatial" or config["policy"]["version"] != "relive-v1-policy-1" or config["spatial"]["intervention"]["operator"] != "opaque_gray" or config["adaptation"]["enabled"]:
        raise ReviewedAnchorInterventionError("FORMAL_SEMANTIC_SPATIAL_CONFIG_INVALID")
    if config["spatial"]["control_count"] != 1 or config["budget"]["max_calls"] < 4 or config["budget"]["max_spatial_proposals"] < 1:
        raise ReviewedAnchorInterventionError("FORMAL_SEMANTIC_SPATIAL_BUDGET_INVALID")
    out = Path(output_dir)
    gate_path, plan_path = out / "warning_adjudication_gate.json", out / "reviewed_anchor_intervention_plan.json"
    _write(gate_path, gate)
    plan = {"format": FORMAT, "status": gate["status"], "formal_cohort_count": len(candidates),
            "formal_variants": list(FORMAL_VARIANTS), "diagnostic_variants_enabled": False,
            "candidates": candidates, "bindings": {**hashes, "warning_adjudication": adjudication_sha,
            "warning_adjudication_gate": sha256_path(gate_path), "semantic_prompt_version": PROMPT_VERSIONS["semantic"],
            "intervention_operator_version": config["spatial"]["intervention"]["operator_version"],
            "intervention_version": INTERVENTION_VERSION, "control_version": CONTROL_VERSION,
            "certificate_policy": config["policy"], "backend_config": config["backend"],
            "backend_config_sha256": stable_hash(config["backend"]), "human_labels_not_in_verifier_prompt": True},
            "model_calls_made": 0, "cache_opened": False, "certificate_created": False,
            "new_verified_count": 0, "gt_used": False}
    plan["plan_content_sha256"] = stable_hash(plan)
    _write(plan_path, plan)
    return {"status": plan["status"], "format": FORMAT, "plan": str(plan_path), "warning_gate": str(gate_path),
            "formal_cohort_count": len(candidates), "model_calls_made": 0, "cache_opened": False,
            "certificate_created": False, "new_verified_count": 0, "gt_used": False}


def _load_plan(root: Path, config_path: str | Path) -> dict[str, Any]:
    plan = _object(root / "reviewed_anchor_intervention_plan.json", "FORMAL_PLAN")
    expected = plan.get("plan_content_sha256")
    check = dict(plan); check.pop("plan_content_sha256", None)
    if not isinstance(expected, str) or stable_hash(check) != expected:
        raise ReviewedAnchorInterventionError("FORMAL_PLAN_TAMPERED")
    if plan.get("bindings", {}).get("config") != sha256_path(config_path):
        raise ReviewedAnchorInterventionError("FORMAL_PLAN_CONFIG_BINDING_INVALID")
    if plan.get("status") != "PASS":
        raise ReviewedAnchorInterventionError("FORMAL_RUN_BLOCKED")
    if plan.get("formal_cohort_count") != EXPECTED_ELIGIBLE or len(plan.get("candidates", [])) != EXPECTED_ELIGIBLE:
        raise ReviewedAnchorInterventionError("FORMAL_PLAN_COHORT_INVALID")
    if plan.get("formal_variants") != list(FORMAL_VARIANTS) or plan.get("diagnostic_variants_enabled") is not False:
        raise ReviewedAnchorInterventionError("FORMAL_PLAN_VARIANT_INVALID")
    return plan


def _candidate_runtime(row: dict[str, Any], plan_sha: str) -> tuple[RuntimeSample, EvidenceCandidate, Claim]:
    frame_path = Path(row["frame_path"])
    if not frame_path.is_file() or sha256_path(frame_path) != row["frame_sha256"]:
        raise ReviewedAnchorInterventionError("SOURCE_FRAME_BINDING_INVALID")
    frame_id = "reviewed-frame-" + stable_hash({"anchor": row["anchor_candidate_id"], "sha": row["frame_sha256"]})[:24]
    stamp = row.get("timestamp_seconds")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or stamp < 0:
        stamp = None
    frame = Frame(frame_id, str(frame_path), 0, float(stamp) if stamp is not None else None,
                  "public_frame_manifest" if stamp is not None else None, row.get("source_frame_reference"))
    claim = Claim(row["observation_claim_id"], row["observation_claim"],
                  time_scope={"frame_ids": [frame_id]}, source="frozen_observation_claim", required_for_question=True)
    sample_id = "reviewed-anchor:" + row["anchor_candidate_id"]
    sample = RuntimeSample(sample_id, "claim_verification", claim.text, (frame,), claim, (),
                           {"dataset_name": "public_reviewed_anchor", "nonofficial_protocol": True},
                           {"source_kind": "public_runtime", "runtime_sha256": plan_sha})
    candidate = EvidenceCandidate(row["candidate_id"], sample_id, (frame_id,), (frame.timestamp,), 0,
                                  "reviewed_anchor_formal_intervention_v1", provenance={"frame_sha256": row["frame_sha256"]})
    return sample, candidate, claim


def _run_candidate(row: dict[str, Any], plan_sha: str, config: dict[str, Any], inference: CachedInference, *, synthetic: bool) -> dict[str, Any]:
    sample, candidate, claim = _candidate_runtime(row, plan_sha)
    generated_control = generate_control_regions(row["support_region"], 1)
    if (not generated_control.get("available") or generated_control.get("regions") != [row["matched_control_region"]]):
        raise ReviewedAnchorInterventionError("FROZEN_MATCHED_CONTROL_BINDING_INVALID")
    runner = SampleRunner(config, inference.store, inference)
    original = runner.infer(sample, candidate, claim, "semantic")
    spatial = None
    if original.semantic_status == SemanticStatus.SUPPORTED:
        proposal = parse_proposal(canonical_json({"support_region": row["support_region"]}), candidate, claim)
        proposal = replace(proposal, provenance={**proposal.provenance, "source": "frozen_reviewed_anchor_region",
                                                  "anchor_candidate_id": row["anchor_candidate_id"], "target_role": row["target_role"]})
        spatial = runner.spatial_with_proposal(sample, candidate, claim, original, proposal)
    certificate = runner.certificate(sample, candidate, claim, original, spatial, None)
    # Synthetic tests exercise orchestration only: they must never be presented
    # as a real certificate, even if a fixture emits supporting verdicts.
    certificate_record = {"certificate_status": "NOT_APPLICABLE_SYNTHETIC_TEST"} if synthetic else to_dict(certificate)
    trace = {"candidate_id": row["candidate_id"], "anchor_candidate_id": row["anchor_candidate_id"],
             "observation_claim_id": row["observation_claim_id"], "target_role": row["target_role"],
             "formal_variants": list(FORMAL_VARIANTS), "original": to_dict(original), "spatial": to_dict(spatial),
             "certificate": certificate_record, "usage": runner.budget.snapshot(),
             "source_frame_sha256": row["frame_sha256"], "support_region": row["support_region"],
             "matched_control_region": row["matched_control_region"], "human_labels_in_model_request": False,
             "gt_used": False, "new_verified_count": 0 if synthetic else int(certificate.final_status.value == "VERIFIED")}
    return trace


def _trace_summary(rows: list[dict[str, Any]], *, mode: str, synthetic: bool) -> dict[str, Any]:
    statuses = Counter()
    for row in rows:
        cert = row["certificate"]
        if isinstance(cert, dict): statuses[cert.get("final_status", cert.get("certificate_status", "UNCERTAIN"))] += 1
    all_drop_supported = bool(rows) and all((row.get("spatial") or {}).get("references", {}).get("drop", {}).get("semantic_status") == "SUPPORTED" for row in rows)
    return {"format": FORMAT, "status": "PASS", "mode": mode, "candidate_count": len(rows),
            "formal_variants": list(FORMAL_VARIANTS), "certificate_distribution": dict(sorted(statuses.items())),
            "new_model_calls": sum(row["usage"].get("new_calls", 0) for row in rows),
            "cache_hits": sum(row["usage"].get("cache_hits", 0) for row in rows),
            "all_drop_target_supported": all_drop_supported,
            "cohort_stop_recommendation": "PAUSE_FORMAL_EXPANSION_DIAGNOSTIC_REQUIRED" if all_drop_supported else None,
            "diagnostic_only_variants_excluded": True, "synthetic_test": synthetic,
            "certificate_created": not synthetic, "new_verified_count": 0 if synthetic else sum(row["new_verified_count"] for row in rows),
            "gt_used": False}


def execute(*, output_dir: str | Path, config_path: str | Path, mode: str,
            smoke_candidate_id: str | None = None, backend_factory=make_backend) -> dict[str, Any]:
    if mode not in {"run", "replay"}:
        raise ReviewedAnchorInterventionError("FORMAL_MODE_INVALID")
    root = Path(output_dir)
    plan = _load_plan(root, config_path)
    selected = list(plan["candidates"])
    if smoke_candidate_id is not None:
        selected = [row for row in selected if row["candidate_id"] == smoke_candidate_id or row["anchor_candidate_id"] == smoke_candidate_id]
        if len(selected) != 1:
            raise ReviewedAnchorInterventionError("FORMAL_SMOKE_CANDIDATE_NOT_IN_FROZEN_COHORT")
    if mode == "replay" and not (root / "run" / "reviewed_anchor_intervention_trace.jsonl").is_file():
        raise ReviewedAnchorInterventionError("REPLAY_REQUIRES_FORMAL_RUN")
    run_dir = root / mode
    if run_dir.exists():
        raise ReviewedAnchorInterventionError("IMMUTABLE_RUN_OUTPUT_EXISTS")
    config = load_config(config_path)
    backend = backend_factory(config["backend"])
    store = ArtifactStore(root / "cache")
    inference = CachedInference(backend, store)
    traces = [_run_candidate(row, plan["plan_content_sha256"], config, inference, synthetic=backend.synthetic) for row in selected]
    if mode == "replay" and sum(row["usage"]["new_calls"] for row in traces) != 0:
        raise ReviewedAnchorInterventionError("REPLAY_NEW_MODEL_CALLS_NONZERO")
    summary = _trace_summary(traces, mode=mode, synthetic=backend.synthetic)
    _write(run_dir / "reviewed_anchor_intervention_trace.jsonl", traces)
    _write(run_dir / "reviewed_anchor_certificate_manifest.jsonl", [{"candidate_id": row["candidate_id"], "certificate": row["certificate"], "new_verified_count": row["new_verified_count"]} for row in traces])
    _write(run_dir / "reviewed_anchor_summary.json", summary)
    report = "# Reviewed-anchor formal intervention\n\n" + "This report records protocol results; it does not assert clinical truth.\n\n" + "```json\n" + canonical_json(summary) + "\n```\n"
    _write_text(run_dir / "reviewed_anchor_report.md", report)
    replay_audit = {"format": FORMAT, "status": "PASS" if mode == "run" or summary["new_model_calls"] == 0 else "FAIL",
                    "mode": mode, "new_model_calls": summary["new_model_calls"], "cache_hits": summary["cache_hits"],
                    "trace_sha256": sha256_path(run_dir / "reviewed_anchor_intervention_trace.jsonl"), "gt_used": False,
                    "certificate_created": summary["certificate_created"], "new_verified_count": summary["new_verified_count"]}
    _write(run_dir / "cache_replay_audit.json", replay_audit)
    return {"status": "PASS", "mode": mode, "summary": summary, "output_dir": str(run_dir)}


def summarize(*, output_dir: str | Path, mode: str = "run") -> dict[str, Any]:
    root = Path(output_dir) / mode
    traces = _rows(root / "reviewed_anchor_intervention_trace.jsonl", "FORMAL_TRACE")
    summary = _trace_summary(traces, mode=mode, synthetic=all(row.get("certificate", {}).get("certificate_status") == "NOT_APPLICABLE_SYNTHETIC_TEST" for row in traces))
    return {"status": "PASS", "summary": summary, "trace_sha256": sha256_path(root / "reviewed_anchor_intervention_trace.jsonl")}
