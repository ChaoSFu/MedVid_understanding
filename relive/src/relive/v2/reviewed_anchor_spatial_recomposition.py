"""One-shot, failure-routed composite-mask R1 for reviewed anchors.

R1 is deliberately downstream of an immutable, complete R0 cohort.  It never
rewrites R0, raw grounding, overlay review or routes.  It consumes only frozen
R0 traces and frozen adjudicated component boxes, then uses the unchanged core
semantic-spatial verifier/certificate policy with a binary union mask.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from relive.backends import make_backend
from relive.config import load_config
from relive.runner import SampleRunner
from relive.storage.artifacts import ArtifactStore, canonical_json, stable_hash
from relive.storage.cache import CachedInference
from relive.types import FinalStatus, SemanticStatus, to_dict
from .reviewed_anchor_formal_intervention import (
    EXPECTED_ELIGIBLE, FORMAL_VARIANTS, FORMAT as R0_FORMAT,
    ReviewedAnchorInterventionError, _candidate_runtime, _object, _rows, _write,
    _write_text, sha256_path, validate_label_resolution_manifest,
)


FORMAT = "reviewed-anchor-spatial-recomposition-r1-v1"
DIAGNOSIS = "COMPOSITE_EVIDENCE_REQUIRED"
ACTION = "SPATIAL_RECOMPOSE"
ACTION_REQUIRED_ROLES = ("OPERATOR_HAND", "BASE_PLATE", "HAND_BASE_INTERFACE")


class ReviewedAnchorRecompositionError(ValueError):
    pass


def _as_error(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ReviewedAnchorInterventionError as exc:
        raise ReviewedAnchorRecompositionError(str(exc)) from exc


def _semantic(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict) or result.get("execution_status") != "OK":
        return None
    return result.get("semantic_status")


def _pixel_audits_ok(spatial: dict[str, Any] | None) -> bool:
    if not isinstance(spatial, dict):
        return False
    refs = spatial.get("pixel_audit_refs")
    if not isinstance(refs, list) or len(refs) != 4:
        return False
    try:
        return all(_object(path, "R0_PIXEL_AUDIT").get("pixel_audit_pass") is True for path in refs)
    except ReviewedAnchorInterventionError:
        return False


def _r0_state(row: dict[str, Any]) -> dict[str, Any]:
    spatial = row.get("spatial") if isinstance(row.get("spatial"), dict) else {}
    refs = spatial.get("references") if isinstance(spatial.get("references"), dict) else {}
    controls = refs.get("controls") if isinstance(refs.get("controls"), list) else []
    values = {"ORIGINAL": _semantic(row.get("original")), "KEEP_TARGET": _semantic(refs.get("keep")),
              "DROP_TARGET": _semantic(refs.get("drop")),
              "DROP_MATCHED_CONTROL": _semantic(controls[0]) if len(controls) == 1 else None}
    references = [refs.get("original"), refs.get("keep"), refs.get("drop"), *controls]
    reference_binding = all(isinstance(item, dict)
                            and isinstance(item.get("input_references"), dict)
                            and item["input_references"].get("candidate_id") == row.get("candidate_id")
                            and item["input_references"].get("claim_id") == row.get("observation_claim_id")
                            for item in references)
    eligible = (values.get("ORIGINAL") == "SUPPORTED"
                and values.get("KEEP_TARGET") in {"INSUFFICIENT", "CONTRADICTED"}
                and values.get("DROP_TARGET") == "SUPPORTED"
                and values.get("DROP_MATCHED_CONTROL") == "SUPPORTED"
                and spatial.get("expected_control_count") == 1 and spatial.get("observed_control_count") == 1
                and _pixel_audits_ok(spatial) and reference_binding)
    return {"semantic_results": values, "pixel_audits_pass": _pixel_audits_ok(spatial),
            "control_count_pass": spatial.get("expected_control_count") == 1 and spatial.get("observed_control_count") == 1,
            "reference_binding_pass": reference_binding,
            "r1_eligible": eligible,
            "diagnosis": DIAGNOSIS if eligible else None, "action": ACTION if eligible else None}


def _validate_plan(plan: dict[str, Any]) -> None:
    expected = plan.get("plan_content_sha256")
    payload = dict(plan); payload.pop("plan_content_sha256", None)
    if not isinstance(expected, str) or stable_hash(payload) != expected:
        raise ReviewedAnchorRecompositionError("R0_PLAN_TAMPERED")
    if plan.get("format") != R0_FORMAT or plan.get("formal_cohort_count") != EXPECTED_ELIGIBLE:
        raise ReviewedAnchorRecompositionError("R0_PLAN_COHORT_INVALID")
    if plan.get("formal_variants") != list(FORMAL_VARIANTS):
        raise ReviewedAnchorRecompositionError("R0_PLAN_VARIANTS_INVALID")


def audit_r0_cohort(*, r0_output_dir: str | Path, output_dir: str | Path,
                    warning_queue: str | Path | None = None,
                    warning_adjudication: str | Path | None = None,
                    derived_review: str | Path | None = None,
                    recomputed_observation_decisions: str | Path | None = None,
                    recomputed_eligible_manifest: str | Path | None = None,
                    label_resolution_manifest: str | Path | None = None) -> dict[str, Any]:
    """Read R0 without opening model/cache; report completeness and label application."""
    root, out = Path(r0_output_dir), Path(output_dir)
    plan = _as_error(_object, root / "reviewed_anchor_intervention_plan.json", "R0_PLAN")
    _validate_plan(plan)
    run = _as_error(_rows, root / "run" / "reviewed_anchor_intervention_trace.jsonl", "R0_RUN_TRACE")
    replay_path = root / "replay" / "reviewed_anchor_intervention_trace.jsonl"
    replay = _as_error(_rows, replay_path, "R0_REPLAY_TRACE") if replay_path.is_file() else []
    expected = {item["candidate_id"] for item in plan["candidates"]}
    run_by_id = {item.get("candidate_id"): item for item in run}
    replay_by_id = {item.get("candidate_id"): item for item in replay}
    complete = set(run_by_id) == expected and set(replay_by_id) == expected
    rows = []
    for candidate in plan["candidates"]:
        trace = run_by_id.get(candidate["candidate_id"])
        state = _r0_state(trace) if isinstance(trace, dict) else {"r1_eligible": False}
        certificate = trace.get("certificate", {}) if isinstance(trace, dict) else {}
        rows.append({"candidate_id": candidate["candidate_id"], "anchor_candidate_id": candidate["anchor_candidate_id"],
                     "observation_claim_id": candidate["observation_claim_id"], "target_role": candidate["target_role"],
                     "roi_area_fraction": (candidate["support_region"][2] - candidate["support_region"][0]) * (candidate["support_region"][3] - candidate["support_region"][1]),
                     **state, "failure_reasons": (trace.get("failure_reasons", []) if isinstance(trace, dict) else ["R0_NOT_RUN"]),
                     "certificate_status": certificate.get("final_status", certificate.get("certificate_status")) if isinstance(certificate, dict) else None,
                     "fresh_model_calls": (trace.get("usage", {}).get("new_calls") if isinstance(trace, dict) else None),
                     "replay_model_calls": (replay_by_id.get(candidate["candidate_id"], {}).get("usage", {}).get("new_calls") if candidate["candidate_id"] in replay_by_id else None)})
    bound = plan.get("bindings", {})
    # R0 plans bind hashes rather than paths.  An audit therefore requires the
    # actual source paths and verifies them before examining human decisions.
    unify = []
    sources_bound = warning_queue is not None and warning_adjudication is not None
    warning_impacts: list[dict[str, Any]] = []
    resolution = None
    resolution_sha = None
    if sources_bound:
        if sha256_path(warning_queue) != bound.get("warning_queue") or sha256_path(warning_adjudication) != bound.get("warning_adjudication"):
            raise ReviewedAnchorRecompositionError("R0_WARNING_ADJUDICATION_BINDING_INVALID")
        formal_roles = {(candidate["anchor_candidate_id"], role): candidate["candidate_id"]
                        for candidate in plan["candidates"] for role in candidate.get("required_roles", [])}
        for warning in _as_error(_rows, warning_queue, "R0_WARNING_QUEUE"):
            if warning.get("warning_code") != "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW":
                continue
            impacted = sorted({formal_roles[(review.get("anchor_candidate_id"), review.get("role"))]
                               for review in warning.get("reviews", []) if isinstance(review, dict)
                               and (review.get("anchor_candidate_id"), review.get("role")) in formal_roles})
            warning_impacts.append({"grounding_signature": warning.get("grounding_signature"),
                                    "affects_formal_required_role": bool(impacted), "affected_candidate_ids": impacted})
        if label_resolution_manifest is not None:
            resolution_sha = sha256_path(label_resolution_manifest)
            if resolution_sha != bound.get("label_resolution_manifest"):
                raise ReviewedAnchorRecompositionError("R0_LABEL_RESOLUTION_BINDING_INVALID")
            resolution = _as_error(validate_label_resolution_manifest, label_resolution_manifest,
                raw_grounding_sha256=bound.get("raw_grounding"), derived_human_review_sha256=bound.get("human_review"),
                validation_report_sha256=bound.get("validation_report"), observation_decisions_sha256=bound.get("observation_decisions"),
                eligible_manifest_sha256=bound.get("eligible_manifest"), warning_queue_sha256=bound.get("warning_queue"),
                warning_adjudication_sha256=bound.get("warning_adjudication"))
        resolved = {item["grounding_signature"]: item["resolved_label"] for item in resolution.get("canonical_labels", [])} if isinstance(resolution, dict) else {}
        for item in _as_error(_rows, warning_adjudication, "R0_WARNING_ADJUDICATION"):
            if item.get("decision") == "UNIFY_LABELS":
                impact = next((record for record in warning_impacts if record["grounding_signature"] == item.get("grounding_signature")), {})
                unify.append({"grounding_signature": item.get("grounding_signature"),
                              "resolved_label_present": item.get("grounding_signature") in resolved,
                              "derived_review_created": resolution is not None or (derived_review is not None and Path(derived_review).is_file()),
                              "routes_recomputed": resolution is not None or (recomputed_observation_decisions is not None and Path(recomputed_observation_decisions).is_file()),
                              "eligible_manifest_recomputed": resolution is not None or (recomputed_eligible_manifest is not None and Path(recomputed_eligible_manifest).is_file()),
                              "affects_formal_required_role": impact.get("affects_formal_required_role", False),
                              "affected_candidate_ids": impact.get("affected_candidate_ids", [])})
    input_adjudication_applied = sources_bound and (not unify or all(
        item["resolved_label_present"] and item["derived_review_created"]
        and item["routes_recomputed"] and item["eligible_manifest_recomputed"]
        for item in unify))
    status = "PASS" if complete else "R0_COHORT_INCOMPLETE"
    report = {"format": FORMAT, "status": status, "r0_output_dir": str(root), "r0_plan_sha256": sha256_path(root / "reviewed_anchor_intervention_plan.json"),
              "r0_run_trace_sha256": sha256_path(root / "run" / "reviewed_anchor_intervention_trace.jsonl"),
              "r0_replay_trace_sha256": sha256_path(replay_path) if replay_path.is_file() else None,
              "formal_cohort_count": EXPECTED_ELIGIBLE, "run_candidate_count": len(run), "replay_candidate_count": len(replay),
              "r0_complete_with_replay": complete, "candidates": rows,
              "duplicate_grounding_signature_impact": sorted(warning_impacts, key=lambda item: str(item["grounding_signature"])),
              "unify_labels_audit": {"warning_sources_bound": sources_bound, "unify_records": unify, "input_adjudication_applied": input_adjudication_applied,
                  "label_resolution_manifest_sha256": resolution_sha,
                  "status": "APPLIED" if input_adjudication_applied else "DIAGNOSTIC_ONLY_INPUT_ADJUDICATION_NOT_APPLIED"},
              "model_calls_made": 0, "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    report["report_content_sha256"] = stable_hash(report)
    _as_error(_write, out / "reviewed_anchor_r0_cohort_audit.json", report)
    _as_error(_write, out / "reviewed_anchor_r0_cohort_rows.jsonl", rows)
    return report


def _r1_candidates(plan: dict[str, Any], r0_rows: list[dict[str, Any]], eligible_manifest: Path) -> list[dict[str, Any]]:
    if sha256_path(eligible_manifest) != plan.get("bindings", {}).get("eligible_manifest"):
        raise ReviewedAnchorRecompositionError("R1_ELIGIBLE_MANIFEST_BINDING_INVALID")
    eligible_by_anchor = {row.get("anchor_candidate_id"): row for row in _as_error(_rows, eligible_manifest, "R1_ELIGIBLE_MANIFEST")}
    original_by_id = {row["candidate_id"]: row for row in plan["candidates"]}
    candidates = []
    for audit in r0_rows:
        if not audit.get("r1_eligible"):
            continue
        frozen = original_by_id.get(audit["candidate_id"]); entry = eligible_by_anchor.get(audit["anchor_candidate_id"])
        if not frozen or not entry:
            raise ReviewedAnchorRecompositionError("R1_R0_ELIGIBLE_BINDING_INVALID")
        if entry.get("observation_role") != "ACTION_CORE_PRESSING":
            raise ReviewedAnchorRecompositionError("R1_OBSERVATION_ROLE_NOT_PREREGISTERED")
        components = {item.get("role"): item for item in entry.get("adjudicated_components", []) if isinstance(item, dict)}
        if any(role not in components or not components[role].get("bbox_usable_for_intervention") for role in ACTION_REQUIRED_ROLES):
            raise ReviewedAnchorRecompositionError("COMPOSITE_INTERVENTION_OPERATOR_UNAVAILABLE")
        regions = [components[role]["adjudicated_bbox"] for role in ACTION_REQUIRED_ROLES]
        candidates.append({**frozen, "r1_candidate_id": "reviewed_anchor_r1_" + stable_hash({"r0": frozen["candidate_id"], "roles": ACTION_REQUIRED_ROLES})[:24],
                           "r0_candidate_id": frozen["candidate_id"], "diagnosis": DIAGNOSIS, "action": ACTION,
                           "composition_geometry": "COMPOSITE_BINARY_MASK_UNION", "component_roles": list(ACTION_REQUIRED_ROLES),
                           "component_regions": regions, "r0_audit_row": audit})
    return candidates


def preflight(*, r0_output_dir: str | Path, eligible_manifest: str | Path, config_path: str | Path,
              output_dir: str | Path, warning_queue: str | Path, warning_adjudication: str | Path,
              derived_review: str | Path | None = None, recomputed_observation_decisions: str | Path | None = None,
              recomputed_eligible_manifest: str | Path | None = None,
              label_resolution_manifest: str | Path | None = None) -> dict[str, Any]:
    r0, out = Path(r0_output_dir), Path(output_dir)
    audit_dir = out / "r0_audit"
    audit = audit_r0_cohort(r0_output_dir=r0, output_dir=audit_dir, warning_queue=warning_queue,
                            warning_adjudication=warning_adjudication, derived_review=derived_review,
                            recomputed_observation_decisions=recomputed_observation_decisions,
                            recomputed_eligible_manifest=recomputed_eligible_manifest,
                            label_resolution_manifest=label_resolution_manifest)
    if audit["status"] != "PASS":
        raise ReviewedAnchorRecompositionError("R0_COHORT_INCOMPLETE")
    if audit["unify_labels_audit"]["status"] != "APPLIED":
        raise ReviewedAnchorRecompositionError("DIAGNOSTIC_ONLY_INPUT_ADJUDICATION_NOT_APPLIED")
    plan = _as_error(_object, r0 / "reviewed_anchor_intervention_plan.json", "R0_PLAN")
    config = load_config(config_path)
    if config["policy"]["name"] != "semantic_spatial" or config["policy"]["version"] != "relive-v1-policy-1" or config["adaptation"]["enabled"]:
        raise ReviewedAnchorRecompositionError("R1_CONFIG_POLICY_OR_ADAPTATION_INVALID")
    if config["spatial"]["intervention"]["operator"] != "opaque_gray" or config["spatial"]["control_count"] != 1 or config["budget"]["max_calls"] < 4 or config["budget"]["max_spatial_proposals"] < 1:
        raise ReviewedAnchorRecompositionError("R1_CONFIG_INTERVENTION_OR_BUDGET_INVALID")
    candidates = _r1_candidates(plan, audit["candidates"], Path(eligible_manifest))
    r1 = {"format": FORMAT, "status": "PASS", "r0_to_diagnosis_to_action": {"r0_format": R0_FORMAT, "diagnosis": DIAGNOSIS, "action": ACTION, "max_rounds": 1, "cycle_guard": "NO_R2_AFTER_R1"},
          "formal_variants": list(FORMAL_VARIANTS), "diagnostic_variants_enabled": False, "candidate_count": len(candidates), "candidates": candidates,
          "bindings": {"r0_plan_sha256": sha256_path(r0 / "reviewed_anchor_intervention_plan.json"), "r0_run_trace_sha256": sha256_path(r0 / "run" / "reviewed_anchor_intervention_trace.jsonl"),
                       "r0_replay_trace_sha256": sha256_path(r0 / "replay" / "reviewed_anchor_intervention_trace.jsonl"),
                       "r0_cohort_audit_sha256": sha256_path(audit_dir / "reviewed_anchor_r0_cohort_audit.json"),
                       "eligible_manifest_sha256": sha256_path(eligible_manifest), "config_sha256": sha256_path(config_path),
                       "label_resolution_manifest_sha256": sha256_path(label_resolution_manifest) if label_resolution_manifest is not None else None,
                       "operator": config["spatial"]["intervention"], "policy": config["policy"], "backend": config["backend"],
                       "human_labels_not_in_verifier_prompt": True, "gt_used": False},
          "model_calls_made": 0, "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    r1["plan_content_sha256"] = stable_hash(r1)
    _as_error(_write, out / "reviewed_anchor_r1_recomposition_plan.json", r1)
    return {"status": "PASS", "format": FORMAT, "plan": str(out / "reviewed_anchor_r1_recomposition_plan.json"),
            "candidate_count": len(candidates), "model_calls_made": 0, "cache_opened": False, "certificate_created": False,
            "new_verified_count": 0, "gt_used": False}


def _load_plan(root: Path, config_path: str | Path) -> dict[str, Any]:
    plan = _as_error(_object, root / "reviewed_anchor_r1_recomposition_plan.json", "R1_PLAN")
    checksum = plan.get("plan_content_sha256"); payload = dict(plan); payload.pop("plan_content_sha256", None)
    if not isinstance(checksum, str) or stable_hash(payload) != checksum:
        raise ReviewedAnchorRecompositionError("R1_PLAN_TAMPERED")
    if plan.get("bindings", {}).get("config_sha256") != sha256_path(config_path):
        raise ReviewedAnchorRecompositionError("R1_PLAN_CONFIG_BINDING_INVALID")
    if plan.get("r0_to_diagnosis_to_action", {}).get("cycle_guard") != "NO_R2_AFTER_R1":
        raise ReviewedAnchorRecompositionError("R1_CYCLE_GUARD_INVALID")
    return plan


def _run_candidate(row: dict[str, Any], plan_sha: str, config: dict[str, Any], inference: CachedInference, *, synthetic: bool) -> dict[str, Any]:
    source = dict(row); source["candidate_id"] = row["r1_candidate_id"]
    sample, candidate, claim = _as_error(_candidate_runtime, source, plan_sha)
    runner = SampleRunner(config, inference.store, inference)
    original = runner.infer(sample, candidate, claim, "semantic")
    spatial = None
    if original.semantic_status == SemanticStatus.SUPPORTED:
        spatial = runner.spatial_with_composite_regions(sample, candidate, claim, original, row["component_regions"],
            component_roles=row["component_roles"], composition_provenance={"source": "frozen_r0_required_roles", "r0_candidate_id": row["r0_candidate_id"], "diagnosis": DIAGNOSIS, "action": ACTION})
    certificate = runner.certificate(sample, candidate, claim, original, spatial, None)
    cert = {"certificate_status": "NOT_APPLICABLE_SYNTHETIC_TEST"} if synthetic else to_dict(certificate)
    return {"r1_candidate_id": row["r1_candidate_id"], "r0_candidate_id": row["r0_candidate_id"],
            "anchor_candidate_id": row["anchor_candidate_id"], "observation_claim_id": row["observation_claim_id"],
            "diagnosis": DIAGNOSIS, "action": ACTION, "component_roles": row["component_roles"], "component_regions": row["component_regions"],
            "formal_variants": list(FORMAL_VARIANTS), "original": to_dict(original), "spatial": to_dict(spatial),
            "certificate": cert, "usage": runner.budget.snapshot(), "human_labels_in_model_request": False,
            "gt_used": False, "new_verified_count": 0 if synthetic else int(certificate.final_status == FinalStatus.VERIFIED)}


def execute(*, output_dir: str | Path, config_path: str | Path, mode: str, backend_factory=make_backend) -> dict[str, Any]:
    if mode not in {"run", "replay"}:
        raise ReviewedAnchorRecompositionError("R1_MODE_INVALID")
    root = Path(output_dir); plan = _load_plan(root, config_path)
    if mode == "replay" and not (root / "run" / "reviewed_anchor_r1_trace.jsonl").is_file():
        raise ReviewedAnchorRecompositionError("R1_REPLAY_REQUIRES_RUN")
    run_dir = root / mode
    if run_dir.exists():
        raise ReviewedAnchorRecompositionError("IMMUTABLE_R1_RUN_OUTPUT_EXISTS")
    config = load_config(config_path); backend = backend_factory(config["backend"])
    store = ArtifactStore(root / "cache"); inference = CachedInference(backend, store)
    traces = [_run_candidate(row, plan["plan_content_sha256"], config, inference, synthetic=backend.synthetic) for row in plan["candidates"]]
    new_calls = sum(row["usage"]["new_calls"] for row in traces)
    if mode == "replay" and new_calls:
        raise ReviewedAnchorRecompositionError("R1_REPLAY_NEW_MODEL_CALLS_NONZERO")
    summary = {"format": FORMAT, "status": "PASS", "mode": mode, "candidate_count": len(traces),
               "new_model_calls": new_calls, "cache_hits": sum(row["usage"]["cache_hits"] for row in traces),
               "certificate_distribution": dict(Counter(row["certificate"].get("final_status", row["certificate"].get("certificate_status")) for row in traces)),
               "r1_rounds_completed": 1, "cycle_guard": "NO_R2_AFTER_R1", "gt_used": False,
               "new_verified_count": sum(row["new_verified_count"] for row in traces), "certificate_created": not backend.synthetic}
    _as_error(_write, run_dir / "reviewed_anchor_r1_trace.jsonl", traces)
    _as_error(_write, run_dir / "reviewed_anchor_r1_certificate_manifest.jsonl", [{"r1_candidate_id": row["r1_candidate_id"], "certificate": row["certificate"]} for row in traces])
    _as_error(_write, run_dir / "reviewed_anchor_r1_summary.json", summary)
    _as_error(_write, run_dir / "cache_replay_audit.json", {"format": FORMAT, "status": "PASS", "mode": mode, "new_model_calls": new_calls,
        "cache_hits": summary["cache_hits"], "trace_sha256": sha256_path(run_dir / "reviewed_anchor_r1_trace.jsonl"), "gt_used": False})
    _as_error(_write_text, run_dir / "reviewed_anchor_r1_report.md", "# Reviewed-anchor R1 recomposition\n\n" + canonical_json(summary) + "\n")
    return {"status": "PASS", "mode": mode, "summary": summary, "output_dir": str(run_dir)}
