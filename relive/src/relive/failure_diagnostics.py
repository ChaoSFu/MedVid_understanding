"""Read-only Phase 2.5 failure-root-cause and refinement-eligibility audit.

This module consumes completed Phase 2 runtime artifacts only.  It never builds
an inference request, opens a VLM, changes a cache, or alters the frozen
candidate pool.  The resulting diagnosis distinguishes an unavailable matched
control from an implementation discrepancy before any future refinement is
considered.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image, ImageChops

from .interventions import audit_control_geometry
from .storage.artifacts import canonical_json, stable_hash

PHASE25_FORMAT = "relive-phase2-failure-diagnostics-v1"


class FailureDiagnosticError(ValueError):
    """Raised for incomplete or incompatible completed-run artifacts."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FailureDiagnosticError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise FailureDiagnosticError(f"JSON object required: {path}")
    return value


def _json_files(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not directory.is_dir():
        return []
    return [(path, _json(path)) for path in sorted(directory.glob("*.json"))]


def _ref(path: Path) -> str:
    return str(path.resolve())


def _cache_entries(cache_dir: Path | None, cache_key: str | None) -> list[str]:
    if cache_dir is None or not cache_key:
        return []
    directory = cache_dir / "raw" / cache_key
    if not directory.is_dir():
        return []
    return [_ref(path) for path in sorted(directory.glob("*.json"))]


def classify_control_count_mismatch(*, expected_count: int | None, generated_count: int | None,
                                    inference_result_count: int | None,
                                    controls_status: str | None) -> tuple[str, str]:
    """Classify mismatch evidence without changing the matched-control policy."""
    if expected_count is None or generated_count is None or inference_result_count is None:
        return "C_UNRESOLVED", "MISSING_CONTROL_ARTIFACT_OR_COUNT"
    if (controls_status == "CONTROL_UNAVAILABLE" and generated_count < expected_count
            and inference_result_count == generated_count):
        return "B_LEGITIMATE_PROTOCOL_UNAVAILABLE", "GEOMETRY_GENERATED_FEWER_THAN_FROZEN_REQUIRED_CONTROLS"
    if generated_count == expected_count and inference_result_count != generated_count:
        return "A_IMPLEMENTATION_OR_ARTIFACT_BUG", "GENERATED_CONTROLS_LACK_MATCHING_INFERENCE_RESULTS"
    if generated_count != expected_count and controls_status != "CONTROL_UNAVAILABLE":
        return "A_IMPLEMENTATION_OR_ARTIFACT_BUG", "CONTROL_ARTIFACT_AVAILABILITY_INCONSISTENT_WITH_COUNTS"
    return "C_UNRESOLVED", "CONTROL_COUNTS_DO_NOT_IDENTIFY_A_SINGLE_CAUSE"


def _image_identical(first: Path, second: Path) -> bool:
    try:
        with Image.open(first) as a, Image.open(second) as b:
            left, right = a.convert("RGB"), b.convert("RGB")
            return left.size == right.size and ImageChops.difference(left, right).getbbox() is None
    except (OSError, ValueError):
        return False


def _is_uniform_fill(path: Path, bbox: list[int] | tuple[int, ...], fill: tuple[int, int, int]) -> bool | None:
    try:
        with Image.open(path) as image:
            source = image.convert("RGB")
            if len(bbox) != 4:
                return None
            crop = source.crop(tuple(bbox))
            if crop.width == 0 or crop.height == 0:
                return None
            colors = crop.getcolors(maxcolors=crop.width * crop.height + 1)
            return colors == [(crop.width * crop.height, fill)]
    except (OSError, ValueError):
        return None


def classify_intervention_no_effect(audits: list[dict[str, Any]], original_paths: list[str]) -> tuple[str, str, list[dict[str, Any]]]:
    """Classify no-effect evidence using pixel/audit artifacts, never semantics."""
    detail: list[dict[str, Any]] = []
    if not audits:
        return "UNRESOLVED", "MISSING_PIXEL_AUDIT_ARTIFACT", detail
    causes: list[str] = []
    for audit in audits:
        bbox = audit.get("pixel_bbox", [])
        frame_id = audit.get("frame_id")
        output = Path(audit["output_path"]) if isinstance(audit.get("output_path"), str) else None
        source_path = None
        # Input paths are in candidate order; a per-frame audit lets us match
        # by index only after the caller supplies the candidate's ordered paths.
        if isinstance(audit.get("source_path"), str):
            source_path = Path(audit["source_path"])
        if not source_path and original_paths:
            source_path = Path(original_paths[min(len(detail), len(original_paths) - 1)])
        row = {"frame_id": frame_id, "pixel_bbox": bbox, "audit_status": audit.get("audit_status"),
               "output_path": str(output) if output else None, "source_path": str(source_path) if source_path else None}
        if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            row["classification"] = "CLIPPED_TO_ZERO_AREA"
            causes.append("CLIPPED_TO_ZERO_AREA")
        elif (audit.get("variant") == "KEEP_TARGET" and isinstance(audit.get("original_size"), list)
              and len(audit["original_size"]) == 2
              and bbox == [0, 0, audit["original_size"][0], audit["original_size"][1]]):
            # KEEP is defined as retaining the ROI while changing its
            # complement.  A full-frame ROI has an empty complement, so a
            # no-effect result is a proposal-geometry failure, not a missing
            # artifact or a broken opaque-gray operator.
            row["classification"] = "FULL_FRAME_OR_NO_COMPLEMENT_ROI"
            causes.append("FULL_FRAME_OR_NO_COMPLEMENT_ROI")
        elif output is None or not output.is_file() or source_path is None or not source_path.is_file():
            row["classification"] = "UNRESOLVED_ARTIFACT_NOT_READABLE"
            causes.append("UNRESOLVED_ARTIFACT_NOT_READABLE")
        elif _image_identical(source_path, output):
            uniform = _is_uniform_fill(source_path, bbox, (127, 127, 127))
            row["classification"] = "UNIFORM_REGION" if uniform else "ARTIFACT_NOT_APPLIED"
            causes.append(row["classification"])
        else:
            # An actual image change together with a no-effect audit is an
            # invariant failure, not evidence that the claim is false.
            row["classification"] = "OPERATOR_OR_AUDIT_BUG"
            causes.append("OPERATOR_OR_AUDIT_BUG")
        detail.append(row)
    for preferred in ("CLIPPED_TO_ZERO_AREA", "FULL_FRAME_OR_NO_COMPLEMENT_ROI", "ARTIFACT_NOT_APPLIED", "OPERATOR_OR_AUDIT_BUG",
                      "UNIFORM_REGION", "UNRESOLVED_ARTIFACT_NOT_READABLE"):
        if preferred in causes:
            return preferred, "PIXEL_AUDIT_AND_IMAGE_COMPARISON", detail
    return "OTHER", "NO_EFFECT_WITHOUT_A_DETERMINISTIC_PIXEL_CLASSIFICATION", detail


def refinement_eligibility(failure_reasons: list[str], *, control_mismatch_class: str | None = None,
                           no_effect_class: str | None = None,
                           control_geometry_refinable: bool = False) -> dict[str, Any]:
    """Apply the frozen Phase 2.5 policy; this function never performs refinement."""
    reasons = set(failure_reasons)
    technical_reasons = {"TECHNICAL_FAILURE", "SPATIAL_PROPOSAL_FAILURE",
                         "INTERVENTION_PIXEL_AUDIT_FAILED", "SPATIAL_REFERENCE_BINDING_MISMATCH"}
    if "CONTROL_RESULT_COUNT_MISMATCH" in reasons:
        return {"root_cause_class": "TECHNICAL_DIAGNOSTIC", "refinement_eligible": False,
                "recommended_action": "TECHNICAL_DIAGNOSTIC_RESOLVE_CONTROL_COUNT_MISMATCH",
                "policy_reason": control_mismatch_class or "CONTROL_RESULT_COUNT_MISMATCH"}
    if reasons & technical_reasons:
        return {"root_cause_class": "TECHNICAL_ERROR", "refinement_eligible": False,
                "recommended_action": "TECHNICAL_DIAGNOSTIC", "policy_reason": sorted(reasons & technical_reasons)}
    if "ORIGINAL_INSUFFICIENT" in reasons or "ORIGINAL_NOT_SUPPORTED" in reasons:
        return {"root_cause_class": "TEMPORAL_OR_SEMANTIC_INSUFFICIENT", "refinement_eligible": False,
                "recommended_action": "TEMPORAL_REACQUIRE", "policy_reason": "ORIGINAL_INSUFFICIENT"}
    if "INTERVENTION_NO_EFFECT" in reasons:
        if no_effect_class in {"CLIPPED_TO_ZERO_AREA", "FULL_FRAME_OR_NO_COMPLEMENT_ROI", "UNIFORM_REGION"}:
            return {"root_cause_class": "SPATIAL_PROPOSAL_OR_ROI_FAILURE", "refinement_eligible": True,
                    "recommended_action": "SPATIAL_REFINE", "policy_reason": no_effect_class}
        if no_effect_class in {"ARTIFACT_NOT_APPLIED", "OPERATOR_OR_AUDIT_BUG"}:
            return {"root_cause_class": "TECHNICAL_ERROR", "refinement_eligible": False,
                    "recommended_action": "TECHNICAL_DIAGNOSTIC", "policy_reason": no_effect_class}
        return {"root_cause_class": "UNRESOLVED", "refinement_eligible": False,
                "recommended_action": "DIAGNOSE_INTERVENTION_NO_EFFECT", "policy_reason": no_effect_class}
    if "CONTROL_UNAVAILABLE" in reasons:
        if control_geometry_refinable:
            return {"root_cause_class": "SPATIAL_PROPOSAL_GEOMETRY", "refinement_eligible": True,
                    "recommended_action": "SPATIAL_REFINE", "policy_reason": "CONTROL_UNAVAILABLE_FROM_GEOMETRY"}
        return {"root_cause_class": "PROTOCOL_UNAVAILABLE", "refinement_eligible": False,
                "recommended_action": "DIAGNOSE_CONTROL_GEOMETRY", "policy_reason": "CONTROL_UNAVAILABLE"}
    if reasons & {"DEPENDENCE_UNRESOLVED", "KEEP_SUPPORT_LOST", "CONTROL_SUPPORT_LOST", "NONSPECIFIC_INTERVENTION_RESPONSE"}:
        return {"root_cause_class": "SPATIAL_DEPENDENCE_OR_SUFFICIENCY", "refinement_eligible": True,
                "recommended_action": "SPATIAL_REFINE", "policy_reason": sorted(reasons & {"DEPENDENCE_UNRESOLVED", "KEEP_SUPPORT_LOST", "CONTROL_SUPPORT_LOST", "NONSPECIFIC_INTERVENTION_RESPONSE"})}
    return {"root_cause_class": "UNRESOLVED", "refinement_eligible": False,
            "recommended_action": "NO_REFINEMENT_UNTIL_DIAGNOSED", "policy_reason": sorted(reasons)}


def _write_immutable(path: Path, value: Any, *, jsonl: bool = False) -> str:
    if jsonl:
        body = "".join(canonical_json(row) + "\n" for row in value).encode("utf-8")
    elif isinstance(value, str):
        body = value.encode("utf-8")
    else:
        body = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise FailureDiagnosticError(f"immutable artifact collision: {path}") from None
    finally:
        os.unlink(temporary)
    return _ref(path)


def _control_event(events: list[tuple[Path, dict[str, Any]]], sample_id: str, proposal_id: str | None) -> tuple[Path, dict[str, Any]] | tuple[None, None]:
    for path, event in events:
        if event.get("sample_id") == sample_id and event.get("proposal_id") == proposal_id:
            return path, event
    return None, None


def _audits_for(audits: list[tuple[Path, dict[str, Any]]], sample_id: str, candidate_id: str,
                proposal_id: str | None) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, event) for path, event in audits if event.get("sample_id") == sample_id
            and event.get("candidate_id") == candidate_id and event.get("proposal_id") == proposal_id]


def _candidate_diagnostic(sample: dict[str, Any], certificate: dict[str, Any], controls_events,
                          pixel_events, certificate_events: dict[str, Path], cache_dir: Path | None) -> dict[str, Any]:
    sample_id, candidate_id = sample.get("sample_id"), certificate.get("candidate_id")
    reasons = list(certificate.get("failure_reasons", []))
    spatial = certificate.get("checks", {}).get("spatial", {})
    spatial = spatial if isinstance(spatial, dict) else {}
    proposal = spatial.get("proposal", {})
    proposal = proposal if isinstance(proposal, dict) else {}
    proposal_id = proposal.get("proposal_id")
    control_path, control_event = _control_event(controls_events, sample_id, proposal_id)
    controls_ref = spatial.get("references", {}).get("controls", []) if isinstance(spatial.get("references"), dict) else []
    controls_ref = controls_ref if isinstance(controls_ref, list) else []
    expected = spatial.get("expected_control_count")
    generated = control_event.get("valid_count") if control_event else None
    inferred = len(controls_ref)
    control_status = control_event.get("status") if control_event else None
    mismatch_class = None
    mismatch_reason = None
    if "CONTROL_RESULT_COUNT_MISMATCH" in reasons:
        mismatch_class, mismatch_reason = classify_control_count_mismatch(
            expected_count=expected, generated_count=generated,
            inference_result_count=inferred, controls_status=control_status)
    control_cache = []
    for item in controls_ref:
        refs = item.get("input_references", {}) if isinstance(item, dict) else {}
        key = refs.get("cache_key") if isinstance(refs, dict) else None
        control_cache.append({"control_index": refs.get("control_index") if isinstance(refs, dict) else None,
                              "cache_key": key, "raw_entries": _cache_entries(cache_dir, key)})
    geometry = None
    region = proposal.get("support_region")
    if isinstance(region, list) and len(region) == 4:
        try:
            geometry = audit_control_geometry(region, int(expected or 0))
        except (ValueError, TypeError):
            geometry = None
    candidate_audits = _audits_for(pixel_events, sample_id, candidate_id, proposal_id)
    original_refs = certificate.get("checks", {}).get("semantic", {}).get("references", {})
    original_input_refs = original_refs.get("input_references", {}) if isinstance(original_refs, dict) else {}
    original_paths = original_input_refs.get("image_paths", []) if isinstance(original_input_refs, dict) else []
    original_paths = original_paths if isinstance(original_paths, list) else []
    original_frame_ids = original_input_refs.get("frame_ids", []) if isinstance(original_input_refs, dict) else []
    original_by_frame = {frame_id: path for frame_id, path in zip(original_frame_ids, original_paths)
                         if isinstance(frame_id, str) and isinstance(path, str)}
    no_effect_class = None
    no_effect_reason = None
    no_effect_detail = []
    if "INTERVENTION_NO_EFFECT" in reasons:
        no_effect_audits = []
        candidate_frame_ids = set(proposal.get("frame_mapping", {}).get("frame_ids", []))
        for _, audit in candidate_audits:
            if audit.get("audit_status") == "INTERVENTION_NO_EFFECT":
                row = dict(audit)
                row["source_path"] = original_by_frame.get(audit.get("frame_id"))
                if audit.get("frame_id") not in candidate_frame_ids:
                    row["mapping_error"] = True
                no_effect_audits.append(row)
        if any(row.get("mapping_error") for row in no_effect_audits):
            no_effect_class, no_effect_reason = "WRONG_COORDINATE_MAPPING", "PIXEL_AUDIT_FRAME_NOT_BOUND_TO_PROPOSAL"
        else:
            no_effect_class, no_effect_reason, no_effect_detail = classify_intervention_no_effect(no_effect_audits, original_paths)
    image_geometry = sorted({tuple(event.get("original_size", [])) for _, event in candidate_audits if event.get("original_size")})
    # A deterministic control geometry failure is potentially spatially fixable;
    # a simultaneous count mismatch remains blocked by the policy below.
    geometry_refinable = bool(geometry and geometry.get("status") == "CONTROL_UNAVAILABLE" and region)
    eligibility = refinement_eligibility(reasons, control_mismatch_class=mismatch_class,
                                         no_effect_class=no_effect_class,
                                         control_geometry_refinable=geometry_refinable)
    return {
        "format": PHASE25_FORMAT,
        "qa_id": sample_id,
        "candidate_id": candidate_id,
        "candidate_rank": next((row.get("candidate_rank") for row in sample.get("candidate_traversal", {}).get("trace", [])
                                if row.get("candidate_id") == candidate_id), None),
        "certificate_id": certificate.get("certificate_id"),
        "certificate_status": certificate.get("final_status"),
        "original_failure_reasons": reasons,
        "root_cause_class": eligibility["root_cause_class"],
        "refinement_eligible": eligibility["refinement_eligible"],
        "recommended_action": eligibility["recommended_action"],
        "policy_reason": eligibility["policy_reason"],
        "control_count_audit": {
            "mismatch_present": "CONTROL_RESULT_COUNT_MISMATCH" in reasons,
            "classification": mismatch_class,
            "classification_reason": mismatch_reason,
            "expected_control_count": expected,
            "generated_control_count": generated,
            "inference_result_count": inferred,
            "control_artifact_status": control_status,
            "control_artifact_ref": _ref(control_path) if control_path else None,
            "control_ids": [item.get("input_references", {}).get("control_index") for item in controls_ref if isinstance(item, dict)],
            "inference_cache_entries": control_cache,
        },
        "control_geometry_audit": {
            "proposal_bbox_normalized_xyxy": region,
            "proposal_area_fraction": spatial.get("support_area_fraction"),
            "image_geometry": [list(item) for item in image_geometry],
            "matched_control_required_geometry": "same normalized width and height; in [0,1]; non-overlapping with target and prior controls",
            "deterministic_placement_audit": geometry,
            "spatial_failure_refinable_if_not_blocked": geometry_refinable,
        },
        "intervention_no_effect_audit": {
            "present": "INTERVENTION_NO_EFFECT" in reasons,
            "classification": no_effect_class,
            "classification_reason": no_effect_reason,
            "details": no_effect_detail,
        },
        "supporting_artifact_references": {
            "certificate": _ref(certificate_events[certificate.get("certificate_id")]) if certificate.get("certificate_id") in certificate_events else None,
            "spatial_proposal_id": proposal_id,
            "pixel_audits": [_ref(path) for path, _ in candidate_audits],
            "controls": _ref(control_path) if control_path else None,
        },
    }


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["# ReliVE Phase 2.5 failure-mode validation and refinement-eligibility audit", "",
             "Read-only audit of existing Phase 2 artifacts. It made no model calls and did not mutate the inference cache.", "",
             "## Summary", "",
             f"- Candidates: {summary['candidate_count']}",
             f"- Technical/protocol: {summary['technical_or_protocol_failures']}",
             f"- Temporal/semantic: {summary['temporal_or_semantic_failures']}",
             f"- Spatial-refinable under policy: {summary['spatial_refinable_failures']}",
             f"- Unresolved: {summary['unresolved_failures']}", "",
             "## Candidate decisions", "",
             "| QA | Rank | Certificate | Root cause | Eligible | Action |", "|---|---:|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['qa_id']}` | {row['candidate_rank']} | {row['certificate_status']} | {row['root_cause_class']} | {row['refinement_eligible']} | {row['recommended_action']} |")
    lines.extend(["", "## Control-count mismatches", "",
                  "| QA | Candidate | Classification | Expected | Generated | Inference results |", "|---|---|---|---:|---:|---:|"])
    for row in rows:
        audit = row["control_count_audit"]
        if audit["mismatch_present"]:
            lines.append(f"| `{row['qa_id']}` | `{row['candidate_id']}` | {audit['classification']} | {audit['expected_control_count']} | {audit['generated_control_count']} | {audit['inference_result_count']} |")
    lines.extend(["", "A `CONTROL_RESULT_COUNT_MISMATCH` stays in technical diagnostic status until resolved, even when its immediate cause is the documented geometry-unavailable protocol state. No refinement was run by this audit.", ""])
    return "\n".join(lines)


def diagnose_phase2_failures(run_dir: Path, output_dir: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    """Write immutable Phase 2.5 artifacts from a completed Phase 2 run directory."""
    run_dir, output_dir = Path(run_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if not run_dir.is_dir() or not (run_dir / "samples").is_dir():
        raise FailureDiagnosticError("run directory must contain completed sample artifacts")
    if cache_dir is not None and not cache_dir.is_dir():
        raise FailureDiagnosticError("cache directory does not exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FailureDiagnosticError("output directory already exists and is nonempty")
    samples = [_json(path) for path in sorted((run_dir / "samples").glob("*.json"))]
    if not samples:
        raise FailureDiagnosticError("run directory has no completed samples")
    controls = _json_files(run_dir / "events" / "controls")
    pixels = _json_files(run_dir / "events" / "pixel_audits")
    certificate_events = {event.get("certificate_id"): path for path, event in _json_files(run_dir / "events" / "certificates")
                          if isinstance(event.get("certificate_id"), str)}
    rows = []
    for sample in samples:
        for certificate in sample.get("certificates", []):
            if isinstance(certificate, dict):
                rows.append(_candidate_diagnostic(sample, certificate, controls, pixels, certificate_events, cache_dir))
    if not rows:
        raise FailureDiagnosticError("run samples have no certificate artifacts")
    rows.sort(key=lambda row: (row["qa_id"], row["candidate_rank"] if row["candidate_rank"] is not None else -1, row["candidate_id"]))
    root_counts = Counter(row["root_cause_class"] for row in rows)
    mismatch_counts = Counter(row["control_count_audit"]["classification"] for row in rows if row["control_count_audit"]["mismatch_present"])
    summary = {
        "format": PHASE25_FORMAT,
        "phase": "ReliVE Phase 2.5 Failure-Mode Validation and Refinement Eligibility Audit",
        "run_directory": _ref(run_dir), "cache_directory": _ref(cache_dir) if cache_dir else None,
        "model_calls_made": 0, "cache_mutated": False,
        "candidate_count": len(rows), "control_count_mismatch_count": sum(mismatch_counts.values()),
        "control_count_mismatch_classification": dict(sorted(mismatch_counts.items())),
        "root_cause_distribution": dict(sorted(root_counts.items())),
        "technical_or_protocol_failures": sum(row["root_cause_class"] in {"TECHNICAL_DIAGNOSTIC", "TECHNICAL_ERROR", "PROTOCOL_UNAVAILABLE"} for row in rows),
        "temporal_or_semantic_failures": sum(row["root_cause_class"] == "TEMPORAL_OR_SEMANTIC_INSUFFICIENT" for row in rows),
        "spatial_refinable_failures": sum(row["refinement_eligible"] for row in rows),
        "unresolved_failures": sum(row["root_cause_class"] == "UNRESOLVED" for row in rows),
        "policy": {"ORIGINAL_INSUFFICIENT": {"action": "TEMPORAL_REACQUIRE", "spatial_refinement_allowed": False},
                   "DEPENDENCE_UNRESOLVED": {"action": "SPATIAL_REFINE", "spatial_refinement_allowed": True},
                   "SUFFICIENCY_UNRESOLVED": {"action": "SPATIAL_REFINE", "spatial_refinement_allowed": True},
                   "CONTROL_UNAVAILABLE": {"action": "SPATIAL_REFINE_ONLY_IF_PROPOSAL_GEOMETRY", "spatial_refinement_allowed": "conditional"},
                   "INTERVENTION_NO_EFFECT": {"action": "SPATIAL_REFINE_ONLY_IF_ROI_OR_PROPOSAL", "spatial_refinement_allowed": "conditional"},
                   "CONTROL_RESULT_COUNT_MISMATCH": {"action": "TECHNICAL_DIAGNOSTIC", "spatial_refinement_allowed": False}},
        "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"],
    }
    _write_immutable(output_dir / "phase2_failure_diagnostics.jsonl", rows, jsonl=True)
    _write_immutable(output_dir / "phase2_failure_diagnostics.summary.json", summary)
    _write_immutable(output_dir / "phase2_failure_diagnostics.md", _markdown(summary, rows))
    return {**summary, "artifacts": {"diagnostics_jsonl": _ref(output_dir / "phase2_failure_diagnostics.jsonl"),
                                      "summary_json": _ref(output_dir / "phase2_failure_diagnostics.summary.json"),
                                      "report_markdown": _ref(output_dir / "phase2_failure_diagnostics.md")},
            "diagnostics_sha256": stable_hash(rows)}
