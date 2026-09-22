"""Immutable application of explicit duplicate-grounding canonical labels.

This module deliberately never infers a label from a reviewer's rationale.  A
v1 duplicate-warning decision records only the human decision to unify.  The
separate v2 canonical-label file supplies the actual label, then this module
creates a *derived* review and re-runs the existing diagnostic-only human
overlay adjudication.  The original review, warning queue, and grounding are
read-only inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .human_overlay_adjudication import (
    ALL_LABELS, AMBIGUOUS_LABELS, HumanOverlayReviewError, NONVISIBLE_LABELS,
    VISIBLE_LABELS, adjudicate,
)
from .reviewed_anchor_formal_intervention import (
    DECISIONS, WARNING_FORMAT, ReviewedAnchorInterventionError, _rows, _write,
    prepare_warning_adjudication_template, sha256_path,
)


FORMAT = "reviewed-anchor-duplicate-grounding-canonical-label-adjudication-v2"
RESOLUTION_FORMAT = "reviewed-anchor-duplicate-grounding-label-resolution-v1"


class ReviewedAnchorLabelResolutionError(ValueError):
    pass


def _as_error(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ReviewedAnchorInterventionError, HumanOverlayReviewError) as exc:
        raise ReviewedAnchorLabelResolutionError(str(exc)) from exc


def _warnings(path: str | Path) -> list[dict[str, Any]]:
    rows = _as_error(_rows, path, "WARNING_QUEUE")
    result = [row for row in rows if row.get("warning_code") == "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW"]
    signatures = [row.get("grounding_signature") for row in result]
    if len(result) != 5 or any(not isinstance(item, str) or not item for item in signatures) or len(set(signatures)) != 5:
        raise ReviewedAnchorLabelResolutionError("DUPLICATE_WARNING_COUNT_OR_SIGNATURE_INVALID")
    return sorted(result, key=lambda row: row["grounding_signature"])


def _v1_adjudications(path: str | Path, queue_sha: str, warning_sigs: set[str]) -> dict[str, dict[str, Any]]:
    rows = _as_error(_rows, path, "WARNING_ADJUDICATION")
    result: dict[str, dict[str, Any]] = {}
    required = {"format", "grounding_signature", "warning_code", "decision", "reviewer_id", "rationale", "source_warning_queue_sha256"}
    for row in rows:
        if set(row) != required or row.get("format") != WARNING_FORMAT or row.get("warning_code") != "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW":
            raise ReviewedAnchorLabelResolutionError("WARNING_ADJUDICATION_SCHEMA_INVALID")
        signature, decision = row.get("grounding_signature"), row.get("decision")
        if signature not in warning_sigs or signature in result or decision not in DECISIONS or row.get("source_warning_queue_sha256") != queue_sha:
            raise ReviewedAnchorLabelResolutionError("WARNING_ADJUDICATION_BINDING_INVALID")
        if decision != "UNRESOLVED" and (not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip() or not isinstance(row.get("rationale"), str) or not row["rationale"].strip()):
            raise ReviewedAnchorLabelResolutionError("WARNING_ADJUDICATION_REVIEWER_OR_RATIONALE_MISSING")
        result[signature] = row
    if set(result) != warning_sigs:
        raise ReviewedAnchorLabelResolutionError("WARNING_ADJUDICATION_MISSING_SIGNATURE")
    return result


def prepare_canonical_label_template(*, warning_adjudication: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create a human-only v2 template from an already completed v1 file."""
    v1 = _as_error(_rows, warning_adjudication, "WARNING_ADJUDICATION")
    if not v1:
        raise ReviewedAnchorLabelResolutionError("WARNING_ADJUDICATION_EMPTY")
    source_sha = sha256_path(warning_adjudication)
    rows = []
    for row in sorted(v1, key=lambda item: str(item.get("grounding_signature"))):
        if row.get("format") != WARNING_FORMAT or row.get("decision") != "UNIFY_LABELS":
            raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_TEMPLATE_REQUIRES_UNIFY_LABELS")
        rows.append({"format": FORMAT, "grounding_signature": row["grounding_signature"],
                     "decision": "UNIFY_LABELS", "resolved_label": "", "reviewer_id": row["reviewer_id"],
                     "rationale": row["rationale"], "source_warning_adjudication_sha256": source_sha,
                     "source_warning_queue_sha256": row["source_warning_queue_sha256"],
                     "warning_code": "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW"})
    _as_error(_write, Path(output_path), rows)
    return {"format": FORMAT, "status": "AWAITING_EXPLICIT_CANONICAL_LABELS", "record_count": len(rows),
            "source_warning_adjudication_sha256": source_sha, "template_sha256": sha256_path(output_path),
            "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
            "certificate_created": False, "new_verified_count": 0, "gt_used": False}


def _canonical_rows(path: str | Path, *, v1_sha: str, queue_sha: str, warning_sigs: set[str], warnings: dict[str, dict[str, Any]], v1: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _as_error(_rows, path, "CANONICAL_LABEL_ADJUDICATION")
    required = {"format", "grounding_signature", "decision", "resolved_label", "reviewer_id", "rationale", "source_warning_adjudication_sha256", "source_warning_queue_sha256", "warning_code"}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != required or row.get("format") != FORMAT or row.get("warning_code") != "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW":
            raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_SCHEMA_INVALID")
        sig, label = row.get("grounding_signature"), row.get("resolved_label")
        if sig not in warning_sigs or sig in result or row.get("decision") != "UNIFY_LABELS" or label not in ALL_LABELS:
            raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_VALUE_OR_BINDING_INVALID")
        if row.get("source_warning_adjudication_sha256") != v1_sha or row.get("source_warning_queue_sha256") != queue_sha:
            raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_SOURCE_HASH_MISMATCH")
        if row.get("reviewer_id") != v1[sig].get("reviewer_id") or row.get("rationale") != v1[sig].get("rationale"):
            raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_REVIEWER_OR_RATIONALE_MISMATCH")
        for review in warnings[sig].get("reviews", []):
            if not isinstance(review, dict) or not isinstance(review.get("model_visibility"), str):
                # The historical warning list has no visibility field.  The
                # derived review loop checks this against the actual review.
                continue
        result[sig] = row
    if set(result) != warning_sigs:
        raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_SIGNATURE_COVERAGE_INVALID")
    return [result[key] for key in sorted(result)]


def apply_canonical_labels(*, raw_grounding: str | Path, human_review: str | Path,
                           warning_queue: str | Path, warning_adjudication: str | Path,
                           canonical_label_adjudication: str | Path, output_dir: str | Path,
                           expected_anchor_count: int = 75) -> dict[str, Any]:
    """Apply reviewer-supplied labels, then rebuild diagnostic-only routes."""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ReviewedAnchorLabelResolutionError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    raw, review, queue, v1_path, canonical = map(Path, (raw_grounding, human_review, warning_queue, warning_adjudication, canonical_label_adjudication))
    raw_sha, review_sha, queue_sha, v1_sha = (sha256_path(path) for path in (raw, review, queue, v1_path))
    warning_rows = _warnings(queue)
    warnings = {row["grounding_signature"]: row for row in warning_rows}
    v1 = _v1_adjudications(v1_path, queue_sha, set(warnings))
    if any(row["decision"] != "UNIFY_LABELS" for row in v1.values()):
        raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_APPLICATION_REQUIRES_ALL_UNIFY")
    canonical_rows = _canonical_rows(canonical, v1_sha=v1_sha, queue_sha=queue_sha, warning_sigs=set(warnings), warnings=warnings, v1=v1)
    canonical_by_signature = {row["grounding_signature"]: row for row in canonical_rows}
    target_keys: dict[tuple[str, str], str] = {}
    for signature, warning in warnings.items():
        for entry in warning.get("reviews", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("anchor_candidate_id"), str) or not isinstance(entry.get("role"), str):
                raise ReviewedAnchorLabelResolutionError("WARNING_QUEUE_REVIEW_BINDING_INVALID")
            key = (entry["anchor_candidate_id"], entry["role"])
            previous = target_keys.setdefault(key, signature)
            if previous != signature:
                raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_OVERLAPPING_REVIEW_INVALID")
    original_rows = _as_error(_rows, review, "HUMAN_REVIEW")
    derived = []
    applied_keys: set[tuple[str, str]] = set()
    changed_count = 0
    for row in original_rows:
        key = (row.get("anchor_candidate_id"), row.get("role"))
        signature = target_keys.get(key)
        if signature is None:
            derived.append(row)
            continue
        label = canonical_by_signature[signature]["resolved_label"]
        # Label validity by visibility remains enforced by the existing
        # adjudicator; fail before emitting a partial artifact.
        visibility = row.get("model_visibility")
        allowed = {"VISIBLE": VISIBLE_LABELS, "NOT_VISIBLE": NONVISIBLE_LABELS, "AMBIGUOUS": AMBIGUOUS_LABELS}.get(visibility)
        if allowed is None or label not in allowed:
            raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_VISIBILITY_INCOMPATIBLE")
        changed_count += int(row.get("human_overlay_label") != label)
        derived.append({**row, "human_overlay_label": label,
                        "reason_code": "CANONICAL_DUPLICATE_GROUNDING_LABEL_APPLIED"})
        applied_keys.add(key)
    if applied_keys != set(target_keys):
        raise ReviewedAnchorLabelResolutionError("CANONICAL_LABEL_REVIEW_TARGET_MISSING")
    derived_path = out / "derived_human_overlay_review.jsonl"
    _as_error(_write, derived_path, derived)
    recomputed_dir = out / "recomputed_human_overlay_adjudication"
    try:
        report = adjudicate(raw_anchor_manifest=raw, review_jsonl=derived_path, output_dir=recomputed_dir, expected_anchor_count=expected_anchor_count)
    except HumanOverlayReviewError as exc:
        raise ReviewedAnchorLabelResolutionError("RECOMPUTED_HUMAN_OVERLAY_ADJUDICATION_INVALID") from exc
    artifact_paths = {name: recomputed_dir / name for name in (
        "adjudicated_anchor_manifest.jsonl", "observation_anchor_decisions.jsonl", "eligible_anchor_manifest.jsonl",
        "schema_issue_candidates.jsonl", "human_review_validation_report.json")}
    manifest = {"format": RESOLUTION_FORMAT, "status": "PASS", "raw_grounding_sha256": raw_sha,
                "source_human_review_sha256": review_sha, "source_warning_queue_sha256": queue_sha,
                "source_warning_adjudication_sha256": v1_sha,
                "canonical_label_adjudication_sha256": sha256_path(canonical),
                "derived_human_review_sha256": sha256_path(derived_path),
                "recomputed_artifact_sha256": {name: sha256_path(path) for name, path in sorted(artifact_paths.items())},
                "canonical_labels": [{"grounding_signature": row["grounding_signature"], "resolved_label": row["resolved_label"]} for row in canonical_rows],
                "canonical_label_count": len(canonical_rows), "review_rows_targeted": len(applied_keys),
                "review_rows_changed": changed_count,
                "source_raw_grounding_unchanged": sha256_path(raw) == raw_sha,
                "source_human_review_unchanged": sha256_path(review) == review_sha,
                "source_warning_queue_unchanged": sha256_path(queue) == queue_sha,
                "source_warning_adjudication_unchanged": sha256_path(v1_path) == v1_sha,
                "recomputed_route_counts": report["observation_route_counts"],
                "recomputed_eligible_anchor_count": report["eligible_anchor_count"],
                "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
                "certificate_created": False, "new_verified_count": 0, "gt_used": False,
                "diagnostic_only": True}
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    _as_error(_write, out / "canonical_label_resolution_manifest.json", manifest)
    return {"status": "PASS", "format": RESOLUTION_FORMAT, "output_dir": str(out),
            "derived_human_review": str(derived_path), "recomputed_output_dir": str(recomputed_dir),
            "resolution_manifest": str(out / "canonical_label_resolution_manifest.json"),
            "recomputed_route_counts": report["observation_route_counts"],
            "recomputed_eligible_anchor_count": report["eligible_anchor_count"],
            "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
            "certificate_created": False, "new_verified_count": 0, "gt_used": False}
