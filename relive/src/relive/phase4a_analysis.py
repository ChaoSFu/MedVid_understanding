"""Read-only Phase 4A diagnostic finalization.

This module deliberately has no backend, cache, certificate, or GT dependency.
It turns an immutable likelihood trace into descriptive classifications and
future-controller fixtures; it never admits evidence or changes a certificate.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from .phase4a0 import ELIGIBLE_MANIFEST_FORMAT
from .storage.artifacts import canonical_json

ANALYSIS_RULE_VERSION = "relive-phase4a-posthoc-descriptive-v1"
VARIANTS = ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL", "FULL_GRAY", "MISMATCHED_PUBLIC")


class Phase4AAnalysisError(ValueError):
    pass


def _sha_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4AAnalysisError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase4AAnalysisError("JSON_OBJECT_REQUIRED")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4AAnalysisError(f"unreadable JSONL: {path}") from exc
    if not values or not all(isinstance(value, dict) for value in values):
        raise Phase4AAnalysisError("NONEMPTY_JSONL_OBJECT_ROWS_REQUIRED")
    return values


def _strict_hash(manifest: dict[str, Any], field: str) -> None:
    if manifest.get(field) != _sha({key: value for key, value in manifest.items() if key != field}):
        raise Phase4AAnalysisError("FROZEN_INPUT_BYTES_CHANGED")


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Phase4AAnalysisError("PHASE4A_MARGIN_MUST_BE_FINITE")
    return float(value)


def classify_case(margins: Mapping[str, float]) -> str:
    """Frozen descriptive ordering; zero is inherent to the support-margin definition."""
    original, keep = margins["ORIGINAL"], margins["KEEP_TARGET"]
    drop, control = margins["DROP_TARGET"], margins["DROP_MATCHED_CONTROL"]
    if original <= 0:
        return "ORIGINAL_LIKELIHOOD_INSUFFICIENT"
    if keep <= 0:
        return "KEEP_PRESERVATION_FAILURE"
    if drop > 0:
        return "DROP_SUPPORT_PERSISTS"
    if control <= 0:
        return "CONTROL_SUPPORT_LOST"
    return "FULL_LOCAL_DEPENDENCE_PATTERN"


def _auxiliary(margins: Mapping[str, float], classification: str) -> str | None:
    if classification != "DROP_SUPPORT_PERSISTS":
        return None
    return ("DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT"
            if margins["FULL_GRAY"] > 0 else "DROP_SUPPORT_PERSISTS_WITH_VISUAL_SENSITIVITY_RETAINED")


def _routing(classification: str, margins: Mapping[str, float]) -> tuple[str, str, str]:
    if classification == "FULL_LOCAL_DEPENDENCE_PATTERN":
        return "NO_FAILURE_DIAGNOSTIC_COMPLETE", "STOP_DIAGNOSTIC", "CREATE_VERIFIED"
    if classification == "ORIGINAL_LIKELIHOOD_INSUFFICIENT":
        return "ORIGINAL_INSUFFICIENT", "TEMPORAL_REACQUIRE", "SPATIAL_REGROUND_ON_SAME_WINDOW"
    if classification == "KEEP_PRESERVATION_FAILURE":
        return "KEEP_SUPPORT_LOST", "RECOMPOSE_SPATIAL_EVIDENCE", "LOWER_KEEP_STANDARD"
    if classification == "CONTROL_SUPPORT_LOST":
        return "CONTROL_SUPPORT_LOST", "INTERVENTION_DIAGNOSTIC", "EXPAND_ROI_BLINDLY"
    if margins["FULL_GRAY"] <= 0:
        return "DROP_SUPPORT_PERSISTS", "LOCALIZE_RESIDUAL_COMPONENT", "CREATE_VERIFIED"
    return "VERIFIER_INSENSITIVITY_UNRESOLVED", "STOP_OR_PREDECLARED_VERIFIER_SWITCH", "UNBOUNDED_R2_R3"


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def finalize(*, phase4a_output_dir: Path, eligibility_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Finalize existing artifacts only; no model, cache write, or certificate call."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Phase4AAnalysisError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    trace_path = phase4a_output_dir / "run" / "phase4a_trace.jsonl"
    frozen_path = phase4a_output_dir / "phase4a_frozen_manifest.json"
    preflight_path = phase4a_output_dir / "phase4a_preflight.json"
    trace, frozen, preflight = _jsonl(trace_path), _json(frozen_path), _json(preflight_path)
    _strict_hash(frozen, "phase4a_frozen_manifest_sha256")
    eligibility = _json(eligibility_manifest_path)
    if eligibility.get("format") != ELIGIBLE_MANIFEST_FORMAT:
        raise Phase4AAnalysisError("INVALID_PHASE4A0_ELIGIBILITY_MANIFEST")
    _strict_hash(eligibility, "manifest_content_sha256")
    legacy_posthoc_gate = "source_mode" not in frozen and "source_mode" not in preflight
    if not legacy_posthoc_gate and (frozen.get("source_mode") != "DEVELOPMENT_POSITIVE_CONTROL" or preflight.get("source_mode") != "DEVELOPMENT_POSITIVE_CONTROL"):
        raise Phase4AAnalysisError("FINALIZER_REQUIRES_DEVELOPMENT_POSITIVE_CONTROL")
    if not legacy_posthoc_gate and preflight.get("eligibility_manifest") != str(eligibility_manifest_path):
        raise Phase4AAnalysisError("FROZEN_INPUT_BYTES_CHANGED")
    if (not legacy_posthoc_gate and frozen.get("formal_positive_control") is not True) or eligibility.get("eligible_count") != len(trace):
        raise Phase4AAnalysisError("ELIGIBILITY_GATE_BINDING_MISMATCH")
    records_path = Path(eligibility.get("eligible_controls_path", ""))
    if not records_path.is_file() or _sha_bytes(records_path) != eligibility.get("eligible_controls_sha256"):
        raise Phase4AAnalysisError("FROZEN_INPUT_BYTES_CHANGED")
    allowed_sources = {f"development:{row['audit_case_id']}" for row in _jsonl(records_path)}
    if {row.get("source_id") for row in trace} != allowed_sources:
        raise Phase4AAnalysisError("ELIGIBILITY_GATE_BINDING_MISMATCH")
    if legacy_posthoc_gate:
        frozen_entries = {row.get("source_id"): row for row in frozen.get("entries", []) if isinstance(row, dict)}
        eligible_by_source = {f"development:{row['audit_case_id']}": row for row in _jsonl(records_path)}
        for source_id, record in eligible_by_source.items():
            entry = frozen_entries.get(source_id)
            if not isinstance(entry, dict) or entry.get("claim_sha256") != record.get("claim_sha256") or entry.get("target_roi") != record.get("frozen_support_region") or entry.get("matched_control_roi") != record.get("matched_control_regions", [None])[0]:
                raise Phase4AAnalysisError("ELIGIBILITY_GATE_BINDING_MISMATCH")

    rows: list[dict[str, Any]] = []
    for item in trace:
        variants = item.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            raise Phase4AAnalysisError("PHASE4A_VARIANT_SET_INVALID")
        margins = {variant: _finite(variants[variant].get("support_margin")) for variant in VARIANTS}
        deltas = {"delta_drop": margins["ORIGINAL"] - margins["DROP_TARGET"],
                  "delta_full_gray": margins["ORIGINAL"] - margins["FULL_GRAY"],
                  "delta_mismatch": margins["ORIGINAL"] - margins["MISMATCHED_PUBLIC"],
                  "delta_specificity": margins["DROP_MATCHED_CONTROL"] - margins["DROP_TARGET"]}
        stored = item.get("deltas")
        if not isinstance(stored, dict) or set(stored) != set(deltas) or any(not math.isclose(_finite(stored[key]), value, rel_tol=0.0, abs_tol=0.0) for key, value in deltas.items()):
            raise Phase4AAnalysisError("PHASE4A_DELTA_BINDING_MISMATCH")
        classification = classify_case(margins)
        rows.append({"case_id": item["source_id"], "claim_id": item["claim_id"], "source_trace_sha256": _sha_bytes(trace_path),
                     "margins": margins, "deltas": deltas, "observed_diagnostic_class": classification,
                     "auxiliary_diagnostic": _auxiliary(margins, classification), "analysis_rule_version": ANALYSIS_RULE_VERSION,
                     "diagnostic_only": True, "certificate_created": False, "certificate_unchanged": True,
                     "new_verified_count": 0, "gt_used": False, "model_calls_made": 0, "backend_loaded": False})

    output_dir.mkdir(parents=True)
    classifications = output_dir / "phase4a_case_classification.jsonl"
    classifications.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    fixture_rows = []
    for row in rows:
        code, allowed, forbidden = _routing(row["observed_diagnostic_class"], row["margins"])
        fixture_rows.append({"case_id": row["case_id"], "source_trace_sha256": row["source_trace_sha256"],
            "observed_diagnostic_class": row["observed_diagnostic_class"], "auxiliary_diagnostic": row["auxiliary_diagnostic"],
            "proposed_v2_failure_code": code, "allowed_action_family": allowed, "forbidden_action_family": forbidden,
            "analysis_rule_version": ANALYSIS_RULE_VERSION, "diagnostic_only": True, "certificate_created": False,
            "certificate_unchanged": True, "new_verified_count": 0, "gt_used": False, "model_calls_made": 0, "backend_loaded": False})
    fixtures = output_dir / "phase4a_failure_routing_fixtures.jsonl"
    fixtures.write_text("".join(canonical_json(row) + "\n" for row in fixture_rows), encoding="utf-8")
    delta_names = ("delta_drop", "delta_full_gray", "delta_mismatch", "delta_specificity")
    summary = {"format": "relive-phase4a-diagnostic-finalization-v1", "status": "PASS", "analysis_rule_version": ANALYSIS_RULE_VERSION,
        "case_count": len(rows), "classification_counts": {name: sum(row["observed_diagnostic_class"] == name for row in rows) for name in sorted({row["observed_diagnostic_class"] for row in rows})},
        "case_ids_by_classification": {name: [row["case_id"] for row in rows if row["observed_diagnostic_class"] == name] for name in sorted({row["observed_diagnostic_class"] for row in rows})},
        "positive_delta_counts": {name: sum(row["deltas"][name] > 0 for row in rows) for name in delta_names},
        "delta_statistics": {name: _stats([row["deltas"][name] for row in rows]) for name in delta_names},
        "source_mode": "DEVELOPMENT_POSITIVE_CONTROL", "eligibility_gate_status": ("POSTHOC_VERIFIED_FOR_DIAGNOSTIC_FINALIZATION_ONLY" if legacy_posthoc_gate else "PASS"), "source_trace_sha256": _sha_bytes(trace_path),
        "frozen_manifest_sha256": frozen["phase4a_frozen_manifest_sha256"], "eligibility_manifest_sha256": eligibility["manifest_content_sha256"],
        "diagnostic_only": True, "certificate_created": False, "certificate_unchanged": True, "new_verified_count": 0,
        "gt_used": False, "model_calls_made": 0, "backend_loaded": False, "legacy_posthoc_gate": legacy_posthoc_gate}
    summary_path = output_dir / "phase4a_descriptive_summary.json"; summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    report = output_dir / "phase4a_diagnostic_report.md"
    report.write_text("# ReliVE Phase 4A diagnostic finalization\n\nPhase 4A demonstrates measurable visual dependence under frozen development controls. It does not establish claim truth, unique evidence localization, certificate validity, or VERIFIED status.\n\nThis report is a read-only descriptive analysis; its future-controller fixtures do not execute adaptation.\n", encoding="utf-8")
    audit = {"format": "relive-phase4a-diagnostic-finalization-v1", "status": "PASS", "analysis_rule_version": ANALYSIS_RULE_VERSION,
        "source_trace_sha256": _sha_bytes(trace_path), "frozen_manifest_sha256": frozen["phase4a_frozen_manifest_sha256"],
        "eligibility_manifest_sha256": eligibility["manifest_content_sha256"], "artifact_sha256": {"phase4a_case_classification.jsonl": _sha_bytes(classifications),
        "phase4a_descriptive_summary.json": _sha_bytes(summary_path), "phase4a_diagnostic_report.md": _sha_bytes(report), "phase4a_failure_routing_fixtures.jsonl": _sha_bytes(fixtures)},
        "diagnostic_only": True, "certificate_created": False, "certificate_unchanged": True, "new_verified_count": 0,
        "gt_used": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False}
    audit["audit_content_sha256"] = _sha(audit)
    (output_dir / "phase4a_finalize_audit.json").write_text(canonical_json(audit) + "\n", encoding="utf-8")
    return audit
