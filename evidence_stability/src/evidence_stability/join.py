from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from evidence_stability.phase_d2 import assert_raw_result_gt_free


CORE_CANDIDATE_LABELS = {"TRUE_SUPPORT", "SPURIOUS_SUPPORT"}
JOINED_OUTPUT_FIELDS = [
    "candidate_id",
    "qa_id",
    "clip_id",
    "window_id",
    "intervention_id",
    "dataset_name",
    "target_action",
    "target_field",
    "candidate_type",
    "original_candidate_label",
    "original_candidate_prediction",
    "gt_duration_total",
    "gt_duration_bin",
    "intervention_family",
    "intervention_type",
    "generation_valid",
    "n_frames",
    "intervened_n_frames",
    "intervened_n_unique_frames",
    "boundary_asymmetric",
    "original_gt_alignment_class",
    "intervention_gt_alignment_class",
    "strict_valid",
    "any_gt_valid",
    "validity_reason",
    "original_evidence_density",
    "intervention_evidence_density",
    "original_gt_evidence_recall",
    "intervention_gt_evidence_recall",
    "original_n_gt_visible",
    "intervention_n_gt_visible",
    "raw_response",
    "intervention_prediction",
    "parsed_prediction",
    "model_name",
    "model_revision",
    "model_identity_hash",
    "prompt_version",
    "prompt_hash",
    "decoding_config",
    "cache_key",
]


def duplicate_ids(rows: list[dict[str, Any]], key: str) -> list[str]:
    counts = Counter(str(row.get(key)) for row in rows)
    return sorted([value for value, count in counts.items() if count > 1])


def index_unique(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    duplicates = duplicate_ids(rows, key)
    if duplicates:
        raise RuntimeError(f"Duplicate {key} in {name}: {duplicates[:5]} total={len(duplicates)}")
    return {str(row[key]): row for row in rows}


def assert_probe_gt_free(probe_rows: list[dict[str, Any]]) -> None:
    for row in probe_rows:
        assert_raw_result_gt_free(row)


def d2_final_prediction_audit(
    manifest_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_duplicates = duplicate_ids(manifest_rows, "intervention_id")
    probe_duplicates = duplicate_ids(probe_rows, "intervention_id")
    manifest_by_id = index_unique(manifest_rows, "intervention_id", "manifest") if not manifest_duplicates else {}
    generation_valid_ids = {
        intervention_id
        for intervention_id, row in manifest_by_id.items()
        if row.get("generation_valid")
    }
    generation_invalid_ids = set(manifest_by_id) - generation_valid_ids
    probe_ids = {str(row.get("intervention_id")) for row in probe_rows}
    error_ids = {str(row.get("intervention_id")) for row in (error_rows or [])}
    prediction_values = Counter(row.get("parsed_prediction") for row in probe_rows)
    invalid_values = sorted(value for value in prediction_values if value not in {"YES", "NO"})
    unresolved_error_ids = sorted((error_ids & generation_valid_ids) - probe_ids)
    return {
        "n_total_interventions": len(manifest_rows),
        "n_generation_valid": len(generation_valid_ids),
        "n_generation_invalid": len(generation_invalid_ids),
        "n_final_predictions": len(probe_rows),
        "missing_generation_valid_predictions": len(generation_valid_ids - probe_ids),
        "extra_predictions": len(probe_ids - generation_valid_ids),
        "predictions_for_generation_invalid": len(probe_ids & generation_invalid_ids),
        "duplicate_prediction_intervention_ids": len(probe_duplicates),
        "duplicate_manifest_intervention_ids": len(manifest_duplicates),
        "prediction_counts": dict(prediction_values),
        "prediction_values_only_yes_no": not invalid_values,
        "invalid_prediction_values": invalid_values,
        "historical_error_records": len(error_rows or []),
        "historical_errors_resolved_by_final_result": len(error_ids & probe_ids),
        "unresolved_errors": len(unresolved_error_ids),
        "unresolved_error_ids": unresolved_error_ids[:20],
    }


def joined_record(manifest: dict[str, Any], probe: dict[str, Any] | None) -> dict[str, Any]:
    if probe is None:
        prediction_fields = {
            "raw_response": "",
            "parsed_prediction": "NOT_RUN_GENERATION_INVALID",
            "model_name": None,
            "model_revision": None,
            "model_identity_hash": None,
            "prompt_version": None,
            "prompt_hash": None,
            "decoding_config": None,
            "cache_key": None,
        }
    else:
        prediction_fields = {
            "raw_response": probe.get("raw_response", ""),
            "parsed_prediction": probe.get("parsed_prediction"),
            "model_name": probe.get("model_name"),
            "model_revision": probe.get("model_revision"),
            "model_identity_hash": probe.get("model_identity_hash"),
            "prompt_version": probe.get("prompt_version"),
            "prompt_hash": probe.get("prompt_hash"),
            "decoding_config": probe.get("decoding_config"),
            "cache_key": probe.get("cache_key"),
        }
    row = {
        "candidate_id": manifest.get("candidate_id") or manifest["window_id"],
        "qa_id": manifest["qa_id"],
        "clip_id": manifest["clip_id"],
        "window_id": manifest["window_id"],
        "intervention_id": manifest["intervention_id"],
        "dataset_name": manifest["dataset_name"],
        "target_action": manifest.get("target_action"),
        "target_field": manifest.get("target_field"),
        "candidate_type": manifest.get("original_candidate_label"),
        "original_candidate_label": manifest.get("original_candidate_label"),
        "original_candidate_prediction": manifest.get("original_candidate_prediction") or manifest.get("parsed_prediction"),
        "gt_duration_total": manifest.get("gt_duration_total"),
        "gt_duration_bin": manifest.get("gt_duration_bin"),
        "intervention_family": manifest.get("intervention_family"),
        "intervention_type": manifest.get("intervention_type"),
        "generation_valid": bool(manifest.get("generation_valid")),
        "n_frames": manifest.get("intervened_n_frames"),
        "intervened_n_frames": manifest.get("intervened_n_frames"),
        "intervened_n_unique_frames": manifest.get("intervened_n_unique_frames"),
        "boundary_asymmetric": bool(manifest.get("boundary_asymmetric")),
        "original_gt_alignment_class": manifest.get("original_gt_alignment_class"),
        "intervention_gt_alignment_class": manifest.get("intervention_gt_alignment_class"),
        "strict_valid": bool(manifest.get("strict_valid")),
        "any_gt_valid": bool(manifest.get("any_gt_valid")),
        "validity_reason": manifest.get("validity_reason"),
        "original_evidence_density": manifest.get("original_evidence_density"),
        "intervention_evidence_density": manifest.get("intervention_evidence_density"),
        "original_gt_evidence_recall": manifest.get("original_gt_evidence_recall"),
        "intervention_gt_evidence_recall": manifest.get("intervention_gt_evidence_recall"),
        "original_n_gt_visible": manifest.get("original_n_gt_visible"),
        "intervention_n_gt_visible": manifest.get("intervention_n_gt_visible"),
        "intervention_prediction": prediction_fields["parsed_prediction"],
        **prediction_fields,
    }
    return {field: row.get(field) for field in JOINED_OUTPUT_FIELDS}


def join_intervention_results(
    manifest_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]] | None = None,
    expected_generation_valid: int | None = None,
    require_complete: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_duplicates = duplicate_ids(manifest_rows, "intervention_id")
    if manifest_duplicates:
        raise RuntimeError(f"Duplicate intervention_id in manifest: {manifest_duplicates[:5]} total={len(manifest_duplicates)}")
    assert_probe_gt_free(probe_rows)
    probe_duplicates = duplicate_ids(probe_rows, "intervention_id")
    if probe_duplicates:
        raise RuntimeError(f"Duplicate intervention_id in probe results: {probe_duplicates[:5]} total={len(probe_duplicates)}")

    manifest_by_id = index_unique(manifest_rows, "intervention_id", "manifest")
    probe_by_id = index_unique(probe_rows, "intervention_id", "probe")
    generation_valid_ids = {
        intervention_id
        for intervention_id, row in manifest_by_id.items()
        if row.get("generation_valid")
    }
    generation_invalid_ids = set(manifest_by_id) - generation_valid_ids
    probe_ids = set(probe_by_id)
    missing = sorted(generation_valid_ids - probe_ids)
    extra = sorted(probe_ids - set(manifest_by_id))
    predicted_for_generation_invalid = sorted(probe_ids & generation_invalid_ids)
    final_audit = d2_final_prediction_audit(manifest_rows, probe_rows, error_rows)
    if expected_generation_valid is not None and len(generation_valid_ids) != expected_generation_valid:
        raise RuntimeError(
            f"generation-valid manifest count mismatch: got {len(generation_valid_ids)}, expected {expected_generation_valid}"
        )
    if require_complete and missing:
        raise RuntimeError(f"Missing generation-valid predictions: {missing[:5]} total={len(missing)}")
    if require_complete and extra:
        raise RuntimeError(f"Extra predictions without manifest rows: {extra[:5]} total={len(extra)}")
    if require_complete and predicted_for_generation_invalid:
        raise RuntimeError(
            "Generation-invalid interventions should not have model predictions: "
            f"{predicted_for_generation_invalid[:5]} total={len(predicted_for_generation_invalid)}"
        )

    joined = []
    for intervention_id, manifest in sorted(manifest_by_id.items()):
        joined.append(joined_record(manifest, probe_by_id.get(intervention_id)))

    labels = Counter(row["original_candidate_label"] for row in manifest_rows)
    candidate_ids_by_label: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        if row.get("original_candidate_label") in CORE_CANDIDATE_LABELS:
            candidate_ids_by_label[row["original_candidate_label"]].add(str(row.get("candidate_id") or row["window_id"]))
    by_type = []
    for intervention_type in sorted({row.get("intervention_type") for row in manifest_rows}):
        sub = [row for row in joined if row["intervention_type"] == intervention_type]
        pred_counts = Counter(row["parsed_prediction"] for row in sub if row["generation_valid"])
        by_type.append(
            {
                "intervention_type": intervention_type,
                "attempted": len(sub),
                "generation_valid": sum(row["generation_valid"] for row in sub),
                "strict_valid": sum(row["strict_valid"] for row in sub if row["generation_valid"]),
                "strict_invalid": sum(row["generation_valid"] and not row["strict_valid"] for row in sub),
                "predicted_YES": pred_counts["YES"],
                "predicted_NO": pred_counts["NO"],
                "predicted_INVALID": pred_counts["INVALID"],
                "predicted_ERROR": pred_counts["ERROR"],
            }
        )

    counts_by_label_and_type = []
    for label in sorted(CORE_CANDIDATE_LABELS):
        for intervention_type in sorted({row.get("intervention_type") for row in manifest_rows}):
            sub = [
                row for row in joined
                if row["candidate_type"] == label and row["intervention_type"] == intervention_type
            ]
            valid = [row for row in sub if row["generation_valid"]]
            strict = [row for row in valid if row["strict_valid"]]
            pred_counts = Counter(row["intervention_prediction"] for row in valid)
            counts_by_label_and_type.append(
                {
                    "candidate_type": label,
                    "intervention_type": intervention_type,
                    "generation_valid": len(valid),
                    "strict_valid": len(strict),
                    "strict_invalid_after_generation_valid": len(valid) - len(strict),
                    "YES": pred_counts["YES"],
                    "NO": pred_counts["NO"],
                    "INVALID": pred_counts["INVALID"],
                    "ERROR": pred_counts["ERROR"],
                }
            )

    summary = {
        "n_manifest_total": len(manifest_rows),
        "n_generation_valid_manifest": len(generation_valid_ids),
        "n_generation_invalid_manifest": len(generation_invalid_ids),
        "n_probe_results": len(probe_rows),
        "n_matched": len(probe_ids & set(manifest_by_id)),
        "n_missing_generation_valid_predictions": len(missing),
        "n_extra_predictions": len(extra),
        "n_predictions_for_generation_invalid_manifest": len(predicted_for_generation_invalid),
        "n_duplicate_intervention_ids_manifest": 0,
        "n_duplicate_intervention_ids_probe": 0,
        "candidate_counts": {label: len(ids) for label, ids in sorted(candidate_ids_by_label.items())},
        "n_true_candidates": len(candidate_ids_by_label["TRUE_SUPPORT"]),
        "n_spurious_candidates": len(candidate_ids_by_label["SPURIOUS_SUPPORT"]),
        "strict_valid_total": sum(row["generation_valid"] and row["strict_valid"] for row in joined),
        "strict_invalid_after_generation_valid_total": sum(row["generation_valid"] and not row["strict_valid"] for row in joined),
        "counts_by_candidate_type_and_intervention_type": counts_by_label_and_type,
        "d2_final_prediction_audit": final_audit,
        "historical_error_records": final_audit["historical_error_records"],
        "historical_errors_resolved_by_final_result": final_audit["historical_errors_resolved_by_final_result"],
        "unresolved_errors": final_audit["unresolved_errors"],
        "n_true_origin_interventions": labels["TRUE_SUPPORT"],
        "n_spurious_origin_interventions": labels["SPURIOUS_SUPPORT"],
        "by_intervention_type": by_type,
    }
    return joined, summary


def paired_qa_ids(rows: list[dict[str, Any]]) -> set[str]:
    by_qa: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = row.get("original_candidate_label")
        if label in CORE_CANDIDATE_LABELS:
            by_qa[row["qa_id"]].add(label)
    return {qa_id for qa_id, labels in by_qa.items() if labels == CORE_CANDIDATE_LABELS}
