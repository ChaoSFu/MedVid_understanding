from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from evidence_stability.phase_d2 import assert_raw_result_gt_free


CORE_CANDIDATE_LABELS = {"TRUE_SUPPORT", "SPURIOUS_SUPPORT"}
JOINED_OUTPUT_FIELDS = [
    "qa_id",
    "clip_id",
    "window_id",
    "intervention_id",
    "dataset_name",
    "target_action",
    "target_field",
    "original_candidate_label",
    "gt_duration_total",
    "gt_duration_bin",
    "intervention_family",
    "intervention_type",
    "generation_valid",
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


def normalize_error_rows(error_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in error_rows:
        item = {
            "qa_id": row.get("qa_id"),
            "clip_id": row.get("clip_id"),
            "window_id": row.get("window_id"),
            "intervention_id": row.get("intervention_id"),
            "dataset_name": row.get("dataset_name"),
            "model_name": row.get("model_name"),
            "model_revision": row.get("model_revision"),
            "prompt_version": row.get("prompt_version"),
            "cache_key": row.get("cache_key"),
            "raw_response": "",
            "parsed_prediction": "ERROR",
            "error_type": row.get("error_type"),
        }
        normalized.append(item)
    return normalized


def assert_probe_gt_free(probe_rows: list[dict[str, Any]]) -> None:
    for row in probe_rows:
        parsed = row.get("parsed_prediction")
        if parsed == "ERROR":
            reduced = {
                "qa_id": row.get("qa_id"),
                "clip_id": row.get("clip_id"),
                "window_id": row.get("window_id"),
                "intervention_id": row.get("intervention_id"),
                "dataset_name": row.get("dataset_name"),
                "target_action": row.get("target_action", ""),
                "intervention_family": row.get("intervention_family", ""),
                "intervention_type": row.get("intervention_type", ""),
                "model_name": row.get("model_name"),
                "model_revision": row.get("model_revision"),
                "model_identity_hash": row.get("model_identity_hash"),
                "prompt_version": row.get("prompt_version"),
                "prompt_hash": row.get("prompt_hash"),
                "decoding_config": row.get("decoding_config", {}),
                "ordered_frame_paths": row.get("ordered_frame_paths", []),
                "n_frames": row.get("n_frames", 0),
                "n_unique_frames": row.get("n_unique_frames", 0),
                "raw_response": row.get("raw_response", ""),
                "parsed_prediction": "ERROR",
                "cache_key": row.get("cache_key"),
                "created_at": row.get("created_at", ""),
            }
            assert_raw_result_gt_free(reduced)
        else:
            assert_raw_result_gt_free(row)


def merge_probe_and_error_rows(
    probe_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return list(probe_rows) + normalize_error_rows(error_rows or [])


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
        "qa_id": manifest["qa_id"],
        "clip_id": manifest["clip_id"],
        "window_id": manifest["window_id"],
        "intervention_id": manifest["intervention_id"],
        "dataset_name": manifest["dataset_name"],
        "target_action": manifest.get("target_action"),
        "target_field": manifest.get("target_field"),
        "original_candidate_label": manifest.get("original_candidate_label"),
        "gt_duration_total": manifest.get("gt_duration_total"),
        "gt_duration_bin": manifest.get("gt_duration_bin"),
        "intervention_family": manifest.get("intervention_family"),
        "intervention_type": manifest.get("intervention_type"),
        "generation_valid": bool(manifest.get("generation_valid")),
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
    probes = merge_probe_and_error_rows(probe_rows, error_rows)
    assert_probe_gt_free(probes)
    probe_duplicates = duplicate_ids(probes, "intervention_id")
    if probe_duplicates:
        raise RuntimeError(f"Duplicate intervention_id in probe results: {probe_duplicates[:5]} total={len(probe_duplicates)}")

    manifest_by_id = index_unique(manifest_rows, "intervention_id", "manifest")
    probe_by_id = index_unique(probes, "intervention_id", "probe")
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

    summary = {
        "n_manifest_total": len(manifest_rows),
        "n_generation_valid_manifest": len(generation_valid_ids),
        "n_generation_invalid_manifest": len(generation_invalid_ids),
        "n_probe_results": len(probes),
        "n_matched": len(probe_ids & set(manifest_by_id)),
        "n_missing_generation_valid_predictions": len(missing),
        "n_extra_predictions": len(extra),
        "n_predictions_for_generation_invalid_manifest": len(predicted_for_generation_invalid),
        "n_duplicate_intervention_ids_manifest": 0,
        "n_duplicate_intervention_ids_probe": 0,
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
