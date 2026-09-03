from __future__ import annotations

import re
import random
from collections import Counter, defaultdict
from typing import Any


FORBIDDEN_MODEL_SIDE_FIELDS = {
    "original_candidate_label",
    "candidate_label",
    "phase_c_label",
    "original_gt_alignment_class",
    "intervention_gt_alignment_class",
    "gt_alignment_class",
    "strict_valid",
    "any_gt_valid",
    "validity_reason",
    "original_evidence_density",
    "intervention_evidence_density",
    "evidence_density",
    "original_gt_evidence_recall",
    "intervention_gt_evidence_recall",
    "gt_evidence_recall",
    "original_n_gt_visible",
    "intervention_n_gt_visible",
    "intervention_n_unique_gt_visible",
    "n_gt_visible",
    "gt_spans",
    "gt_duration",
    "gt_duration_total",
    "gt_duration_bin",
    "WEAKENED_GT_SUPPORT",
    "GT_EVIDENCE_REMOVED",
    "GT_CONTAMINATED",
}

FORBIDDEN_FIELD_PREFIXES = (
    "gt_",
    "ground_truth",
    "evidence_density",
    "gt_evidence_recall",
)

FORBIDDEN_FIELD_EXACT = {
    "gt",
    "candidate_label",
    "original_candidate_label",
    "true_support",
    "spurious_support",
    "strict_valid",
    "any_gt_valid",
    "validity_reason",
    "gt_contaminated",
    "weakened_gt_support",
    "gt_evidence_removed",
}

RAW_RESULT_ALLOWED_FIELDS = {
    "qa_id",
    "clip_id",
    "window_id",
    "intervention_id",
    "dataset_name",
    "target_action",
    "intervention_family",
    "intervention_type",
    "model_name",
    "model_revision",
    "model_identity_hash",
    "prompt_version",
    "prompt_hash",
    "decoding_config",
    "ordered_frame_paths",
    "n_frames",
    "n_unique_frames",
    "raw_response",
    "parsed_prediction",
    "cache_key",
    "created_at",
    "processor_metadata",
}

PROCESSOR_METADATA_ALLOWED_FIELDS = {
    "expected_frame_count",
    "input_frame_count",
    "input_token_length",
    "input_ids_shape",
    "image_grid_thw",
    "image_grid_thw_shape",
    "image_grid_thw_value",
    "image_grid_thw_rows",
    "video_grid_thw",
    "video_grid_thw_shape",
    "video_grid_thw_value",
    "video_grid_thw_rows",
    "pixel_values_shape",
    "input_keys",
    "image_sizes",
}


def is_generation_valid_intervention(row: dict[str, Any]) -> bool:
    return bool(row.get("generation_valid"))


def select_generation_valid_interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if is_generation_valid_intervention(row)]


def project_intervention_for_model(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("generation_valid"):
        raise ValueError(f"Cannot project generation-invalid intervention: {row.get('intervention_id')}")
    projection = {
        "qa_id": row["qa_id"],
        "clip_id": row["clip_id"],
        "window_id": row["window_id"],
        "intervention_id": row["intervention_id"],
        "dataset_name": row["dataset_name"],
        "target_action": row["target_action"],
        "target_field": row.get("target_field"),
        "intervention_family": row["intervention_family"],
        "intervention_type": row["intervention_type"],
        "ordered_frame_paths": list(row["intervened_frame_paths"]),
        "n_frames": int(row["intervened_n_frames"]),
        "n_unique_frames": int(row.get("intervened_n_unique_frames", len(set(row["intervened_frame_paths"])))),
        "generation_valid": bool(row["generation_valid"]),
    }
    assert_no_forbidden_model_fields(projection, "model inference projection")
    if projection["n_frames"] != len(projection["ordered_frame_paths"]):
        raise ValueError(f"Frame count mismatch for {projection['intervention_id']}")
    return projection


def field_name_tokens(field_name: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", field_name.lower())
    return [part for part in parts if part]


def is_forbidden_field_name(field_name: str) -> bool:
    normalized = field_name.lower()
    leaf = re.split(r"[.\[]", normalized)[-1].rstrip("]")
    leaf_tokens = field_name_tokens(leaf)
    if leaf in FORBIDDEN_MODEL_SIDE_FIELDS or leaf in FORBIDDEN_FIELD_EXACT:
        return True
    if any(leaf.startswith(prefix) for prefix in FORBIDDEN_FIELD_PREFIXES):
        return True
    if "gt" in leaf_tokens:
        return True
    if "ground" in leaf_tokens and "truth" in leaf_tokens:
        return True
    if "candidate" in leaf_tokens and "label" in leaf_tokens:
        return True
    if "true" in leaf_tokens and "support" in leaf_tokens:
        return True
    if "spurious" in leaf_tokens and "support" in leaf_tokens:
        return True
    if "strict" in leaf_tokens and "valid" in leaf_tokens:
        return True
    if "any" in leaf_tokens and "gt" in leaf_tokens and "valid" in leaf_tokens:
        return True
    if "evidence" in leaf_tokens and "density" in leaf_tokens:
        return True
    if "gt" in leaf_tokens and "evidence" in leaf_tokens and "recall" in leaf_tokens:
        return True
    if "n" in leaf_tokens and "gt" in leaf_tokens and "visible" in leaf_tokens:
        return True
    if "visible" in leaf_tokens and "gt" in leaf_tokens:
        return True
    if "validity" in leaf_tokens and "reason" in leaf_tokens:
        return True
    if "contaminated" in leaf_tokens and "gt" in leaf_tokens:
        return True
    if "weakened" in leaf_tokens and "gt" in leaf_tokens and "support" in leaf_tokens:
        return True
    if "removed" in leaf_tokens and "gt" in leaf_tokens and "evidence" in leaf_tokens:
        return True
    return False


def assert_no_forbidden_model_fields(row: dict[str, Any], context: str) -> None:
    def iter_field_names(value: Any, prefix: str = ""):
        if isinstance(value, dict):
            for key, nested in value.items():
                field = f"{prefix}.{key}" if prefix else str(key)
                yield field
                yield from iter_field_names(nested, field)
        elif isinstance(value, list):
            for i, nested in enumerate(value):
                yield from iter_field_names(nested, f"{prefix}[{i}]")

    fields = list(iter_field_names(row))
    root_fields = {field.split(".", 1)[0].split("[", 1)[0] for field in fields}
    forbidden = sorted(root_fields & FORBIDDEN_MODEL_SIDE_FIELDS)
    token_hits = sorted(field for field in fields if is_forbidden_field_name(field))
    hits = sorted(set(forbidden + token_hits))
    if hits:
        raise RuntimeError(f"Forbidden GT/label-derived fields in {context}: {hits}")


def sanitize_processor_metadata(metadata: dict[str, Any] | None, expected_frame_count: int | None = None) -> dict[str, Any]:
    source = dict(metadata or {})
    if expected_frame_count is not None:
        source["expected_frame_count"] = int(expected_frame_count)
    sanitized = {
        key: value
        for key, value in source.items()
        if key in PROCESSOR_METADATA_ALLOWED_FIELDS
    }
    assert_no_forbidden_model_fields(sanitized, "processor metadata")
    disallowed_allowed_object_fields = sorted(set(sanitized) - PROCESSOR_METADATA_ALLOWED_FIELDS)
    if disallowed_allowed_object_fields:
        raise RuntimeError(f"Unexpected processor metadata fields: {disallowed_allowed_object_fields}")
    return sanitized


def assert_raw_result_gt_free(row: dict[str, Any]) -> None:
    disallowed = sorted(set(row) - RAW_RESULT_ALLOWED_FIELDS)
    if disallowed:
        raise RuntimeError(f"Unexpected raw intervention result fields: {disallowed}")
    processor_metadata = row.get("processor_metadata")
    if processor_metadata is not None:
        unexpected_processor_fields = sorted(set(processor_metadata) - PROCESSOR_METADATA_ALLOWED_FIELDS)
        if unexpected_processor_fields:
            raise RuntimeError(f"Unexpected processor metadata fields: {unexpected_processor_fields}")
    assert_no_forbidden_model_fields(row, "raw intervention result")


def frame_count_distribution(rows: list[dict[str, Any]]) -> dict[int, int]:
    return dict(sorted(Counter(int(row["n_frames"]) for row in rows).items()))


def deterministic_smoke_select(
    rows: list[dict[str, Any]],
    n: int = 12,
    seed: int = 42,
) -> list[dict[str, Any]]:
    valid = select_generation_valid_interventions(rows)
    if n < 1:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_first(predicate) -> None:
        candidates = [r for r in valid if r["intervention_id"] not in selected_ids and predicate(r)]
        candidates.sort(key=lambda r: (r.get("dataset_name", ""), r.get("qa_id", ""), r.get("intervention_id", "")))
        if candidates and len(selected) < n:
            row = candidates[0]
            selected.append(row)
            selected_ids.add(row["intervention_id"])

    for frame_count in (8, 12, 16, 24):
        add_first(lambda r, fc=frame_count: int(r.get("intervened_n_frames", 0)) == fc)
    for origin_label in ("TRUE_SUPPORT", "SPURIOUS_SUPPORT"):
        add_first(lambda r, label=origin_label: r.get("original_candidate_label") == label)
    for target_field in ("action", "phase"):
        add_first(lambda r, tf=target_field: r.get("target_field") == tf)
    for intervention_type in ("RESAMPLE_50", "RESAMPLE_75", "SHIFT_LEFT_2", "SHIFT_RIGHT_2", "CONTEXT_1P5X"):
        add_first(lambda r, it=intervention_type: r.get("intervention_type") == it)

    rng = random.Random(seed)
    remaining = [r for r in valid if r["intervention_id"] not in selected_ids]
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= n:
            break
        selected.append(row)
        selected_ids.add(row["intervention_id"])

    selected.sort(key=lambda r: r["intervention_id"])
    return selected


def smoke_selection_artifact(selected: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "n_selected_interventions": len(selected),
        "interventions": [
            {
                "intervention_id": row["intervention_id"],
                "intervention_type": row["intervention_type"],
                "intervention_family": row["intervention_family"],
                "dataset_name": row["dataset_name"],
                "target_field": row.get("target_field"),
                "n_frames": int(row["intervened_n_frames"]),
            }
            for row in selected
        ],
    }


def audit_cache_collisions(rows: list[dict[str, Any]], cache_keys: dict[str, str]) -> dict[str, Any]:
    sequence_by_id = {
        row["intervention_id"]: tuple(row["ordered_frame_paths"])
        for row in rows
    }
    reverse: dict[str, list[str]] = defaultdict(list)
    for intervention_id, cache_key in cache_keys.items():
        reverse[cache_key].append(intervention_id)
    collisions = []
    for cache_key, ids in reverse.items():
        if len(ids) < 2:
            continue
        sequences = {sequence_by_id[i] for i in ids}
        if len(sequences) > 1 or len(set(ids)) > 1:
            collisions.append({"cache_key": cache_key, "intervention_ids": ids})
    if collisions:
        raise RuntimeError(f"Intervention cache collisions detected: {collisions[:3]}")
    return {"cache_collisions": 0}


def memory_leak_flag(memory_rows: list[dict[str, Any]]) -> bool:
    values = [
        row.get("current_allocated_bytes")
        for row in memory_rows
        if isinstance(row.get("current_allocated_bytes"), int)
    ]
    if len(values) < 4:
        return False
    return all(later > earlier for earlier, later in zip(values, values[1:]))


def frame_count_processor_audit(debug_records: list[dict[str, Any]]) -> dict[int, bool | None]:
    out: dict[int, bool | None] = {}
    for record in debug_records:
        expected = int(record["expected_frame_count"])
        metadata = record.get("processor_metadata") or {}
        represented: bool | None = None
        for key in ("image_grid_thw_rows", "video_grid_thw_rows"):
            rows = metadata.get(key)
            if rows is not None:
                represented = int(rows) == expected
                break
        out[expected] = represented
    return out
