from __future__ import annotations

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

FORBIDDEN_FIELD_TOKENS = (
    "gt",
    "ground_truth",
    "alignment",
    "true_support",
    "spurious_support",
    "candidate_label",
    "strict_valid",
    "any_gt_valid",
    "evidence_density",
    "recall",
    "visible_gt",
    "contaminated",
    "validity_reason",
)


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
    token_hits = sorted(
        field
        for field in fields
        if any(token in field.lower() for token in FORBIDDEN_FIELD_TOKENS)
    )
    hits = sorted(set(forbidden + token_hits))
    if hits:
        raise RuntimeError(f"Forbidden GT/label-derived fields in {context}: {hits}")


def assert_raw_result_gt_free(row: dict[str, Any]) -> None:
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
