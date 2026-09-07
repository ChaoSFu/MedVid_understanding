from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from evidence_stability.cache import stable_hash
from evidence_stability.prompts import PROMPT_VERSION, build_evidence_presence_prompt


PROTOCOL_VERSION = "h3_dynamic_v1"
STUDY_STAGE = "h3_dynamic_discovery"
GLOBAL_SEED = 3407
INTERVENTION_TYPES = ["FULL_SHUFFLE_V1", "BLOCK_SHUFFLE_4X4_V1", "FREEZE_MID_V1"]
PRIMARY_INTERVENTIONS = ["FULL_SHUFFLE_V1", "BLOCK_SHUFFLE_4X4_V1"]
SECONDARY_INTERVENTION = "FREEZE_MID_V1"
BLOCK_ORDER = [2, 0, 3, 1]
BLOCK_SIZE = 4
FREEZE_MID_IDX = 7
CORE_CANDIDATE_TYPES = ["TRUE_SUPPORT", "SPURIOUS_SUPPORT"]
BINARY_PREDICTIONS = {"YES", "NO"}

FORBIDDEN_PROMPT_TERMS = [
    "freeze",
    "shuffle",
    "block_shuffle",
    "intervention",
    "dynamic",
    "gt",
    "true_support",
    "spurious_support",
]

MODEL_MANIFEST_FORBIDDEN_FIELDS = {
    "candidate_type",
    "gt_span",
    "gt_spans",
    "gt_alignment_class",
    "evidence_density",
    "gt_evidence_recall",
    "h2_stability",
    "strict_valid",
    "previous_h2_score",
    "candidate_label",
    "original_candidate_label",
}


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def nan() -> float:
    return float("nan")


def clean_mean(values: list[Any]) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
    return mean(clean) if clean else nan()


def clean_median(values: list[Any]) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
    return median(clean) if clean else nan()


def stable_int_seed(payload: dict[str, Any]) -> int:
    digest = stable_hash(payload)
    return int(digest[:16], 16)


def ordered_frame_hash(paths: list[str]) -> str:
    return stable_hash({"ordered_frame_paths": list(paths)})


def multiset_preserved(left: list[str], right: list[str]) -> bool:
    return Counter(left) == Counter(right)


def adjacent_pairs_preserved(permutation: list[int]) -> int:
    return sum(1 for a, b in zip(permutation, permutation[1:]) if b == a + 1)


def make_full_shuffle_permutation(
    candidate_id: str,
    n_frames: int = 16,
    global_seed: int = GLOBAL_SEED,
    max_retries: int = 100,
) -> dict[str, Any]:
    if n_frames != 16:
        return {
            "generation_valid": False,
            "generation_reason": f"FULL_SHUFFLE_EXPECTED_16_FRAMES_GOT_{n_frames}",
        }
    original = list(range(n_frames))
    base_seed = stable_int_seed(
        {
            "protocol": "H3_FULL_SHUFFLE_V1",
            "candidate_id": candidate_id,
            "global_seed": global_seed,
        }
    )
    for retry_index in range(max_retries + 1):
        seed = base_seed + retry_index
        permutation = list(original)
        random.Random(seed).shuffle(permutation)
        n_changed = sum(a != b for a, b in zip(original, permutation))
        n_adjacent = adjacent_pairs_preserved(permutation)
        if permutation != original and n_changed >= 12 and (15 - n_adjacent) >= 8:
            return {
                "generation_valid": True,
                "generation_reason": "OK",
                "original_positions": original,
                "shuffled_positions": permutation,
                "permutation": permutation,
                "seed": seed,
                "retry_index": retry_index,
                "n_positions_changed": n_changed,
                "n_original_adjacent_pairs_preserved": n_adjacent,
            }
    return {
        "generation_valid": False,
        "generation_reason": "FULL_SHUFFLE_CONSTRAINTS_UNSATISFIED",
        "seed": base_seed,
        "max_retries": max_retries,
    }


def apply_block_shuffle(paths: list[str]) -> tuple[list[str], dict[str, Any]]:
    if len(paths) != 16:
        return [], {
            "generation_valid": False,
            "generation_reason": f"BLOCK_SHUFFLE_EXPECTED_16_FRAMES_GOT_{len(paths)}",
            "block_size": BLOCK_SIZE,
            "block_order": BLOCK_ORDER,
        }
    blocks = [paths[i : i + BLOCK_SIZE] for i in range(0, 16, BLOCK_SIZE)]
    shuffled = [frame for block_index in BLOCK_ORDER for frame in blocks[block_index]]
    return shuffled, {
        "generation_valid": True,
        "generation_reason": "OK",
        "block_size": BLOCK_SIZE,
        "block_order": BLOCK_ORDER,
    }


def apply_freeze_mid(paths: list[str]) -> tuple[list[str], dict[str, Any]]:
    if len(paths) != 16:
        return [], {
            "generation_valid": False,
            "generation_reason": f"FREEZE_EXPECTED_16_FRAMES_GOT_{len(paths)}",
            "mid_idx": FREEZE_MID_IDX,
        }
    frozen = [paths[FREEZE_MID_IDX] for _ in paths]
    return frozen, {
        "generation_valid": True,
        "generation_reason": "OK",
        "mid_idx": FREEZE_MID_IDX,
        "n_repeated_logical_entries": len(frozen),
    }


def generate_dynamic_interventions_for_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    original_paths = list(candidate["ordered_frame_paths"])
    out = []
    for intervention_type in INTERVENTION_TYPES:
        if intervention_type == "FULL_SHUFFLE_V1":
            provenance = make_full_shuffle_permutation(candidate["candidate_id"], len(original_paths))
            paths = [original_paths[i] for i in provenance.get("permutation", [])] if provenance["generation_valid"] else []
        elif intervention_type == "BLOCK_SHUFFLE_4X4_V1":
            paths, provenance = apply_block_shuffle(original_paths)
        elif intervention_type == "FREEZE_MID_V1":
            paths, provenance = apply_freeze_mid(original_paths)
        else:
            raise ValueError(f"Unsupported H3 intervention type: {intervention_type}")

        generation_valid = bool(provenance["generation_valid"])
        checks = {
            "same_length": len(paths) == len(original_paths) if generation_valid else False,
            "same_frame_multiset": multiset_preserved(original_paths, paths) if generation_valid else False,
            "same_duplicate_multiplicity": multiset_preserved(original_paths, paths) if generation_valid else False,
            "no_new_or_dropped_frames": multiset_preserved(original_paths, paths) if generation_valid else False,
        }
        if intervention_type == "FREEZE_MID_V1" and generation_valid:
            checks["freeze_mid_idx_is_7"] = provenance["mid_idx"] == FREEZE_MID_IDX
            checks["sixteen_repeated_logical_entries"] = len(paths) == 16 and len(set(paths)) == 1
        if intervention_type == "BLOCK_SHUFFLE_4X4_V1" and generation_valid:
            checks["block_order_matches"] = provenance["block_order"] == BLOCK_ORDER
            checks["within_block_order_preserved"] = all(
                paths[out_start : out_start + BLOCK_SIZE] == original_paths[src_start : src_start + BLOCK_SIZE]
                for out_start, src_start in zip(range(0, 16, BLOCK_SIZE), [i * BLOCK_SIZE for i in BLOCK_ORDER])
            )
        if intervention_type == "FULL_SHUFFLE_V1" and generation_valid:
            checks["ordering_changed"] = provenance["permutation"] != list(range(16))
            checks["at_least_12_positions_changed"] = provenance["n_positions_changed"] >= 12

        out.append(
            {
                "intervention_id": f"{candidate['candidate_id']}::{intervention_type}",
                "candidate_id": candidate["candidate_id"],
                "qa_id": candidate["qa_id"],
                "clip_id": candidate["clip_id"],
                "window_id": candidate["window_id"],
                "dataset_name": candidate["dataset_name"],
                "target_action": candidate["target_action"],
                "target_field": candidate.get("target_field"),
                "intervention_type": intervention_type,
                "intervention_family": "ORDER" if intervention_type in PRIMARY_INTERVENTIONS else "FREEZE",
                "ordered_frame_paths": paths,
                "n_frames": len(paths),
                "n_unique_frames": len(set(paths)),
                "prompt_version": PROMPT_VERSION,
                "generation_valid": generation_valid,
                "generation_reason": provenance["generation_reason"],
                "gt_information_used_in_model_inference": False,
                "intervention_provenance": {
                    "protocol_version": PROTOCOL_VERSION,
                    "global_seed": GLOBAL_SEED,
                    "original_ordered_frame_hash": ordered_frame_hash(original_paths),
                    "intervened_ordered_frame_hash": ordered_frame_hash(paths) if generation_valid else None,
                    "checks": checks,
                    **provenance,
                },
            }
        )
    return out


def assert_dynamic_intervention_valid(row: dict[str, Any], original_paths: list[str]) -> None:
    if not row.get("generation_valid"):
        return
    paths = list(row["ordered_frame_paths"])
    if len(paths) != 16 or int(row["n_frames"]) != 16:
        raise RuntimeError(f"H3 intervention frame length mismatch: {row['intervention_id']}")
    if row["intervention_type"] in {"FULL_SHUFFLE_V1", "BLOCK_SHUFFLE_4X4_V1"}:
        if not multiset_preserved(original_paths, paths):
            raise RuntimeError(f"H3 order intervention changed frame multiset: {row['intervention_id']}")
    if row["intervention_type"] == "FULL_SHUFFLE_V1":
        provenance = row["intervention_provenance"]
        if provenance["permutation"] == list(range(16)):
            raise RuntimeError(f"FULL_SHUFFLE identity permutation: {row['intervention_id']}")
        if provenance["n_positions_changed"] < 12:
            raise RuntimeError(f"FULL_SHUFFLE changed too few positions: {row['intervention_id']}")
    if row["intervention_type"] == "BLOCK_SHUFFLE_4X4_V1":
        provenance = row["intervention_provenance"]
        if provenance["block_order"] != BLOCK_ORDER:
            raise RuntimeError(f"BLOCK_SHUFFLE block order changed: {row['intervention_id']}")
    if row["intervention_type"] == "FREEZE_MID_V1":
        if paths != [original_paths[FREEZE_MID_IDX]] * 16:
            raise RuntimeError(f"FREEZE_MID did not repeat original frame 7: {row['intervention_id']}")


def assert_model_manifest_gt_free(row: dict[str, Any], require_declaration: bool = True) -> None:
    illegal = [field for field in MODEL_MANIFEST_FORBIDDEN_FIELDS if field in row]
    if illegal:
        raise RuntimeError(f"GT/candidate-label field leaked into H3 model manifest: {illegal}")
    if require_declaration and row.get("gt_information_used_in_model_inference") is not False:
        raise RuntimeError(f"H3 manifest must declare GT-free inference: {row.get('intervention_id')}")


def prompt_hash_for_target(target_action: str) -> str:
    prompt = build_evidence_presence_prompt(target_action)
    lowered = prompt.lower()
    forbidden = [term for term in FORBIDDEN_PROMPT_TERMS if term in lowered]
    if forbidden:
        raise RuntimeError(f"Forbidden H3 intervention/GT term in frozen prompt: {forbidden}")
    return stable_hash({"prompt": prompt})


def reconstruct_h3_candidates(
    phase_c_labeled_rows: list[dict[str, Any]],
    paired_qa_rows: list[dict[str, Any]],
    *,
    expected_total: int | None = 158,
    expected_true: int | None = 64,
    expected_spurious: int | None = 94,
    expected_primary_action_qa: int | None = 9,
    expected_all_paired_qa: int | None = 16,
    expected_phase_only_qa: int | None = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_window = {str(row["window_id"]): row for row in phase_c_labeled_rows}
    candidates: list[dict[str, Any]] = []
    paired_ids = {str(row["qa_id"]) for row in paired_qa_rows}
    for qa in paired_qa_rows:
        for candidate_type, key in (
            ("TRUE_SUPPORT", "true_support_window_ids"),
            ("SPURIOUS_SUPPORT", "spurious_support_window_ids"),
        ):
            for window_id in qa.get(key, []):
                source = by_window.get(str(window_id))
                if source is None:
                    raise RuntimeError(f"Frozen Phase C window missing for H3 candidate: {window_id}")
                paths = list(source.get("frame_paths") or source.get("source_frame_paths") or [])
                row = {
                    "candidate_id": str(window_id),
                    "qa_id": str(source["qa_id"]),
                    "clip_id": str(source["clip_id"]),
                    "window_id": str(window_id),
                    "dataset_name": source.get("dataset_name"),
                    "target_action": source.get("target_action"),
                    "target_field": source.get("target_field"),
                    "ordered_frame_paths": paths,
                    "n_frames": len(paths),
                    "ordered_frame_hash": ordered_frame_hash(paths),
                    "original_prediction": source.get("parsed_prediction"),
                    "prompt_version": source.get("prompt_version", PROMPT_VERSION),
                    "prompt_hash": source.get("prompt_hash") or prompt_hash_for_target(str(source.get("target_action"))),
                    "candidate_type": candidate_type,
                    "source_phase_c_cache_key": source.get("cache_key"),
                }
                if row["qa_id"] != str(qa["qa_id"]):
                    raise RuntimeError(f"QA mismatch for H3 candidate {window_id}")
                candidates.append(row)

    ids = [row["candidate_id"] for row in candidates]
    duplicate_ids = [value for value, count in Counter(ids).items() if count > 1]
    label_counts = Counter(row["candidate_type"] for row in candidates)
    prediction_counts = Counter(row.get("original_prediction") for row in candidates)
    frame_count_counts = Counter(row.get("n_frames") for row in candidates)
    target_counts = Counter(row.get("target_field") for row in paired_qa_rows)
    audit = {
        "n_total_candidates": len(candidates),
        "candidate_type_counts": dict(label_counts),
        "original_prediction_counts": dict(prediction_counts),
        "frame_count_counts": dict(frame_count_counts),
        "n_duplicate_candidate_ids": len(duplicate_ids),
        "duplicate_candidate_ids": duplicate_ids[:20],
        "n_all_paired_qa": len(paired_ids),
        "n_primary_action_qa": sum(1 for row in paired_qa_rows if row.get("target_field") == "action"),
        "n_phase_only_qa": sum(1 for row in paired_qa_rows if row.get("target_field") == "phase"),
        "paired_qa_target_field_counts": dict(target_counts),
        "dataset_counts": dict(Counter(row.get("dataset_name") for row in candidates)),
    }
    failures = []
    if expected_total is not None and len(candidates) != expected_total:
        failures.append(f"total={len(candidates)} expected={expected_total}")
    if expected_true is not None and label_counts["TRUE_SUPPORT"] != expected_true:
        failures.append(f"TRUE={label_counts['TRUE_SUPPORT']} expected={expected_true}")
    if expected_spurious is not None and label_counts["SPURIOUS_SUPPORT"] != expected_spurious:
        failures.append(f"SPURIOUS={label_counts['SPURIOUS_SUPPORT']} expected={expected_spurious}")
    if any(prediction != "YES" for prediction in prediction_counts):
        failures.append(f"original predictions not all YES: {dict(prediction_counts)}")
    if duplicate_ids:
        failures.append(f"duplicate candidate IDs: {duplicate_ids[:5]}")
    if any(frame_count != 16 for frame_count in frame_count_counts):
        failures.append(f"frame counts not all 16: {dict(frame_count_counts)}")
    if expected_primary_action_qa is not None and audit["n_primary_action_qa"] != expected_primary_action_qa:
        failures.append(f"primary action QA={audit['n_primary_action_qa']} expected={expected_primary_action_qa}")
    if expected_all_paired_qa is not None and audit["n_all_paired_qa"] != expected_all_paired_qa:
        failures.append(f"all paired QA={audit['n_all_paired_qa']} expected={expected_all_paired_qa}")
    if expected_phase_only_qa is not None and audit["n_phase_only_qa"] != expected_phase_only_qa:
        failures.append(f"phase-only QA={audit['n_phase_only_qa']} expected={expected_phase_only_qa}")
    if failures:
        audit["failures"] = failures
        raise RuntimeError(f"H3 frozen candidate reconstruction failed: {failures}")
    audit["failures"] = []
    return candidates, audit


def deterministic_smoke_candidate_ids(candidates: list[dict[str, Any]], n: int = 12, seed: int = GLOBAL_SEED) -> list[str]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_dataset[str(row["dataset_name"])].append(row)

    def key(row: dict[str, Any]) -> str:
        return stable_hash({"seed": seed, "dataset_name": row.get("dataset_name"), "candidate_id": row["candidate_id"]})

    selected: list[str] = []
    for dataset in sorted(by_dataset):
        first = sorted(by_dataset[dataset], key=key)[0]
        selected.append(first["candidate_id"])
    for row in sorted(candidates, key=key):
        if len(selected) >= n:
            break
        if row["candidate_id"] not in selected:
            selected.append(row["candidate_id"])
    return selected[:n]


def dynamic_sensitivity(prediction: str | None) -> float:
    if prediction == "YES":
        return 0.0
    if prediction == "NO":
        return 1.0
    return nan()


def candidate_dynamic_rows(joined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        by_candidate[str(row["candidate_id"])].append(row)
    out = []
    for candidate_id, rows in sorted(by_candidate.items()):
        first = rows[0]
        by_type = {row["intervention_type"]: row for row in rows}
        pred_full = by_type.get("FULL_SHUFFLE_V1", {}).get("prediction")
        pred_block = by_type.get("BLOCK_SHUFFLE_4X4_V1", {}).get("prediction")
        pred_freeze = by_type.get("FREEZE_MID_V1", {}).get("prediction")
        d_full = dynamic_sensitivity(pred_full)
        d_block = dynamic_sensitivity(pred_block)
        d_freeze = dynamic_sensitivity(pred_freeze)
        d_order = clean_mean([d_full, d_block])
        d_macro = clean_mean([d_order, d_freeze])
        out.append(
            {
                "candidate_id": candidate_id,
                "qa_id": first["qa_id"],
                "clip_id": first.get("clip_id"),
                "dataset_name": first.get("dataset_name"),
                "target_action": first.get("target_action"),
                "target_field": first.get("target_field"),
                "candidate_type": first.get("candidate_type"),
                "original_prediction": first.get("original_prediction"),
                "pred_full_shuffle": pred_full,
                "pred_block_shuffle": pred_block,
                "pred_freeze": pred_freeze,
                "D_full_shuffle": d_full,
                "D_block_shuffle": d_block,
                "D_order": d_order,
                "D_freeze": d_freeze,
                "D_macro_exploratory": d_macro,
                "n_valid_order": sum(not is_nan(v) for v in [d_full, d_block]),
                "n_dynamic_flips": sum(v == 1.0 for v in [d_full, d_block, d_freeze]),
                "intervention_ids": {row["intervention_type"]: row["intervention_id"] for row in rows},
            }
        )
    return out


def qa_dynamic_rows(candidate_rows: list[dict[str, Any]], paired_qa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired_meta = {str(row["qa_id"]): row for row in paired_qa_rows}
    by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_qa[str(row["qa_id"])].append(row)
    out = []
    for qa_id, rows in sorted(by_qa.items()):
        first = rows[0]
        true_rows = [row for row in rows if row.get("candidate_type") == "TRUE_SUPPORT"]
        spurious_rows = [row for row in rows if row.get("candidate_type") == "SPURIOUS_SUPPORT"]
        meta = paired_meta.get(qa_id, {})
        row = {
            "qa_id": qa_id,
            "dataset": first.get("dataset_name"),
            "dataset_name": first.get("dataset_name"),
            "target_action": first.get("target_action"),
            "target_field": first.get("target_field"),
            "n_true_candidates": len(true_rows),
            "n_spurious_candidates": len(spurious_rows),
        }
        for name, field in [
            ("full_shuffle", "D_full_shuffle"),
            ("block_shuffle", "D_block_shuffle"),
            ("order", "D_order"),
            ("freeze", "D_freeze"),
            ("macro", "D_macro_exploratory"),
        ]:
            true_value = clean_mean([r[field] for r in true_rows])
            spur_value = clean_mean([r[field] for r in spurious_rows])
            row[f"mean_true_D_{name}"] = true_value
            row[f"mean_spur_D_{name}"] = spur_value
            delta_field = f"delta_D_{name}" if name in {"order", "freeze", "macro"} else f"delta_{name}"
            row[delta_field] = true_value - spur_value if not is_nan(true_value) and not is_nan(spur_value) else nan()
        row["paired_h2_eligible"] = meta.get("paired_h2_eligible")
        out.append(row)
    return out


def cohort_summary(rows: list[dict[str, Any]], cohort_name: str, role: str) -> dict[str, Any]:
    values = [row.get("delta_D_order") for row in rows]
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
    summary = {
        "cohort": cohort_name,
        "role": role,
        "n_QA": len(rows),
        "delta_D_order_positive": sum(v > 0 for v in clean),
        "delta_D_order_zero": sum(v == 0 for v in clean),
        "delta_D_order_negative": sum(v < 0 for v in clean),
        "mean_delta_D_order": clean_mean(values),
        "median_delta_D_order": clean_median(values),
        "min_delta_D_order": min(clean) if clean else nan(),
        "max_delta_D_order": max(clean) if clean else nan(),
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
    }
    for label, field in [
        ("full_shuffle", "delta_full_shuffle"),
        ("block_shuffle", "delta_block_shuffle"),
        ("freeze", "delta_D_freeze"),
        ("macro_exploratory", "delta_D_macro"),
    ]:
        deltas = [row.get(field) for row in rows]
        summary[f"mean_{label}_delta"] = clean_mean(deltas)
        summary[f"median_{label}_delta"] = clean_median(deltas)
    return summary


def persistence_diagnostic_rows(joined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = {
        "overall": lambda row: "ALL",
        "dataset": lambda row: row.get("dataset_name"),
        "target_field": lambda row: row.get("target_field"),
        "intervention_type": lambda row: row.get("intervention_type"),
        "candidate_type": lambda row: row.get("candidate_type"),
        "dataset_x_intervention_type_x_candidate_type": lambda row: (
            f"{row.get('dataset_name')}::{row.get('intervention_type')}::{row.get('candidate_type')}"
        ),
    }
    out = []
    rows = [row for row in joined_rows if row.get("prediction") in BINARY_PREDICTIONS]
    for dimension, getter in specs.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(getter(row))].append(row)
        for group, sub in sorted(groups.items()):
            yes = sum(row["prediction"] == "YES" for row in sub)
            no = sum(row["prediction"] == "NO" for row in sub)
            out.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": len(sub),
                    "YES": yes,
                    "NO": no,
                    "P_YES": yes / len(sub) if sub else nan(),
                    "dynamic_sensitivity_rate": no / len(sub) if sub else nan(),
                }
            )
    return out


def protocol_provenance() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "study_stage": "discovery",
        "study_stage_full": STUDY_STAGE,
        "confirmation_status": "NOT_RUN",
        "current_candidate_pool": "reused frozen H2 discovery cohort",
        "source_candidate_protocol": "H2 frozen 158-candidate cohort",
        "independent_confirmation": False,
        "model": "Qwen3-VL-8B-Instruct",
        "prompt_version": PROMPT_VERSION,
        "primary_unit": "QA",
        "primary_target": "action",
        "primary_score": "D_order",
        "primary_interventions": PRIMARY_INTERVENTIONS,
        "secondary_intervention": SECONDARY_INTERVENTION,
        "exploratory_combined_score": "D_macro",
        "global_seed": GLOBAL_SEED,
        "block_order": BLOCK_ORDER,
        "freeze_mid_idx": FREEZE_MID_IDX,
        "no_training": True,
        "gt_used_in_model_inference": False,
        "formal_statistics_run": False,
    }


def final_records_by_intervention(
    result_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    expected_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_results = []
    for row in result_rows:
        intervention_id = str(row.get("intervention_id"))
        if intervention_id in by_id:
            duplicate_results.append(intervention_id)
        by_id[intervention_id] = row
    if duplicate_results:
        raise RuntimeError(f"Duplicate H3 intervention results: {duplicate_results[:5]}")
    final_errors = {}
    historical_errors = 0
    resolved = 0
    for row in error_rows:
        intervention_id = str(row.get("intervention_id"))
        historical_errors += 1
        if intervention_id in by_id:
            resolved += 1
        elif intervention_id in expected_ids:
            final_errors[intervention_id] = row
    final = {k: v for k, v in by_id.items() if k in expected_ids}
    missing = sorted(expected_ids - set(final) - set(final_errors))
    extra = sorted(set(by_id) - expected_ids)
    audit = {
        "expected": len(expected_ids),
        "n_result_rows": len(result_rows),
        "n_final_success_results": len(final),
        "historical_errors": historical_errors,
        "resolved_by_final_result": resolved,
        "unresolved_errors": len(final_errors),
        "missing": missing[:20],
        "extra": extra[:20],
        "duplicate": duplicate_results[:20],
    }
    return final, audit


def assert_no_intervention_terms_in_prompt(target_action: str) -> str:
    prompt = build_evidence_presence_prompt(target_action)
    lowered = prompt.lower()
    forbidden = [term for term in FORBIDDEN_PROMPT_TERMS if term in lowered]
    if forbidden:
        raise RuntimeError(f"Forbidden term in model prompt: {forbidden}")
    return prompt


def write_contact_sheet(
    original_paths: list[str],
    variants: dict[str, list[str]],
    output_path: str | Path,
    thumb_size: tuple[int, int] = (160, 120),
) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    all_rows = {"ORIGINAL": original_paths, **variants}
    for paths in all_rows.values():
        if any(not Path(path).exists() for path in paths):
            return False
    width = thumb_size[0] * 4
    label_h = 24
    height = (thumb_size[1] * 4 + label_h) * len(all_rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for label, paths in all_rows.items():
        draw.text((4, y + 4), label, fill="black")
        y += label_h
        for i, path in enumerate(paths):
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail(thumb_size)
                x0 = (i % 4) * thumb_size[0]
                y0 = y + (i // 4) * thumb_size[1]
                sheet.paste(img, (x0, y0))
                draw.text((x0 + 3, y0 + 3), str(i), fill="yellow")
        y += thumb_size[1] * 4
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return True
