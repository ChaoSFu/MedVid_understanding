from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any


STABILITY_DEFINITION_VERSION = "stability_v1"
ANALYSIS_PROTOCOL_VERSION = "h2_protocol_v1"
INTERVENTION_TYPES = ["SHIFT_LEFT_2", "SHIFT_RIGHT_2", "RESAMPLE_75", "RESAMPLE_50", "CONTEXT_1P5X"]
FAMILY_TYPES = {
    "SHIFT": ["SHIFT_LEFT_2", "SHIFT_RIGHT_2"],
    "RESAMPLE": ["RESAMPLE_75", "RESAMPLE_50"],
    "CONTEXT": ["CONTEXT_1P5X"],
}
FAMILY_SCORE_FIELDS = {
    "SHIFT": "S_shift",
    "RESAMPLE": "S_resample",
    "CONTEXT": "S_context",
}
BINARY_PREDICTIONS = {"YES", "NO"}
CORE_CANDIDATE_LABELS = ["TRUE_SUPPORT", "SPURIOUS_SUPPORT"]
GT_DURATION_BINS = ["<2 sec", "2-4 sec", "4-10 sec", "10-30 sec", ">=30 sec", "UNKNOWN"]


def nan() -> float:
    return float("nan")


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def clean_mean(values: list[Any]) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
    return mean(clean) if clean else nan()


def clean_median(values: list[Any]) -> float:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not is_nan(v)]
    return median(clean) if clean else nan()


def score(n_yes: int, n_valid: int) -> float:
    return n_yes / n_valid if n_valid else nan()


def candidate_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["qa_id"]), str(row["window_id"])


def include_in_primary_denominator(row: dict[str, Any], validity_mode: str = "strict") -> bool:
    if not row.get("generation_valid"):
        return False
    label = row.get("original_candidate_label")
    if validity_mode == "strict":
        return bool(row.get("strict_valid"))
    if validity_mode == "any_gt":
        if label == "TRUE_SUPPORT":
            return bool(row.get("any_gt_valid"))
        if label == "SPURIOUS_SUPPORT":
            return bool(row.get("strict_valid"))
    raise ValueError(f"Unsupported validity_mode: {validity_mode}")


def binary_retention(row: dict[str, Any], validity_mode: str = "strict") -> int | None:
    if not include_in_primary_denominator(row, validity_mode):
        return None
    pred = row.get("parsed_prediction")
    if pred == "YES":
        return 1
    if pred == "NO":
        return 0
    return None


def family_stats(rows: list[dict[str, Any]], family: str, validity_mode: str = "strict") -> dict[str, Any]:
    sub = [row for row in rows if row.get("intervention_family") == family]
    denominator_rows = [row for row in sub if include_in_primary_denominator(row, validity_mode)]
    binary_values = [binary_retention(row, validity_mode) for row in denominator_rows]
    binary_values = [v for v in binary_values if v is not None]
    n_valid = len(binary_values)
    n_yes = sum(binary_values)
    n_missing = sum(row.get("parsed_prediction") not in BINARY_PREDICTIONS for row in denominator_rows)
    return {
        "n_valid": n_valid,
        "n_yes": n_yes,
        "score": score(n_yes, n_valid),
        "n_missing_binary_outputs": n_missing,
        "n_generation_invalid": sum(not row.get("generation_valid") for row in sub),
        "n_validity_invalid": sum(row.get("generation_valid") and not include_in_primary_denominator(row, validity_mode) for row in sub),
        "n_model_invalid": sum(
            include_in_primary_denominator(row, validity_mode) and row.get("parsed_prediction") == "INVALID"
            for row in sub
        ),
        "n_model_error": sum(
            include_in_primary_denominator(row, validity_mode) and row.get("parsed_prediction") == "ERROR"
            for row in sub
        ),
    }


def canonical_gt_duration_bin(row: dict[str, Any]) -> str:
    existing = row.get("gt_duration_bin")
    if existing:
        return str(existing)
    duration = row.get("gt_duration_total")
    if duration is None:
        return "UNKNOWN"
    duration = float(duration)
    if duration < 2:
        return "<2 sec"
    if duration < 4:
        return "2-4 sec"
    if duration < 10:
        return "4-10 sec"
    if duration < 30:
        return "10-30 sec"
    return ">=30 sec"


def candidate_stability_rows(
    joined_rows: list[dict[str, Any]],
    validity_mode: str = "strict",
    stable_hallucination_threshold: float | None = None,
    fragile_true_threshold: float | None = None,
) -> list[dict[str, Any]]:
    by_candidate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        by_candidate[candidate_key(row)].append(row)

    out = []
    for key, rows in sorted(by_candidate.items()):
        first = rows[0]
        family = {name: family_stats(rows, name, validity_mode) for name in FAMILY_TYPES}
        type_retention = {}
        for intervention_type in INTERVENTION_TYPES:
            matches = [row for row in rows if row.get("intervention_type") == intervention_type]
            type_retention[f"R_{intervention_type}"] = binary_retention(matches[0], validity_mode) if matches else nan()
            if type_retention[f"R_{intervention_type}"] is None:
                type_retention[f"R_{intervention_type}"] = nan()
        n_valid_total = sum(family[name]["n_valid"] for name in FAMILY_TYPES)
        n_yes_total = sum(family[name]["n_yes"] for name in FAMILY_TYPES)
        s_micro = score(n_yes_total, n_valid_total)
        family_scores = [family[name]["score"] for name in FAMILY_TYPES]
        s_macro = clean_mean(family_scores)
        label = first.get("original_candidate_label")
        stable_hallucination = None
        if stable_hallucination_threshold is not None and label == "SPURIOUS_SUPPORT" and not is_nan(s_macro):
            stable_hallucination = s_macro >= stable_hallucination_threshold
        fragile_true = None
        if fragile_true_threshold is not None and label == "TRUE_SUPPORT" and not is_nan(s_macro):
            fragile_true = s_macro <= fragile_true_threshold
        row = {
            "qa_id": first["qa_id"],
            "clip_id": first["clip_id"],
            "window_id": first["window_id"],
            "dataset_name": first.get("dataset_name"),
            "target_action": first.get("target_action"),
            "target_field": first.get("target_field"),
            "original_candidate_label": label,
            "gt_duration": first.get("gt_duration_total"),
            "gt_duration_bin": canonical_gt_duration_bin(first),
            "original_evidence_density": first.get("original_evidence_density"),
            "original_gt_evidence_recall": first.get("original_gt_evidence_recall"),
            "n_shift_valid": family["SHIFT"]["n_valid"],
            "n_shift_yes": family["SHIFT"]["n_yes"],
            "S_shift": family["SHIFT"]["score"],
            "n_resample_valid": family["RESAMPLE"]["n_valid"],
            "n_resample_yes": family["RESAMPLE"]["n_yes"],
            "S_resample": family["RESAMPLE"]["score"],
            "n_context_valid": family["CONTEXT"]["n_valid"],
            "n_context_yes": family["CONTEXT"]["n_yes"],
            "S_context": family["CONTEXT"]["score"],
            **type_retention,
            "n_valid_total": n_valid_total,
            "n_yes_total": n_yes_total,
            "S_overall_micro": s_micro,
            "S_overall_macro": s_macro,
            "n_missing_binary_outputs": sum(family[name]["n_missing_binary_outputs"] for name in FAMILY_TYPES),
            "stable_hallucination_candidate": stable_hallucination,
            "fragile_true_evidence": fragile_true,
        }
        out.append(row)
    return out


def qa_stability_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_qa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_qa[row["qa_id"]].append(row)
    out = []
    for qa_id, rows in sorted(by_qa.items()):
        first = rows[0]
        true_rows = [row for row in rows if row["original_candidate_label"] == "TRUE_SUPPORT"]
        spurious_rows = [row for row in rows if row["original_candidate_label"] == "SPURIOUS_SUPPORT"]
        record = {
            "qa_id": qa_id,
            "clip_id": first.get("clip_id"),
            "dataset_name": first.get("dataset_name"),
            "target_action": first.get("target_action"),
            "target_field": first.get("target_field"),
            "n_true_candidates": len(true_rows),
            "mean_S_true_macro": clean_mean([row["S_overall_macro"] for row in true_rows]),
            "median_S_true_macro": clean_median([row["S_overall_macro"] for row in true_rows]),
            "mean_S_true_micro": clean_mean([row["S_overall_micro"] for row in true_rows]),
            "median_S_true_micro": clean_median([row["S_overall_micro"] for row in true_rows]),
            "mean_S_true_shift": clean_mean([row["S_shift"] for row in true_rows]),
            "mean_S_true_resample": clean_mean([row["S_resample"] for row in true_rows]),
            "mean_S_true_context": clean_mean([row["S_context"] for row in true_rows]),
            "n_spurious_candidates": len(spurious_rows),
            "mean_S_spurious_macro": clean_mean([row["S_overall_macro"] for row in spurious_rows]),
            "median_S_spurious_macro": clean_median([row["S_overall_macro"] for row in spurious_rows]),
            "mean_S_spurious_micro": clean_mean([row["S_overall_micro"] for row in spurious_rows]),
            "median_S_spurious_micro": clean_median([row["S_overall_micro"] for row in spurious_rows]),
            "mean_S_spurious_shift": clean_mean([row["S_shift"] for row in spurious_rows]),
            "mean_S_spurious_resample": clean_mean([row["S_resample"] for row in spurious_rows]),
            "mean_S_spurious_context": clean_mean([row["S_context"] for row in spurious_rows]),
        }
        for suffix in ["macro", "micro", "shift", "resample", "context"]:
            left = record[f"mean_S_true_{suffix}"]
            right = record[f"mean_S_spurious_{suffix}"]
            record[f"delta_S_{suffix}_mean" if suffix in {"macro", "micro"} else f"delta_S_{suffix}"] = (
                left - right if not is_nan(left) and not is_nan(right) else nan()
            )
        record["paired_macro_usable"] = any(not is_nan(row["S_overall_macro"]) for row in true_rows) and any(
            not is_nan(row["S_overall_macro"]) for row in spurious_rows
        )
        record["paired_shift_analysis_usable"] = any(not is_nan(row["S_shift"]) for row in true_rows) and any(
            not is_nan(row["S_shift"]) for row in spurious_rows
        )
        record["paired_resample_analysis_usable"] = any(not is_nan(row["S_resample"]) for row in true_rows) and any(
            not is_nan(row["S_resample"]) for row in spurious_rows
        )
        record["paired_context_analysis_usable"] = any(not is_nan(row["S_context"]) for row in true_rows) and any(
            not is_nan(row["S_context"]) for row in spurious_rows
        )
        out.append(record)
    return out


def retention_matrix_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in candidate_rows:
        out.append(
            {
                "qa_id": row["qa_id"],
                "clip_id": row["clip_id"],
                "window_id": row["window_id"],
                "dataset_name": row["dataset_name"],
                "target_field": row["target_field"],
                "original_candidate_label": row["original_candidate_label"],
                **{name: row[f"R_{name}"] for name in INTERVENTION_TYPES},
            }
        )
    return out


def missingness_rows(joined_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], validity_mode: str = "strict") -> list[dict[str, Any]]:
    by_family_candidate: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidate_rows:
        for family, score_field in FAMILY_SCORE_FIELDS.items():
            by_family_candidate[(candidate["qa_id"], candidate["window_id"], family)] = {
                "family_nan": is_nan(candidate[score_field]),
                "no_valid_denominator": candidate[f"n_{family.lower()}_valid"] == 0,
            }
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        groups[
            (
                str(row.get("dataset_name")),
                str(row.get("target_field")),
                str(row.get("original_candidate_label")),
                str(row.get("intervention_family")),
            )
        ].append(row)
    out = []
    for (dataset, target_field, label, family), rows in sorted(groups.items()):
        candidate_keys = {(row["qa_id"], row["window_id"], family) for row in rows}
        family_status = [by_family_candidate[key] for key in candidate_keys if key in by_family_candidate]
        out.append(
            {
                "dataset_name": dataset,
                "target_field": target_field,
                "candidate_label": label,
                "intervention_family": family,
                "n_rows": len(rows),
                "generation_invalid": sum(not row.get("generation_valid") for row in rows),
                "strict_invalid": sum(row.get("generation_valid") and not row.get("strict_valid") for row in rows),
                "model_INVALID": sum(
                    include_in_primary_denominator(row, validity_mode) and row.get("parsed_prediction") == "INVALID"
                    for row in rows
                ),
                "model_ERROR": sum(
                    include_in_primary_denominator(row, validity_mode) and row.get("parsed_prediction") == "ERROR"
                    for row in rows
                ),
                "no_valid_denominator": sum(item["no_valid_denominator"] for item in family_status),
                "family_NaN": sum(item["family_nan"] for item in family_status),
            }
        )
    return out


def summarize_candidate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_QA": len({row["qa_id"] for row in rows}),
        "n_candidates": len(rows),
        "mean_S_shift": clean_mean([row["S_shift"] for row in rows]),
        "median_S_shift": clean_median([row["S_shift"] for row in rows]),
        "mean_S_resample": clean_mean([row["S_resample"] for row in rows]),
        "median_S_resample": clean_median([row["S_resample"] for row in rows]),
        "mean_S_context": clean_mean([row["S_context"] for row in rows]),
        "median_S_context": clean_median([row["S_context"] for row in rows]),
        "mean_S_overall_macro": clean_mean([row["S_overall_macro"] for row in rows]),
        "median_S_overall_macro": clean_median([row["S_overall_macro"] for row in rows]),
        "mean_S_overall_micro": clean_mean([row["S_overall_micro"] for row in rows]),
        "median_S_overall_micro": clean_median([row["S_overall_micro"] for row in rows]),
    }


def stability_by_group_rows(
    candidate_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    group_field: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(str(row.get(group_field)), str(row.get("original_candidate_label")))].append(row)
    usable_by_group = Counter()
    for qa in qa_rows:
        if qa.get("paired_macro_usable"):
            usable_by_group[str(qa.get(group_field))] += 1
    out = []
    for (group_value, label), rows in sorted(groups.items()):
        out.append(
            {
                group_field: group_value,
                "original_candidate_label": label,
                **summarize_candidate_group(rows),
                "paired_QA_usable_count": usable_by_group[group_value],
            }
        )
    return out


def phase_e_summary(
    candidate_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    validity_mode: str,
) -> dict[str, Any]:
    labels = Counter(row["original_candidate_label"] for row in candidate_rows)
    target_fields = Counter(row["target_field"] for row in qa_rows)
    return {
        "stability_definition_version": STABILITY_DEFINITION_VERSION,
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "validity_mode": validity_mode,
        "n_candidates": len(candidate_rows),
        "n_true_candidates": labels["TRUE_SUPPORT"],
        "n_spurious_candidates": labels["SPURIOUS_SUPPORT"],
        "n_qa": len(qa_rows),
        "n_action_qa": target_fields["action"],
        "n_phase_qa": target_fields["phase"],
        "paired_macro_usable_qa": sum(row["paired_macro_usable"] for row in qa_rows),
        "paired_shift_analysis_usable_qa": sum(row["paired_shift_analysis_usable"] for row in qa_rows),
        "paired_resample_analysis_usable_qa": sum(row["paired_resample_analysis_usable"] for row in qa_rows),
        "paired_context_analysis_usable_qa": sum(row["paired_context_analysis_usable"] for row in qa_rows),
        "primary_h2_analysis": {
            "target_type": "action",
            "primary_unit": "qa",
            "preferred_summary_score": "S_overall_macro",
            "candidate_level_analysis": "secondary_correlated_window_analysis",
            "inferential_statistics_run": False,
        },
    }


def analysis_protocol_snapshot(validity_mode: str) -> dict[str, Any]:
    return {
        "stability_definition_version": STABILITY_DEFINITION_VERSION,
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "primary_target_type": "action",
        "primary_validity_mode": validity_mode,
        "preferred_summary_score": "S_overall_macro",
        "also_report_micro": True,
        "family_scores": ["shift", "resample", "context"],
        "invalid_intervention_policy": "exclude",
        "model_invalid_policy": "exclude_and_report",
        "missing_family_policy": "nan",
        "primary_unit": "qa",
        "candidate_level_analysis": "secondary_correlated_window_analysis",
        "primary_rationale": (
            "Action-only paired QA is the cleaner localized evidence test because Phase C showed stronger "
            "separation between GT-aligned and no-GT-overlap windows for action targets, while phase targets "
            "had higher model-positive rates outside GT and target type is partially confounded with dataset."
        ),
        "inferential_statistics_run": False,
        "thresholds": {
            "stable_hallucination_candidate": None,
            "fragile_true_evidence": None,
        },
    }
