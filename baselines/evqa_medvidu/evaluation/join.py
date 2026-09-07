from __future__ import annotations

from collections import Counter
from typing import Any

from baselines.evqa_medvidu.manifest import canonical_task, extract_human_question, make_sample_id


def gt_index(data: list[dict[str, Any]], task: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, sample in enumerate(data):
        if not isinstance(sample, dict) or canonical_task(sample.get("qa_type")) != task:
            continue
        out[make_sample_id(sample, i, task)] = sample
    return out


def prediction_index(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    counts = Counter(str(row.get("sample_id")) for row in rows)
    duplicates = sorted(k for k, v in counts.items() if v > 1)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid not in out:
            out[sid] = row
    return out, duplicates


def join_audit(gt: dict[str, Any], pred: dict[str, Any], duplicates: list[str]) -> dict[str, Any]:
    gt_ids = set(gt)
    pred_ids = set(pred)
    return {
        "n_GT": len(gt_ids),
        "n_predictions": len(pred_ids),
        "matched": len(gt_ids & pred_ids),
        "missing": len(gt_ids - pred_ids),
        "extra": len(pred_ids - gt_ids),
        "duplicates": len(duplicates),
        "missing_preview": sorted(gt_ids - pred_ids)[:20],
        "extra_preview": sorted(pred_ids - gt_ids)[:20],
        "duplicate_preview": duplicates[:20],
    }


def leaderboard_record(sample_id: str, gt_sample: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "qa_type": gt_sample.get("qa_type"),
        "question": extract_human_question(gt_sample),
        "answer": prediction.get("answer", ""),
        "gnd": _assistant_answer(gt_sample),
        "struc_info": gt_sample.get("struc_info", []),
        "metadata": gt_sample.get("metadata", {}),
        "data_source": gt_sample.get("data_source"),
        "dataset_name": gt_sample.get("dataset_name"),
    }


def _assistant_answer(sample: dict[str, Any]) -> str:
    for turn in sample.get("conversations") or []:
        if turn.get("from") in {"gpt", "assistant"}:
            return str(turn.get("value", "") or "")
    return str(sample.get("answer", "") or sample.get("gnd", "") or "")

