from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

from .config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_ROOT, REPO_ROOT
from .io_utils import read_json, read_jsonl, write_json, write_jsonl


EVAL_DIR = REPO_ROOT / "MedVidBench-Leaderboard" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from eval_tal import evaluate_tal_record  # noqa: E402


def sample_id_for(sample: dict[str, Any], original_index: int) -> str:
    return f"{original_index:06d}::{sample.get('id', '')}::tal"


def gt_spans(sample: dict[str, Any]) -> list[dict[str, float]]:
    spans: list[dict[str, float]] = []
    for item in sample.get("struc_info") or []:
        if isinstance(item, dict):
            for span in item.get("spans") or []:
                spans.append({"start": float(span["start"]), "end": float(span["end"])})
    return spans


def pred_spans(prediction: dict[str, Any]) -> list[dict[str, float]]:
    spans = []
    for start, end in prediction.get("parsed_timestamps") or []:
        spans.append({"start": float(start), "end": float(end)})
    return spans


def evaluate_predictions(prediction_path: Path, data_path: Path, output_root: Path, split_name: str) -> dict[str, Any]:
    preds = read_jsonl(prediction_path)
    pred_by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for pred in preds:
        sid = str(pred.get("sample_id"))
        if sid in pred_by_id:
            duplicates.append(sid)
        pred_by_id[sid] = pred

    data = read_json(data_path)
    gt_by_id = {
        sample_id_for(sample, i): sample
        for i, sample in enumerate(data)
        if isinstance(sample, dict) and sample.get("qa_type") == "tal"
    }

    matched_ids = sorted(set(pred_by_id).intersection(gt_by_id))
    missing = sorted(set(gt_by_id).difference(pred_by_id))
    extra = sorted(set(pred_by_id).difference(gt_by_id))

    per_sample_rows: list[dict[str, Any]] = []
    records_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sid in matched_ids:
        sample = gt_by_id[sid]
        pred = pred_by_id[sid]
        record = {"prediction": pred_spans(pred), "ground_truth": gt_spans(sample)}
        dataset_name = str(sample.get("dataset_name") or sample.get("data_source") or "Unknown")
        records_by_dataset[dataset_name].append(record)
        row = {
            "sample_id": sid,
            "dataset_name": dataset_name,
            "parse_valid": bool(pred.get("parse_valid")),
            "n_predicted_intervals": len(record["prediction"]),
            "n_gt_intervals": len(record["ground_truth"]),
            "Recall@0.30": evaluate_tal_record([record], tiou_thresh=0.3)["Recall@0.30"],
            "meanIoU@0.30": evaluate_tal_record([record], tiou_thresh=0.3)["meanIoU@0.30"],
            "Recall@0.50": evaluate_tal_record([record], tiou_thresh=0.5)["Recall@0.50"],
            "meanIoU@0.50": evaluate_tal_record([record], tiou_thresh=0.5)["meanIoU@0.50"],
        }
        per_sample_rows.append(row)

    per_dataset: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for dataset_name, records in sorted(records_by_dataset.items()):
        all_records.extend(records)
        per_dataset[dataset_name] = {
            **evaluate_tal_record(records, tiou_thresh=0.3),
            **evaluate_tal_record(records, tiou_thresh=0.5),
            "count": len(records),
        }
    overall = {
        **evaluate_tal_record(all_records, tiou_thresh=0.3),
        **evaluate_tal_record(all_records, tiou_thresh=0.5),
        "count": len(all_records),
    }

    audit = {
        "n_predictions": len(preds),
        "n_GT": len(gt_by_id),
        "matched": len(matched_ids),
        "missing": len(missing),
        "extra": len(extra),
        "duplicate": len(duplicates),
        "missing_sample_ids": missing[:50],
        "extra_sample_ids": extra[:50],
        "duplicate_sample_ids": duplicates[:50],
        "prediction_parse_status_counts": dict(sorted(Counter(str(p.get("parse_status")) for p in preds).items())),
    }
    result = {"audit": audit, "overall": overall, "per_dataset": per_dataset}
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / f"{split_name}_eval.json", result)
    write_json(output_root / "per_dataset_metrics.json", per_dataset)
    write_jsonl(output_root / "per_sample_metrics.jsonl", per_sample_rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TimeLens-8B MedVidU TAL predictions with MedVidBench TAL matching.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT_ROOT / "predictions" / "timelens8b_tal_smoke_predictions.jsonl")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "evaluation")
    parser.add_argument("--split-name", default="smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_predictions(args.predictions, args.data_path, args.output_root, args.split_name)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

