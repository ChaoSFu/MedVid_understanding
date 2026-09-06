from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

from .config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_ROOT, REPO_ROOT, RunConfig
from .io_utils import read_json, read_jsonl, write_json, write_jsonl


EVAL_DIR = REPO_ROOT / "MedVidBench-Leaderboard" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from eval_tal import compute_iou, evaluate_tal_record  # noqa: E402


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
    if not prediction.get("parse_valid"):
        return []
    span = prediction.get("official_parsed_span") or [0, 0]
    if not isinstance(span, list) or len(span) < 2:
        return []
    return [{"start": float(span[0]), "end": float(span[1])}]


def evaluate_predictions(prediction_path: Path, data_path: Path, output_root: Path) -> dict[str, Any]:
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

    records_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_sample_rows: list[dict[str, Any]] = []
    for sid in matched_ids:
        sample = gt_by_id[sid]
        pred = pred_by_id[sid]
        record = {"prediction": pred_spans(pred), "ground_truth": gt_spans(sample)}
        dataset_name = str(sample.get("dataset_name") or sample.get("data_source") or "Unknown")
        records_by_dataset[dataset_name].append(record)
        best_iou = max((compute_iou(p, g) for p in record["prediction"] for g in record["ground_truth"]), default=0.0)
        per_sample_rows.append(
            {
                "sample_id": sid,
                "qa_id": sample.get("id"),
                "dataset_name": dataset_name,
                "parse_valid": bool(pred.get("parse_valid")),
                "n_predicted_intervals": len(record["prediction"]),
                "n_gt_intervals": len(record["ground_truth"]),
                "best_iou": best_iou,
                **evaluate_tal_record([record], tiou_thresh=0.3),
                **evaluate_tal_record([record], tiou_thresh=0.5),
                **evaluate_tal_record([record], tiou_thresh=0.7),
            }
        )

    per_dataset: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for dataset_name, records in sorted(records_by_dataset.items()):
        all_records.extend(records)
        per_dataset[dataset_name] = {
            **evaluate_tal_record(records, tiou_thresh=0.3),
            **evaluate_tal_record(records, tiou_thresh=0.5),
            **evaluate_tal_record(records, tiou_thresh=0.7),
            "count": len(records),
        }

    official_metrics = {
        **evaluate_tal_record(all_records, tiou_thresh=0.3),
        **evaluate_tal_record(all_records, tiou_thresh=0.5),
        **evaluate_tal_record(all_records, tiou_thresh=0.7),
        "count": len(all_records),
    }
    secondary = {
        "mIoU": sum(row["best_iou"] for row in per_sample_rows) / len(per_sample_rows) if per_sample_rows else 0.0,
        "R@1_IoU_0.3": official_metrics.get("Recall@0.30", 0.0),
        "R@1_IoU_0.5": official_metrics.get("Recall@0.50", 0.0),
        "R@1_IoU_0.7": official_metrics.get("Recall@0.70", 0.0),
        "parse_valid": sum(1 for pred in preds if pred.get("parse_valid")),
        "parse_invalid": sum(1 for pred in preds if not pred.get("parse_valid")),
    }
    audit = {
        "n_GT": len(gt_by_id),
        "n_predictions": len(preds),
        "matched": len(matched_ids),
        "missing": len(missing),
        "extra": len(extra),
        "duplicate": len(duplicates),
        "missing_sample_ids": missing[:50],
        "extra_sample_ids": extra[:50],
        "duplicate_sample_ids": duplicates[:50],
        "parse_valid_counts": dict(sorted(Counter(bool(pred.get("parse_valid")) for pred in preds).items())),
    }
    result = {"join_audit": audit, "medvidu_official_tal_metrics": official_metrics, "secondary": secondary, "per_dataset": per_dataset}
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "medvidu_official_tal_metrics.json", result)
    write_json(output_root / "secondary_grounding_metrics.json", secondary)
    write_json(output_root / "per_dataset_metrics.json", per_dataset)
    write_jsonl(output_root / "per_sample_metrics.jsonl", per_sample_rows)
    if duplicates or missing or extra:
        write_json(output_root / "join_audit_requires_review.json", audit)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline MedVidU official TAL evaluation for TimeChat GT-free predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT_ROOT / "predictions" / "full_predictions.jsonl")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root.parents[0] if args.output_root.name == "evaluation" else DEFAULT_OUTPUT_ROOT)
    cfg.evaluation_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_predictions(args.predictions, args.data_path, args.output_root)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
