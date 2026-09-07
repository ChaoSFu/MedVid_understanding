from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from baselines.evqa_medvidu.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_ROOT
from baselines.evqa_medvidu.evaluation.join import gt_index, join_audit, leaderboard_record, prediction_index
from baselines.evqa_medvidu.io_utils import read_json, read_jsonl, write_json


def evaluate_predictions(pred_path: Path, data_path: Path, output_path: Path) -> dict[str, Any]:
    data = read_json(data_path)
    preds = read_jsonl(pred_path)
    gt = gt_index(data, "stg")
    pred, duplicates = prediction_index(preds)
    records = {sid: leaderboard_record(sid, gt[sid], pred[sid]) for sid in sorted(set(gt) & set(pred))}
    try:
        from MedVidBench_Leaderboard.evaluation.eval_stg import group_records_by_dataset, evaluate_dataset_stg  # type: ignore
    except Exception:
        import importlib.util

        module_path = Path(__file__).resolve().parents[3] / "MedVidBench-Leaderboard" / "evaluation" / "eval_stg.py"
        spec = importlib.util.spec_from_file_location("medvidu_eval_stg", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        group_records_by_dataset = module.group_records_by_dataset
        evaluate_dataset_stg = module.evaluate_dataset_stg
    grouped = group_records_by_dataset(records)
    per_dataset = {dataset: evaluate_dataset_stg(dataset, rows) for dataset, rows in grouped.items()}
    result = {"join_audit": join_audit(gt, pred, duplicates), "medvidu_official_stg_metrics": per_dataset}
    write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate E-VQA MedVidU STG predictions with the official MedVidU STG evaluator.")
    parser.add_argument("--pred-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "predictions" / "medvidu" / "stg.jsonl")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "evaluation" / "stg_official.json")
    args = parser.parse_args()
    print(evaluate_predictions(args.pred_path, args.data_path, args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

