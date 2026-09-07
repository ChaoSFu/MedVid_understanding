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
    gt = gt_index(data, "cvs")
    pred, duplicates = prediction_index(preds)
    records = {sid: leaderboard_record(sid, gt[sid], pred[sid]) for sid in sorted(set(gt) & set(pred))}
    import importlib.util

    module_path = Path(__file__).resolve().parents[3] / "MedVidBench-Leaderboard" / "evaluation" / "eval_cvs_assessment.py"
    spec = importlib.util.spec_from_file_location("medvidu_eval_cvs", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grouped = module.group_records_by_dataset(records)
    per_dataset = {dataset: module.evaluate_cvs_assessment(rows) for dataset, rows in grouped.items()}
    result = {"join_audit": join_audit(gt, pred, duplicates), "medvidu_official_cvs_metrics": per_dataset}
    write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate E-VQA MedVidU CVS predictions with the official MedVidU CVS evaluator.")
    parser.add_argument("--pred-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "predictions" / "medvidu" / "cvs.jsonl")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "evaluation" / "cvs_official.json")
    args = parser.parse_args()
    print(evaluate_predictions(args.pred_path, args.data_path, args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

