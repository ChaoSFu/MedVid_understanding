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
    gt = gt_index(data, "rc")
    pred, duplicates = prediction_index(preds)
    records = {sid: leaderboard_record(sid, gt[sid], pred[sid]) for sid in sorted(set(gt) & set(pred))}
    result = {
        "join_audit": join_audit(gt, pred, duplicates),
        "medvidu_official_rc_metrics": None,
        "status": "READY_FOR_OFFICIAL_LLM_JUDGE",
        "records_for_official_judge": records,
        "note": "RC official evaluator uses external LLM judge / semantic scorer; this adapter prepares exact joined records but does not call an LLM with GT during inference.",
    }
    write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/evaluate E-VQA MedVidU RC predictions for the official MedVidU RC evaluator.")
    parser.add_argument("--pred-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "predictions" / "medvidu" / "rc.jsonl")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "evaluation" / "rc_official.json")
    args = parser.parse_args()
    print(evaluate_predictions(args.pred_path, args.data_path, args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

