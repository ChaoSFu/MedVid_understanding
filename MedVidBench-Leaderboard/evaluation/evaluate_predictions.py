"""Auto-detect prediction format and evaluate with ground-truth merging if needed."""

import json
import sys
import argparse
import os
from pathlib import Path

# Add evaluation directory to path to import evaluate_all_pai
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

import evaluate_all_pai


def detect_has_ground_truth(data):
    """Detect if prediction file already contains ground-truth.

    Args:
        data: Loaded JSON data (dict or list)

    Returns:
        bool: True if ground-truth is present, False otherwise
    """
    # Handle both dict and list formats
    if isinstance(data, dict):
        # Check first record
        first_key = next(iter(data))
        sample = data[first_key]
    elif isinstance(data, list):
        if not data:
            return False
        sample = data[0]
    else:
        return False

    # Check for ground-truth indicators
    # results.json format has: question, gnd, answer, struc_info, metadata, qa_type, data_source
    has_question = 'question' in sample
    has_gnd = 'gnd' in sample
    has_struc_info = 'struc_info' in sample
    has_metadata_dict = isinstance(sample.get('metadata'), dict)

    # predictions_only.json format has: id, qa_type, prediction
    has_id = 'id' in sample
    has_prediction = 'prediction' in sample

    # If it has id + prediction format, it's prediction-only
    if has_id and has_prediction and not has_gnd:
        return False

    # If it has question + gnd + struc_info, it's already merged
    if has_question and has_gnd and has_struc_info:
        return True

    # Default: assume needs merging if unclear
    return False


def parse_id(id_str):
    """Parse ID string into components.

    Format: video_id&&start_frame&&end_frame&&fps
    Example: "kcOqlifSukA&&22425&&25124&&1.0"

    Returns:
        dict: {'video_id': str, 'input_video_start_frame': str,
               'input_video_end_frame': str, 'fps': str}
    """
    parts = id_str.split('&&')
    if len(parts) != 4:
        raise ValueError(f"Invalid ID format: {id_str}")

    return {
        'video_id': parts[0],
        'input_video_start_frame': parts[1],
        'input_video_end_frame': parts[2],
        'fps': parts[3]
    }


def merge_with_ground_truth(predictions_file, ground_truth_file):
    """Merge prediction-only file with ground-truth by array index.

    Args:
        predictions_file: Path to predictions JSON (array format, same order as ground truth)
        ground_truth_file: Path to ground-truth JSON

    Returns:
        dict: Merged data in results.json format
    """
    print(f"[EvaluationWrapper] Loading predictions from {predictions_file}")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    print(f"[EvaluationWrapper] Loading ground-truth from {ground_truth_file}")
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    print(f"[EvaluationWrapper] Predictions: {len(predictions)} records")
    print(f"[EvaluationWrapper] Ground-truth: {len(ground_truth)} records")

    # Check lengths match
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) != ground truth ({len(ground_truth)}). "
            f"Predictions must be in the same order as ground truth."
        )

    # Merge predictions with ground-truth by index
    merged = {}
    mismatched_qa_types = []

    for i, (pred, gt_record) in enumerate(zip(predictions, ground_truth)):
        # Validate prediction has 'prediction' field
        if 'prediction' not in pred:
            raise ValueError(f"Prediction at index {i} missing 'prediction' field")

        # Optional: check qa_type matches
        if 'qa_type' in pred and pred['qa_type'] != gt_record.get('qa_type'):
            mismatched_qa_types.append(i)

        # Extract question and ground truth from conversations
        question = ''
        gnd = ''
        if 'conversations' in gt_record:
            for msg in gt_record['conversations']:
                if msg.get('from') in ['human', 'user']:
                    # Remove <video> token to match original format
                    question = msg.get('value', '').replace('<video>\n', '').replace('<video>', '')
                elif msg.get('from') in ['gpt', 'assistant']:
                    gnd = msg.get('value', '')

        # Get data_source
        data_source = gt_record.get('data_source', 'Unknown')
        if data_source == 'Unknown' or not data_source:
            data_source = gt_record.get('dataset_name', 'Unknown')

        # Create merged record
        merged_record = {
            'metadata': gt_record.get('metadata', {}),
            'qa_type': gt_record.get('qa_type', ''),
            'struc_info': gt_record.get('struc_info', []),
            'question': question,
            'gnd': gnd,
            'answer': pred.get('prediction', ''),  # Model prediction
            'data_source': data_source
        }

        # Use sequential keys like results.json
        merged[str(i)] = merged_record

    if mismatched_qa_types:
        print(f"[EvaluationWrapper] ⚠️  Warning: {len(mismatched_qa_types)} samples with mismatched qa_type")

    print(f"[EvaluationWrapper] ✓ Successfully merged {len(merged)}/{len(predictions)} predictions")

    return merged


def _parse_metrics_from_output(output):
    """Parse leaderboard metrics from evaluation stdout.

    Mirrors app.py's parse_evaluation_output() logic.
    Returns dict with keys: tag_miou_03, tag_miou_05, stg_miou, nap_acc,
                             sa_acc, cvs_acc, dvc_f1, dvc_llm, vs_llm, rc_llm
    """
    metrics = {}
    lines = output.split('\n')
    current_task = None
    current_iou_section = None

    for line in lines:
        line = line.strip()

        # Detect task sections
        # NOTE: Order matters — check CVS before VS (since "CVS" contains "VS")
        if ("CVS" in line and "Overall" in line) or "CVS Assessment" in line:
            current_task = "cvs_assessment"
        elif ("SKILL" in line and "Overall" in line) or "Skill Assessment" in line:
            current_task = "skill_assessment"
        elif "TAL" in line and "Overall" in line:
            current_task = "tal"
        elif "STG" in line and "Overall" in line:
            current_task = "stg"
        elif ("NEXT_ACTION" in line and "Overall" in line) or "Next Action" in line:
            current_task = "next_action"
        elif ("DVC" in line and "Overall" in line) or "Dense Video Captioning" in line:
            current_task = "dvc"
        elif ("RC" in line and "Overall" in line) or "Region Caption" in line:
            current_task = "rc"
        elif ("VS" in line and "Overall" in line) or "Video Summary" in line:
            current_task = "vs"

        if current_task == "tal":
            if "IoU_0.3:" in line:
                current_iou_section = "0.3"
            elif "IoU_0.5:" in line:
                current_iou_section = "0.5"

        if not current_task:
            continue

        try:
            if current_task == "tal":
                if "meanIoU@0.3" in line or "mIoU@0.3" in line:
                    metrics["tag_miou_03"] = float(line.split(":")[-1].strip())
                elif "meanIoU@0.5" in line or "mIoU@0.5" in line:
                    metrics["tag_miou_05"] = float(line.split(":")[-1].strip())
                elif current_iou_section and "meanIoU:" in line and "meanIoU@" not in line:
                    val = float(line.split(":")[-1].strip())
                    if current_iou_section == "0.3":
                        metrics["tag_miou_03"] = val
                    elif current_iou_section == "0.5":
                        metrics["tag_miou_05"] = val

            elif current_task == "stg" and ("mean_iou" in line.lower() or "miou" in line.lower()):
                metrics["stg_miou"] = float(line.split(":")[-1].strip())

            elif current_task == "next_action" and "accuracy" in line.lower():
                metrics["nap_acc"] = float(line.split(":")[-1].strip())

            elif current_task == "dvc":
                if "caption_score" in line.lower() or "caption score" in line.lower():
                    metrics["dvc_llm"] = float(line.split(":")[-1].strip())
                elif "temporal_f1" in line.lower() or "temporal f1" in line.lower():
                    metrics["dvc_f1"] = float(line.split(":")[-1].strip())

            elif current_task == "vs" and ("score" in line.lower() or "average" in line.lower()):
                val_str = line.split(":")[-1].strip().split("(")[0].strip()
                metrics["vs_llm"] = float(val_str)

            elif current_task == "rc" and ("score" in line.lower() or "average" in line.lower()):
                val_str = line.split(":")[-1].strip().split("(")[0].strip()
                metrics["rc_llm"] = float(val_str)

            elif current_task == "skill_assessment" and "aspect_balanced_accuracy" in line.lower():
                metrics["sa_acc"] = float(line.split(":")[1].split("(")[0].strip())

            elif current_task == "cvs_assessment" and "component_balanced_accuracy" in line:
                metrics["cvs_acc"] = float(line.split(":")[-1].strip())
        except (ValueError, IndexError):
            pass

    return metrics


def _print_leaderboard_summary(captured_output, skip_llm_judge=False):
    """Print a clean leaderboard metrics summary parsed from evaluation stdout.

    Printed in patterns that match app.py's parse_evaluation_output() triggers,
    so both human readers and the leaderboard parser can consume this output.

    Non-LLM metrics: tag_miou_03, tag_miou_05, stg_miou, nap_acc, sa_acc, cvs_acc, dvc_f1
    LLM-judge metrics (omitted when skip_llm_judge=True): dvc_llm, vs_llm, rc_llm
    """
    metrics = _parse_metrics_from_output(captured_output)

    print(f"\n{'='*80}", flush=True)
    print("LEADERBOARD METRICS SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    METRIC_LABELS = [
        ("cvs_acc",     "CVS Assessment - Overall Evaluation",      "cvs_assessment",    "  component_balanced_accuracy: {v:.4f}"),
        ("nap_acc",     "Next Action - Overall Evaluation",         "next_action",        "  accuracy: {v:.4f}"),
        ("sa_acc",      "Skill Assessment - Overall Evaluation",    "skill_assessment",   "  aspect_balanced_accuracy: {v:.4f}"),
        ("stg_miou",    "STG - Overall Evaluation",                 "stg",                "  mean_iou: {v:.4f}"),
        ("tag_miou_03", "TAL - Overall Evaluation",                 "tal",                "  mIoU@0.3: {v:.4f}"),
        ("tag_miou_05", None,                                        None,                 "  mIoU@0.5: {v:.4f}"),
        ("dvc_f1",      "Dense Video Captioning Metrics",           "dvc",                "  temporal_f1: {v:.4f}"),
    ]
    if not skip_llm_judge:
        METRIC_LABELS += [
            ("dvc_llm",  None,                               None,   "  caption_score: {v:.4f}"),
            ("vs_llm",  "VS - Overall Evaluation",           "vs",   "  score: {v:.4f}"),
            ("rc_llm",  "RC - Overall Evaluation",           "rc",   "  score: {v:.4f}"),
        ]

    last_task = None
    for metric_key, header, task_tag, fmt in METRIC_LABELS:
        if metric_key not in metrics:
            continue
        v = metrics[metric_key]
        if header and task_tag != last_task:
            print(f"\n{header}", flush=True)
            last_task = task_tag
        print(fmt.format(v=v), flush=True)

    print(f"\n{'='*80}", flush=True)
    print("END LEADERBOARD METRICS SUMMARY", flush=True)
    print(f"{'='*80}\n", flush=True)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate predictions with automatic ground-truth merging"
    )
    parser.add_argument("predictions_file",
                       help="Path to predictions JSON file (can be merged or prediction-only format)")
    parser.add_argument("--ground-truth",
                       default="/root/code/MedVidBench-Leaderboard/data/ground_truth.json",
                       help="Path to ground-truth JSON file (default: data/ground_truth.json)")
    parser.add_argument("--tasks", nargs="+",
                       choices=["dvc", "tal", "next_action", "stg", "rc", "vs",
                               "skill_assessment", "cvs_assessment", "gemini_structured", "gpt_structured"],
                       help="Specific tasks to evaluate (default: all available tasks)")
    parser.add_argument("--grouping", choices=["per-dataset", "overall"], default="overall",
                       help="Grouping strategy: 'per-dataset' or 'overall' (default: overall)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze the file structure without running evaluations")
    parser.add_argument("--skip-llm-judge", default=False, action="store_true",
                       help="Skip LLM judge evaluation for caption tasks (use when LLM scores are pre-computed)")

    args = parser.parse_args()

    # Load predictions
    print(f"[EvaluationWrapper] Loading predictions from {args.predictions_file}", flush=True)
    with open(args.predictions_file, 'r') as f:
        predictions_data = json.load(f)

    # Auto-detect format
    has_ground_truth = detect_has_ground_truth(predictions_data)

    if has_ground_truth:
        print("[EvaluationWrapper] ✓ Detected: Predictions already contain ground-truth", flush=True)
        print("[EvaluationWrapper] Using predictions file directly for evaluation", flush=True)
        eval_file = args.predictions_file
    else:
        print("[EvaluationWrapper] ✓ Detected: Prediction-only format (id, qa_type, prediction)", flush=True)
        print("[EvaluationWrapper] Merging with ground-truth...", flush=True)

        # Check ground-truth file exists
        if not os.path.exists(args.ground_truth):
            print(f"[EvaluationWrapper] ❌ ERROR: Ground-truth file not found: {args.ground_truth}", flush=True)
            sys.exit(1)

        # Merge predictions with ground-truth
        merged_data = merge_with_ground_truth(args.predictions_file, args.ground_truth)

        # Save merged data to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(merged_data, f, indent=2)
            eval_file = f.name

        print(f"[EvaluationWrapper] ✓ Merged data saved to temporary file: {eval_file}", flush=True)

    # Call evaluate_all_pai with the appropriate file
    print(f"\n[EvaluationWrapper] {'='*80}", flush=True)
    print(f"[EvaluationWrapper] Starting evaluation with evaluate_all_pai.py", flush=True)
    print(f"[EvaluationWrapper] {'='*80}\n", flush=True)

    # Set sys.argv for evaluate_all_pai
    eval_args = [eval_file]
    if args.tasks:
        eval_args.extend(["--tasks"] + args.tasks)
    if args.grouping:
        eval_args.extend(["--grouping", args.grouping])
    if args.analyze_only:
        eval_args.append("--analyze-only")
    if args.skip_llm_judge:
        eval_args.append("--skip-llm-judge")

    original_argv = sys.argv
    sys.argv = ["evaluate_all_pai.py"] + eval_args

    try:
        # Run evaluation
        if args.analyze_only:
            qa_type_counts, dataset_counts = evaluate_all_pai.analyze_output_file(eval_file)
            # Determine available tasks
            available_tasks = []
            if any("dense_captioning" in qa_type or qa_type == "dc" for qa_type in qa_type_counts):
                available_tasks.append("dvc")
            if qa_type_counts.get("tal", 0) > 0:
                available_tasks.append("tal")
            if qa_type_counts.get("next_action", 0) > 0:
                available_tasks.append("next_action")
            if qa_type_counts.get("stg", 0) > 0:
                available_tasks.append("stg")
            if any("region_caption" in qa_type for qa_type in qa_type_counts):
                available_tasks.append("rc")
            if any("video_summary" in qa_type for qa_type in qa_type_counts):
                available_tasks.append("vs")
            if qa_type_counts.get("skill_assessment", 0) > 0:
                available_tasks.append("skill_assessment")
            if qa_type_counts.get("cvs_assessment", 0) > 0:
                available_tasks.append("cvs_assessment")

            evaluate_all_pai.print_evaluation_results_csv(eval_file, available_tasks)
        else:
            import io
            silent_eval = (args.grouping == "overall")
            # Tee stdout: capture a copy for metric parsing while still printing live
            captured = io.StringIO()
            original_stdout = sys.stdout

            class _TeeWriter:
                def write(self, s):
                    original_stdout.write(s)
                    captured.write(s)
                def flush(self):
                    original_stdout.flush()

            sys.stdout = _TeeWriter()
            try:
                evaluate_all_pai.run_evaluation(
                    eval_file,
                    args.tasks,
                    grouping=args.grouping,
                    silent_eval=silent_eval,
                    skip_llm_judge=args.skip_llm_judge
                )
            finally:
                sys.stdout = original_stdout

            _print_leaderboard_summary(captured.getvalue(), args.skip_llm_judge)
    finally:
        sys.argv = original_argv

        # Clean up temporary file if we created one
        if not has_ground_truth and os.path.exists(eval_file):
            os.unlink(eval_file)
            print(f"\n[EvaluationWrapper] ✓ Cleaned up temporary file: {eval_file}")


if __name__ == "__main__":
    main()
