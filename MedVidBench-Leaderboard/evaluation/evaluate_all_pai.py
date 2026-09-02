"""Main Evaluation Script for All Tasks and Multiple Datasets."""

import json
import sys
import argparse
from collections import defaultdict

# Import task-specific evaluation modules using importlib to avoid path conflicts
import importlib.util
import os

def load_eval_module(module_name):
    """Load evaluation module from the current directory using importlib."""
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(eval_dir, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_output_file(output_file):
    """Analyze the output file to determine what tasks and datasets are present."""
    print(f"Analyzing output file: {output_file}")
    
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Count different QA types
    qa_type_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        records = data.values()
    elif isinstance(data, list):
        records = data
    else:
        print(f"Unexpected data format: {type(data)}")
        return {}, {}
    
    for record in records:
        qa_type = record.get("qa_type", "unknown")
        qa_type_counts[qa_type] += 1
        
        # Get dataset from data_source field if available
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            video_id = record.get("metadata", {}).get("video_id", "")
            dataset = detect_dataset_from_video_id(video_id)
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record.get("question", ""))
        
        dataset_counts[dataset] += 1
    
    print(f"\nFound QA types:", flush=True)
    for qa_type, count in qa_type_counts.items():
        print(f"  {qa_type}: {count} records", flush=True)

    print(f"\nFound datasets:", flush=True)
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} records", flush=True)

    return qa_type_counts, dataset_counts


def detect_dataset_from_video_id(video_id):
    """Detect dataset from video ID patterns."""
    video_id = str(video_id).lower()
    
    # AVOS dataset - YouTube video IDs
    if len(video_id) == 11 and any(c.isalpha() for c in video_id):
        return "AVOS"
    
    # CoPESD dataset - numerical IDs with parts
    if "_part" in video_id and video_id.replace("_part", "").split("_")[0].isdigit():
        return "CoPESD"
    
    # CholecT50 dataset
    if "video" in video_id.lower() and any(c.isdigit() for c in video_id):
        return "CholecT50"
    
    # NurViD dataset - specific patterns
    if any(keyword in video_id for keyword in ["nur", "nursing", "medical"]):
        return "NurViD"
    
    return "Unknown"


def detect_dataset_from_question(question):
    """Detect dataset from question text patterns."""
    question_lower = question.lower()
    
    if "avos" in question_lower:
        return "AVOS"
    elif "copesd" in question_lower:
        return "CoPESD"
    elif "cholect50" in question_lower or "cholec" in question_lower:
        return "CholecT50"
    elif "nurvid" in question_lower or "nursing" in question_lower:
        return "NurViD"
    
    # Check for dataset-specific action patterns
    if any(action in question_lower for action in ["cutting", "tying", "suturing"]):
        return "AVOS"
    elif "forceps" in question_lower and "knife" in question_lower:
        return "CoPESD"
    
    return "Unknown"





def print_evaluation_results_csv_with_real_results(output_file, tasks, all_task_results):
    """Print evaluation results in CSV format with real captured results."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS SUMMARY (NEW CSV FORMAT) - WITH REAL RESULTS")
    print(f"{'='*80}")
    
    # Convert the task results to the format expected by the internal function
    converted_results = {}
    
    # Load the data to get FPS information
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Group records by dataset, fps, and task to match structure
    dataset_fps_task_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'count': 0, 'videos': set()
    })))
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        records = data.values()
    elif isinstance(data, list):
        records = data
    else:
        print(f"Unexpected data format in print_evaluation_results_csv_with_real_results: {type(data)}")
        return
    
    for record in records:
        qa_type = record.get("qa_type", "unknown")
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            video_id = record.get("metadata", {}).get("video_id", "")
            dataset = detect_dataset_from_video_id(video_id)
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record.get("question", ""))
        
        fps = record.get("metadata", {}).get("fps", "unknown")
        video_id = record.get("metadata", {}).get("video_id", "unknown")
        
        # Map qa_type to task name for consistency
        task_name = "unknown"
        if any("dense_captioning" in qa_type or qa_type == "dc" for _ in [qa_type]):
            task_name = "dvc"
        elif qa_type == "tal":
            task_name = "tal"
        elif qa_type == "next_action":
            task_name = "next_action"
        elif qa_type == "stg":
            task_name = "stg"
        elif "region_caption" in qa_type:
            task_name = "rc"
        elif "video_summary" in qa_type:
            task_name = "vs"
        elif qa_type == "skill_assessment":
            task_name = "skill_assessment"
        elif qa_type == "cvs_assessment":
            task_name = "cvs_assessment"
        
        # Only include tasks that were evaluated
        if task_name in tasks or task_name == "unknown":
            dataset_fps_task_stats[dataset][fps][task_name]['count'] += 1
            dataset_fps_task_stats[dataset][fps][task_name]['videos'].add(video_id)
    
    # Convert real evaluation results to expected format
    for task_name, task_results in all_task_results.items():
        for dataset_name, dataset_results in task_results.items():
            # For each FPS in this dataset
            for fps in dataset_fps_task_stats[dataset_name].keys():
                if task_name in dataset_fps_task_stats[dataset_name][fps]:
                    eval_key = f"{dataset_name}_{task_name}_{fps}"
                    
                    # Extract metrics based on task type
                    if task_name == "dvc":
                        # DVC format: extract CIDER, METEOR, Precision_Mean, Recall_Mean, F1_Score
                        metrics = []
                        if isinstance(dataset_results, dict):
                            metrics.append(dataset_results.get('CIDER', 0.0))
                            metrics.append(dataset_results.get('METEOR', 0.0))
                            metrics.append(dataset_results.get('Precision_Mean', 0.0))
                            metrics.append(dataset_results.get('Recall_Mean', 0.0))
                            metrics.append(dataset_results.get('F1_Score', 0.0))
                            metrics.append(dataset_results.get('SODA_c_1', 0.0))
                        converted_results[eval_key] = {'metrics': metrics}
                        
                    elif task_name == "tal":
                        # TAL format: extract precision and recall at different IoU thresholds
                        metrics = []
                        if isinstance(dataset_results, dict):
                            # Look for IoU thresholds
                            metrics.append(dataset_results.get('0.3', {}).get('Precision', 0.0))
                            metrics.append(dataset_results.get('0.3', {}).get('Recall', 0.0))
                            metrics.append(dataset_results.get('0.5', {}).get('Precision', 0.0))
                            metrics.append(dataset_results.get('0.5', {}).get('Recall', 0.0))
                            metrics.append(dataset_results.get('mAP@0.5', 0.0))
                        converted_results[eval_key] = {'metrics': metrics}
                        
                    elif task_name == "next_action":
                        # Next Action format: extract overall accuracy
                        metrics = []
                        if isinstance(dataset_results, dict) and 'overall' in dataset_results:
                            overall = dataset_results['overall']
                            metrics.append(overall.get('accuracy', 0.0))
                            metrics.append(0.0)  # Per_class_avg placeholder
                            metrics.append(0.0)  # Weighted_F1 placeholder
                        converted_results[eval_key] = {'metrics': metrics}
                        
                    elif task_name == "stg":
                        # STG format: extract IoU metrics
                        metrics = []
                        if isinstance(dataset_results, dict):
                            # Use overall metrics if available
                            if 'overall' in dataset_results:
                                overall = dataset_results['overall']
                                mean_iou = overall.get('mean_iou', 0.0)
                                metrics = [mean_iou, mean_iou, mean_iou, mean_iou]  # IoU@0.3, 0.5, 0.7, mIoU
                            else:
                                # Use FPS-specific metrics
                                fps_result = dataset_results.get(str(fps), {})
                                mean_iou = fps_result.get('mean_iou', 0.0)
                                metrics = [mean_iou, mean_iou, mean_iou, mean_iou]
                        converted_results[eval_key] = {'metrics': metrics}
    
    # Use the existing function but pass the converted real evaluation results
    print_evaluation_results_csv_internal(output_file, tasks, converted_results)


def print_evaluation_results_csv(output_file, tasks):
    """Print evaluation results in new CSV format: Dataset → Task → Metrics."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS SUMMARY (NEW CSV FORMAT)")
    print(f"{'='*80}")
    
    # Call internal function with empty evaluation results (for analyze-only mode)
    print_evaluation_results_csv_internal(output_file, tasks, {})


def print_evaluation_results_csv_internal(output_file, tasks, evaluation_results):
    """Internal function to print CSV results with optional real evaluation results."""
    # Load the data to analyze structure
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Define metrics for each task type (these will be populated from actual evaluation results)
    task_metrics = {
        'dvc': ['CIDER', 'METEOR', 'Precision@0.5', 'Recall@0.5', 'F1_Score'],
        'tal': ['Precision@0.3', 'Recall@0.3', 'Precision@0.5', 'Recall@0.5', 'mAP@0.5'],
        'next_action': ['Accuracy', 'Per_class_avg', 'Weighted_F1'],
        'stg': ['IoU@0.3', 'IoU@0.5', 'IoU@0.7', 'mIoU'],
        'rc': ['BLEU4', 'METEOR', 'CIDEr', 'ROUGE_L'],
        'vs': ['BLEU4', 'METEOR', 'CIDEr', 'ROUGE_L'],
        'skill_assessment': ['Accuracy', 'Macro_F1', 'Weighted_F1'],
        'cvs_assessment': ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    }
    
    # Group records by dataset, fps, and task
    dataset_fps_task_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'count': 0, 'videos': set()
    })))
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        records = data.values()
    elif isinstance(data, list):
        records = data
    else:
        print(f"Unexpected data format in print_evaluation_results_csv_internal: {type(data)}")
        return
    
    for record in records:
        qa_type = record.get("qa_type", "unknown")
        dataset = record.get("data_source", "Unknown")
        
        # Fallback to detection methods if data_source is not available
        if dataset == "Unknown" or not dataset:
            video_id = record.get("metadata", {}).get("video_id", "")
            dataset = detect_dataset_from_video_id(video_id)
            if dataset == "Unknown":
                dataset = detect_dataset_from_question(record.get("question", ""))
        
        fps = record.get("metadata", {}).get("fps", "unknown")
        video_id = record.get("metadata", {}).get("video_id", "unknown")
        
        # Map qa_type to task name for consistency
        task_name = "unknown"
        if any("dense_captioning" in qa_type or qa_type == "dc" for _ in [qa_type]):
            task_name = "dvc"
        elif qa_type == "tal":
            task_name = "tal"
        elif qa_type == "next_action":
            task_name = "next_action"
        elif qa_type == "stg":
            task_name = "stg"
        elif "region_caption" in qa_type:
            task_name = "rc"
        elif "video_summary" in qa_type:
            task_name = "vs"
        elif qa_type == "skill_assessment":
            task_name = "skill_assessment"
        elif qa_type == "cvs_assessment":
            task_name = "cvs_assessment"
        
        # Only include tasks that were evaluated
        if task_name in tasks or task_name == "unknown":
            dataset_fps_task_stats[dataset][fps][task_name]['count'] += 1
            dataset_fps_task_stats[dataset][fps][task_name]['videos'].add(video_id)
    
    # Get all unique tasks that have data
    available_tasks = set()
    for dataset_stats in dataset_fps_task_stats.values():
        for fps_stats in dataset_stats.values():
            available_tasks.update(fps_stats.keys())
    
    # Print results for each dataset
    for dataset_name in sorted(dataset_fps_task_stats.keys()):
        print(f"\n{dataset_name}")
        
        # For each task in this dataset
        dataset_tasks = set()
        for fps_stats in dataset_fps_task_stats[dataset_name].values():
            dataset_tasks.update(fps_stats.keys())
        
        for task_name in sorted(dataset_tasks):
            print(f"{task_name}")
            
            # Print headers for this task
            metrics = task_metrics.get(task_name, ['Count', 'Videos'])
            header = "fps, qa_instances, " + ", ".join(metrics)
            print(header)
            
            # Store metrics for overall average calculation
            task_overall_metrics = []
            task_overall_count = 0
            
            # Print data rows for each FPS
            for fps in sorted(dataset_fps_task_stats[dataset_name].keys()):
                fps_stats = dataset_fps_task_stats[dataset_name][fps]
                
                if task_name in fps_stats:
                    task_stats = fps_stats[task_name]
                    count = task_stats['count']
                    video_count = len(task_stats['videos'])
                    
                    # Get real evaluation results if available
                    eval_key = f"{dataset_name}_{task_name}_{fps}"
                    if eval_key in evaluation_results:
                        values = evaluation_results[eval_key]['metrics']
                        task_overall_metrics.append(values)
                        task_overall_count += count
                        
                        # Format values as strings
                        value_strs = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]
                        row = f"{fps}, {count}, " + ", ".join(value_strs)
                        print(row)
                    else:
                        print(f"No real results for {eval_key}, missing!!!")
            
            # Add overall average line if we have metrics
            if task_overall_metrics and task_overall_count > 0:
                # Calculate weighted average across all fps
                num_metrics = len(task_overall_metrics[0])
                overall_avg = [0.0] * num_metrics
                for metrics in task_overall_metrics:
                    for i, val in enumerate(metrics):
                        if isinstance(val, (int, float)):
                            overall_avg[i] += val
                
                # Average the metrics
                for i in range(num_metrics):
                    overall_avg[i] /= len(task_overall_metrics)
                
                avg_strs = [f"{v:.3f}" for v in overall_avg]
                avg_row = f"Overall, {task_overall_count}, " + ", ".join(avg_strs)
                print(avg_row)
    
    # Print combined summary
    print(f"\nCombined Summary")
    
    for task_name in sorted(available_tasks):
        print(f"{task_name}")
        
        # Aggregate across all datasets for this task
        task_fps_stats = defaultdict(lambda: {'count': 0, 'videos': set()})
        
        for dataset_stats in dataset_fps_task_stats.values():
            for fps, fps_stats in dataset_stats.items():
                if task_name in fps_stats:
                    task_fps_stats[fps]['count'] += fps_stats[task_name]['count']
                    task_fps_stats[fps]['videos'].update(fps_stats[task_name]['videos'])
        
        # Print headers
        metrics = task_metrics.get(task_name, ['Count', 'Videos'])
        header = "fps, qa_instances, " + ", ".join(metrics)
        print(header)
        
        # Store metrics for overall average calculation
        combined_task_metrics = []
        combined_task_count = 0
        
        # Print data rows
        for fps in sorted(task_fps_stats.keys()):
            fps_data = task_fps_stats[fps]
            count = fps_data['count']
            video_count = len(fps_data['videos'])
            

        
        # Add overall average line for combined summary
        if combined_task_metrics and combined_task_count > 0:
            # Calculate average across all fps for this task
            num_metrics = len(combined_task_metrics[0])
            combined_avg = [0.0] * num_metrics
            for metrics in combined_task_metrics:
                for i, val in enumerate(metrics):
                    if isinstance(val, (int, float)):
                        combined_avg[i] += val
            
            # Average the metrics
            for i in range(num_metrics):
                combined_avg[i] /= len(combined_task_metrics)
            
            avg_strs = [f"{v:.3f}" for v in combined_avg]
            avg_row = f"Overall, {combined_task_count}, " + ", ".join(avg_strs)
            print(avg_row)


def print_overall_evaluation_results(output_file, tasks, all_task_results, skip_llm_judge=False):
    """Print evaluation results in overall mode using cached per-dataset results.

    Aggregates per-dataset results from _run_task_eval (pooled across all datasets)
    so that each data point is only evaluated once.
    """
    import numpy as np

    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - OVERALL (Dataset-Agnostic)")
    print(f"{'='*80}")

    for task_name in sorted(tasks):
        print(f"\n{'='*80}")
        print(f"{task_name.upper()} - Overall Evaluation (All Datasets Combined)")
        print(f"{'='*80}")

        cached = all_task_results.get(task_name, {})
        if not cached:
            print(f"No results found for {task_name}")
            continue

        try:
            if task_name == "tal":
                per_dataset = cached.get('per_dataset', {})
                if per_dataset:
                    # Pool all per-sample meanIoU across datasets and FPS groups
                    all_miou_03 = []
                    all_miou_05 = []
                    for ds_name, fps_results in per_dataset.items():
                        for fps_key, metrics in fps_results.items():
                            if isinstance(metrics, dict) and 'meanIoU@0.3' in metrics:
                                count = metrics.get('count', 1)
                                all_miou_03.extend([metrics['meanIoU@0.3']] * count)
                                all_miou_05.extend([metrics['meanIoU@0.5']] * count)
                    print(f"\n  mIoU@0.3: {np.mean(all_miou_03):.4f}" if all_miou_03 else "\n  mIoU@0.3: 0.0000")
                    print(f"  mIoU@0.5: {np.mean(all_miou_05):.4f}" if all_miou_05 else "  mIoU@0.5: 0.0000")
                else:
                    print(f"  mIoU@0.3: {cached.get('meanIoU@0.3', 0.0):.4f}")
                    print(f"  mIoU@0.5: {cached.get('meanIoU@0.5', 0.0):.4f}")

            elif task_name == "stg":
                per_dataset = cached.get('per_dataset', {})
                if per_dataset:
                    # Pool all per-sample IoUs across datasets
                    all_ious = []
                    for ds_name, fps_results in per_dataset.items():
                        if 'overall' in fps_results:
                            count = fps_results['overall'].get('valid_records', 1)
                            miou = fps_results['overall'].get('mean_iou', 0.0)
                            all_ious.extend([miou] * count)
                        else:
                            for fps_key, metrics in fps_results.items():
                                if isinstance(metrics, dict) and 'mIoU' in metrics:
                                    count = metrics.get('count', 1)
                                    all_ious.extend([metrics['mIoU']] * count)
                    print(f"\nmean_iou: {np.mean(all_ious):.4f}" if all_ious else "\nmean_iou: 0.0000")
                else:
                    print(f"\nmean_iou: {cached.get('mean_iou', 0.0):.4f}")

            elif task_name in ["rc", "vs"]:
                # LLM judge — use cached results directly (already pooled)
                if 'score' in cached:
                    print(f"Method: {cached['method']}")
                    print(f"Score: {cached['score']:.4f} ({cached['scale']} scale)")
                    if 'aspect_scores' in cached:
                        print("Aspect Scores:")
                        for aspect, score in sorted(cached['aspect_scores'].items()):
                            print(f"  {aspect}: {score:.3f}")
                else:
                    print(f"No LLM judge results available")

            elif task_name == "next_action":
                per_dataset = cached.get('per_dataset', {})
                if per_dataset:
                    # Pool per-sample correct/total across datasets
                    total_correct = 0
                    total_samples = 0
                    for ds_name, fps_results in per_dataset.items():
                        if 'overall' in fps_results:
                            acc = fps_results['overall'].get('accuracy', 0.0)
                            count = fps_results['overall'].get('count', 0)
                            total_correct += round(acc * count)
                            total_samples += count
                    if total_samples > 0:
                        print(f"\n  accuracy: {total_correct / total_samples:.4f}")
                    else:
                        print(f"\n  accuracy: 0.0000")
                else:
                    print(f"\n  accuracy: {cached.get('accuracy', 0.0):.4f}")

            elif task_name == "dvc":
                per_dataset = cached.get('per_dataset', {})
                print(f"\nDense Video Captioning Metrics:")
                if per_dataset:
                    # Pool caption_score and temporal_f1 weighted by sample count
                    total_caption = 0.0
                    total_f1 = 0.0
                    total_count = 0
                    for ds_name, ds_results in per_dataset.items():
                        if ds_results and 'overall' in ds_results:
                            overall = ds_results['overall']
                            count = overall.get('count', 0)
                            total_caption += overall.get('caption_score', 0.0) * count
                            total_f1 += overall.get('temporal_f1', 0.0) * count
                            total_count += count
                    if total_count > 0:
                        print(f"  caption_score: {total_caption / total_count:.4f}")
                        print(f"  temporal_f1: {total_f1 / total_count:.4f}")
                else:
                    for metric_name in ['caption_score', 'temporal_f1']:
                        if metric_name in cached and isinstance(cached[metric_name], (int, float)):
                            print(f"  {metric_name}: {cached[metric_name]:.4f}")

            elif task_name == "cvs_assessment":
                per_dataset = cached.get('per_dataset', {})
                if per_dataset:
                    print(f"\n  component_balanced_accuracy: {cached.get('component_balanced_accuracy', 0.0):.4f}")
                else:
                    print(f"\n  component_balanced_accuracy: {cached.get('component_balanced_accuracy', 0.0):.4f}")

            elif task_name == "skill_assessment":
                per_dataset = cached.get('per_dataset', {})
                if per_dataset:
                    print(f"\n  aspect_balanced_accuracy: {cached.get('aspect_balanced_accuracy', 0.0):.4f}")
                else:
                    print(f"\n  aspect_balanced_accuracy: {cached.get('aspect_balanced_accuracy', 0.0):.4f}")

            else:
                print(f"Overall evaluation not implemented for {task_name} yet")

        except Exception as e:
            print(f"Error printing overall evaluation for {task_name}: {e}")
            import traceback
            traceback.print_exc()


def _run_task_eval(task, output_file, skip_llm_judge=False):
    """Helper function to run a single task evaluation.

    Args:
        task: Task name (e.g., 'tal', 'stg')
        output_file: Path to results JSON
        skip_llm_judge: If True, skip LLM judge for caption tasks (DVC, VS, RC)

    Returns:
        Dictionary of evaluation results
    """
    import sys

    # Save original sys.argv
    original_argv = sys.argv.copy()

    try:
        # Set sys.argv for main() functions
        sys.argv = ["eval_script", output_file]
        if skip_llm_judge:
            sys.argv.append("--skip-llm-judge")

        if task == "dvc":
            module = load_eval_module("eval_dvc")
            task_results = module.main()
        elif task == "tal":
            module = load_eval_module("eval_tal")
            task_results = module.main()
        elif task == "next_action":
            module = load_eval_module("eval_next_action")
            task_results = module.main()
        elif task == "stg":
            module = load_eval_module("eval_stg")
            task_results = module.main()
        elif task == "rc":
            module = load_eval_module("eval_caption_llm_judge")
            # Evaluate region caption using LLM judge
            task_results = module.evaluate_caption_task(output_file, "region_caption")
        elif task == "vs":
            module = load_eval_module("eval_caption_llm_judge")
            # Evaluate video summary using LLM judge
            task_results = module.evaluate_caption_task(output_file, "video_summary")
        elif task == "skill_assessment":
            module = load_eval_module("eval_skill_assessment")
            task_results = module.main()
        elif task == "cvs_assessment":
            module = load_eval_module("eval_cvs_assessment")
            task_results = module.main()
        elif task == "gemini_structured":
            module = load_eval_module("eval_gemini_structured")
            task_results = module.main()
        elif task == "gpt_structured":
            module = load_eval_module("eval_gpt_structured")
            task_results = module.main()
        else:
            print(f"Unknown task: {task}")
            task_results = {}

        return task_results
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def run_evaluation(output_file, tasks=None, grouping="per-dataset", silent_eval=False, skip_llm_judge=False):
    """Run evaluation for specified tasks and capture real results.

    Args:
        output_file: Path to inference results JSON
        tasks: List of tasks to evaluate (None = auto-detect)
        grouping: 'per-dataset' or 'overall' - how to group results
        silent_eval: If True, suppress intermediate per-dataset output
        skip_llm_judge: If True, skip LLM judge evaluation for caption tasks (DVC, VS, RC)
    """
    import sys

    # Analyze the file first
    qa_type_counts, dataset_counts = analyze_output_file(output_file)


    # Determine which tasks to run
    if tasks is None:
        # Run all available tasks based on what's in the file
        available_tasks = []
        
        # Check for dense captioning (various naming patterns)
        if any("dense_captioning" in qa_type or qa_type == "dc" for qa_type in qa_type_counts):
            available_tasks.append("dvc")
        
        # Check for TAL
        if qa_type_counts.get("tal", 0) > 0:
            available_tasks.append("tal")
            
        # Check for next action
        if qa_type_counts.get("next_action", 0) > 0:
            available_tasks.append("next_action")
            
        # Check for STG
        if qa_type_counts.get("stg", 0) > 0:
            available_tasks.append("stg")
            
        # Check for region caption and video summary (various naming patterns)
        if any("region_caption" in qa_type for qa_type in qa_type_counts):
            available_tasks.append("rc")
        if any("video_summary" in qa_type for qa_type in qa_type_counts):
            available_tasks.append("vs")
            
        # Check for skill assessment
        if qa_type_counts.get("skill_assessment", 0) > 0:
            available_tasks.append("skill_assessment")
            
        # Check for CVS assessment
        if qa_type_counts.get("cvs_assessment", 0) > 0:
            available_tasks.append("cvs_assessment")
        tasks = available_tasks

    print(f"\nRunning evaluation for tasks: {tasks}", flush=True)
    print(f"Total tasks to evaluate: {len(tasks)}", flush=True)

    # Filter out LLM judge tasks if skip flag is set (but keep DVC for temporal F1)
    if skip_llm_judge:
        original_tasks = tasks.copy()
        tasks = [t for t in tasks if t not in ['vs', 'rc']]  # Keep 'dvc' for temporal F1
        if len(tasks) < len(original_tasks):
            print(f"Skipping LLM judge caption evaluation for: {[t for t in original_tasks if t not in tasks]}", flush=True)
            print(f"Note: DVC will compute temporal F1 but skip caption quality", flush=True)
            print(f"Evaluating {len(tasks)} tasks: {tasks}", flush=True)

    # Dictionary to store all evaluation results
    all_task_results = {}

    # Save original sys.argv to restore later
    original_argv = sys.argv.copy()

    # Redirect stdout if silent mode (for overall grouping)
    import io
    import contextlib

    try:
        # Run each task evaluation and capture returned results
        for task_idx, task in enumerate(tasks, 1):
            print(f"\n[Progress] Task {task_idx}/{len(tasks)}: {task.upper()}", flush=True)
            # Skip VS and RC if LLM judge flag is set (but not DVC - it has temporal F1)
            if skip_llm_judge and task in ['vs', 'rc']:
                if not silent_eval:
                    print(f"\n{'='*80}", flush=True)
                    print(f"SKIPPING {task.upper()} EVALUATION (LLM judge pre-computed)", flush=True)
                    print(f"{'='*80}", flush=True)
                # Store empty results - metrics will come from pre-computed scores
                all_task_results[task] = {}
                continue

            if not silent_eval:
                print(f"\n{'='*80}", flush=True)
                print(f"RUNNING {task.upper()} EVALUATION", flush=True)
                print(f"{'='*80}", flush=True)
            else:
                # Even in silent mode, show progress
                print(f"Evaluating {task.upper()}...", flush=True)

            # Load the module dynamically and call main to get results
            try:
                # Optionally suppress output from eval modules
                # Note: Disabled redirect to show metrics even in silent mode
                task_results = _run_task_eval(task, output_file, skip_llm_judge=skip_llm_judge)

                # Store the results for this task
                all_task_results[task] = task_results if task_results else {}
                print(f"[Progress] ✓ Completed {task.upper()} evaluation (Task {task_idx}/{len(tasks)})", flush=True)

            except Exception as e:
                print(f"Error running {task} evaluation: {e}", flush=True)
                all_task_results[task] = {}

    finally:
        # Restore original sys.argv
        sys.argv = original_argv

    # Print results based on grouping mode
    if grouping == "overall":
        print_overall_evaluation_results(output_file, tasks, all_task_results, skip_llm_judge=skip_llm_judge)
    else:  # per-dataset
        print_evaluation_results_csv_with_real_results(output_file, tasks, all_task_results)

    return all_task_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate multiple tasks on video understanding results")
    parser.add_argument("output_file",
                       help="Path to the JSON output file containing inference results")
    parser.add_argument("--tasks", nargs="+",
                       choices=["dvc", "tal", "next_action", "stg", "rc", "vs", "skill_assessment", "cvs_assessment", "gemini_structured", "gpt_structured"],
                       help="Specific tasks to evaluate (default: all available tasks)")
    parser.add_argument("--grouping", choices=["per-dataset", "overall"], default="per-dataset",
                       help="Grouping strategy: 'per-dataset' shows results per dataset, 'overall' aggregates all datasets (default: per-dataset)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze the file structure without running evaluations")
    parser.add_argument("--structured", choices=["gemini", "gpt"],
                       help="Evaluate structured outputs from Gemini or GPT models")
    parser.add_argument("--skip-llm-judge", action="store_true",
                       help="Skip LLM judge evaluation for caption tasks (DVC, VS, RC) - use when LLM scores are pre-computed")

    args = parser.parse_args()
    
    if args.analyze_only:
        qa_type_counts, dataset_counts = analyze_output_file(args.output_file)
        # Print CSV-style results summary for analyze-only mode
        # Determine available tasks based on what's in the file
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

        print_evaluation_results_csv(args.output_file, available_tasks)
    else:
        # Handle structured evaluation
        # Enable silent mode when using overall grouping
        silent_eval = (args.grouping == "overall")

        if args.structured:
            tasks = [f"{args.structured}_structured"]
            run_evaluation(args.output_file, tasks, grouping=args.grouping, silent_eval=silent_eval, skip_llm_judge=args.skip_llm_judge)
        else:
            run_evaluation(args.output_file, args.tasks, grouping=args.grouping, silent_eval=silent_eval, skip_llm_judge=args.skip_llm_judge)


if __name__ == "__main__":
    main()
