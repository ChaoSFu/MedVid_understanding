#!/usr/bin/env python3
"""
VS/RC Evaluation with LLM Judge V4 Best5 + Semantic Similarity
For 10k_model_eval checkpoints (steps 150, 165, 180, 195, 209)
"""

import json
import numpy as np
import csv
import re
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Constants
BEST5_ASPECTS = ['R2', 'R3', 'R8', 'R5', 'R4']
MAX_WORKERS = 20  # Parallel API calls

# Initialize models
print("Loading SentenceTransformer model...")
sem_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Initializing OpenAI client...")
openai_client = OpenAI()

# Thread-safe counter
progress_lock = Lock()
completed_calls = 0
total_calls = 0


def create_llm_judge_v4_prompt_vs(prediction: str, ground_truth: str) -> str:
    """Create LLM Judge V4 prompt for Video Summary evaluation."""
    prompt = f"""You are an expert medical evaluator. Assess the quality of the predicted video summary against the ground truth.

**Ground Truth Summary:** {ground_truth}

**Predicted Summary:** {prediction}

Evaluate the prediction on these 5 aspects (scale 1-5):

**R2 - Medical Terminology Accuracy (1-5):**
1 = Incorrect or inappropriate medical terms
2 = Some correct terms but frequent errors
3 = Generally correct with minor terminology issues
4 = Accurate terminology with rare imprecisions
5 = Precise and appropriate medical terminology throughout

**R3 - Anatomical/Instrument Detail (1-5):**
1 = Missing or incorrect anatomy/instrument details
2 = Vague or partially correct identifications
3 = Basic correct identification with some missing details
4 = Detailed and mostly comprehensive descriptions
5 = Comprehensive with precise anatomical/instrument specifics

**R8 - Specificity and Detail Level (1-5):**
1 = Extremely vague or generic
2 = Some specific elements but mostly general
3 = Moderately detailed with balanced specificity
4 = Detailed with good specifics on key elements
5 = Highly detailed and specific throughout

**R5 - Clinical Context Relevance (1-5):**
1 = Clinically irrelevant or misleading
2 = Minimal clinical relevance
3 = Somewhat relevant with missing context
4 = Clinically relevant with good context
5 = Highly relevant with comprehensive clinical context

**R4 - Action/State Accuracy (1-5):**
1 = Actions/states completely incorrect
2 = Some correct actions but frequent errors
3 = Generally correct with minor inaccuracies
4 = Accurate with rare minor issues
5 = Precisely captures all actions/states

**Response Format (required):**
R2: [score]
R3: [score]
R8: [score]
R5: [score]
R4: [score]
"""
    return prompt


def create_llm_judge_v4_prompt_rc(prediction: str, ground_truth: str) -> str:
    """Create LLM Judge V4 prompt for Region Caption evaluation."""
    prompt = f"""You are an expert medical evaluator. Assess the quality of the predicted region caption against the ground truth.

**Ground Truth Caption:** {ground_truth}

**Predicted Caption:** {prediction}

Evaluate the prediction on these 5 aspects (scale 1-5):

**R2 - Medical Terminology Accuracy (1-5):**
1 = Incorrect or inappropriate medical terms
2 = Some correct terms but frequent errors
3 = Generally correct with minor terminology issues
4 = Accurate terminology with rare imprecisions
5 = Precise and appropriate medical terminology throughout

**R3 - Anatomical/Instrument Detail (1-5):**
1 = Missing or incorrect anatomy/instrument details
2 = Vague or partially correct identifications
3 = Basic correct identification with some missing details
4 = Detailed and mostly comprehensive descriptions
5 = Comprehensive with precise anatomical/instrument specifics

**R8 - Specificity and Detail Level (1-5):**
1 = Extremely vague or generic
2 = Some specific elements but mostly general
3 = Moderately detailed with balanced specificity
4 = Detailed with good specifics on key elements
5 = Highly detailed and specific throughout

**R5 - Clinical Context Relevance (1-5):**
1 = Clinically irrelevant or misleading
2 = Minimal clinical relevance
3 = Somewhat relevant with missing context
4 = Clinically relevant with good context
5 = Highly relevant with comprehensive clinical context

**R4 - Action/State Accuracy (1-5):**
1 = Actions/states completely incorrect
2 = Some correct actions but frequent errors
3 = Generally correct with minor inaccuracies
4 = Accurate with rare minor issues
5 = Precisely captures all actions/states

**Response Format (required):**
R2: [score]
R3: [score]
R8: [score]
R5: [score]
R4: [score]
"""
    return prompt


def call_llm_judge(prediction: str, ground_truth: str, task_type: str, max_retries=3) -> dict:
    """Call LLM judge API (GPT-4.1) to evaluate a caption."""
    global completed_calls, total_calls

    # Select prompt based on task type
    if 'video_summary' in task_type:
        prompt = create_llm_judge_v4_prompt_vs(prediction, ground_truth)
    else:  # region_caption
        prompt = create_llm_judge_v4_prompt_rc(prediction, ground_truth)

    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            raw_response = response.choices[0].message.content

            # Parse scores
            scores = {}
            for aspect in BEST5_ASPECTS:
                pattern = f'{aspect}:\\s*(\\d+)'
                match = re.search(pattern, raw_response)
                if match:
                    scores[aspect] = int(match.group(1))

            if len(scores) == len(BEST5_ASPECTS):
                scores['api_success'] = True
                scores['raw_response'] = raw_response

                with progress_lock:
                    completed_calls += 1
                    if completed_calls % 50 == 0:
                        print(f"  Progress: {completed_calls}/{total_calls} API calls completed")

                return scores
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue

    # Failed
    with progress_lock:
        completed_calls += 1

    return {aspect: 0 for aspect in BEST5_ASPECTS} | {'api_success': False, 'raw_response': ''}


def compute_semantic_similarity(pred: str, ref: str):
    """Compute SentenceBERT cosine similarity."""
    if not pred or not ref:
        return None

    pred_emb = sem_model.encode(pred, convert_to_tensor=True)
    ref_emb = sem_model.encode(ref, convert_to_tensor=True)

    similarity = torch.nn.functional.cosine_similarity(
        pred_emb.unsqueeze(0),
        ref_emb.unsqueeze(0)
    ).item()

    return similarity


def filter_vs_rc_samples(data):
    """Filter samples to only VS and RC tasks."""
    print("\nFiltering VS/RC samples...")

    filtered = []
    task_counts = defaultdict(int)
    dataset_counts = defaultdict(int)

    for sample_id, item in data.items():
        qa_type = item.get('qa_type', '')

        # Filter for VS/RC only
        if not any(task in qa_type for task in ['video_summary', 'region_caption']):
            continue

        # Determine task type (remove gpt/gemini suffix)
        if 'video_summary' in qa_type:
            task = 'video_summary'
        elif 'region_caption' in qa_type:
            task = 'region_caption'
        else:
            continue

        dataset = item.get('data_source', 'Unknown')
        pred = item.get('answer', '')
        gt = item.get('gnd', '')

        if not pred or not gt:
            continue

        filtered.append({
            'sample_id': sample_id,
            'task': task,
            'dataset': dataset,
            'qa_type': qa_type,
            'prediction': pred,
            'ground_truth': gt
        })

        task_counts[task] += 1
        dataset_counts[dataset] += 1

    print(f"  ✓ Filtered {len(filtered)} VS/RC samples")
    print(f"\n  Task breakdown:")
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count}")
    print(f"\n  Dataset breakdown:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"    {dataset}: {count}")

    return filtered


def evaluate_single_sample(sample):
    """Evaluate a single sample with both metrics."""
    llm_result = call_llm_judge(
        sample['prediction'],
        sample['ground_truth'],
        sample['task']
    )
    sem_sim = compute_semantic_similarity(
        sample['prediction'],
        sample['ground_truth']
    )

    return {
        'sample_id': sample['sample_id'],
        'task': sample['task'],
        'dataset': sample['dataset'],
        'qa_type': sample['qa_type'],
        'prediction': sample['prediction'],
        'ground_truth': sample['ground_truth'],
        'llm_judge': llm_result,
        'semantic_similarity': sem_sim
    }


def evaluate_samples_parallel(filtered_samples, model_name):
    """Evaluate all samples in parallel."""
    global completed_calls, total_calls

    total_calls = len(filtered_samples)
    completed_calls = 0

    print(f"\nEvaluating {total_calls} samples with LLM Judge + Semantic Similarity...")
    print(f"  Using {MAX_WORKERS} parallel workers")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_sample = {
            executor.submit(evaluate_single_sample, sample): sample
            for sample in filtered_samples
        }

        for future in as_completed(future_to_sample):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                sample = future_to_sample[future]
                print(f"  ⚠️ Error evaluating {sample['sample_id']}: {e}")
                results.append({
                    'sample_id': sample['sample_id'],
                    'task': sample['task'],
                    'dataset': sample['dataset'],
                    'llm_judge': {aspect: 0 for aspect in BEST5_ASPECTS} | {'api_success': False},
                    'semantic_similarity': None
                })

    print(f"  ✓ Completed {completed_calls}/{total_calls} evaluations")
    return results


def aggregate_results(evaluated_samples):
    """Aggregate results by task and dataset."""
    # Overall aggregation
    overall_llm = {aspect: [] for aspect in BEST5_ASPECTS}
    overall_sem = []
    overall_api_success = []

    # Per-task aggregation
    per_task = defaultdict(lambda: {
        'llm': {aspect: [] for aspect in BEST5_ASPECTS},
        'sem': [],
        'api_success': []
    })

    # Per-dataset aggregation
    per_dataset = defaultdict(lambda: {
        'llm': {aspect: [] for aspect in BEST5_ASPECTS},
        'sem': [],
        'api_success': []
    })

    # Per-task-dataset aggregation
    per_task_dataset = defaultdict(lambda: defaultdict(lambda: {
        'llm': {aspect: [] for aspect in BEST5_ASPECTS},
        'sem': [],
        'api_success': []
    }))

    for sample in evaluated_samples:
        task = sample['task']
        dataset = sample['dataset']
        llm_result = sample['llm_judge']
        sem_sim = sample['semantic_similarity']

        # Collect LLM scores
        if llm_result['api_success']:
            for aspect in BEST5_ASPECTS:
                score = llm_result[aspect]
                overall_llm[aspect].append(score)
                per_task[task]['llm'][aspect].append(score)
                per_dataset[dataset]['llm'][aspect].append(score)
                per_task_dataset[task][dataset]['llm'][aspect].append(score)

        # Collect semantic similarity
        if sem_sim is not None:
            overall_sem.append(sem_sim)
            per_task[task]['sem'].append(sem_sim)
            per_dataset[dataset]['sem'].append(sem_sim)
            per_task_dataset[task][dataset]['sem'].append(sem_sim)

        # Collect API success
        overall_api_success.append(llm_result['api_success'])
        per_task[task]['api_success'].append(llm_result['api_success'])
        per_dataset[dataset]['api_success'].append(llm_result['api_success'])
        per_task_dataset[task][dataset]['api_success'].append(llm_result['api_success'])

    # Compute overall metrics
    overall_metrics = {
        'llm_judge': {aspect: np.mean(scores) if scores else None for aspect, scores in overall_llm.items()},
        'semantic_similarity': np.mean(overall_sem) if overall_sem else None,
        'api_success_rate': np.mean(overall_api_success) if overall_api_success else 0.0,
        'num_samples': len(evaluated_samples)
    }
    overall_metrics['llm_judge']['average'] = np.mean([v for v in overall_metrics['llm_judge'].values() if v is not None])

    # Compute per-task metrics
    per_task_metrics = {}
    for task, data in per_task.items():
        per_task_metrics[task] = {
            'llm_judge': {aspect: np.mean(scores) if scores else None for aspect, scores in data['llm'].items()},
            'semantic_similarity': np.mean(data['sem']) if data['sem'] else None,
            'api_success_rate': np.mean(data['api_success']) if data['api_success'] else 0.0,
            'num_samples': len(data['api_success'])
        }
        per_task_metrics[task]['llm_judge']['average'] = np.mean([v for v in per_task_metrics[task]['llm_judge'].values() if v is not None])

    # Compute per-dataset metrics
    per_dataset_metrics = {}
    for dataset, data in per_dataset.items():
        per_dataset_metrics[dataset] = {
            'llm_judge': {aspect: np.mean(scores) if scores else None for aspect, scores in data['llm'].items()},
            'semantic_similarity': np.mean(data['sem']) if data['sem'] else None,
            'api_success_rate': np.mean(data['api_success']) if data['api_success'] else 0.0,
            'num_samples': len(data['api_success'])
        }
        per_dataset_metrics[dataset]['llm_judge']['average'] = np.mean([v for v in per_dataset_metrics[dataset]['llm_judge'].values() if v is not None])

    # Compute per-task-dataset metrics
    per_task_dataset_metrics = {}
    for task, datasets in per_task_dataset.items():
        per_task_dataset_metrics[task] = {}
        for dataset, data in datasets.items():
            per_task_dataset_metrics[task][dataset] = {
                'llm_judge': {aspect: np.mean(scores) if scores else None for aspect, scores in data['llm'].items()},
                'semantic_similarity': np.mean(data['sem']) if data['sem'] else None,
                'api_success_rate': np.mean(data['api_success']) if data['api_success'] else 0.0,
                'num_samples': len(data['api_success'])
            }
            per_task_dataset_metrics[task][dataset]['llm_judge']['average'] = np.mean([v for v in per_task_dataset_metrics[task][dataset]['llm_judge'].values() if v is not None])

    return {
        'overall': overall_metrics,
        'per_task': per_task_metrics,
        'per_dataset': per_dataset_metrics,
        'per_task_dataset': per_task_dataset_metrics
    }


def process_model(model_name, file_path):
    """Process a single model: filter + evaluate."""
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"{'='*80}")

    print(f"Loading data from: {file_path}")
    with open(file_path) as f:
        data = json.load(f)

    filtered_samples = filter_vs_rc_samples(data)

    if not filtered_samples:
        print(f"  ⚠️ No VS/RC samples found!")
        return None

    evaluated_samples = evaluate_samples_parallel(filtered_samples, model_name)
    aggregated = aggregate_results(evaluated_samples)

    output_dir = Path('/root/code/Qwen2.5-VL/my_eval/results/vs_rc_llm_judge_v4_best5_10k')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{model_name}_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model_name': model_name,
            'evaluated_samples': evaluated_samples,
            'aggregated_results': aggregated
        }, f, indent=2)

    print(f"\n✓ Saved results: {output_file}")
    return aggregated


def generate_csv(all_results, output_dir):
    """Generate comprehensive CSV."""
    csv_file = output_dir / 'vs_rc_llm_judge_v4_best5_10k_complete.csv'
    print(f"\nGenerating CSV: {csv_file}")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Overall LLM Judge
        writer.writerow(['=== VS/RC LLM JUDGE V4 BEST5 - OVERALL ==='])
        writer.writerow(['Model'] + BEST5_ASPECTS + ['Average', 'Num Samples'])
        for model_name in sorted(all_results.keys()):
            row = [model_name]
            overall = all_results[model_name]['overall']['llm_judge']
            for aspect in BEST5_ASPECTS:
                val = overall[aspect]
                row.append(f"{val:.3f}" if val is not None else "N/A")
            row.append(f"{overall['average']:.3f}" if overall['average'] is not None else "N/A")
            row.append(all_results[model_name]['overall']['num_samples'])
            writer.writerow(row)

        writer.writerow([])

        # Overall Semantic
        writer.writerow(['=== VS/RC SEMANTIC SIMILARITY - OVERALL ==='])
        writer.writerow(['Model', 'Cosine Similarity', 'Num Samples'])
        for model_name in sorted(all_results.keys()):
            sem_sim = all_results[model_name]['overall']['semantic_similarity']
            num_samples = all_results[model_name]['overall']['num_samples']
            writer.writerow([
                model_name,
                f"{sem_sim:.4f}" if sem_sim is not None else "N/A",
                num_samples
            ])

        writer.writerow([])

        # Per-Task LLM Judge
        writer.writerow(['=== VS/RC LLM JUDGE V4 BEST5 - PER TASK ==='])
        writer.writerow(['Model', 'Task'] + BEST5_ASPECTS + ['Average', 'Num Samples'])
        for model_name in sorted(all_results.keys()):
            for task in ['video_summary', 'region_caption']:
                if task in all_results[model_name]['per_task']:
                    row = [model_name, task]
                    task_data = all_results[model_name]['per_task'][task]['llm_judge']
                    for aspect in BEST5_ASPECTS:
                        val = task_data[aspect]
                        row.append(f"{val:.3f}" if val is not None else "N/A")
                    row.append(f"{task_data['average']:.3f}" if task_data['average'] is not None else "N/A")
                    row.append(all_results[model_name]['per_task'][task]['num_samples'])
                    writer.writerow(row)

        writer.writerow([])

        # Per-Task Semantic
        writer.writerow(['=== VS/RC SEMANTIC SIMILARITY - PER TASK ==='])
        writer.writerow(['Model', 'Task', 'Cosine Similarity', 'Num Samples'])
        for model_name in sorted(all_results.keys()):
            for task in ['video_summary', 'region_caption']:
                if task in all_results[model_name]['per_task']:
                    sem_sim = all_results[model_name]['per_task'][task]['semantic_similarity']
                    num_samples = all_results[model_name]['per_task'][task]['num_samples']
                    writer.writerow([
                        model_name,
                        task,
                        f"{sem_sim:.4f}" if sem_sim is not None else "N/A",
                        num_samples
                    ])

        writer.writerow([])

        # Per-Dataset LLM Judge
        writer.writerow(['=== VS/RC LLM JUDGE V4 BEST5 - PER DATASET ==='])
        writer.writerow(['Model', 'Dataset'] + BEST5_ASPECTS + ['Average', 'Num Samples'])

        # Get all unique datasets
        all_datasets = set()
        for model_results in all_results.values():
            all_datasets.update(model_results['per_dataset'].keys())

        for model_name in sorted(all_results.keys()):
            for dataset in sorted(all_datasets):
                if dataset in all_results[model_name]['per_dataset']:
                    row = [model_name, dataset]
                    dataset_data = all_results[model_name]['per_dataset'][dataset]['llm_judge']
                    for aspect in BEST5_ASPECTS:
                        val = dataset_data[aspect]
                        row.append(f"{val:.3f}" if val is not None else "N/A")
                    row.append(f"{dataset_data['average']:.3f}" if dataset_data['average'] is not None else "N/A")
                    row.append(all_results[model_name]['per_dataset'][dataset]['num_samples'])
                    writer.writerow(row)

        writer.writerow([])

        # Per-Dataset Semantic
        writer.writerow(['=== VS/RC SEMANTIC SIMILARITY - PER DATASET ==='])
        writer.writerow(['Model', 'Dataset', 'Cosine Similarity', 'Num Samples'])
        for model_name in sorted(all_results.keys()):
            for dataset in sorted(all_datasets):
                if dataset in all_results[model_name]['per_dataset']:
                    sem_sim = all_results[model_name]['per_dataset'][dataset]['semantic_similarity']
                    num_samples = all_results[model_name]['per_dataset'][dataset]['num_samples']
                    writer.writerow([
                        model_name,
                        dataset,
                        f"{sem_sim:.4f}" if sem_sim is not None else "N/A",
                        num_samples
                    ])

    print(f"✓ Saved CSV: {csv_file}")


def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("VS/RC LLM JUDGE V4 BEST5 + SEMANTIC SIMILARITY EVALUATION")
    print("10k_model_eval checkpoints (steps 150, 165, 180, 195, 209)")
    print("="*80)

    # 5 checkpoints from 10k_model_eval
    models = {
        'step_150': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_150/results.json',
        'step_165': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_165/results.json',
        'step_180': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_180/results.json',
        'step_195': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_195/results.json',
        'step_209': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_209/results.json',
    }

    all_results = {}

    for model_name, file_path in models.items():
        try:
            aggregated = process_model(model_name, file_path)
            if aggregated:
                all_results[model_name] = aggregated
        except Exception as e:
            print(f"\n⚠️ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        output_dir = Path('/root/code/Qwen2.5-VL/my_eval/results/vs_rc_llm_judge_v4_best5_10k')
        generate_csv(all_results, output_dir)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total models evaluated: {len(all_results)}")


if __name__ == "__main__":
    main()
