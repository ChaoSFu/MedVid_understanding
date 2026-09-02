#!/usr/bin/env python3
"""
DVC Evaluation with LLM Judge V4 Best5 + Semantic Similarity
For 10k_model_eval checkpoints (steps 150, 165, 180, 195, 209)
Optimized: Match first, then parallel API calls
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
IOU_THRESHOLDS = [0.3, 0.5, 0.7]
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


def process_raw_output(raw_descriptions: str):
    """Parse DVC segments from string format."""
    if not raw_descriptions or not isinstance(raw_descriptions, str):
        return []

    text = raw_descriptions.strip()
    pattern = r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s+seconds?:\s+(.*?)(?=\n\d+(?:\.\d+)?-\d+(?:\.\d+)?\s+seconds?:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    segments = []
    for start_str, end_str, caption in matches:
        try:
            start = float(start_str)
            end = float(end_str)
            caption = caption.strip()
            if caption:
                segments.append({'start': start, 'end': end, 'caption': caption})
        except ValueError:
            continue

    return segments


def iou(segment1, segment2):
    """Compute Intersection over Union for two temporal segments."""
    start1, end1 = segment1['start'], segment1['end']
    start2, end2 = segment2['start'], segment2['end']
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    return intersection / union if union > 0 else 0.0


def match_captions_at_threshold(pred_segments, gt_segments, threshold):
    """Match predicted to ground truth segments at a specific IOU threshold."""
    matched_pairs = []
    for pred_seg in pred_segments:
        best_iou = 0.0
        best_gt_caption = None
        for gt_seg in gt_segments:
            current_iou = iou(pred_seg, gt_seg)
            if current_iou >= threshold and current_iou > best_iou:
                best_iou = current_iou
                best_gt_caption = gt_seg['caption']
        if best_gt_caption is not None:
            matched_pairs.append((pred_seg['caption'], best_gt_caption))
    return matched_pairs


def create_llm_judge_v4_prompt(prediction: str, ground_truth: str) -> str:
    """Create LLM Judge V4 prompt for DVC caption evaluation."""
    prompt = f"""You are an expert medical evaluator. Assess the quality of the predicted dense video caption against the ground truth.

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


def call_llm_judge(prediction: str, ground_truth: str, max_retries=3) -> dict:
    """Call LLM judge API (GPT-4.1) to evaluate a caption pair."""
    global completed_calls, total_calls

    prompt = create_llm_judge_v4_prompt(prediction, ground_truth)

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


def match_all_samples(data, data_type='baseline'):
    """Phase 1: Match all DVC samples at all IOU thresholds."""
    print(f"\nPhase 1: Matching segments at {len(IOU_THRESHOLDS)} IOU thresholds...")

    matched_samples = []

    if data_type == 'baseline':
        for sample_id, item in data.items():
            qa_type = item.get('qa_type', '')
            if 'dense_captioning' not in qa_type:
                continue

            pred_text = item.get('answer', '')
            gt_text = item.get('gnd', '')
            dataset = item.get('data_source', 'Unknown')

            pred_segments = process_raw_output(pred_text)
            gt_segments = process_raw_output(gt_text)

            if not pred_segments or not gt_segments:
                continue

            matched_pairs = {}
            for threshold in IOU_THRESHOLDS:
                pairs = match_captions_at_threshold(pred_segments, gt_segments, threshold)
                matched_pairs[threshold] = pairs

            matched_samples.append({
                'sample_id': sample_id,
                'dataset': dataset,
                'matched_pairs': matched_pairs
            })

    print(f"  ✓ Matched {len(matched_samples)} DVC samples")
    total_pairs = sum(sum(len(pairs) for pairs in sample['matched_pairs'].values()) for sample in matched_samples)
    print(f"  ✓ Total matched pairs across all thresholds: {total_pairs}")

    return matched_samples


def evaluate_single_pair(pred, gt):
    """Evaluate a single caption pair with both metrics."""
    llm_result = call_llm_judge(pred, gt)
    sem_sim = compute_semantic_similarity(pred, gt)
    return {'prediction': pred, 'ground_truth': gt, 'llm_judge': llm_result, 'semantic_similarity': sem_sim}


def evaluate_matched_pairs_parallel(matched_samples, model_name):
    """Phase 2: Evaluate all matched pairs in parallel."""
    global completed_calls, total_calls

    total_calls = sum(sum(len(pairs) for pairs in sample['matched_pairs'].values()) for sample in matched_samples)
    completed_calls = 0

    print(f"\nPhase 2: Evaluating {total_calls} caption pairs with LLM Judge + Semantic Similarity...")
    print(f"  Using {MAX_WORKERS} parallel workers")

    evaluation_tasks = []
    for sample_idx, sample in enumerate(matched_samples):
        for threshold in IOU_THRESHOLDS:
            for pair_idx, (pred, gt) in enumerate(sample['matched_pairs'][threshold]):
                evaluation_tasks.append((sample_idx, threshold, pair_idx, pred, gt))

    results = [{'sample_id': sample['sample_id'], 'dataset': sample['dataset'], 'evaluations': {t: [] for t in IOU_THRESHOLDS}} for sample in matched_samples]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(evaluate_single_pair, pred, gt): (sample_idx, threshold, pair_idx) for sample_idx, threshold, pair_idx, pred, gt in evaluation_tasks}

        for future in as_completed(future_to_task):
            sample_idx, threshold, pair_idx = future_to_task[future]
            try:
                eval_result = future.result()
                results[sample_idx]['evaluations'][threshold].append(eval_result)
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
                results[sample_idx]['evaluations'][threshold].append({
                    'llm_judge': {aspect: 0 for aspect in BEST5_ASPECTS} | {'api_success': False},
                    'semantic_similarity': None
                })

    print(f"  ✓ Completed {completed_calls}/{total_calls} evaluations")
    return results


def aggregate_results(evaluated_samples):
    """Aggregate results across thresholds."""
    per_threshold_results = {}

    for threshold in IOU_THRESHOLDS:
        llm_scores = {aspect: [] for aspect in BEST5_ASPECTS}
        sem_scores = []
        api_successes = []

        for sample in evaluated_samples:
            for eval_result in sample['evaluations'][threshold]:
                if eval_result['llm_judge']['api_success']:
                    for aspect in BEST5_ASPECTS:
                        llm_scores[aspect].append(eval_result['llm_judge'][aspect])
                api_successes.append(eval_result['llm_judge']['api_success'])

                if eval_result['semantic_similarity'] is not None:
                    sem_scores.append(eval_result['semantic_similarity'])

        per_threshold_results[threshold] = {
            'llm_judge': {aspect: np.mean(scores) if scores else None for aspect, scores in llm_scores.items()},
            'semantic_similarity': np.mean(sem_scores) if sem_scores else None,
            'api_success_rate': np.mean(api_successes) if api_successes else 0.0,
            'num_pairs': len(api_successes)
        }

        valid_scores = [v for v in per_threshold_results[threshold]['llm_judge'].values() if v is not None]
        per_threshold_results[threshold]['llm_judge']['average'] = np.mean(valid_scores) if valid_scores else None

    # Overall (threshold-averaged)
    overall_llm = {aspect: [] for aspect in BEST5_ASPECTS}
    overall_sem = []

    for threshold_data in per_threshold_results.values():
        for aspect in BEST5_ASPECTS:
            if threshold_data['llm_judge'][aspect] is not None:
                overall_llm[aspect].append(threshold_data['llm_judge'][aspect])
        if threshold_data['semantic_similarity'] is not None:
            overall_sem.append(threshold_data['semantic_similarity'])

    overall_results = {
        'llm_judge': {aspect: np.mean(scores) if scores else None for aspect, scores in overall_llm.items()},
        'semantic_similarity': np.mean(overall_sem) if overall_sem else None
    }

    valid_scores = [v for v in overall_results['llm_judge'].values() if v is not None]
    overall_results['llm_judge']['average'] = np.mean(valid_scores) if valid_scores else None

    return {'overall': overall_results, 'per_threshold': per_threshold_results}


def process_model(model_name, file_path, data_type='baseline'):
    """Process a single model: match + evaluate."""
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"{'='*80}")

    print(f"Loading data from: {file_path}")
    with open(file_path) as f:
        data = json.load(f)

    matched_samples = match_all_samples(data, data_type)

    if not matched_samples:
        print(f"  ⚠️ No DVC samples found!")
        return None

    evaluated_samples = evaluate_matched_pairs_parallel(matched_samples, model_name)
    aggregated = aggregate_results(evaluated_samples)

    output_dir = Path('/root/code/Qwen2.5-VL/my_eval/results/dvc_llm_judge_v4_best5_10k')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{model_name}_results.json'
    with open(output_file, 'w') as f:
        json.dump({'model_name': model_name, 'evaluated_samples': evaluated_samples, 'aggregated_results': aggregated}, f, indent=2)

    print(f"\n✓ Saved results: {output_file}")
    return aggregated


def generate_csv(all_results, output_dir):
    """Generate comprehensive CSV."""
    csv_file = output_dir / 'dvc_llm_judge_v4_best5_10k_complete.csv'
    print(f"\nGenerating CSV: {csv_file}")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Overall LLM Judge
        writer.writerow(['=== DVC LLM JUDGE V4 BEST5 - OVERALL (THRESHOLD-AVERAGED) ==='])
        writer.writerow(['Model'] + BEST5_ASPECTS + ['Average'])
        for model_name in sorted(all_results.keys()):
            row = [model_name]
            overall = all_results[model_name]['overall']['llm_judge']
            for aspect in BEST5_ASPECTS:
                val = overall[aspect]
                row.append(f"{val:.3f}" if val is not None else "N/A")
            row.append(f"{overall['average']:.3f}" if overall['average'] is not None else "N/A")
            writer.writerow(row)

        writer.writerow([])

        # Overall Semantic
        writer.writerow(['=== DVC SEMANTIC SIMILARITY - OVERALL (THRESHOLD-AVERAGED) ==='])
        writer.writerow(['Model', 'Cosine Similarity'])
        for model_name in sorted(all_results.keys()):
            sem_sim = all_results[model_name]['overall']['semantic_similarity']
            writer.writerow([model_name, f"{sem_sim:.4f}" if sem_sim is not None else "N/A"])

        writer.writerow([])

        # Per-Threshold LLM Judge
        writer.writerow(['=== DVC LLM JUDGE V4 BEST5 - PER THRESHOLD ==='])
        writer.writerow(['Model', 'IOU Threshold'] + BEST5_ASPECTS + ['Average'])
        for model_name in sorted(all_results.keys()):
            for threshold in IOU_THRESHOLDS:
                row = [model_name, f"{threshold:.1f}"]
                threshold_data = all_results[model_name]['per_threshold'][threshold]['llm_judge']
                for aspect in BEST5_ASPECTS:
                    val = threshold_data[aspect]
                    row.append(f"{val:.3f}" if val is not None else "N/A")
                row.append(f"{threshold_data['average']:.3f}" if threshold_data['average'] is not None else "N/A")
                writer.writerow(row)

        writer.writerow([])

        # Per-Threshold Semantic
        writer.writerow(['=== DVC SEMANTIC SIMILARITY - PER THRESHOLD ==='])
        writer.writerow(['Model', 'IOU Threshold', 'Cosine Similarity'])
        for model_name in sorted(all_results.keys()):
            for threshold in IOU_THRESHOLDS:
                sem_sim = all_results[model_name]['per_threshold'][threshold]['semantic_similarity']
                writer.writerow([model_name, f"{threshold:.1f}", f"{sem_sim:.4f}" if sem_sim is not None else "N/A"])

    print(f"✓ Saved CSV: {csv_file}")


def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("DVC LLM JUDGE V4 BEST5 + SEMANTIC SIMILARITY EVALUATION")
    print("10k_model_eval checkpoints (steps 150, 165, 180, 195, 209)")
    print("="*80)

    # 5 checkpoints from 10k_model_eval
    models = {
        'step_150': {
            'file': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_150/results.json',
            'type': 'baseline'
        },
        'step_165': {
            'file': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_165/results.json',
            'type': 'baseline'
        },
        'step_180': {
            'file': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_180/results.json',
            'type': 'baseline'
        },
        'step_195': {
            'file': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_195/results.json',
            'type': 'baseline'
        },
        'step_209': {
            'file': '/root/code/Qwen2.5-VL/my_vllm_infer/experiments/10k_model_eval/results/step_209/results.json',
            'type': 'baseline'
        },
    }

    all_results = {}

    for model_name, model_info in models.items():
        try:
            aggregated = process_model(model_name, model_info['file'], model_info['type'])
            if aggregated:
                all_results[model_name] = aggregated
        except Exception as e:
            print(f"\n⚠️ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        output_dir = Path('/root/code/Qwen2.5-VL/my_eval/results/dvc_llm_judge_v4_best5_10k')
        generate_csv(all_results, output_dir)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total models evaluated: {len(all_results)}")


if __name__ == "__main__":
    main()
