#!/usr/bin/env python3
"""
Server-Side LLM Judge Evaluator for DVC, VS, RC tasks

Runs LLM judge evaluation on the server for consistent scoring.
Uses OpenAI GPT-4.1 with V4 Best5 prompts (R2, R3, R8, R5, R4).

Falls back to semantic similarity if API key not available.
"""

import json
import sys
import os
import numpy as np
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from sentence_transformers import SentenceTransformer

# Try to import OpenAI (may not be available in all environments)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ OpenAI library not available, will use semantic similarity fallback")

# Constants for LLM judge
BEST5_ASPECTS = ['R2', 'R3', 'R8', 'R5', 'R4']
MAX_WORKERS = 10  # Parallel API calls
DEBUG_LOGGING = False  # Set to True to enable debug output for troubleshooting
progress_lock = Lock()
completed_calls = 0
total_calls = 0


def create_llm_judge_prompt(prediction: str, ground_truth: str, task_type: str) -> str:
    """Create LLM Judge V4 prompt for caption evaluation."""
    prompt = f"""You are an expert medical evaluator. Assess the quality of the predicted {task_type.replace('_', ' ')} against the ground truth.

**Ground Truth:** {ground_truth}

**Prediction:** {prediction}

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

**IMPORTANT - Response Format:**
You MUST respond with ONLY the scores in this exact format (no explanations):

R2: [score]
R3: [score]
R8: [score]
R5: [score]
R4: [score]

Example response:
R2: 4
R3: 3
R8: 5
R5: 4
R4: 5
"""
    return prompt


def call_llm_judge_api(prediction: str, ground_truth: str, task_type: str, api_key: str, max_retries=5) -> dict:
    """
    Call OpenAI API to evaluate a caption pair with retry logic.

    Args:
        prediction: Model's prediction text
        ground_truth: Ground truth text
        task_type: Task type (dense_captioning, video_summary, region_caption)
        api_key: OpenAI API key
        max_retries: Maximum number of retry attempts (default: 5)

    Returns:
        dict: Scores for each aspect + api_success flag
    """
    global completed_calls, total_calls

    if not OPENAI_AVAILABLE:
        return {aspect: 0 for aspect in BEST5_ASPECTS} | {'api_success': False}

    try:
        client = OpenAI(api_key=api_key)
        prompt = create_llm_judge_prompt(prediction, ground_truth, task_type)

        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1",  # GPT-4.1 (released April 2025)
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    timeout=30.0  # 30 second timeout per request
                )

                raw_response = response.choices[0].message.content

                # DEBUG: Log raw response for troubleshooting (controlled by DEBUG_LOGGING flag)
                if DEBUG_LOGGING and attempt == 0:  # Only log on first attempt to reduce noise
                    print(f"\n  [DEBUG] Raw GPT-4 response:\n{raw_response}\n")

                # Parse scores with flexible regex (handles variations like "R2: 4", "R2 = 4", "R2:4", etc.)
                scores = {}
                for aspect in BEST5_ASPECTS:
                    # Try multiple patterns in order of preference
                    patterns = [
                        f'{aspect}:\\s*(\\d+)',           # R2: 4
                        f'{aspect}\\s*=\\s*(\\d+)',       # R2 = 4
                        f'{aspect}\\s*:\\s*(\\d+)',       # R2 : 4
                        f'{aspect}\\s+(\\d+)',            # R2 4
                        f'\\*\\*{aspect}\\*\\*\\s*:\\s*(\\d+)',  # **R2**: 4
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, raw_response)
                        if match:
                            scores[aspect] = int(match.group(1))
                            break

                if len(scores) == len(BEST5_ASPECTS):
                    scores['api_success'] = True
                    scores['raw_response'] = raw_response

                    with progress_lock:
                        completed_calls += 1
                        if total_calls > 0 and completed_calls % 50 == 0:
                            print(f"  Progress: {completed_calls}/{total_calls} API calls completed")

                    return scores
                else:
                    # Failed to parse all scores - retry
                    last_error = f"Incomplete parsing: got {len(scores)}/{len(BEST5_ASPECTS)} scores"
                    print(f"  ⚠ {last_error}, parsed: {list(scores.keys())}")

                    # DEBUG: Show what we got vs what we expected (controlled by DEBUG_LOGGING flag)
                    if DEBUG_LOGGING and attempt == 0:
                        print(f"  [DEBUG] Expected aspects: {BEST5_ASPECTS}")
                        print(f"  [DEBUG] Parsed aspects: {list(scores.keys())}")
                        print(f"  [DEBUG] Raw response:\n{raw_response}\n")

                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 16)  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                        print(f"  Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

            except Exception as e:
                last_error = str(e)
                error_type = type(e).__name__

                # Determine if error is retryable
                is_rate_limit = 'rate_limit' in last_error.lower() or 'RateLimitError' in error_type
                is_timeout = 'timeout' in last_error.lower() or 'TimeoutError' in error_type
                is_network = 'connection' in last_error.lower() or 'ConnectionError' in error_type

                if attempt < max_retries - 1:
                    # Exponential backoff with longer waits for rate limits
                    if is_rate_limit:
                        wait_time = min(2 ** (attempt + 2), 60)  # 4, 8, 16, 32, 60 seconds for rate limits
                        print(f"  ⚠ Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    elif is_timeout or is_network:
                        wait_time = min(2 ** attempt, 16)
                        print(f"  ⚠ {error_type}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    else:
                        wait_time = min(2 ** attempt, 16)
                        print(f"  ⚠ API error: {error_type}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")

                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed
                    print(f"  ❌ API call failed after {max_retries} attempts: {error_type}")

    except Exception as e:
        print(f"  ❌ LLM Judge API error (client setup): {e}")

    # Failed after all retries
    with progress_lock:
        completed_calls += 1

    return {aspect: 0 for aspect in BEST5_ASPECTS} | {'api_success': False, 'error': last_error}


def run_llm_judge_evaluation(results_data, task_type, api_key):
    """
    Run server-side LLM judge evaluation on caption pairs.

    Args:
        results_data: List or dict of result records
        task_type: 'dense_captioning', 'video_summary', or 'region_caption'
        api_key: OpenAI API key

    Returns:
        dict with average scores and aspect breakdown
    """
    global completed_calls, total_calls

    if not OPENAI_AVAILABLE or not api_key:
        print("⚠ OpenAI not available, skipping LLM judge")
        return None

    # Determine records format
    if isinstance(results_data, dict):
        records = list(results_data.values())
    elif isinstance(results_data, list):
        records = results_data
    else:
        return None

    # Filter by task type
    task_records = [r for r in records if task_type in r.get('qa_type', '')]

    if not task_records:
        return None

    print(f"\n🔄 Running server-side LLM Judge evaluation on {len(task_records)} samples...")
    print(f"   Using OpenAI GPT-4 with V4 Best5 prompts")

    # Collect prediction-ground truth pairs
    caption_pairs = []
    for record in task_records:
        prediction = record.get('answer', record.get('response', ''))
        ground_truth = record.get('gnd', record.get('ground_truth', ''))

        # Handle list format for dense captioning
        if isinstance(prediction, list):
            prediction = ' '.join([item.get('caption', '') if isinstance(item, dict) else str(item) for item in prediction])
        if isinstance(ground_truth, list):
            ground_truth = ' '.join([item.get('caption', '') if isinstance(item, dict) else str(item) for item in ground_truth])

        if prediction and ground_truth:
            caption_pairs.append((prediction, ground_truth))

    if not caption_pairs:
        return None

    # Run parallel API calls
    total_calls = len(caption_pairs)
    completed_calls = 0

    all_scores = defaultdict(list)
    api_successes = []
    api_failures = []

    print(f"   Running {total_calls} API calls with {MAX_WORKERS} parallel workers...")
    print(f"   Max retries per call: 5 (with exponential backoff)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_llm_judge_api, pred, gt, task_type, api_key): i
                   for i, (pred, gt) in enumerate(caption_pairs)}

        for future in as_completed(futures):
            result = future.result()
            if result['api_success']:
                for aspect in BEST5_ASPECTS:
                    all_scores[aspect].append(result[aspect])
                api_successes.append(True)
            else:
                api_successes.append(False)
                if 'error' in result:
                    api_failures.append(result['error'])

    if not all_scores:
        print(f"❌ All API calls failed")
        if api_failures:
            print(f"   Sample errors: {api_failures[:3]}")
        return None

    # Compute averages
    aspect_averages = {aspect: np.mean(scores) for aspect, scores in all_scores.items()}
    overall_average = np.mean(list(aspect_averages.values()))

    success_rate = np.mean(api_successes) if api_successes else 0.0
    num_successes = sum(api_successes)
    num_failures = len(api_successes) - num_successes

    print(f"✓ LLM Judge completed: {num_successes}/{len(caption_pairs)} successful API calls")
    if num_failures > 0:
        print(f"  ⚠ {num_failures} calls failed after all retries")
        if api_failures:
            print(f"  Sample errors: {api_failures[:3]}")

    return {
        'average_score': overall_average,
        'aspect_scores': aspect_averages,
        'num_samples': len(caption_pairs),
        'api_success_rate': success_rate
    }


def extract_llm_judge_scores(results_data, task_type):
    """
    Extract pre-computed LLM judge scores from results file.

    Supports two formats:
    1. Direct format: record['llm_judge_scores'] = {R2: 4, R3: 3, ...}
    2. Nested format: record['evaluations'][threshold][idx]['llm_judge'] = {R2: 4, ...}

    Args:
        results_data: List or dict of result records
        task_type: 'dense_captioning', 'video_summary', or 'region_caption'

    Returns:
        dict: {
            'has_llm_judge': bool,
            'average_score': float (1-5 scale) or None,
            'num_samples': int,
            'aspect_scores': dict of {aspect: score}
        }
    """
    # Determine records format
    if isinstance(results_data, dict):
        records = list(results_data.values())
    elif isinstance(results_data, list):
        records = results_data
    else:
        return {'has_llm_judge': False, 'average_score': None, 'num_samples': 0}

    # Filter by task type
    task_records = [r for r in records if task_type in r.get('qa_type', '')]

    if not task_records:
        return {'has_llm_judge': False, 'average_score': None, 'num_samples': 0}

    # Try to extract LLM judge scores
    all_scores = defaultdict(list)
    samples_with_scores = 0

    for record in task_records:
        # Format 1: Direct llm_judge_scores field
        if 'llm_judge_scores' in record and isinstance(record['llm_judge_scores'], dict):
            scores = record['llm_judge_scores']
            for aspect, score in scores.items():
                if aspect != 'api_success' and isinstance(score, (int, float)):
                    all_scores[aspect].append(score)
            if len(scores) > 0:
                samples_with_scores += 1

        # Format 2: Nested evaluations structure (from eval_dvc_llm_judge_v4_best5_10k.py)
        elif 'evaluations' in record and isinstance(record['evaluations'], dict):
            # Check each IoU threshold
            for threshold, eval_list in record['evaluations'].items():
                if not isinstance(eval_list, list):
                    continue
                for eval_item in eval_list:
                    if 'llm_judge' in eval_item and isinstance(eval_item['llm_judge'], dict):
                        scores = eval_item['llm_judge']
                        for aspect, score in scores.items():
                            if aspect not in ['api_success', 'raw_response'] and isinstance(score, (int, float)):
                                all_scores[aspect].append(score)
                        if scores.get('api_success', False):
                            samples_with_scores += 1
                        break  # Only count first threshold per record

    if not all_scores or samples_with_scores == 0:
        return {'has_llm_judge': False, 'average_score': None, 'num_samples': len(task_records)}

    # Compute average across all aspects
    aspect_averages = {aspect: np.mean(scores) for aspect, scores in all_scores.items()}
    overall_average = np.mean(list(aspect_averages.values()))

    return {
        'has_llm_judge': True,
        'average_score': overall_average,
        'num_samples': len(task_records),
        'samples_with_scores': samples_with_scores,
        'aspect_scores': aspect_averages
    }


def compute_semantic_similarity_fallback(results_data, task_type):
    """
    Compute semantic similarity as fallback when LLM judge scores unavailable.

    Args:
        results_data: List or dict of result records
        task_type: 'dense_captioning', 'video_summary', or 'region_caption'

    Returns:
        float: Average semantic similarity score (0-1 scale)
    """
    # Load sentence transformer model
    print(f"Loading SentenceTransformer for {task_type} fallback evaluation...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Determine records format
    if isinstance(results_data, dict):
        records = list(results_data.values())
    elif isinstance(results_data, list):
        records = results_data
    else:
        return 0.0

    # Filter by task type
    task_records = [r for r in records if task_type in r.get('qa_type', '')]

    if not task_records:
        return 0.0

    similarities = []

    for record in task_records:
        prediction = record.get('answer', record.get('response', ''))
        ground_truth = record.get('gnd', record.get('ground_truth', ''))

        if not prediction or not ground_truth:
            continue

        # Handle list format (for dense captioning segments)
        if isinstance(prediction, list):
            prediction = ' '.join([item.get('caption', '') if isinstance(item, dict) else str(item) for item in prediction])
        if isinstance(ground_truth, list):
            ground_truth = ' '.join([item.get('caption', '') if isinstance(item, dict) else str(item) for item in ground_truth])

        if prediction and ground_truth:
            pred_emb = model.encode(prediction, convert_to_tensor=True)
            gt_emb = model.encode(ground_truth, convert_to_tensor=True)

            import torch
            similarity = torch.nn.functional.cosine_similarity(pred_emb.unsqueeze(0), gt_emb.unsqueeze(0)).item()
            similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.0


def evaluate_caption_task(results_file, task_type, api_key=None):
    """
    Main evaluation function for caption tasks.

    Priority order:
    1. Run server-side LLM judge (if API key provided)
    2. Extract pre-computed LLM judge scores (if available in results)
    3. Fall back to semantic similarity

    Args:
        results_file: Path to results JSON file
        task_type: 'dense_captioning', 'video_summary', or 'region_caption'
        api_key: OpenAI API key (optional, reads from env if not provided)

    Returns:
        dict: Evaluation results with score and method used
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {task_type.replace('_', ' ').title()}")
    print(f"{'='*60}")

    with open(results_file, 'r') as f:
        results_data = json.load(f)

    # Try to get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    # Option 1: Run server-side LLM judge (PREFERRED for consistency)
    if api_key:
        llm_results = run_llm_judge_evaluation(results_data, task_type, api_key)
        if llm_results:
            print(f"✓ Server-side LLM Judge completed")
            print(f"  Average Score: {llm_results['average_score']:.3f}")
            print(f"  API Success Rate: {llm_results['api_success_rate']:.1%}")
            print(f"  Aspect Scores:")
            for aspect, score in sorted(llm_results['aspect_scores'].items()):
                print(f"    {aspect}: {score:.3f}")

            return {
                'method': 'llm_judge_server',
                'score': llm_results['average_score'],
                'scale': '1-5',
                'aspect_scores': llm_results['aspect_scores'],
                'num_samples': llm_results['num_samples'],
                'api_success_rate': llm_results['api_success_rate']
            }

    # Option 2: Extract pre-computed scores (backward compatibility)
    llm_results = extract_llm_judge_scores(results_data, task_type)
    if llm_results['has_llm_judge']:
        print(f"✓ Found pre-computed LLM judge scores")
        print(f"  Samples with scores: {llm_results['samples_with_scores']}/{llm_results['num_samples']}")
        print(f"  Average LLM Judge Score: {llm_results['average_score']:.3f}")
        print(f"  Aspect Scores:")
        for aspect, score in sorted(llm_results['aspect_scores'].items()):
            print(f"    {aspect}: {score:.3f}")

        return {
            'method': 'llm_judge_precomputed',
            'score': llm_results['average_score'],
            'scale': '1-5',
            'aspect_scores': llm_results['aspect_scores'],
            'num_samples': llm_results['num_samples']
        }

    # Option 3: Semantic similarity fallback
    print(f"⚠ No API key and no pre-computed scores, using semantic similarity fallback")
    similarity = compute_semantic_similarity_fallback(results_data, task_type)
    print(f"  Semantic Similarity: {similarity:.4f}")

    return {
        'method': 'semantic_similarity',
        'score': similarity,
        'scale': '0-1',
        'num_samples': llm_results['num_samples']
    }


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 3:
        print("Usage: python eval_caption_hybrid.py <results_file> <task_type>")
        print("  task_type: dense_captioning, video_summary, or region_caption")
        sys.exit(1)

    results_file = sys.argv[1]
    task_type = sys.argv[2]

    result = evaluate_caption_task(results_file, task_type)

    print(f"\n{'='*60}")
    print(f"Final Result: {result['score']:.4f} ({result['method']}, {result['scale']} scale)")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    main()
