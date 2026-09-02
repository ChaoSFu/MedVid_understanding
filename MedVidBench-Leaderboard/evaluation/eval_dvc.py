"""Dense Video Captioning evaluation using LLM judge + temporal F1.

LLM judge uses IoU-matched segment pairs (matching original Qwen2.5-VL/llm_judge/):
- Match predicted segments to GT segments at IoU thresholds (0.3, 0.5, 0.7)
- Only judge matched pairs individually (not concatenated)
- Average across matched pairs, then across thresholds

Temporal F1 algorithm matches Qwen2.5-VL/my_eval/eval_dvc.py exactly:
- process_raw_output() + flatten_overlapping_segments() for parsing
- Frame-based coordinates (multiply by FPS)
- Many-to-many threshold matching across IoU (0.3, 0.5, 0.7)
- F1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
"""

import json
import os
import re
import sys
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from eval_caption_llm_judge import (
    call_llm_judge_api, BEST5_ASPECTS, OPENAI_AVAILABLE,
    compute_semantic_similarity_fallback
)


# =============================================================================
# Ported from Qwen2.5-VL/my_eval_old/eval_dvc.py - exact same algorithms
# =============================================================================

def zs_parse_multi_segment_annotations(raw_text: str):
    """Parse raw multiline string with multiple timestamped captions per line."""
    all_segments = []
    lines = raw_text.strip().split('\n')
    for line in lines:
        matches = re.findall(
            r"(?:\*\*Start Time:\*\*|Start\s*\(?Time\)?|Time\s*Range:|Time\s*Interval:|^|\n)\s*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*seconds?.*?(?:\*\*Description:\*\*|-)\s*(.+?)(?=\n\d|$)",
            line, flags=re.DOTALL
        )
        for start, end, caption in matches:
            all_segments.append({
                "start": float(start),
                "end": float(end),
                "caption": caption.strip().rstrip('.')
            })
    return all_segments


def process_raw_output(raw_descriptions: str):
    """Process raw frame-wise descriptions into structured segments."""
    pattern = r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s+seconds?:\s+(.*?)(?=\n\d+(?:\.\d+)?-\d+(?:\.\d+)?\s+seconds?:|\Z)"
    matches = re.findall(pattern, raw_descriptions, re.DOTALL)

    segments = []
    for start, end, desc in matches:
        segments.append({
            "start": float(start),
            "end": float(end),
            "caption": desc.strip().replace("\n", " ")
        })

    # Remove duplicate (start, end) segments
    seen = set()
    unique_segments = []
    for seg in segments:
        key = (seg["start"], seg["end"])
        if key not in seen:
            seen.add(key)
            unique_segments.append(seg)

    if not unique_segments:
        unique_segments = zs_parse_multi_segment_annotations(raw_descriptions)

    return unique_segments


def check_for_overlaps(segments):
    """Check a list of temporal segments for any overlaps."""
    sorted_segs = sorted(segments, key=lambda x: (x['start'], x['end']))
    overlaps = []
    for i in range(len(sorted_segs) - 1):
        seg1 = sorted_segs[i]
        seg2 = sorted_segs[i + 1]
        if seg2["start"] < seg1["end"]:
            overlaps.append((seg1, seg2))
    return overlaps


def flatten_overlapping_segments(segments, caption_strategy="longest"):
    """Split overlapping segments into non-overlapping intervals."""
    time_points = sorted(set([s["start"] for s in segments] + [s["end"] for s in segments]))
    result = []
    for i in range(len(time_points) - 1):
        start = time_points[i]
        end = time_points[i + 1]
        overlapping = []
        for s in segments:
            if s["start"] < end and s["end"] > start:
                overlapping.append(s)
        if not overlapping:
            continue
        if caption_strategy == "longest":
            selected = max(overlapping, key=lambda x: x["end"] - x["start"])
        elif caption_strategy == "first":
            selected = overlapping[0]
        else:
            raise ValueError("Unsupported strategy")
        result.append({
            "start": start,
            "end": end,
            "caption": selected["caption"]
        })
    return result


def iou(interval_1, interval_2):
    """Compute IoU between two intervals - matches old eval exactly."""
    start_1, end_1 = min(*interval_1), max(*interval_1)
    start_2, end_2 = min(*interval_2), max(*interval_2)

    intersection = max(0, min(end_1, end_2) - max(start_1, start_2))
    union = min(
        max(end_1, end_2) - min(start_1, start_2),
        end_1 - start_1 + end_2 - start_2)
    result = float(intersection) / (union + 1e-8)
    return result


def evaluate_detections(predicted_segments, gt_segments, splits,
                        iou_thresholds=(0.3, 0.5, 0.7, 0.9)):
    """Compute P/R between predicted and ground truth segments.

    Many-to-many matching: any pred-gt pair exceeding threshold counts as covered.
    """
    best_recall = []
    best_precision = []

    predicted_shape = predicted_segments.shape[0]

    for split in set(splits):
        metrics = {}
        for threshold in iou_thresholds:
            metrics[str(threshold)] = {
                'gt_covered': set(),
                'pred_covered': set(),
            }
        split_idx = np.where(splits == split)[0]
        split_gt_segments = np.array([gt_segments[idx] for idx in split_idx])
        gt_shape = split_gt_segments.shape[0]

        for idx_g, gt_segment in enumerate(split_gt_segments):
            for idx_p, segment in enumerate(predicted_segments):
                sample_iou = iou(segment, gt_segment)
                for threshold in iou_thresholds:
                    if sample_iou > threshold:
                        metrics[str(threshold)]['pred_covered'].add(idx_p)
                        metrics[str(threshold)]['gt_covered'].add(idx_g)

        for threshold, m in metrics.items():
            pred_covered = m['pred_covered']
            gt_covered = m['gt_covered']
            m['precision'] = float(len(pred_covered)) / max(float(predicted_shape), 1.0)
            m['recall'] = float(len(gt_covered)) / float(gt_shape)

        precision = [m['precision'] for m in metrics.values()]
        recall = [m['recall'] for m in metrics.values()]
        if best_precision:
            best_precision = [max(precision[i], best_precision[i]) for i in range(len(precision))]
            best_recall = [max(recall[i], best_recall[i]) for i in range(len(recall))]
        else:
            best_precision, best_recall = precision, recall

    return best_precision, best_recall


def compute_temporal_f1_single(predicted_segments, gt_segments, splits,
                               iou_thresholds=(0.3, 0.5, 0.7)):
    """Compute temporal F1 for a single sample using the old eval algorithm.

    Returns dict with Precision_Mean, Recall_Mean, F1_Score.
    """
    if predicted_segments.shape[0] == 0 or gt_segments.shape[0] == 0:
        return {'Precision_Mean': 0.0, 'Recall_Mean': 0.0, 'F1_Score': 0.0}

    detection_precision, detection_recall = evaluate_detections(
        predicted_segments, gt_segments, splits, iou_thresholds
    )

    mean_precision = sum(detection_precision) / len(detection_precision)
    mean_recall = sum(detection_recall) / len(detection_recall)
    f1 = 2 * mean_recall * mean_precision / (mean_recall + mean_precision) \
        if (mean_recall + mean_precision) > 0 else 0.0

    return {
        'Precision_Mean': float(mean_precision),
        'Recall_Mean': float(mean_recall),
        'F1_Score': float(f1),
    }


# =============================================================================
# Dataset grouping and evaluation (LlamaFactory specific)
# =============================================================================

def group_records_by_dataset(data):
    """Group DVC records by dataset for per-dataset evaluation."""
    dataset_groups = defaultdict(list)

    for key, record in data.items():
        qa_type = record.get('qa_type', '')
        # Match any dense_captioning variant (dense_captioning, dense_captioning_gpt, dense_captioning_gemini, dc)
        if not any(x in qa_type.lower() for x in ['dense_captioning', 'dense_caption', 'dc']):
            continue

        # Check data_source first (leaderboard format), then fall back to dataset/dataset_name
        dataset = record.get('data_source', record.get('dataset', record.get('dataset_name', record.get('metadata', {}).get('dataset', 'Unknown'))))
        video_id = record.get('video_id', record.get('metadata', {}).get('video_id', ''))

        if dataset == 'Unknown' and video_id:
            video_id_lower = str(video_id).lower()
            if len(video_id) == 11 and any(c.isalpha() for c in video_id):
                dataset = "AVOS"
            elif "_part" in video_id_lower:
                dataset = "CoPESD"
            elif "video" in video_id_lower:
                dataset = "CholecT50"

        dataset_groups[dataset].append(record)

    return dict(dataset_groups)


def _extract_gt_segments(record):
    """Extract ground truth segments from struc_info, matching Qwen2.5-VL logic."""
    struc_info = record.get('struc_info', [])

    if isinstance(struc_info, list) and len(struc_info) > 0:
        if isinstance(struc_info[0], list):
            # Format: [[{segments...}]]
            gnd = struc_info[0]
        elif isinstance(struc_info[0], dict) and 'dc_segments' in struc_info[0]:
            # NurViD format: [{'dc_segments': [...]}]
            gnd = struc_info[0]['dc_segments']
        else:
            # Format: [{segments...}]
            gnd = struc_info
    else:
        gnd = struc_info

    return gnd


DVC_IOU_THRESHOLDS = [0.3, 0.5, 0.7]
DVC_MAX_WORKERS = 20

# Thread-safe progress counter for DVC LLM judge
_dvc_progress_lock = Lock()
_dvc_completed = 0
_dvc_total = 0


def _segment_iou(seg1, seg2):
    """Compute IoU for two temporal segments (dicts with 'start' and 'end')."""
    intersection = max(0, min(seg1['end'], seg2['end']) - max(seg1['start'], seg2['start']))
    union = (seg1['end'] - seg1['start']) + (seg2['end'] - seg2['start']) - intersection
    return intersection / union if union > 0 else 0.0


def _match_captions_at_threshold(pred_segments, gt_segments, threshold):
    """Match predicted to ground truth segments at a specific IoU threshold.

    Returns list of (pred_caption, gt_caption) pairs.
    """
    matched_pairs = []
    for pred_seg in pred_segments:
        best_iou = 0.0
        best_gt_caption = None
        for gt_seg in gt_segments:
            current_iou = _segment_iou(pred_seg, gt_seg)
            if current_iou >= threshold and current_iou > best_iou:
                best_iou = current_iou
                best_gt_caption = gt_seg['caption']
        if best_gt_caption is not None:
            matched_pairs.append((pred_seg['caption'], best_gt_caption))
    return matched_pairs


def _evaluate_dvc_caption_iou_matched(records, api_key):
    """Evaluate DVC captions using IoU-matched segment pairs + LLM judge.

    Matches the original Qwen2.5-VL/llm_judge/ approach:
    1. Parse pred and GT into segments
    2. Match at IoU thresholds (0.3, 0.5, 0.7)
    3. Judge each matched pair individually
    4. Average across pairs, then across thresholds
    """
    global _dvc_completed, _dvc_total

    # Phase 1: Match all samples at all thresholds
    print(f"  Phase 1: Matching segments at IoU thresholds {DVC_IOU_THRESHOLDS}...")
    all_matched = []

    for record in records:
        pred_text = record.get('answer', '')
        gt_text = record.get('gnd', '')

        pred_segments = process_raw_output(pred_text)
        gt_segments = _extract_gt_segments(record)

        if not isinstance(gt_segments, list):
            continue

        # Ensure gt_segments are dicts with caption
        gt_segs = [g for g in gt_segments if isinstance(g, dict) and 'start' in g and 'end' in g and 'caption' in g]

        if not pred_segments or not gt_segs:
            continue

        matched_pairs = {}
        for threshold in DVC_IOU_THRESHOLDS:
            pairs = _match_captions_at_threshold(pred_segments, gt_segs, threshold)
            matched_pairs[threshold] = pairs

        all_matched.append(matched_pairs)

    total_pairs = sum(sum(len(pairs) for pairs in m.values()) for m in all_matched)
    print(f"  ✓ Matched {len(all_matched)} samples, {total_pairs} total pairs across all thresholds")

    if total_pairs == 0:
        return 0.0, 'llm_judge_iou_matched', 0.0

    # Phase 2: Evaluate all matched pairs in parallel
    _dvc_total = total_pairs
    _dvc_completed = 0

    print(f"  Phase 2: Evaluating {total_pairs} pairs with LLM Judge ({DVC_MAX_WORKERS} workers)...")

    # Collect all tasks: (sample_idx, threshold, pred_caption, gt_caption)
    tasks = []
    for sample_idx, matched_pairs in enumerate(all_matched):
        for threshold in DVC_IOU_THRESHOLDS:
            for pred_cap, gt_cap in matched_pairs[threshold]:
                tasks.append((sample_idx, threshold, pred_cap, gt_cap))

    # Store results per threshold
    threshold_scores = {t: {aspect: [] for aspect in BEST5_ASPECTS} for t in DVC_IOU_THRESHOLDS}
    api_successes = 0

    def _judge_pair(pred_cap, gt_cap):
        global _dvc_completed
        result = call_llm_judge_api(pred_cap, gt_cap, 'dense_captioning', api_key)
        with _dvc_progress_lock:
            _dvc_completed += 1
            if _dvc_completed % 50 == 0:
                print(f"    Progress: {_dvc_completed}/{_dvc_total} API calls completed")
        return result

    with ThreadPoolExecutor(max_workers=DVC_MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(_judge_pair, pred_cap, gt_cap): (sample_idx, threshold)
            for sample_idx, threshold, pred_cap, gt_cap in tasks
        }

        for future in as_completed(future_to_task):
            _, threshold = future_to_task[future]
            try:
                result = future.result()
                if result.get('api_success', False):
                    for aspect in BEST5_ASPECTS:
                        threshold_scores[threshold][aspect].append(result[aspect])
                    api_successes += 1
            except Exception as e:
                print(f"    ⚠ Error: {e}")

    # Phase 3: Aggregate — average per threshold, then across thresholds
    per_threshold_avg = {}
    for threshold in DVC_IOU_THRESHOLDS:
        aspect_avgs = {}
        for aspect in BEST5_ASPECTS:
            scores = threshold_scores[threshold][aspect]
            aspect_avgs[aspect] = np.mean(scores) if scores else 0.0
        valid = [v for v in aspect_avgs.values() if v > 0]
        per_threshold_avg[threshold] = np.mean(valid) if valid else 0.0

    # Overall: average across thresholds
    valid_thresholds = [v for v in per_threshold_avg.values() if v > 0]
    overall_score = np.mean(valid_thresholds) if valid_thresholds else 0.0
    success_rate = api_successes / total_pairs if total_pairs > 0 else 0.0

    print(f"  ✓ LLM Judge completed: {api_successes}/{total_pairs} successful")
    for t in DVC_IOU_THRESHOLDS:
        print(f"    IoU@{t}: {per_threshold_avg[t]:.3f}")
    print(f"    Overall (threshold-averaged): {overall_score:.3f}")

    return overall_score, 'llm_judge_iou_matched', success_rate


def evaluate_dataset_dvc(dataset_name, records, skip_llm_judge=False):
    """Evaluate DVC for a specific dataset using caption quality + temporal F1."""
    print(f"\nEvaluating {dataset_name} ({len(records)} records)...")

    # Step 1: Evaluate caption quality using IoU-matched LLM judge
    if skip_llm_judge:
        print(f"  Skipping LLM judge caption evaluation (--skip-llm-judge flag)")
        caption_score = 0.0
        caption_method = 'skipped'
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and OPENAI_AVAILABLE:
            caption_score, caption_method, _ = _evaluate_dvc_caption_iou_matched(records, api_key)
        else:
            print(f"  ⚠ No API key, using semantic similarity fallback")
            import tempfile
            temp_data = {str(i): record for i, record in enumerate(records)}
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(temp_data, f)
                temp_file = f.name
            try:
                caption_score = compute_semantic_similarity_fallback(temp_data, 'dense_captioning')
                caption_method = 'semantic_similarity'
            finally:
                os.unlink(temp_file)

    # Step 2: Compute temporal F1 matching Qwen2.5-VL algorithm exactly
    all_f1_scores = []
    all_precision_scores = []
    all_recall_scores = []

    for record in records:
        # Get FPS
        fps = record.get('fps', record.get('metadata', {}).get('fps', 1.0))
        if isinstance(fps, str):
            fps = float(fps)

        # Parse predicted segments using process_raw_output (same as Qwen2.5-VL)
        raw_answer = record.get('answer', '')
        processed_answer = process_raw_output(raw_answer)
        overlaps = check_for_overlaps(processed_answer)
        if overlaps:
            processed_answer = flatten_overlapping_segments(processed_answer, caption_strategy="longest")

        # Get ground truth segments
        gnd = _extract_gt_segments(record)

        # Convert both to frame-based coordinates (multiply by fps, cast to int)
        # IMPORTANT: require 'caption' field to match Qwen2.5-VL's prepare_eval_arrays
        gt_segments = []
        if isinstance(gnd, list):
            for g in gnd:
                if isinstance(g, dict) and 'start' in g and 'end' in g and 'caption' in g:
                    gt_segments.append([int(float(g['start']) * fps), int(float(g['end']) * fps)])

        pred_segments = []
        if isinstance(processed_answer, list):
            for p in processed_answer:
                if isinstance(p, dict) and 'start' in p and 'end' in p and 'caption' in p:
                    pred_segments.append([int(p['start'] * fps), int(p['end'] * fps)])

        # Compute F1 using many-to-many matching across IoU thresholds (0.3, 0.5, 0.7)
        if pred_segments and gt_segments:
            pred_np = np.array(pred_segments)
            gt_np = np.array(gt_segments)
            splits = np.ones(len(gt_segments), dtype=int)

            result = compute_temporal_f1_single(pred_np, gt_np, splits,
                                                iou_thresholds=(0.3, 0.5, 0.7))
            all_f1_scores.append(result['F1_Score'])
            all_precision_scores.append(result['Precision_Mean'])
            all_recall_scores.append(result['Recall_Mean'])

    # Aggregate scores
    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
    avg_precision = np.mean(all_precision_scores) if all_precision_scores else 0.0
    avg_recall = np.mean(all_recall_scores) if all_recall_scores else 0.0

    return {
        'overall': {
            'caption_score': caption_score,
            'caption_method': caption_method,
            'temporal_f1': avg_f1,
            'temporal_precision': avg_precision,
            'temporal_recall': avg_recall,
            'count': len(records),
            'f1_samples': len(all_f1_scores)
        }
    }


def main():
    """Main evaluation function for DVC."""
    if len(sys.argv) < 2:
        print("Usage: python eval_dvc.py <results_json_file> [--skip-llm-judge]")
        print("Example: python eval_dvc.py results/model_results.json")
        print("Example: python eval_dvc.py results/model_results.json --skip-llm-judge")
        sys.exit(1)

    output_file = sys.argv[1]
    skip_llm_judge = '--skip-llm-judge' in sys.argv

    print(f"Loading results from: {output_file}")
    if skip_llm_judge:
        print("  --skip-llm-judge flag detected: Skipping caption evaluation, computing temporal F1 only")

    with open(output_file, "r") as f:
        infer_output = json.load(f)

    dataset_records = group_records_by_dataset(infer_output)

    print(f"\nFound datasets: {list(dataset_records.keys())}")
    for dataset, records in dataset_records.items():
        print(f"  {dataset}: {len(records)} DVC records")

    if not any(dataset_records.values()):
        print("No DVC records found!")
        return {}

    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:
            results = evaluate_dataset_dvc(dataset_name, records, skip_llm_judge=skip_llm_judge)
            all_results[dataset_name] = results

    print(f"\n{'='*80}")
    print("DENSE VIDEO CAPTIONING EVALUATION SUMMARY")
    print(f"{'='*80}")

    all_caption_scores = []
    all_f1_scores = []

    for dataset_name, results in all_results.items():
        if results:
            print(f"\n{dataset_name}:")
            for key, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"  Caption Score ({metrics.get('caption_method', 'unknown')}): {metrics.get('caption_score', 0):.4f}")
                    print(f"  Temporal F1: {metrics.get('temporal_f1', 0):.4f}")
                    print(f"  Temporal Precision: {metrics.get('temporal_precision', 0):.4f}")
                    print(f"  Temporal Recall: {metrics.get('temporal_recall', 0):.4f}")
                    print(f"  Total samples: {metrics.get('count', 0)}")
                    print(f"  F1 computed on: {metrics.get('f1_samples', 0)} samples")

                    all_caption_scores.append(metrics.get('caption_score', 0))
                    all_f1_scores.append(metrics.get('temporal_f1', 0))

    return {
        'per_dataset': all_results,
        'caption_score': np.mean(all_caption_scores) if all_caption_scores else 0.0,
        'temporal_f1': np.mean(all_f1_scores) if all_f1_scores else 0.0,
        'method': all_results[list(all_results.keys())[0]]['overall'].get('caption_method', 'unknown') if all_results else 'unknown'
    }


if __name__ == "__main__":
    main()
