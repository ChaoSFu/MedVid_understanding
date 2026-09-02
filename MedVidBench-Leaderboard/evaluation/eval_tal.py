import json
import re
import numpy as np
from typing import Tuple
from collections import defaultdict


def extract_time_segments(text):
    print("="*50)
    print(text)
    segments = []

    # Match: from 12.1 to 117.0 / from 113.2s to 163.4s / from 10.0 seconds to 15.0 seconds
    pattern1 = re.findall(
        r'(?:from|is from|takes place from)?\s*'  # optional "from"
        r'(\d+(?:\.\d+)?)(?:s| seconds?)?\s*'
        r'to\s*'
        r'(\d+(?:\.\d+)?)(?:s| seconds?)?', text, flags=re.IGNORECASE)

    # Match: 00:00:00 to 00:00:08
    pattern2 = re.findall(
        r'(\d+):(\d+):(\d+)\s+to\s+(\d+):(\d+):(\d+)', text, flags=re.IGNORECASE)

    for start, end in pattern1:
        try:
            segments.append({
                'start': round(float(start), 2),
                'end': round(float(end), 2)
            })
        except:
            continue

    for h1, m1, s1, h2, m2, s2 in pattern2:
        start_sec = int(h1) * 3600 + int(m1) * 60 + int(s1)
        end_sec = int(h2) * 3600 + int(m2) * 60 + int(s2)
        segments.append({
            'start': float(start_sec),
            'end': float(end_sec)
        })

    return segments




def extract_segments_from_text(text):
    # Match patterns like 379-419 or 540-540
    # pattern = re.findall(r'(\d+)\s*-\s*(\d+)', text)
    pattern = re.findall(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', text)
    segments = []
    for start, end in pattern:
        segments.append({'start': float(start), 'end': float(end)})
    
    if not segments:
        # process raw, usually zero-shot answer
        segments = extract_time_segments(text)
        if not segments:
            print(f"Warning: No valid segments found in text: {text}")
    return segments


def compute_iou(seg1, seg2):
    inter_start = max(seg1['start'], seg2['start'])
    inter_end = min(seg1['end'], seg2['end'])
    inter = max(0, inter_end - inter_start)
    union = max(seg1['end'], seg2['end']) - min(seg1['start'], seg2['start'])
    return inter / union if union > 0 else 0.0

def evaluate_pair(preds, gts, tiou_thresh=0.5):
    gt_matched = [False] * len(gts)
    pred_matched = [False] * len(preds)
    matched_ious = []

    for i, gt in enumerate(gts):
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(preds):
            if pred_matched[j]:  # avoid multiple GTs matching same pred
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= tiou_thresh:
            gt_matched[i] = True
            pred_matched[best_j] = True
            matched_ious.append(best_iou)

    recall = sum(gt_matched) / len(gts) if gts else 0.0
    precision = sum(pred_matched) / len(preds) if preds else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    # print(f"types: recall={type(recall)}, precision={type(precision)}, f1={type(f1)}, mean_iou={type(mean_iou)}")

    return recall, precision, f1, mean_iou


def evaluate_tal_record(tal_record, tiou_thresh=0.5):
    recalls, precisions, f1s, mean_ious = [], [], [], []

    for entry in tal_record:
        preds = entry['prediction']
        gts = entry['ground_truth']
        recall, precision, f1, mean_iou = evaluate_pair(preds, gts, tiou_thresh)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        mean_ious.append(mean_iou)

        # for i, (r, p, f, mi) in enumerate(zip(recalls, precisions, f1s, mean_ious)):
        #     print(f"[{i}] types: recall={type(r)}, precision={type(p)}, f1={type(f)}, mean_iou={type(mi)}")
    
    def avg(x): return sum(x) / len(x) if x else 0.0

    return {
        f"Recall@{tiou_thresh:.2f}": avg(recalls),
        # f"Precision@{tiou_thresh:.2f}": avg(precisions),
        # f"F1@{tiou_thresh:.2f}": avg(f1s),
        f"meanIoU@{tiou_thresh:.2f}": avg(mean_ious),
    }
    
    
def pretty_print_summary(summary, label):
    # print(f"\n📊 {label}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


def group_records_by_dataset(data):
    """Group TAL records by dataset for per-dataset evaluation."""
    dataset_groups = defaultdict(list)

    for key, record in data.items():
        qa_type = record.get('qa_type', '')
        if 'tal' not in qa_type.lower():
            continue

        # Detect dataset from video_id or other fields
        video_id = record.get('video_id', '')
        # Check data_source first (used in leaderboard format), then fall back to dataset/dataset_name
        dataset = record.get('data_source', record.get('dataset', record.get('dataset_name', 'Unknown')))

        if dataset == 'Unknown' and video_id:
            # Try to infer from video_id patterns
            video_id_lower = str(video_id).lower()
            if len(video_id) == 11 and any(c.isalpha() for c in video_id):
                dataset = "AVOS"
            elif "_part" in video_id_lower:
                dataset = "CoPESD"
            elif "video" in video_id_lower:
                dataset = "CholecT50"

        dataset_groups[dataset].append(record)

    return dict(dataset_groups)


def evaluate_dataset_tal(dataset_name, records):
    """Evaluate TAL for a specific dataset."""
    print(f"\nEvaluating {dataset_name} ({len(records)} records)...")

    results_by_fps = defaultdict(list)

    for record in records:
        fps = record.get('fps', record.get('metadata', {}).get('fps', 1.0))
        if isinstance(fps, str):
            fps = float(fps)

        # Extract prediction and ground truth from record
        # Parse prediction from 'answer' field
        raw_answer = record.get('answer', '')
        predicted_segments = extract_segments_from_text(raw_answer)

        # Get ground truth spans from struc_info
        struc_info = record.get('struc_info', [])
        gt_spans = []
        if isinstance(struc_info, list):
            for item in struc_info:
                if 'spans' in item:
                    gt_spans.extend(item['spans'])

        # Convert to frames
        for seg in predicted_segments:
            seg['start'] = float(seg['start'] * fps)
            seg['end'] = float(seg['end'] * fps)
        for span in gt_spans:
            span['start'] = float(span['start'] * fps)
            span['end'] = float(span['end'] * fps)

        # Create formatted record for evaluation
        formatted_record = [{
            'prediction': predicted_segments,
            'ground_truth': gt_spans
        }]

        # Evaluate this record at both IoU thresholds
        result_03 = evaluate_tal_record(formatted_record, tiou_thresh=0.3)
        result_05 = evaluate_tal_record(formatted_record, tiou_thresh=0.5)
        results_by_fps[fps].append({'0.3': result_03, '0.5': result_05})

    # Aggregate results
    aggregated = {}
    for fps, results_list in results_by_fps.items():
        # Extract metrics from results at both thresholds
        all_recalls_03 = [r['0.3'].get(f'Recall@0.30', 0) for r in results_list if r]
        all_mean_ious_03 = [r['0.3'].get(f'meanIoU@0.30', 0) for r in results_list if r]
        all_recalls_05 = [r['0.5'].get(f'Recall@0.50', 0) for r in results_list if r]
        all_mean_ious_05 = [r['0.5'].get(f'meanIoU@0.50', 0) for r in results_list if r]

        if all_recalls_03:
            aggregated[f'fps_{fps}'] = {
                'recall@0.3': np.mean(all_recalls_03),
                'meanIoU@0.3': np.mean(all_mean_ious_03),
                'recall@0.5': np.mean(all_recalls_05),
                'meanIoU@0.5': np.mean(all_mean_ious_05),
                'count': len(all_recalls_03)
            }

    # Compute overall metrics (combining all FPS) if we have multiple FPS values
    if len(results_by_fps) > 1:
        # Collect all records' results across all FPS
        all_results_03 = []
        all_results_05 = []

        for fps, results_list in results_by_fps.items():
            for r in results_list:
                if r:
                    all_results_03.append(r['0.3'])
                    all_results_05.append(r['0.5'])

        # Compute overall averages
        if all_results_03:
            overall_recalls_03 = [r.get('Recall@0.30', 0) for r in all_results_03]
            overall_mean_ious_03 = [r.get('meanIoU@0.30', 0) for r in all_results_03]
            overall_recalls_05 = [r.get('Recall@0.50', 0) for r in all_results_05]
            overall_mean_ious_05 = [r.get('meanIoU@0.50', 0) for r in all_results_05]

            aggregated['IoU_0.3'] = {
                'Recall': np.mean(overall_recalls_03),
                'meanIoU': np.mean(overall_mean_ious_03),
            }
            aggregated['IoU_0.5'] = {
                'Recall': np.mean(overall_recalls_05),
                'meanIoU': np.mean(overall_mean_ious_05),
            }

    return aggregated


def main():
    """Main evaluation function for TAL."""
    import sys

    # Require results file via command line argument
    if len(sys.argv) < 2:
        print("Usage: python eval_tal.py <results_json_file>")
        print("Example: python eval_tal.py results/model_results.json")
        sys.exit(1)

    output_file = sys.argv[1]
    print(f"Loading results from: {output_file}")

    with open(output_file, "r") as f:
        infer_output = json.load(f)

    # Group by dataset
    dataset_records = group_records_by_dataset(infer_output)

    print(f"\nFound datasets: {list(dataset_records.keys())}")
    for dataset, records in dataset_records.items():
        print(f"  {dataset}: {len(records)} TAL records")

    if not any(dataset_records.values()):
        print("No TAL records found!")
        return

    # Evaluate each dataset
    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:
            results = evaluate_dataset_tal(dataset_name, records)
            all_results[dataset_name] = results

    # Print summary
    print(f"\n{'='*80}")
    print("TAL EVALUATION SUMMARY")
    print(f"{'='*80}")

    # Aggregate metrics across all datasets
    all_miou_03 = []
    all_miou_05 = []

    for dataset_name, fps_results in all_results.items():
        if fps_results:
            print(f"\n{dataset_name}:")
            for fps_key, metrics in sorted(fps_results.items()):
                print(f"  {fps_key}:")
                for metric_name, value in metrics.items():
                    if metric_name != 'count':
                        print(f"    {metric_name}: {value:.4f}")
                    else:
                        print(f"    samples: {value}")

                # Collect for overall average
                if 'meanIoU@0.3' in metrics:
                    all_miou_03.append(metrics['meanIoU@0.3'])
                if 'meanIoU@0.5' in metrics:
                    all_miou_05.append(metrics['meanIoU@0.5'])

    # Return per-dataset results for caching + macro averages
    return {
        'per_dataset': all_results,
        'meanIoU@0.3': np.mean(all_miou_03) if all_miou_03 else 0.0,
        'meanIoU@0.5': np.mean(all_miou_05) if all_miou_05 else 0.0
    }


if __name__ == "__main__":
    main()
