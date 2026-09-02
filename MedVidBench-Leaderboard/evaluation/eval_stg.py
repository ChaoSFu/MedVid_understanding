import json
import re
import numpy as np
from typing import Tuple
from collections import defaultdict

'''
dict_keys(['struc_info', 'metadata', 'qa_type', 'question', 'answer', 'gnd'])

'''
def extract_boxes(raw_output):
    print("="*50)
    print(raw_output)
    '''
    for raw output
    '''
    pattern = re.compile(r"\[\[([\d\s,]+)\]\]")
    matches = pattern.findall(raw_output)

    boxes = []
    for match in matches:
        try:
            box = [int(x.strip()) for x in match.split(',')]
            if len(box) == 4:
                boxes.append(box)
        except:
            continue
    return boxes


# def post_process_pred(raw_output):
#     parsed_prediction = {}
    
#     # pattern = r"(\d+)\s+seconds:\s+\[([^\]]+)\]"
#     pattern = r"(\d+(?:\.\d+)?)\s+seconds:\s*\[([^\]]+)\]"
#     matches = re.findall(pattern, raw_output)

#     if not matches:
#         # print("No valid matches found in prediction output.")
#         # print(f"Raw output: {raw_output}")
#         boxes = extract_boxes(raw_output)
#         # print(f"Extracted boxes: {boxes}")
#         return boxes  # or return None, or raise ValueError
    
#     parsed_prediction = {
#         k: [float(num) for num in v.split(', ')]
#         for k, v in matches
#     }

#     return parsed_prediction

def post_process_pred(raw_output):
    """
    Parses STG-style prediction text into a dictionary {time_key: [x1, y1, x2, y2]}.

    Supports float second keys like '8.0 seconds: [x1, y1, x2, y2]'

    If parsing fails, fall back to extract_boxes().
    """
    pattern = r"(\d+(?:\.\d+)?)\s+seconds:\s*\[([^\]]+)\]"
    matches = re.findall(pattern, raw_output)

    if not matches:
        # Fall back to raw box list extraction
        return extract_boxes(raw_output)
    # print(raw_output)
    # print(matches)
    # print()
    # parsed_prediction = {
    #     str(float(k)): [float(num) for num in v.split(',') if num.strip()]
    #     for k, v in matches
    # }
    parsed_prediction = {}
    last_valid_box = None
    for k, v in matches:
        try:
            nums = []
            for num in v.split(','):
                num_clean = num.strip().lstrip('[').rstrip(']')
                nums.append(float(num_clean))
            if len(nums) != 4:
                raise ValueError("Box should have 4 values.")
            parsed_prediction[str(float(k))] = nums
            last_valid_box = nums
        except ValueError:
            print(f"[Outlier] Failed to parse entry at time {k}: {v}")
            print(f"Raw output line: {k} seconds: [{v}]")
            print("---")
            if last_valid_box is not None:
                parsed_prediction[str(float(k))] = last_valid_box
            else:
                print(f"[Warning] No valid box available to copy for time {k}")

    return parsed_prediction
    
    
    
    
    # print(f"Parsed prediction: {parsed_prediction}")
    return parsed_prediction

def is_valid_box(box):
    return isinstance(box, list) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box)


def np_box_area(boxes: np.array) -> np.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = np_box_area(boxes1)
    area2 = np_box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def np_box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

def validate_prediction_and_gt(pred_dict, gt_dict):
    pred_keys = set(pred_dict.keys())
    gt_keys = set(gt_dict.keys())

    if pred_keys != gt_keys:
        missing_in_pred = gt_keys - pred_keys
        missing_in_gt = pred_keys - gt_keys
        print("Key mismatch:")
        if missing_in_pred:
            print(" - Missing in prediction:", missing_in_pred)
        if missing_in_gt:
            print(" - Missing in ground truth:", missing_in_gt)
        return False

    for k in pred_keys:
        if not is_valid_box(pred_dict[k]):
            print(f"Invalid prediction box for key {k}: {pred_dict[k]}")
            return False
        if not is_valid_box(gt_dict[k]):
            print(f"Invalid ground truth box for key {k}: {gt_dict[k]}")
            return False

    # print("✅ All keys match and all boxes are valid.")
    return True


def compute_iou_batch(boxes1, boxes2):
    """
    boxes1, boxes2: (N, 4) arrays where each row is [x1, y1, x2, y2]
    """
    # print(boxes1, boxes2)
    xA = np.maximum(boxes1[:, 0], boxes2[:, 0])
    yA = np.maximum(boxes1[:, 1], boxes2[:, 1])
    xB = np.minimum(boxes1[:, 2], boxes2[:, 2])
    yB = np.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_w = np.clip(xB - xA, 0, None)
    inter_h = np.clip(yB - yA, 0, None)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / np.clip(union_area, 1e-6, None)
    return iou


def group_records_by_dataset(data):
    """Group STG records by dataset for per-dataset evaluation."""
    dataset_groups = defaultdict(list)

    for key, record in data.items():
        qa_type = record.get('qa_type', '')
        if 'stg' not in qa_type.lower():
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


def evaluate_dataset_stg(dataset_name, records):
    """Evaluate STG for a specific dataset."""
    print(f"\nEvaluating {dataset_name} ({len(records)} records)...")

    results_by_fps = defaultdict(list)
    all_ious = []

    for record in records:
        fps = record.get('fps', record.get('metadata', {}).get('fps', 1.0))
        if isinstance(fps, str):
            fps = float(fps)

        # Parse prediction from 'answer' field
        raw_answer = record.get('answer', '')
        processed_pred = post_process_pred(raw_answer)

        # Extract ground truth from struc_info - handle bbox_dict format
        struc_info = record.get('struc_info', {})
        if isinstance(struc_info, list) and len(struc_info) > 0:
            struc_item = struc_info[0]
            if isinstance(struc_item, dict) and 'bbox_dict' in struc_item:
                gt_dict = struc_item['bbox_dict']
            else:
                gt_dict = struc_item
        elif isinstance(struc_info, dict):
            if 'bbox_dict' in struc_info:
                gt_dict = struc_info['bbox_dict']
            else:
                gt_dict = struc_info
        else:
            gt_dict = {}

        # Convert prediction list to dict if needed
        if isinstance(processed_pred, list):
            key_list = list(gt_dict.keys())
            processed_pred = {key: box for key, box in zip(key_list[:len(processed_pred)], processed_pred)}

        # Process boxes
        pred_boxes = []
        gt_boxes = []
        
        for i, key in enumerate(gt_dict.keys()):
            gt_boxes.append(gt_dict[key])
            key_str = f"{float(key):.1f}"
            pred_box = processed_pred.get(key_str, [0, 0, 0, 0]) if isinstance(processed_pred, dict) else [0, 0, 0, 0]
            if pred_box == [0, 0, 0, 0] and i > 0:
                pred_box = pred_boxes[i - 1]  # Use previous box if current is invalid
            pred_boxes.append(pred_box)

        # Validate and compute IoU
        valid_pred_boxes = []
        valid_gt_boxes = []
        for pred_box, gt_box in zip(pred_boxes, gt_boxes):
            if is_valid_box(pred_box) and is_valid_box(gt_box):
                valid_pred_boxes.append(pred_box)
                valid_gt_boxes.append(gt_box)

        if valid_pred_boxes and valid_gt_boxes:
            pred_boxes_array = np.array(valid_pred_boxes)
            gt_boxes_array = np.array(valid_gt_boxes)
            iou = compute_iou_batch(pred_boxes_array, gt_boxes_array)
            
            if len(iou) > 0:
                mean_iou = iou.mean()
                results_by_fps[fps].append(mean_iou)
                all_ious.append(mean_iou)

    # Aggregate results per FPS
    aggregated = {}
    for fps, iou_list in results_by_fps.items():
        if iou_list:
            aggregated[f'fps_{fps}'] = {
                'iou@0.3': np.mean([1 if x >= 0.3 else 0 for x in iou_list]),
                'iou@0.5': np.mean([1 if x >= 0.5 else 0 for x in iou_list]),
                'iou@0.7': np.mean([1 if x >= 0.7 else 0 for x in iou_list]),
                'mIoU': np.mean(iou_list),
                'count': len(iou_list)
            }
    
    # Add overall result
    if all_ious:
        overall_mean_iou = np.mean(all_ious)
        aggregated['overall'] = {
            'mean_iou': overall_mean_iou,
            'valid_records': len(all_ious),
            'total_records': len(records)
        }

    return aggregated


def main():
    """Main evaluation function for STG."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_stg.py <results_json_file>")
        print("Example: python eval_stg.py results/model_results.json")
        sys.exit(1)

    output_file = sys.argv[1]
    print(f"Loading results from: {output_file}")

    with open(output_file, "r") as f:
        infer_output = json.load(f)

    dataset_records = group_records_by_dataset(infer_output)

    print(f"\nFound datasets: {list(dataset_records.keys())}")
    for dataset, records in dataset_records.items():
        print(f"  {dataset}: {len(records)} STG records")

    if not any(dataset_records.values()):
        print("No STG records found!")
        return

    all_results = {}
    for dataset_name, records in dataset_records.items():
        if records:
            results = evaluate_dataset_stg(dataset_name, records)
            all_results[dataset_name] = results

    print(f"\n{'='*80}")
    print("STG EVALUATION SUMMARY")
    print(f"{'='*80}")

    all_ious = []
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
            if 'overall' in fps_results:
                all_ious.append(fps_results['overall'].get('mean_iou', 0.0))

    return {
        'per_dataset': all_results,
        'mean_iou': np.mean(all_ious) if all_ious else 0.0
    }


if __name__ == "__main__":
    main()
