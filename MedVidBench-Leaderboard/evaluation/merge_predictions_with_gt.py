"""Merge user predictions with ground truth for evaluation.

This script is used server-side to combine user-submitted predictions
with the private ground truth data before running evaluation.
"""

import json
import sys
from pathlib import Path
from typing import Tuple


def merge_predictions_with_ground_truth(
    predictions_file: str,
    ground_truth_file: str,
    output_file: str
) -> Tuple[bool, str]:
    """
    Merge user predictions with server-side ground truth by array index.

    Args:
        predictions_file: Path to user's predictions JSON array (same order as ground truth)
        ground_truth_file: Path to ground truth JSON (struc_info, GPT responses)
        output_file: Path to save merged JSON for evaluation

    Returns:
        (success, message)
    """
    try:
        # Load predictions
        print(f"Loading predictions from: {predictions_file}")
        with open(predictions_file) as f:
            predictions = json.load(f)

        # Load ground truth
        print(f"Loading ground truth from: {ground_truth_file}")
        with open(ground_truth_file) as f:
            ground_truth = json.load(f)

        print(f"Loaded {len(predictions)} predictions")
        print(f"Loaded {len(ground_truth)} ground truth samples")

        # Validate predictions format
        if not isinstance(predictions, list):
            return False, "Predictions must be a JSON array"

        # Check lengths match for index-based merging
        if len(predictions) != len(ground_truth):
            return False, f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) must have the same length"

        # Validate predictions have required fields
        for i, pred in enumerate(predictions):
            if 'prediction' not in pred:
                return False, f"Prediction at index {i} missing 'prediction' field"

        # Merge predictions with ground truth by index
        merged = {}
        mismatched_qa_types = []

        for idx, gt_sample in enumerate(ground_truth):
            pred = predictions[idx]

            # Verify qa_type matches (optional validation)
            if 'qa_type' in pred and pred['qa_type'] != gt_sample.get('qa_type'):
                mismatched_qa_types.append({
                    'index': idx,
                    'predicted': pred.get('qa_type'),
                    'actual': gt_sample.get('qa_type')
                })

            # Create minimal format matching original results.json
            # Only include essential fields: metadata, qa_type, struc_info, question, gnd, answer, data_source
            merged_sample = {
                'metadata': gt_sample.get('metadata', {}),
                'qa_type': gt_sample.get('qa_type', ''),
                'struc_info': gt_sample.get('struc_info', []),
                'question': '',  # Extract from conversations if present
                'gnd': '',  # Extract from conversations if present
                'answer': pred['prediction'],
                'data_source': gt_sample.get('data_source', '')
            }

            # Extract question and ground truth answer from conversations
            if 'conversations' in gt_sample:
                for msg in gt_sample['conversations']:
                    if msg.get('from') in ['human', 'user']:
                        # Remove <video> token from question to match original format
                        question = msg.get('value', '')
                        merged_sample['question'] = question.replace('<video>\n', '').replace('<video>', '')
                    elif msg.get('from') in ['gpt', 'assistant']:
                        merged_sample['gnd'] = msg.get('value', '')

            # Use numeric string key to match original format
            merged[str(idx)] = merged_sample

        # Save merged data as dict with numeric string keys
        print(f"Saving merged data to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(merged, f, indent=2)

        # Build result message
        message_parts = [
            f"Successfully merged {len(merged)} samples"
        ]

        if mismatched_qa_types:
            message_parts.append(
                f"Warning: {len(mismatched_qa_types)} samples with mismatched qa_type"
            )
            for mismatch in mismatched_qa_types[:5]:  # Show first 5
                print(f"  Mismatch at index {mismatch['index']}: predicted: {mismatch['predicted']}, actual: {mismatch['actual']}")

        message = ". ".join(message_parts)
        print(message)

        return True, message

    except FileNotFoundError as e:
        return False, f"File not found: {e.filename}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Error merging: {str(e)}"


def main():
    """Command-line interface."""
    if len(sys.argv) != 4:
        print("Usage: python merge_predictions_with_gt.py predictions.json ground_truth.json output.json")
        print()
        print("Arguments:")
        print("  predictions.json  - User's predictions array (same length/order as ground truth)")
        print("  ground_truth.json - Server's ground truth (struc_info, GPT responses)")
        print("  output.json       - Merged output for evaluation")
        print()
        print("Note: Predictions and ground truth are merged by array index (0-based).")
        sys.exit(1)

    predictions_file = sys.argv[1]
    ground_truth_file = sys.argv[2]
    output_file = sys.argv[3]

    success, msg = merge_predictions_with_ground_truth(
        predictions_file,
        ground_truth_file,
        output_file
    )

    print()
    print("=" * 80)
    if success:
        print("✓ SUCCESS:", msg)
        sys.exit(0)
    else:
        print("✗ FAILED:", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
