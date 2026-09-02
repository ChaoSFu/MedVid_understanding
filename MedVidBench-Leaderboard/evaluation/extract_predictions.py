#!/usr/bin/env python3
"""Extract predictions from results.json for user submission format.

This script extracts only the prediction-related fields from results.json,
creating a format that users would submit (without ground truth data).
"""

import json
import sys
from pathlib import Path


def extract_predictions(results_file: str, output_file: str) -> None:
    """
    Extract predictions from results.json.

    Args:
        results_file: Path to results.json (dict format with numeric keys)
        output_file: Path to save predictions (list format)
    """
    print(f"Loading results from: {results_file}")
    with open(results_file) as f:
        results = json.load(f)

    # results.json is a dict with numeric string keys ("0", "1", "2", ...)
    # We need to convert to list format with proper IDs

    print(f"Loaded {len(results)} results")

    # Extract predictions
    predictions = []
    for idx, (key, result) in enumerate(results.items()):
        # Create ID from metadata
        metadata = result.get('metadata', {})
        video_id = metadata.get('video_id', '')

        # Try both naming conventions for frame numbers
        start_frame = metadata.get('input_video_start_frame', '') or metadata.get('start_frame', '')
        end_frame = metadata.get('input_video_end_frame', '') or metadata.get('end_frame', '')
        fps = metadata.get('fps', '')

        # ID format: video_id&&start_frame&&end_frame&&fps
        sample_id = f"{video_id}&&{start_frame}&&{end_frame}&&{fps}"

        prediction = {
            'id': sample_id,
            'qa_type': result.get('qa_type', ''),
            'prediction': result.get('answer', '')
        }

        predictions.append(prediction)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} predictions...")

    # Save predictions
    print(f"Saving {len(predictions)} predictions to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"✓ Successfully extracted {len(predictions)} predictions")

    # Show sample
    if predictions:
        print("\nSample prediction (first entry):")
        print(json.dumps(predictions[0], indent=2))


def main():
    """Command-line interface."""
    if len(sys.argv) != 3:
        print("Usage: python extract_predictions.py results.json predictions.json")
        print()
        print("Arguments:")
        print("  results.json      - Input results file (with ground truth)")
        print("  predictions.json  - Output predictions file (user format)")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_predictions(results_file, output_file)


if __name__ == "__main__":
    main()
