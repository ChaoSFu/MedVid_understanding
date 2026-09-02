#!/usr/bin/env python3

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        required=True,
        help="Original trainval JSON containing ground truth"
    )

    parser.add_argument(
        "--prediction_path",
        required=True,
        help="Qwen inference results.json"
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Merged file for MedVidBench evaluation"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load trainval GT
    # --------------------------------------------------

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --------------------------------------------------
    # Load Qwen predictions
    # --------------------------------------------------

    with open(args.prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    merged = {}

    missing = []

    for idx, sample in enumerate(data):

        key = str(idx)

        if key not in predictions:
            missing.append(idx)
            continue

        pred = predictions[key]

        conversations = sample.get("conversations", [])

        # question
        question = conversations[0]["value"]

        # ground truth
        if len(conversations) < 2:
            raise RuntimeError(
                f"Sample {idx} does not contain ground truth "
                f"in conversations[1]"
            )

        gnd = conversations[1]["value"]

        merged[key] = {
            "metadata": sample.get(
                "metadata",
                pred.get("metadata")
            ),

            "qa_type": sample.get(
                "qa_type",
                pred.get("qa_type")
            ),

            "struc_info": sample.get(
                "struc_info"
            ),

            "question": question,

            # model prediction
            "answer": pred["answer"],

            # official ground truth
            "gnd": gnd,

            "data_source": sample.get(
                "data_source",
                pred.get("data_source")
            ),
        }

    os.makedirs(
        os.path.dirname(
            os.path.abspath(args.output_path)
        ),
        exist_ok=True
    )

    with open(
        args.output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            merged,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 80)
    print("Trainval evaluation file created")
    print("=" * 80)

    print(f"GT samples:         {len(data)}")
    print(f"Prediction samples: {len(predictions)}")
    print(f"Merged samples:     {len(merged)}")
    print(f"Missing prediction: {len(missing)}")

    if missing:
        print(
            "First missing indices:",
            missing[:20]
        )

    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
