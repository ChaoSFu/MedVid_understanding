from __future__ import annotations

import argparse
import json

from .atomic_claim_freeze import freeze_reviewed_atomic_claims


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply explicit author feedback and freeze reviewed atomic claims."
    )
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--review-jsonl", required=True)
    parser.add_argument("--feedback-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    args = parser.parse_args()
    result = freeze_reviewed_atomic_claims(
        runtime_manifest=args.runtime_manifest,
        candidates_jsonl=args.candidates_jsonl,
        review_jsonl=args.review_jsonl,
        feedback_json=args.feedback_json,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
