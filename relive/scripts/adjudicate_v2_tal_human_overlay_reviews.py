#!/usr/bin/env python
"""Validate and integrate immutable Stage 3G human overlay reviews."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from relive.v2.human_overlay_adjudication import HumanOverlayReviewError, adjudicate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-anchor-manifest", required=True, type=Path)
    parser.add_argument("--review-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = adjudicate(raw_anchor_manifest=args.raw_anchor_manifest, review_jsonl=args.review_jsonl, output_dir=args.output_dir)
    except HumanOverlayReviewError as exc:
        print(f"ReliVE-v2 human-overlay adjudication error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
