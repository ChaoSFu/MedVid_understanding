#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relive.v2.temporal_search_plan import TemporalSearchPlanError, freeze_temporal_search_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirement-freeze-dir", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--video-index-dir", required=True, type=Path)
    parser.add_argument("--public-timestamp-manifest", required=True, type=Path)
    parser.add_argument("--public-timestamp-provenance", required=True, type=Path)
    parser.add_argument("--temporal-policy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = freeze_temporal_search_plan(requirement_dir=args.requirement_freeze_dir, selection_manifest_path=args.selection_manifest,
                                              video_index_dir=args.video_index_dir, timestamp_manifest_path=args.public_timestamp_manifest,
                                              timestamp_provenance_path=args.public_timestamp_provenance, policy_path=args.temporal_policy,
                                              output_dir=args.output_dir)
    except TemporalSearchPlanError as exc:
        print(f"ReliVE-v2 TAL temporal-search freeze error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
