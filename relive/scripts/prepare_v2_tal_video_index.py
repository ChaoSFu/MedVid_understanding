#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from relive.v2.video_index import VideoIndexError, freeze_video_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirement-freeze-dir", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--frame-root", required=True, type=Path)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--timebase-policy", required=True, type=Path)
    parser.add_argument("--public-timestamp-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        audit = freeze_video_index(requirement_dir=args.requirement_freeze_dir, selection_manifest_path=args.selection_manifest,
                                   source_json=args.source_json, frame_root=args.frame_root, source_prefix=args.source_prefix,
                                   timebase_policy_path=args.timebase_policy, output_dir=args.output_dir,
                                   public_timestamp_manifest_path=args.public_timestamp_manifest)
    except VideoIndexError as exc:
        print(f"ReliVE-v2 TAL video-index freeze error: {exc}")
        return 2
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
