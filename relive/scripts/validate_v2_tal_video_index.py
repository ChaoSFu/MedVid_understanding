#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from relive.v2.video_index import VideoIndexError, validate_video_index_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--materialize-frames", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_video_index_artifacts(args.output_dir, materialize_frames=args.materialize_frames)
    except VideoIndexError as exc:
        print(f"ReliVE-v2 TAL video-index validation error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
