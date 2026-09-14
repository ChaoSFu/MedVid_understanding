#!/usr/bin/env python3
"""Create a zero-model ROI worksheet for a new Phase 4A-0 control cohort."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relive.phase4a0_development_controls import DevelopmentControlError, prepare_development_controls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--source-prefix", default="/root/data")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = prepare_development_controls(selection_path=Path(args.selection), source_json=Path(args.source_json),
                                              frame_root=Path(args.frame_root), source_prefix=args.source_prefix,
                                              output_dir=Path(args.output_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, DevelopmentControlError) as exc:
        print(f"ReliVE Phase 4A-0 development-control preparation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
