#!/usr/bin/env python3
"""Export fresh public records and complete contact sheets for human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from relive.public_record_candidates import PublicRecordCandidateError, export_public_record_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 3.5 zero-model public-record candidate exporter")
    parser.add_argument("--public-selector", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--source-prefix", default="/root/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--thumbnail-width", type=int, default=256)
    parser.add_argument("--thumbnail-height", type=int, default=192)
    args = parser.parse_args()
    try:
        result = export_public_record_candidates(public_selector=Path(args.public_selector).resolve(), source_json=Path(args.source_json).resolve(),
                                                  frame_root=Path(args.frame_root).resolve(), source_prefix=args.source_prefix,
                                                  output_dir=Path(args.output_dir).resolve(), repo_root=Path(args.repo_root).resolve(),
                                                  batch_size=args.batch_size, page_size=args.page_size, columns=args.columns,
                                                  thumbnail_width=args.thumbnail_width, thumbnail_height=args.thumbnail_height)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, PublicRecordCandidateError) as exc:
        print(f"ReliVE Phase 3.5 public-record export error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
