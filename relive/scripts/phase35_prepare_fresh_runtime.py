#!/usr/bin/env python3
"""Prepare and freeze a fresh public LOCAL_ATOMIC runtime; never run a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from relive.fresh_public_runtime import FreshRuntimeError, prepare_fresh_public_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 3.5 fresh public-window runtime preparation")
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--source-prefix", default="/root/data")
    parser.add_argument("--fresh-claims", required=True, help="Strict three-field public claim JSONL")
    parser.add_argument("--window-confirmations", required=True, help="Human public-window confirmation JSONL; no ROI allowed")
    parser.add_argument("--config", required=True, help="Read only; its SHA-256 is frozen in the report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    try:
        report = prepare_fresh_public_runtime(source_json=Path(args.source_json).resolve(), frame_root=Path(args.frame_root).resolve(),
                                              source_prefix=args.source_prefix, fresh_claims=Path(args.fresh_claims).resolve(),
                                              window_confirmations=Path(args.window_confirmations).resolve(), config=Path(args.config).resolve(),
                                              output_dir=Path(args.output_dir).resolve(), repo_root=Path(args.repo_root).resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, FreshRuntimeError) as exc:
        print(f"ReliVE Phase 3.5 fresh-runtime error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
