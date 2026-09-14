#!/usr/bin/env python3
"""Phase 4A-0: zero-model blinded claim--evidence eligibility audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from relive.phase4a0 import Phase4A0Error, prepare, validate_reviews

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "validate"), required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--candidate-manifest")
    parser.add_argument("--review", action="append", default=[]); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        result = prepare(Path(args.candidate_manifest), Path(args.output_dir), seed=args.seed) if args.mode == "prepare" else validate_reviews(Path(args.output_dir), [Path(p) for p in args.review])
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result.get("status") in {"PASS", "AWAITING_HUMAN_REVIEWS", "REQUIRES_ADJUDICATION"} else 2
    except (OSError, ValueError, Phase4A0Error) as exc:
        print(f"ReliVE Phase 4A-0 error: {exc}", file=sys.stderr); return 2
if __name__ == "__main__": raise SystemExit(main())
