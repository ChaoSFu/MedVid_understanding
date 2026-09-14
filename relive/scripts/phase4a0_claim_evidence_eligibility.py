#!/usr/bin/env python3
"""Phase 4A-0: zero-model blinded claim--evidence eligibility audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from relive.phase4a0 import Phase4A0Error, export_historical_pilots, prepare, validate_reviews

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("export-historical", "prepare", "validate"), required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--candidate-manifest")
    parser.add_argument("--review", action="append", default=[]); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime"); parser.add_argument("--prospective-manifest"); parser.add_argument("--phase35-v3-run-dir"); parser.add_argument("--phase36-run-dir")
    args = parser.parse_args()
    try:
        if args.mode == "export-historical":
            if not all((args.runtime, args.prospective_manifest, args.phase35_v3_run_dir, args.phase36_run_dir)):
                raise Phase4A0Error("FROZEN_ARTIFACT_PATHS_REQUIRED")
            result = export_historical_pilots(runtime_path=Path(args.runtime), prospective_manifest_path=Path(args.prospective_manifest), phase35_v3_run_dir=Path(args.phase35_v3_run_dir), phase36_run_dir=Path(args.phase36_run_dir), output_dir=Path(args.output_dir))
        elif args.mode == "prepare":
            if not args.candidate_manifest: raise Phase4A0Error("CANDIDATE_MANIFEST_REQUIRED")
            result = prepare(Path(args.candidate_manifest), Path(args.output_dir), seed=args.seed)
        else: result = validate_reviews(Path(args.output_dir), [Path(p) for p in args.review])
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result.get("status") in {"PASS", "AWAITING_HUMAN_REVIEWS", "REQUIRES_ADJUDICATION"} else 2
    except (OSError, ValueError, Phase4A0Error) as exc:
        print(f"ReliVE Phase 4A-0 error: {exc}", file=sys.stderr); return 2
if __name__ == "__main__": raise SystemExit(main())
