#!/usr/bin/env python3
"""Finalize immutable Phase 4A likelihood diagnostics without model access."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from relive.phase4a_analysis import Phase4AAnalysisError, finalize

parser = argparse.ArgumentParser()
parser.add_argument("--phase4a-output-dir", required=True)
parser.add_argument("--eligibility-manifest", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(finalize(phase4a_output_dir=Path(args.phase4a_output_dir), eligibility_manifest_path=Path(args.eligibility_manifest), output_dir=Path(args.output_dir)), ensure_ascii=False, indent=2))
except (OSError, ValueError, Phase4AAnalysisError) as exc:
    print(f"ReliVE Phase 4A finalizer error: {exc}", file=sys.stderr)
    raise SystemExit(2)
