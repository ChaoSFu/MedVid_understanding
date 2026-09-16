#!/usr/bin/env python3
"""Freeze or validate ReliVE-v2 Stage 3F typed spatial evidence plans."""
import argparse
import json
from pathlib import Path

from relive.v2.spatial_evidence_planning import SpatialPlanError, prepare, validate

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=("prepare", "validate"))
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--stage3c-dir", type=Path)
parser.add_argument("--stage3d-dir", type=Path)
parser.add_argument("--stage3e-dir", type=Path)
parser.add_argument("--video-index-dir", type=Path)
parser.add_argument("--policy", type=Path)
args = parser.parse_args()
try:
    result = prepare(stage3c_dir=args.stage3c_dir, stage3d_dir=args.stage3d_dir, stage3e_dir=args.stage3e_dir,
                     video_index_dir=args.video_index_dir, policy_path=args.policy, output_dir=args.output_dir) if args.mode == "prepare" else validate(args.output_dir)
except (SpatialPlanError, ValueError) as exc:
    print(f"ReliVE-v2 TAL spatial-plan error: {exc}")
    raise SystemExit(2)
print(json.dumps(result, sort_keys=True))
