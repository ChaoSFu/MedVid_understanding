#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from relive.v2.coarse_hypothesis import CoarseHypothesisError, execute, preflight, prepare, validate

parser = argparse.ArgumentParser(); parser.add_argument("--mode", required=True, choices=("prepare", "preflight", "run", "replay", "validate")); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--config", type=Path); parser.add_argument("--requirement-freeze-dir", type=Path); parser.add_argument("--selection-manifest", type=Path); parser.add_argument("--stage3a-dir", type=Path); parser.add_argument("--video-index-dir", type=Path); parser.add_argument("--public-timestamp-manifest", type=Path); parser.add_argument("--public-timestamp-provenance", type=Path); args = parser.parse_args()
try:
    if args.mode == "prepare":
        if not all((args.config, args.requirement_freeze_dir, args.selection_manifest, args.stage3a_dir, args.video_index_dir, args.public_timestamp_manifest, args.public_timestamp_provenance)): raise CoarseHypothesisError("PREPARE_INPUTS_REQUIRED")
        result = prepare(requirement_dir=args.requirement_freeze_dir, selection_manifest_path=args.selection_manifest, stage3a_dir=args.stage3a_dir, video_index_dir=args.video_index_dir, timestamp_manifest_path=args.public_timestamp_manifest, timestamp_provenance_path=args.public_timestamp_provenance, config_path=args.config, output_dir=args.output_dir)
    elif args.mode == "preflight":
        if not args.config: raise CoarseHypothesisError("CONFIG_REQUIRED")
        result = preflight(output_dir=args.output_dir, config_path=args.config)
    elif args.mode in {"run", "replay"}:
        if not args.config: raise CoarseHypothesisError("CONFIG_REQUIRED")
        result = execute(output_dir=args.output_dir, config_path=args.config, mode=args.mode)
    else: result = validate(args.output_dir)
except (CoarseHypothesisError, RuntimeError, ValueError) as exc:
    print(f"ReliVE-v2 TAL coarse-hypothesis error: {exc}"); raise SystemExit(2)
print(json.dumps(result, sort_keys=True))
