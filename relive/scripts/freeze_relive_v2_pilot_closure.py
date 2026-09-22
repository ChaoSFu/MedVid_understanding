#!/usr/bin/env python
"""Create a read-only closure manifest for a completed R0/R1 pilot."""
from __future__ import annotations
import argparse
import json
from relive.v2.pilot_closure import PilotClosureError, freeze_pilot_closure

parser = argparse.ArgumentParser()
parser.add_argument("--r0-output-dir", required=True)
parser.add_argument("--r1-output-dir", required=True)
parser.add_argument("--label-resolution-manifest", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--expected-formal-cohort", type=int, default=5)
args = parser.parse_args()
try:
    print(json.dumps(freeze_pilot_closure(r0_output_dir=args.r0_output_dir, r1_output_dir=args.r1_output_dir,
        label_resolution_manifest=args.label_resolution_manifest, output_dir=args.output_dir,
        expected_formal_cohort=args.expected_formal_cohort), sort_keys=True))
except PilotClosureError as exc:
    print("relive-v2 pilot closure error: " + str(exc)); raise SystemExit(2)
