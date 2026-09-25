#!/usr/bin/env python
"""Audit a documented CoPESD image/time mapping before keyframe selection."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, copesd_timebase_audit

parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--data-roots", required=True)
parser.add_argument("--timebase-manifest")
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(copesd_timebase_audit(cases_dir=args.cases_dir, data_roots=args.data_roots, timebase_manifest=args.timebase_manifest, output_dir=args.output_dir), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev CoPESD timebase audit error: " + str(exc)); raise SystemExit(2)
