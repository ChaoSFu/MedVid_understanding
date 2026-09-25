#!/usr/bin/env python
"""Read-only, fail-closed portable frame-pattern resolver."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, resolve_frame_patterns

parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--data-roots", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(resolve_frame_patterns(cases_dir=args.cases_dir, data_roots=args.data_roots, output_dir=args.output_dir), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev frame resolver error: " + str(exc)); raise SystemExit(2)
