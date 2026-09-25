#!/usr/bin/env python
"""Enumerate EGO frame-path candidates without selecting one."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, ego_candidate_path_resolution

parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--data-roots", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--render-previews", action="store_true")
args = parser.parse_args()
try:
    print(json.dumps(ego_candidate_path_resolution(cases_dir=args.cases_dir, data_roots=args.data_roots, output_dir=args.output_dir, render_previews=args.render_previews), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev EGO candidate inspection error: " + str(exc)); raise SystemExit(2)
