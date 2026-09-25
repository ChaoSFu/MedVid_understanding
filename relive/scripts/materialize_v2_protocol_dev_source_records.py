#!/usr/bin/env python
"""Build public-field source-record hashes for protocol-dev cases."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, materialize_source_records

parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--qa-file", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(materialize_source_records(cases_dir=args.cases_dir, qa_path=args.qa_file, output_dir=args.output_dir), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev source materialization error: " + str(exc)); raise SystemExit(2)
