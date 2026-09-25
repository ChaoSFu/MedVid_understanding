#!/usr/bin/env python
"""Static-only protocol-dev readiness audit; it has no model authority."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, audit_cases

parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--oracle-dir")
parser.add_argument("--data-roots")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--strict", action="store_true")
args = parser.parse_args()
try:
    print(json.dumps(audit_cases(cases_dir=args.cases_dir, oracle_dir=args.oracle_dir, data_roots=args.data_roots,
        output_dir=args.output_dir, strict=args.strict), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev case audit error: " + str(exc)); raise SystemExit(2)
