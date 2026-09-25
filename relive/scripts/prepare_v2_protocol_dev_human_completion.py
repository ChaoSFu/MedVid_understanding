#!/usr/bin/env python
"""Generate human-only completion forms and coordinate-free oracle templates."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, prepare_human_completion

parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--reviews-dir", required=True)
parser.add_argument("--oracle-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(prepare_human_completion(cases_dir=args.cases_dir, reviews_dir=args.reviews_dir, oracle_dir=args.oracle_dir), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev human completion error: " + str(exc)); raise SystemExit(2)
