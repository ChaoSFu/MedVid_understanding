#!/usr/bin/env python
"""Bind confirmed public QA records to immutable protocol-dev case copies."""
from __future__ import annotations

import argparse
import json

from relive.v2.protocol_dev_cases import ProtocolDevCaseError, apply_source_record_confirmations


parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--qa-file", required=True)
parser.add_argument("--confirmation-jsonl", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

try:
    result = apply_source_record_confirmations(cases_dir=args.cases_dir, qa_path=args.qa_file,
                                               confirmation_path=args.confirmation_jsonl,
                                               output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev source-record confirmation error: " + str(exc))
    raise SystemExit(2)
