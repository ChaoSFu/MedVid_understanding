#!/usr/bin/env python
"""Normalize a human protocol-dev draft into strict per-case portable files."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, normalize_draft

parser = argparse.ArgumentParser()
parser.add_argument("--draft", required=True)
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--manifest", required=True)
args = parser.parse_args()
try:
    print(json.dumps(normalize_draft(draft_path=args.draft, cases_dir=args.cases_dir, manifest_path=args.manifest), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev normalization error: " + str(exc)); raise SystemExit(2)
