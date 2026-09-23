#!/usr/bin/env python
"""Freeze accepted differential-policy and closed-pilot references."""
from __future__ import annotations
import argparse
import json
from relive.v2.protocol_dev_manifests import ProtocolDevManifestError, freeze_preconditions

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--acceptance-dir", required=True)
parser.add_argument("--pilot-closure", required=True)
parser.add_argument("--accepted-code-commit", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(freeze_preconditions(policy_path=args.policy, acceptance_dir=args.acceptance_dir,
        pilot_closure=args.pilot_closure, accepted_code_commit=args.accepted_code_commit,
        output_dir=args.output_dir), sort_keys=True))
except ProtocolDevManifestError as exc:
    print("relive-v2 protocol-dev precondition error: " + str(exc)); raise SystemExit(2)
