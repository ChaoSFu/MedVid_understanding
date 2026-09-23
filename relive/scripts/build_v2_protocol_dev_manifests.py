#!/usr/bin/env python
"""Build protocol-dev manifests; incomplete human review stays PREVIEW_ONLY."""
from __future__ import annotations
import argparse
import json
from relive.v2.protocol_dev_manifests import ProtocolDevManifestError, build_protocol_dev_manifests

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--precondition-freeze-report", required=True)
parser.add_argument("--stage-a", required=True)
parser.add_argument("--stage-b", required=True)
parser.add_argument("--external-split-registry")
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(build_protocol_dev_manifests(policy_path=args.policy,
        precondition_freeze_report=args.precondition_freeze_report, stage_a_path=args.stage_a,
        stage_b_path=args.stage_b, external_split_registry=args.external_split_registry,
        output_dir=args.output_dir), sort_keys=True))
except ProtocolDevManifestError as exc:
    print("relive-v2 protocol-dev manifest error: " + str(exc)); raise SystemExit(2)
