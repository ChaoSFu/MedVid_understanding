#!/usr/bin/env python
"""Validate the three disjoint, hash-bound protocol split manifests."""
from __future__ import annotations
import argparse
import json
from relive.v2.protocol_manifests import ProtocolManifestError, validate_split_manifests

parser = argparse.ArgumentParser()
parser.add_argument("--protocol-dev", required=True)
parser.add_argument("--calibration", required=True)
parser.add_argument("--blind-test", required=True)
args = parser.parse_args()
try:
    print(json.dumps(validate_split_manifests(protocol_dev=args.protocol_dev, calibration=args.calibration, blind_test=args.blind_test), sort_keys=True))
except ProtocolManifestError as exc:
    print("relive-v2 protocol manifest error: " + str(exc)); raise SystemExit(2)
