#!/usr/bin/env python
"""Read-only validation for frozen or PREVIEW_ONLY protocol-dev outputs."""
from __future__ import annotations
import argparse
import json
from relive.v2.protocol_dev_manifests import ProtocolDevManifestError, validate_protocol_dev_freeze

parser = argparse.ArgumentParser()
parser.add_argument("--freeze-manifest", required=True)
parser.add_argument("--policy", required=True)
args = parser.parse_args()
try:
    print(json.dumps(validate_protocol_dev_freeze(freeze_manifest=args.freeze_manifest, policy_path=args.policy), sort_keys=True))
except ProtocolDevManifestError as exc:
    print("relive-v2 protocol-dev validation error: " + str(exc)); raise SystemExit(2)
