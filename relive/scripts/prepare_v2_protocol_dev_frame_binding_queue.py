#!/usr/bin/env python
"""Generate the non-model human/environment frame-binding queue."""
from __future__ import annotations
import argparse, json
from relive.v2.protocol_dev_cases import ProtocolDevCaseError, prepare_frame_binding_completion_queue

parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(prepare_frame_binding_completion_queue(cases_dir=args.cases_dir, output_dir=args.output_dir), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev frame-binding queue error: " + str(exc)); raise SystemExit(2)
