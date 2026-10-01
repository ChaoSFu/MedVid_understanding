#!/usr/bin/env python
"""Generate the human-only addendum for missing entity and claim components."""
from __future__ import annotations

import argparse
import json

from relive.v2.protocol_dev_cases import ProtocolDevCaseError, prepare_entity_component_completion


parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--completion-queue", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(prepare_entity_component_completion(cases_dir=args.cases_dir,
          completion_queue=args.completion_queue, output_dir=args.output_dir), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev entity-component completion error: " + str(exc))
    raise SystemExit(2)
