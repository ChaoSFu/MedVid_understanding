#!/usr/bin/env python
"""Apply completed human-review fields into immutable reviewed protocol-dev cases."""
from __future__ import annotations

import argparse
import json

from relive.v2.protocol_dev_cases import ProtocolDevCaseError, apply_human_completion


parser = argparse.ArgumentParser()
parser.add_argument("--cases-dir", required=True)
parser.add_argument("--completion-queue", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    print(json.dumps(apply_human_completion(cases_dir=args.cases_dir, completion_queue=args.completion_queue,
                                             output_dir=args.output_dir), sort_keys=True))
except ProtocolDevCaseError as exc:
    print("relive-v2 protocol-dev human-completion application error: " + str(exc))
    raise SystemExit(2)
