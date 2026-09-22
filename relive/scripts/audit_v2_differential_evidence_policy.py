#!/usr/bin/env python
"""Run the fixture-only ReliVE-v2 differential policy acceptance audit."""
from __future__ import annotations
import argparse
import json
from relive.v2.differential_policy_acceptance import DifferentialAcceptanceError, run_acceptance_audit
from relive.v2.differential_evidence import DifferentialEvidenceError

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True); p.add_argument("--fixture", required=True)
p.add_argument("--output-dir", required=True); p.add_argument("--pilot-closure")
a = p.parse_args()
try:
    print(json.dumps(run_acceptance_audit(policy_path=a.policy, fixture_path=a.fixture,
        output_dir=a.output_dir, pilot_closure=a.pilot_closure), sort_keys=True))
except (DifferentialAcceptanceError, DifferentialEvidenceError) as exc:
    print("differential policy acceptance error: " + str(exc)); raise SystemExit(2)
