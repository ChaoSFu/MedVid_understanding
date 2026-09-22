#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from relive.v2.oracle_feasibility import OracleFeasibilityError, prepare_runtime

p=argparse.ArgumentParser()
p.add_argument("--dry-run", action="store_true")
p.add_argument("--runtime-manifest", required=True); p.add_argument("--ground-truth-manifest", required=True)
p.add_argument("--policy", required=True); p.add_argument("--calibration-artifact")
p.add_argument("--output-root", required=True); p.add_argument("--cache-root", required=True); p.add_argument("--model-path")
a=p.parse_args()
try:
    # Ground truth remains an evaluator input and is intentionally not opened
    # by this runtime command, including dry-run mode.
    print(json.dumps(prepare_runtime(runtime_manifest=a.runtime_manifest, policy_path=a.policy,
        calibration_artifact=a.calibration_artifact, output_root=a.output_root, cache_root=a.cache_root,
        model_path=a.model_path, dry_run=a.dry_run), sort_keys=True))
except (OracleFeasibilityError, DifferentialEvidenceError) as exc:
    print("oracle feasibility error: " + str(exc)); raise SystemExit(2)
