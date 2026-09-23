#!/usr/bin/env python
"""Create a non-authoritative, incomplete Stage-A review form from a prefilter CSV."""
from __future__ import annotations
import argparse
import json
from relive.v2.protocol_dev_manifests import ProtocolDevManifestError, derive_stage_a_source_uids, prefilter_csv_to_stage_a_preview, stage_b_template

parser = argparse.ArgumentParser()
parser.add_argument("--prefilter-csv")
parser.add_argument("--stage-a")
parser.add_argument("--mode", choices=("from-prefilter", "canonicalize-source-uids", "stage-b-template"), required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
try:
    if args.mode == "from-prefilter":
        if not args.prefilter_csv: raise ProtocolDevManifestError("PREFILTER_CSV_REQUIRED")
        result = prefilter_csv_to_stage_a_preview(prefilter_csv=args.prefilter_csv, output_dir=args.output_dir)
    elif args.mode == "canonicalize-source-uids":
        if not args.stage_a: raise ProtocolDevManifestError("STAGE_A_REQUIRED")
        from pathlib import Path
        path = Path(args.output_dir) / "protocol_dev_stage_a_candidate_review.derived.jsonl"
        result = derive_stage_a_source_uids(stage_a_path=args.stage_a, output_path=path)
    else:
        if not args.stage_a: raise ProtocolDevManifestError("STAGE_A_REQUIRED")
        from pathlib import Path
        path = Path(args.output_dir) / "protocol_dev_stage_b_case_definition.jsonl"
        result = stage_b_template(stage_a_path=args.stage_a, output_path=path)
    print(json.dumps(result, sort_keys=True))
except ProtocolDevManifestError as exc:
    print("relive-v2 protocol-dev review-template error: " + str(exc)); raise SystemExit(2)
