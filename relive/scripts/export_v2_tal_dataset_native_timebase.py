#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from relive.v2.dataset_native_timebase import DatasetNativeTimebaseError, audit_dataset_native_timebase, export_dataset_native_timebase

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset-json",required=True,type=Path);parser.add_argument("--selection-manifest",required=True,type=Path)
    parser.add_argument("--requirement-freeze-dir",required=True,type=Path);parser.add_argument("--media-audit-dir",required=True,type=Path)
    parser.add_argument("--frame-root",required=True,type=Path);parser.add_argument("--source-prefix",required=True)
    parser.add_argument("--output-dir",required=True,type=Path);parser.add_argument("--audit-output-dir",type=Path)
    parser.add_argument("--frame-bank-layout",required=True,choices=["frames_2fps"]); args=parser.parse_args()
    try:
        if args.audit_output_dir:
            audit_dataset_native_timebase(dataset_json=args.dataset_json,frame_root=args.frame_root,source_prefix=args.source_prefix,output_dir=args.audit_output_dir,frame_bank_layout=args.frame_bank_layout)
        result=export_dataset_native_timebase(dataset_json=args.dataset_json,selection_manifest=args.selection_manifest,requirement_freeze_dir=args.requirement_freeze_dir,media_audit_dir=args.media_audit_dir,frame_root=args.frame_root,source_prefix=args.source_prefix,output_dir=args.output_dir,frame_bank_layout=args.frame_bank_layout)
    except DatasetNativeTimebaseError as exc: print(json.dumps({"status":"UNRESOLVED_TIMEBASE_SOURCE","reason_code":str(exc),"model_calls_made":0,"gt_used":False},sort_keys=True));return 2
    print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
