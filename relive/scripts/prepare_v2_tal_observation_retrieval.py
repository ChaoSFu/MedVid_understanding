#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from relive.v2.observation_retrieval import ObservationRetrievalError, execute, preflight, prepare, validate

parser=argparse.ArgumentParser(); parser.add_argument("--mode",required=True,choices=("prepare","preflight","run","replay","validate")); parser.add_argument("--output-dir",required=True,type=Path); parser.add_argument("--config",type=Path); parser.add_argument("--requirement-freeze-dir",type=Path); parser.add_argument("--selection-manifest",type=Path); parser.add_argument("--video-index-dir",type=Path); parser.add_argument("--stage3c-dir",type=Path); parser.add_argument("--policy",type=Path); args=parser.parse_args()
try:
    if args.mode=="prepare":
        if not all((args.config,args.requirement_freeze_dir,args.selection_manifest,args.video_index_dir,args.stage3c_dir,args.policy)): raise ObservationRetrievalError("PREPARE_INPUTS_REQUIRED")
        result=prepare(requirement_dir=args.requirement_freeze_dir,selection_manifest_path=args.selection_manifest,video_index_dir=args.video_index_dir,stage3c_dir=args.stage3c_dir,config_path=args.config,policy_path=args.policy,output_dir=args.output_dir)
    elif args.mode=="preflight":
        if not all((args.config,args.stage3c_dir)): raise ObservationRetrievalError("PREFLIGHT_INPUTS_REQUIRED")
        result=preflight(output_dir=args.output_dir,config_path=args.config,stage3c_dir=args.stage3c_dir)
    elif args.mode in {"run","replay"}:
        if not args.config: raise ObservationRetrievalError("CONFIG_REQUIRED")
        result=execute(output_dir=args.output_dir,config_path=args.config,mode=args.mode)
    else: result=validate(args.output_dir)
except (ObservationRetrievalError,RuntimeError,ValueError) as exc:
    print(f"ReliVE-v2 TAL observation-retrieval error: {exc}"); raise SystemExit(2)
print(json.dumps(result,sort_keys=True))
