#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from relive.v2.observation_planning import ObservationPlanningError,prepare,validate
p=argparse.ArgumentParser();p.add_argument("--mode",required=True,choices=("prepare","validate"));p.add_argument("--output-dir",required=True,type=Path);p.add_argument("--requirement-freeze-dir",type=Path);p.add_argument("--selection-manifest",type=Path);p.add_argument("--video-index-dir",type=Path);p.add_argument("--stage3b-dir",type=Path);p.add_argument("--independent-repeat-comparison",type=Path);p.add_argument("--policy",type=Path);a=p.parse_args()
try:
 if a.mode=="prepare":
  if not all((a.requirement_freeze_dir,a.selection_manifest,a.video_index_dir,a.stage3b_dir,a.independent_repeat_comparison,a.policy)):raise ObservationPlanningError("PREPARE_INPUTS_REQUIRED")
  r=prepare(requirement_dir=a.requirement_freeze_dir,selection_manifest=a.selection_manifest,video_index_dir=a.video_index_dir,stage3b_dir=a.stage3b_dir,comparison_path=a.independent_repeat_comparison,policy_path=a.policy,output_dir=a.output_dir)
 else:r=validate(a.output_dir)
except ObservationPlanningError as e: print(f"ReliVE-v2 Stage 3C error: {e}");raise SystemExit(2)
print(json.dumps(r,sort_keys=True))
