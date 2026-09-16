#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from relive.v2.temporal_evidence_composition import prepare,validate,TemporalCompositionError
p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('prepare','validate'));p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--stage3c-dir',type=Path);p.add_argument('--stage3d-dir',type=Path);p.add_argument('--video-index-dir',type=Path);p.add_argument('--policy',type=Path);a=p.parse_args()
try:
 r=prepare(stage3c_dir=a.stage3c_dir,stage3d_dir=a.stage3d_dir,video_index_dir=a.video_index_dir,policy_path=a.policy,output_dir=a.output_dir) if a.mode=='prepare' else validate(a.output_dir)
except (TemporalCompositionError,ValueError) as e: print(f'ReliVE-v2 TAL temporal-composition error: {e}');raise SystemExit(2)
print(json.dumps(r,sort_keys=True))
