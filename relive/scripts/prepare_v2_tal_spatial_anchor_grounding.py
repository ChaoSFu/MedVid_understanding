#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from relive.v2.spatial_anchor_grounding import SpatialAnchorGroundingError, preflight, execute, validate
p=argparse.ArgumentParser()
p.add_argument('--mode',required=True,choices=('preflight','run','replay','validate'));p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--config',type=Path);p.add_argument('--policy',type=Path);p.add_argument('--stage3f-dir',type=Path);p.add_argument('--stage3c-dir',type=Path);p.add_argument('--stage3d-dir',type=Path);p.add_argument('--stage3e-dir',type=Path);p.add_argument('--video-index-dir',type=Path)
a=p.parse_args()
try:
 if a.mode=='preflight':r=preflight(stage3f_dir=a.stage3f_dir,stage3c_dir=a.stage3c_dir,stage3d_dir=a.stage3d_dir,stage3e_dir=a.stage3e_dir,video_index_dir=a.video_index_dir,config_path=a.config,policy_path=a.policy,output_dir=a.output_dir)
 elif a.mode in ('run','replay'):r=execute(output_dir=a.output_dir,config_path=a.config,policy_path=a.policy,mode=a.mode)
 else:r=validate(output_dir=a.output_dir)
except (SpatialAnchorGroundingError,ValueError) as e:print(f'ReliVE-v2 TAL spatial-anchor-grounding error: {e}');raise SystemExit(2)
print(json.dumps(r,sort_keys=True))
