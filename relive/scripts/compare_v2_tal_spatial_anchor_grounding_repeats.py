#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from relive.v2.spatial_anchor_grounding_repeat import compare,SpatialAnchorRepeatError
p=argparse.ArgumentParser();p.add_argument('--run-a-dir',required=True,type=Path);p.add_argument('--run-b-dir',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
try:r=compare(run_a=a.run_a_dir,run_b=a.run_b_dir,output_dir=a.output_dir)
except SpatialAnchorRepeatError as e:print(f'ReliVE-v2 TAL spatial-anchor repeat error: {e}');raise SystemExit(2)
print(json.dumps(r,sort_keys=True))
