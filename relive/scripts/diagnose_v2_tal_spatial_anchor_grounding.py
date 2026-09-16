#!/usr/bin/env python
import argparse,json
from pathlib import Path
from relive.v2.spatial_anchor_grounding_v2_diagnostic import diagnose
p=argparse.ArgumentParser();p.add_argument('--v2-output-dir',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--max-examples-per-failure',type=int,default=3);a=p.parse_args()
try:r=diagnose(v2_output_dir=a.v2_output_dir,output_dir=a.output_dir,max_examples_per_failure=a.max_examples_per_failure)
except ValueError as e:print(f'ReliVE-v2 Stage 3G v2 diagnostic error: {e}');raise SystemExit(2)
print(json.dumps(r,sort_keys=True))
