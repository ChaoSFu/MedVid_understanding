#!/usr/bin/env python
import argparse,json
from pathlib import Path
from relive.v2.spatial_anchor_grounding_v31_smoke import freeze
p=argparse.ArgumentParser();p.add_argument('--v3-selection-manifest',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
try:r=freeze(v3_selection_manifest=a.v3_selection_manifest,output_dir=a.output_dir)
except ValueError as e:print(f'ReliVE-v2 Stage 3G v3.1 smoke selection error: {e}');raise SystemExit(2)
print(json.dumps(r,sort_keys=True))
