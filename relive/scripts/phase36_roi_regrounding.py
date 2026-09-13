#!/usr/bin/env python3
"""ReliVE Phase 3.6 failure-conditioned automatic ROI re-grounding."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from relive.phase36 import Phase36Error, execute, preflight

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight','run','replay'), required=True)
    parser.add_argument('--config', required=True); parser.add_argument('--runtime', required=True)
    parser.add_argument('--prospective-manifest', required=True); parser.add_argument('--phase35-v3-run-dir', required=True)
    parser.add_argument('--output-dir', required=True); args = parser.parse_args()
    kwargs = {'config_path': Path(args.config).resolve(), 'runtime_path': Path(args.runtime).resolve(),
              'prospective_manifest_path': Path(args.prospective_manifest).resolve(),
              'phase35_v3_run_dir': Path(args.phase35_v3_run_dir).resolve(), 'output_dir': Path(args.output_dir).resolve()}
    try:
        report = preflight(**kwargs) if args.mode == 'preflight' else execute(**kwargs, mode=args.mode)
        print(json.dumps(report, ensure_ascii=False, indent=2)); return 0 if report.get('status') == 'PASS' else 2
    except (OSError, ValueError, Phase36Error) as exc:
        print(f'ReliVE Phase 3.6 error: {exc}', file=sys.stderr); return 2
if __name__ == '__main__': raise SystemExit(main())
