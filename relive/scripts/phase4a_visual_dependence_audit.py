#!/usr/bin/env python3
"""Run the isolated Phase 4A frozen visual-dependence audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from relive.phase4a import Phase4AError, execute, preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 4A frozen visual-dependence audit")
    parser.add_argument("--mode", choices=("preflight", "run", "replay"), required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--runtime", required=True)
    parser.add_argument("--prospective-manifest", required=True); parser.add_argument("--phase35-v3-run-dir", required=True)
    parser.add_argument("--phase36-run-dir", required=True); parser.add_argument("--mismatched-manifest", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--calibration-manifest")
    args = parser.parse_args()
    kwargs = {"config_path": Path(args.config).resolve(), "runtime_path": Path(args.runtime).resolve(), "prospective_manifest_path": Path(args.prospective_manifest).resolve(),
              "phase35_v3_run_dir": Path(args.phase35_v3_run_dir).resolve(), "phase36_run_dir": Path(args.phase36_run_dir).resolve(),
              "mismatched_manifest_path": Path(args.mismatched_manifest).resolve(), "output_dir": Path(args.output_dir).resolve(),
              "calibration_manifest_path": Path(args.calibration_manifest).resolve() if args.calibration_manifest else None}
    try:
        result = preflight(**kwargs) if args.mode == "preflight" else execute(**kwargs, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result.get("status") == "PASS" else 2
    except (OSError, ValueError, RuntimeError, Phase4AError) as exc:
        print(f"ReliVE Phase 4A error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
