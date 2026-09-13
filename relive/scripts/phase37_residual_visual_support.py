#!/usr/bin/env python3
"""Run the frozen, one-attempt Phase 3.7 residual-support diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from relive.phase37 import Phase37Error, execute, preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 3.7 residual visual support diagnostic")
    parser.add_argument("--mode", choices=("preflight", "run", "replay"), required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--runtime", required=True)
    parser.add_argument("--prospective-manifest", required=True); parser.add_argument("--phase36-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        kwargs = {"config_path": Path(args.config).resolve(), "runtime_path": Path(args.runtime).resolve(),
                  "prospective_manifest_path": Path(args.prospective_manifest).resolve(),
                  "phase36_run_dir": Path(args.phase36_run_dir).resolve(), "output_dir": Path(args.output_dir).resolve()}
        result = preflight(**kwargs) if args.mode == "preflight" else execute(**kwargs, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result.get("status") == "PASS" else 2
    except (OSError, ValueError, Phase37Error) as exc:
        print(f"ReliVE Phase 3.7 error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
