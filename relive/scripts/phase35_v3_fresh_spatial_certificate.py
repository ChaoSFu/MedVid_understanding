#!/usr/bin/env python3
"""Phase 3-v3 fresh LOCAL_ATOMIC fixed-window spatial-certificate slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from relive.phase35_v3 import Phase35V3Error, execute, preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 3-v3 fresh LOCAL_ATOMIC spatial-certificate vertical slice")
    parser.add_argument("--mode", choices=("preflight", "run", "replay"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--prospective-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        kwargs = {"config_path": Path(args.config).resolve(), "runtime_path": Path(args.runtime).resolve(),
                  "prospective_manifest_path": Path(args.prospective_manifest).resolve(),
                  "output_dir": Path(args.output_dir).resolve()}
        report = preflight(**kwargs) if args.mode == "preflight" else execute(**kwargs, mode=args.mode)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("status") == "PASS" else 2
    except (OSError, ValueError, Phase35V3Error) as exc:
        print(f"ReliVE Phase 3-v3 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
