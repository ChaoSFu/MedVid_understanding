#!/usr/bin/env python3
"""Read-only Phase 2.5 failure root-cause audit for a completed Phase 2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from relive.failure_diagnostics import FailureDiagnosticError, diagnose_phase2_failures


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 2.5 read-only failure-mode audit")
    parser.add_argument("--run-dir", required=True, help="Completed Phase 2 run directory, not its parent")
    parser.add_argument("--cache-dir", required=True, help="Existing Phase 2 cache directory; read only")
    parser.add_argument("--output-dir", required=True, help="New empty diagnostic output directory")
    args = parser.parse_args()
    try:
        report = diagnose_phase2_failures(Path(args.run_dir), Path(args.output_dir), Path(args.cache_dir))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, FailureDiagnosticError) as exc:
        print(f"ReliVE Phase 2.5 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
