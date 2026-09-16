#!/usr/bin/env python3
"""Print a safe, non-GT Stage 3B candidate summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relive.v2.independent_repeat import IndependentRepeatError, safe_candidate_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = safe_candidate_summary(run_dir=args.run_dir)
    except IndependentRepeatError as exc:
        print(f"ReliVE-v2 Stage 3B safe-summary error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
