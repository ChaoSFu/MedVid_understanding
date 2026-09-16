#!/usr/bin/env python3
"""Compare two completed Stage 3B fresh runs without model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relive.v2.independent_repeat import IndependentRepeatError, compare_independent_runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a-dir", required=True, type=Path)
    parser.add_argument("--run-b-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare_independent_runs(run_a_dir=args.run_a_dir, run_b_dir=args.run_b_dir, output_dir=args.output_dir)
    except IndependentRepeatError as exc:
        print(f"ReliVE-v2 Stage 3B independent-repeat error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
