#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relive.v2.temporal_search_plan import TemporalSearchPlanError, validate_temporal_search_plan_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate_temporal_search_plan_artifacts(args.output_dir)
    except TemporalSearchPlanError as exc:
        print(f"ReliVE-v2 TAL temporal-search validation error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
