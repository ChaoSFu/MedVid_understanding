#!/usr/bin/env python3
"""Read-only validator for previously frozen TAL RequirementSpec artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from relive.v2.requirement_freeze import RequirementFreezeError, validate_requirement_freeze_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate_requirement_freeze_artifacts(args.output_dir)
    except RequirementFreezeError as exc:
        print(f"ReliVE-v2 TAL requirement validation error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
