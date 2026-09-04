from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import repair_jsonl, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair interrupted JSONL artifacts by keeping valid JSON object lines.")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--backup-suffix", default=".corrupt.bak")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    reports = [
        repair_jsonl(path, backup_suffix=args.backup_suffix)
        for path in args.paths
    ]
    if args.report is not None:
        write_json(args.report, reports)
    for report in reports:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
