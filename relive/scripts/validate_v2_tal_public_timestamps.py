#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from relive.v2.timestamp_provenance import TimestampProvenanceError, validate_export

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try: result = validate_export(args.output_dir)
    except TimestampProvenanceError as exc:
        print(f"ReliVE-v2 TAL public timestamp validation error: {exc}"); return 2
    print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
