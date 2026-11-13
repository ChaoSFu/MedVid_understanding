from __future__ import annotations

import argparse
import json

from .posthoc import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc GT evaluation for a completed ReliVE-Spatial runtime.")
    parser.add_argument("--runtime-results", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--oracle-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = evaluate(runtime_results=args.runtime_results, runtime_manifest=args.runtime_manifest,
                      oracle_manifest=args.oracle_manifest, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
