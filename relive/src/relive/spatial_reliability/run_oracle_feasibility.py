from __future__ import annotations

import argparse
import json

from .oracle import run_oracle_feasibility


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated GT-bbox oracle feasibility audit.")
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--oracle-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    result = run_oracle_feasibility(runtime_manifest=args.runtime_manifest, oracle_manifest=args.oracle_manifest,
                                    config_path=args.config, cache_dir=args.cache_dir, output_dir=args.output_dir,
                                    split=args.split, max_samples=args.max_samples)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
