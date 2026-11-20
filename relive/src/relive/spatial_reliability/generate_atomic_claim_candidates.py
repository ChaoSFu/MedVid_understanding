from __future__ import annotations

import argparse
import json

from .atomic_claims import generate_atomic_claim_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate review-only atomic claims from bound RC captions."
    )
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument("--execution-mode", choices=("run", "replay"), default="run")
    args = parser.parse_args()
    result = generate_atomic_claim_candidates(
        runtime_manifest=args.runtime_manifest,
        source_json=args.source_json,
        config_path=args.config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        execution_mode=args.execution_mode,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
