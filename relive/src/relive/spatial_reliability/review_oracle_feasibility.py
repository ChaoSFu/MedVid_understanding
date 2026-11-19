from __future__ import annotations

import argparse
import json

from .oracle_review import prepare_oracle_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create read-only overlays and a human-review template for frozen Oracle results."
    )
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--oracle-manifest", required=True)
    parser.add_argument("--oracle-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = prepare_oracle_review(
        runtime_manifest=args.runtime_manifest,
        oracle_manifest=args.oracle_manifest,
        oracle_results=args.oracle_results,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
