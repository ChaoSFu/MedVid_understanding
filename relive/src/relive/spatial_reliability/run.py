from __future__ import annotations

import argparse
import json

from .runtime import run_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen ReliVE-Spatial runtime without oracle inputs.")
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["one_shot", "verify_only", "generic_retry", "relive_spatial"], default="relive_spatial")
    parser.add_argument("--execution-mode", choices=["run", "replay"], default="run")
    parser.add_argument("--split", default="development")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    result = run_runtime(runtime_manifest=args.runtime_manifest, config_path=args.config, cache_dir=args.cache_dir,
                         output_dir=args.output_dir, mode=args.mode, split=args.split,
                         max_samples=args.max_samples, execution_mode=args.execution_mode)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
