from __future__ import annotations

import argparse
import json

from .dataset import prepare_rc_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare GT-isolated ReliVE-Spatial Region Caption manifests.")
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-prefix")
    parser.add_argument("--frame-root")
    parser.add_argument("--dev-count", type=int, default=20)
    args = parser.parse_args()
    result = prepare_rc_dataset(source_json=args.source_json, output_dir=args.output_dir,
                                source_prefix=args.source_prefix, frame_root=args.frame_root,
                                dev_count=args.dev_count)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
