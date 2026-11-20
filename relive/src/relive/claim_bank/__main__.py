"""python -m relive.claim_bank --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import ClaimBankError, QA_TYPES
from .data import audit_dataset, load_json
from .pipeline import generate, validate_bank


def main() -> int:
    parser = argparse.ArgumentParser(description="GT-isolated question-conditioned MedVidU Claim Bank; no verification or QA answers.")
    parser.add_argument("--mode", required=True, choices=("audit", "prepare", "stage1", "stage2", "validate"))
    parser.add_argument("--protocol-version", choices=("v1", "v1.1", "v1.2"),
                        help="Defaults: stage1 v1.1, stage2 v1.2. Validation detects the frozen version.")
    parser.add_argument("--dataset-json")
    parser.add_argument("--expected-count", type=int, default=6270)
    parser.add_argument("--runtime-manifest")
    parser.add_argument("--stage1-dir")
    parser.add_argument("--config")
    parser.add_argument("--model", help="Must match the frozen config model identity; never overrides it.")
    parser.add_argument("--cache-dir")
    parser.add_argument("--data-root", help="Local image root replacing /root/data; never substitutes unrelated images.")
    parser.add_argument("--path-map", help="JSON object mapping exact source directory prefixes to local roots.")
    parser.add_argument("--qa-type", action="append", choices=QA_TYPES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, help="Exclusive source record index.")
    parser.add_argument("--per-qa-type", type=int, help="Freeze first N source records of each qa_type; engineering smoke only.")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Zero model calls; preflight only, not generated claims.")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--replay-from", help="Frozen prior run directory; requires exactly matching inputs/config/selection.")
    args = parser.parse_args()
    try:
        path_map = load_json(args.path_map) if args.path_map else None
        if path_map is not None and (not isinstance(path_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in path_map.items())):
            raise ClaimBankError("PATH_MAP_INVALID")
        if args.mode in {"audit", "prepare"}:
            if not args.dataset_json:
                parser.error("--dataset-json is required")
            result = audit_dataset(args.dataset_json, args.output_dir, data_root=args.data_root,
                                   path_map=path_map, expected_count=args.expected_count, prepare=args.mode == "prepare")
            result = {k: result[k] for k in ("record_count", "dataset_sha256", "count_matches", "qa_type_counts", "duplicate_id_group_count", "existing_unique_frame_paths", "missing_unique_frame_paths", "model_calls_made", "cache_writes", "certificate_writes", "new_verified_count")}
            result["dataset_audit"] = str(Path(args.output_dir) / "dataset_audit.json")
        elif args.mode == "validate":
            result = validate_bank(Path(args.output_dir))
        else:
            if not args.runtime_manifest or not args.config or not args.cache_dir:
                parser.error("--runtime-manifest, --config and --cache-dir are required")
            result = generate(runtime_manifest=args.runtime_manifest, output_dir=args.output_dir,
                              stage=1 if args.mode == "stage1" else 2, config_path=args.config,
                              cache_dir=args.cache_dir, stage1_dir=args.stage1_dir, data_root=args.data_root,
                              path_map=path_map, max_frames=args.max_frames, qa_type=args.qa_type,
                              start_index=args.start_index, end_index=args.end_index, limit=args.limit,
                              per_qa_type=args.per_qa_type, resume=args.resume, replay_only=args.replay_only,
                              replay_from=args.replay_from, dry_run=args.dry_run, model=args.model,
                              protocol_version=args.protocol_version or ("v1.2" if args.mode == "stage2" else "v1.1"))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Do not print request bodies, forbidden values, credentials or full source records.
        code = str(exc).split(":", 1)[0] if isinstance(exc, ClaimBankError) else type(exc).__name__
        print(json.dumps({"status": "FAIL", "error": code, "new_verified_count": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
