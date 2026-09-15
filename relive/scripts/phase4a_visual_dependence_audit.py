#!/usr/bin/env python3
"""Run the isolated Phase 4A frozen visual-dependence audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from relive.phase4a import Phase4AError, execute, freeze_development_mismatched_public, preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 4A frozen visual-dependence audit")
    parser.add_argument("--mode", choices=("freeze-development-mismatch", "preflight", "run", "replay"), required=True)
    parser.add_argument("--config"); parser.add_argument("--runtime")
    parser.add_argument("--prospective-manifest"); parser.add_argument("--phase35-v3-run-dir")
    parser.add_argument("--phase36-run-dir"); parser.add_argument("--mismatched-manifest")
    parser.add_argument("--output-dir", required=True); parser.add_argument("--calibration-manifest")
    parser.add_argument("--development-manifest"); parser.add_argument("--eligibility-manifest")
    parser.add_argument("--source-mode", choices=("HISTORICAL_EXPLORATORY", "DEVELOPMENT_POSITIVE_CONTROL"))
    args = parser.parse_args()
    try:
        if args.mode == "freeze-development-mismatch":
            if not args.development_manifest and not args.eligibility_manifest: raise Phase4AError("development or eligibility manifest is required")
            result = freeze_development_mismatched_public(development_manifest_path=Path(args.development_manifest).resolve() if args.development_manifest else None,
                                                          eligibility_manifest_path=Path(args.eligibility_manifest).resolve() if args.eligibility_manifest else None,
                                                          output_path=Path(args.output_dir).resolve())
        else:
            if not args.config or not args.mismatched_manifest: raise Phase4AError("config and mismatched manifest are required")
            kwargs = {"config_path": Path(args.config).resolve(), "runtime_path": Path(args.runtime).resolve() if args.runtime else None,
                      "prospective_manifest_path": Path(args.prospective_manifest).resolve() if args.prospective_manifest else None,
                      "phase35_v3_run_dir": Path(args.phase35_v3_run_dir).resolve() if args.phase35_v3_run_dir else None,
                      "phase36_run_dir": Path(args.phase36_run_dir).resolve() if args.phase36_run_dir else None,
                      "mismatched_manifest_path": Path(args.mismatched_manifest).resolve(), "output_dir": Path(args.output_dir).resolve(),
                      "calibration_manifest_path": Path(args.calibration_manifest).resolve() if args.calibration_manifest else None,
                      "development_manifest_path": Path(args.development_manifest).resolve() if args.development_manifest else None,
                      "eligibility_manifest_path": Path(args.eligibility_manifest).resolve() if args.eligibility_manifest else None,
                      "source_mode": args.source_mode}
            result = preflight(**kwargs) if args.mode == "preflight" else execute(**kwargs, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result.get("status") == "PASS" else 2
    except (OSError, ValueError, RuntimeError, Phase4AError) as exc:
        print(f"ReliVE Phase 4A error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
