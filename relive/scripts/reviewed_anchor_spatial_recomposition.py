#!/usr/bin/env python
"""Read-only R0 audit and one-round reviewed-anchor composite R1."""
from __future__ import annotations

import argparse
import json

from relive.v2.reviewed_anchor_spatial_recomposition import (
    ReviewedAnchorRecompositionError, audit_r0_cohort, execute, preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("audit-r0", "preflight", "run", "replay"))
    parser.add_argument("--r0-output-dir")
    parser.add_argument("--eligible-manifest")
    parser.add_argument("--warning-queue")
    parser.add_argument("--warning-adjudication")
    parser.add_argument("--derived-review")
    parser.add_argument("--recomputed-observation-decisions")
    parser.add_argument("--recomputed-eligible-manifest")
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "audit-r0":
            if not args.r0_output_dir:
                parser.error("--r0-output-dir is required")
            result = audit_r0_cohort(r0_output_dir=args.r0_output_dir, output_dir=args.output_dir,
                warning_queue=args.warning_queue, warning_adjudication=args.warning_adjudication,
                derived_review=args.derived_review, recomputed_observation_decisions=args.recomputed_observation_decisions,
                recomputed_eligible_manifest=args.recomputed_eligible_manifest)
        elif args.mode == "preflight":
            needed = ("r0_output_dir", "eligible_manifest", "warning_queue", "warning_adjudication", "config")
            if any(getattr(args, key) is None for key in needed):
                parser.error("preflight requires R0, eligible, warning, adjudication and config inputs")
            result = preflight(r0_output_dir=args.r0_output_dir, eligible_manifest=args.eligible_manifest,
                config_path=args.config, output_dir=args.output_dir, warning_queue=args.warning_queue,
                warning_adjudication=args.warning_adjudication, derived_review=args.derived_review,
                recomputed_observation_decisions=args.recomputed_observation_decisions,
                recomputed_eligible_manifest=args.recomputed_eligible_manifest)
        else:
            if not args.config:
                parser.error("--config is required")
            result = execute(output_dir=args.output_dir, config_path=args.config, mode=args.mode)
    except ReviewedAnchorRecompositionError as exc:
        print("reviewed_anchor_spatial_recomposition error: " + str(exc)); return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
