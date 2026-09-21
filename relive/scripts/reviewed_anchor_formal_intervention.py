#!/usr/bin/env python
"""CLI for reviewed-anchor formal semantic-spatial intervention."""
from __future__ import annotations

import argparse
import json
from relive.v2.reviewed_anchor_formal_intervention import (
    ReviewedAnchorInterventionError, execute, preflight,
    prepare_warning_adjudication_template, summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("prepare-warning-adjudication", "preflight", "run", "replay", "summarize"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eligible-manifest")
    parser.add_argument("--observation-decisions")
    parser.add_argument("--raw-grounding")
    parser.add_argument("--human-review")
    parser.add_argument("--validation-report")
    parser.add_argument("--warning-queue")
    parser.add_argument("--warning-adjudication")
    parser.add_argument("--warning-template-output")
    parser.add_argument("--config")
    parser.add_argument("--smoke-candidate-id")
    args = parser.parse_args()
    try:
        if args.mode == "prepare-warning-adjudication":
            if not args.warning_queue or not args.warning_template_output:
                parser.error("--warning-queue and --warning-template-output are required")
            result = prepare_warning_adjudication_template(args.warning_queue, args.warning_template_output)
        elif args.mode == "preflight":
            needed = ("eligible_manifest", "observation_decisions", "raw_grounding", "human_review", "validation_report", "warning_queue", "config")
            if any(getattr(args, name) is None for name in needed): parser.error("preflight inputs are required")
            result = preflight(eligible_manifest=args.eligible_manifest, observation_decisions=args.observation_decisions,
                raw_grounding=args.raw_grounding, human_review=args.human_review, validation_report=args.validation_report,
                warning_queue=args.warning_queue, warning_adjudication=args.warning_adjudication, config_path=args.config,
                output_dir=args.output_dir)
        elif args.mode in {"run", "replay"}:
            if not args.config: parser.error("--config is required")
            result = execute(output_dir=args.output_dir, config_path=args.config, mode=args.mode, smoke_candidate_id=args.smoke_candidate_id)
        else:
            result = summarize(output_dir=args.output_dir)
    except ReviewedAnchorInterventionError as exc:
        print("reviewed_anchor_formal_intervention error: " + str(exc)); return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
