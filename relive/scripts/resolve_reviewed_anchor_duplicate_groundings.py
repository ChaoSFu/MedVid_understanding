#!/usr/bin/env python
"""Create and apply explicit canonical labels for duplicate grounding warnings."""
from __future__ import annotations

import argparse
import json

from relive.v2.reviewed_anchor_label_resolution import (
    ReviewedAnchorLabelResolutionError, apply_canonical_labels, prepare_canonical_label_template,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "apply"), required=True)
    parser.add_argument("--warning-adjudication", required=True)
    parser.add_argument("--canonical-label-adjudication")
    parser.add_argument("--warning-queue")
    parser.add_argument("--raw-grounding")
    parser.add_argument("--human-review")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--template-output")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            if not args.template_output:
                parser.error("prepare requires --template-output")
            result = prepare_canonical_label_template(warning_adjudication=args.warning_adjudication, output_path=args.template_output)
        else:
            needed = ("canonical_label_adjudication", "warning_queue", "raw_grounding", "human_review")
            if any(getattr(args, name) is None for name in needed):
                parser.error("apply requires canonical labels, warning queue, raw grounding, and human review")
            result = apply_canonical_labels(raw_grounding=args.raw_grounding, human_review=args.human_review,
                warning_queue=args.warning_queue, warning_adjudication=args.warning_adjudication,
                canonical_label_adjudication=args.canonical_label_adjudication, output_dir=args.output_dir)
    except ReviewedAnchorLabelResolutionError as exc:
        print("reviewed_anchor_label_resolution error: " + str(exc)); return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
