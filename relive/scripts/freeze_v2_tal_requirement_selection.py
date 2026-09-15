#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from relive.v2.task_selection import TALSelectionError, freeze_explicit_public_identity, freeze_selector_order, write_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-question-selector", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max-samples", type=int)
    group.add_argument("--identity-manifest", type=Path)
    args = parser.parse_args()
    try:
        selection = (freeze_selector_order(selector_path=args.public_question_selector, max_samples=args.max_samples)
                     if args.max_samples is not None else
                     freeze_explicit_public_identity(selector_path=args.public_question_selector, identity_manifest_path=args.identity_manifest))
        write_selection(selection, args.output)
    except TALSelectionError as exc:
        print(f"ReliVE-v2 TAL selection error: {exc}")
        return 2
    print(json.dumps({"status": "PASS", "selection": str(args.output), "selection_basis": selection.selection_basis,
                      "selection_content_sha256": selection.to_canonical_dict()["selection_content_sha256"],
                      "model_calls_made": 0, "frames_read": 0, "videos_read": 0, "gt_used": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
