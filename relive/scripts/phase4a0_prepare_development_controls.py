#!/usr/bin/env python3
"""Create a zero-model ROI worksheet for a new Phase 4A-0 control cohort."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relive.phase4a0_development_controls import (DevelopmentControlError, apply_target_roi_overrides,
                                                   prepare_development_controls)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "apply-target-rois"), default="prepare")
    parser.add_argument("--selection")
    parser.add_argument("--source-json")
    parser.add_argument("--frame-root")
    parser.add_argument("--source-prefix", default="/root/data")
    parser.add_argument("--output-dir")
    parser.add_argument("--roi-template")
    parser.add_argument("--target-roi-overrides")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            if not all((args.selection, args.source_json, args.frame_root, args.output_dir)):
                raise DevelopmentControlError("PREPARE_REQUIRES_SELECTION_SOURCE_JSON_FRAME_ROOT_AND_OUTPUT_DIR")
            result = prepare_development_controls(selection_path=Path(args.selection), source_json=Path(args.source_json),
                                                  frame_root=Path(args.frame_root), source_prefix=args.source_prefix,
                                                  output_dir=Path(args.output_dir))
        else:
            if not all((args.roi_template, args.target_roi_overrides, args.output)):
                raise DevelopmentControlError("APPLY_TARGET_ROIS_REQUIRES_ROI_TEMPLATE_OVERRIDES_AND_OUTPUT")
            result = apply_target_roi_overrides(roi_template_path=Path(args.roi_template),
                                                target_roi_overrides_path=Path(args.target_roi_overrides),
                                                output_path=Path(args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, DevelopmentControlError) as exc:
        print(f"ReliVE Phase 4A-0 development-control preparation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
