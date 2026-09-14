#!/usr/bin/env python3
"""Create a zero-model ROI worksheet for a new Phase 4A-0 control cohort."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relive.phase4a0_development_controls import (DevelopmentControlError, apply_target_roi_overrides,
                                                   apply_matched_control_overrides, freeze_development_controls,
                                                   prepare_development_controls)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "apply-target-rois", "apply-matched-controls", "freeze"), default="prepare")
    parser.add_argument("--selection")
    parser.add_argument("--source-json")
    parser.add_argument("--frame-root")
    parser.add_argument("--source-prefix", default="/root/data")
    parser.add_argument("--output-dir")
    parser.add_argument("--roi-template")
    parser.add_argument("--target-roi-overrides")
    parser.add_argument("--target-roi-worksheet")
    parser.add_argument("--matched-control-overrides")
    parser.add_argument("--ready-worksheet")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            if not all((args.selection, args.source_json, args.frame_root, args.output_dir)):
                raise DevelopmentControlError("PREPARE_REQUIRES_SELECTION_SOURCE_JSON_FRAME_ROOT_AND_OUTPUT_DIR")
            result = prepare_development_controls(selection_path=Path(args.selection), source_json=Path(args.source_json),
                                                  frame_root=Path(args.frame_root), source_prefix=args.source_prefix,
                                                  output_dir=Path(args.output_dir))
        elif args.mode == "apply-target-rois":
            if not all((args.roi_template, args.target_roi_overrides, args.output)):
                raise DevelopmentControlError("APPLY_TARGET_ROIS_REQUIRES_ROI_TEMPLATE_OVERRIDES_AND_OUTPUT")
            result = apply_target_roi_overrides(roi_template_path=Path(args.roi_template),
                                                target_roi_overrides_path=Path(args.target_roi_overrides),
                                                output_path=Path(args.output))
        elif args.mode == "apply-matched-controls":
            if not all((args.target_roi_worksheet, args.matched_control_overrides, args.output)):
                raise DevelopmentControlError("APPLY_MATCHED_CONTROLS_REQUIRES_TARGET_WORKSHEET_OVERRIDES_AND_OUTPUT")
            result = apply_matched_control_overrides(target_roi_worksheet_path=Path(args.target_roi_worksheet),
                                                     matched_control_overrides_path=Path(args.matched_control_overrides),
                                                     output_path=Path(args.output))
        else:
            if not all((args.ready_worksheet, args.output_dir)):
                raise DevelopmentControlError("FREEZE_REQUIRES_READY_WORKSHEET_AND_OUTPUT_DIR")
            result = freeze_development_controls(ready_worksheet_path=Path(args.ready_worksheet),
                                                 output_dir=Path(args.output_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, DevelopmentControlError) as exc:
        print(f"ReliVE Phase 4A-0 development-control preparation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
