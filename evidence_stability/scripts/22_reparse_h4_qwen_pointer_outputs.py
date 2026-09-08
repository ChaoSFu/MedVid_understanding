#!/usr/bin/env python3
"""Create a separate, auditable compatibility parse of frozen H4 pointer outputs."""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
import sys

sys.path.insert(0, str(SRC_DIR))

from evidence_stability.spatial import SPATIAL_POINTER_COMPAT_PARSER_VERSION, parse_qwen_list_wrapped_bbox  # noqa: E402
from evidence_stability.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def latest_by_candidate(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if candidate_id:
            latest[candidate_id] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility-reparse frozen Qwen H4 spatial pointer responses.")
    parser.add_argument("--pointer_predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    source = Path(args.pointer_predictions)
    out_dir = Path(args.output_dir)
    predictions_dir = out_dir / "predictions"
    summary_dir = out_dir / "summary"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = latest_by_candidate(read_jsonl(source))
    reparsed: list[dict] = []
    methods: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    strict_invalid = 0
    for row in rows:
        if not row.get("bbox_valid"):
            strict_invalid += 1
        parsed = parse_qwen_list_wrapped_bbox(row.get("raw_response"))
        methods[parsed.get("parse_method") or "NONE"] += 1
        if not parsed["bbox_valid"]:
            reasons[parsed.get("reason") or "UNKNOWN"] += 1
        reparsed.append(
            {
                **row,
                "strict_bbox_valid": row.get("bbox_valid"),
                "strict_bbox_invalid_reason": row.get("bbox_invalid_reason"),
                "bbox_valid": bool(parsed["bbox_valid"]),
                "predicted_bbox_norm": parsed.get("bbox"),
                "bbox_invalid_reason": None if parsed["bbox_valid"] else parsed.get("reason"),
                "bbox_area_fraction": parsed.get("bbox_area_fraction") if parsed["bbox_valid"] else None,
                "bbox_parser_version": SPATIAL_POINTER_COMPAT_PARSER_VERSION,
                "bbox_parse_method": parsed.get("parse_method"),
                "source_pointer_predictions": str(source),
                "reparsed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    output_path = predictions_dir / "spatial_pointer_predictions_compat.jsonl"
    write_jsonl(output_path, reparsed)
    audit = {
        "purpose": "posthoc_format_compatibility_parse_only",
        "model_inference_executed": False,
        "gt_used": False,
        "source_pointer_predictions": str(source),
        "output_pointer_predictions": str(output_path),
        "parser_version": SPATIAL_POINTER_COMPAT_PARSER_VERSION,
        "n_source_rows": len(rows),
        "n_strict_invalid": strict_invalid,
        "n_compat_valid": sum(bool(row["bbox_valid"]) for row in reparsed),
        "n_compat_invalid": sum(not bool(row["bbox_valid"]) for row in reparsed),
        "parse_method_counts": dict(methods),
        "invalid_reason_counts": dict(reasons),
    }
    audit_path = summary_dir / "h4_pointer_compat_reparse_audit.json"
    write_json(audit_path, audit)
    print(f"Wrote {output_path}")
    print(f"Wrote {audit_path}")
    print(audit)


if __name__ == "__main__":
    main()
