from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_ROOT
from .io_utils import read_jsonl, write_json
from .timestamp_adapters import (
    STRICT_SINGLE_FPS_ADAPTER,
    TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER,
    adapter_scientific_profile,
    audit_adapter_for_row,
)


ADAPTERS = (STRICT_SINGLE_FPS_ADAPTER, TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER)


def summarize(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_adapter: dict[str, Any] = {}
    full_timing_status = Counter(row.get("timestamp_spacing_audit", {}).get("status") for row in rows)
    for adapter in ADAPTERS:
        subset = [r for r in records if r["adapter"] == adapter]
        applicable = [r for r in subset if r["applicable"]]
        skipped = [r for r in subset if not r["applicable"]]
        by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
        for r in subset:
            by_dataset[str(r.get("dataset_name") or "Unknown")][r["status"]] += 1
        by_adapter[adapter] = {
            "profile": adapter_scientific_profile(adapter),
            "n_records": len(subset),
            "applicable": len(applicable),
            "skipped": len(skipped),
            "coverage_fraction": len(applicable) / len(subset) if subset else 0.0,
            "status_counts": dict(sorted(Counter(r["status"] for r in subset).items())),
            "qwen_processing_status_counts": dict(sorted(Counter(str(r.get("qwen_processing_status")) for r in subset).items())),
            "by_dataset_status_counts": {k: dict(sorted(v.items())) for k, v in sorted(by_dataset.items())},
            "official_prompt_exact_match_all_applicable": all(r["official_prompt_exact_match"] for r in applicable) if applicable else False,
            "official_prompt_included_verbatim_all_applicable": all(r["official_prompt_included_verbatim"] for r in applicable) if applicable else False,
            "uses_sample_video_only_all": all(r["uses_sample_video_only"] for r in subset),
            "additional_temporal_resampling_any": any(r["additional_temporal_resampling"] for r in subset),
            "example_skip_reasons": [r["reason"] for r in skipped[:5]],
        }
    return {
        "n_smoke_rows": len(rows),
        "smoke_timestamp_status_counts": dict(sorted(full_timing_status.items())),
        "adapters": by_adapter,
        "recommendation": {
            "primary_external_baseline": TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER,
            "diagnostic_subset": STRICT_SINGLE_FPS_ADAPTER,
            "rationale": (
                "strict_single_fps_subset is closest to official TimeLens-8B behavior but cannot cover MedVidU TAL. "
                "textual_timestamp_image_sequence preserves MedVidU-provided frame evidence and original GT-free local_time for all smoke samples, "
                "so it is the better primary MedVidU external baseline if named as an adapted TimeLens variant rather than an official TimeLens result."
            ),
        },
    }


def compare_adapters(
    manifest_path: Path,
    output_path: Path,
    limit: int = 20,
    process_qwen: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)[:limit]
    records = []
    for row in rows:
        for adapter in ADAPTERS:
            records.append(audit_adapter_for_row(row, adapter, process_qwen=process_qwen).to_dict())
    payload = {
        "audit_name": "timestamp_adapter_preflight_comparison",
        "manifest_path": str(manifest_path),
        "limit": limit,
        "process_qwen": process_qwen,
        "records": records,
        "summary": summarize(records, rows),
    }
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare strict single-fps and textual timestamp TimeLens-MedVidU adapters.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifest" / "medvidu_tal_timelens_manifest_gt_free.smoke.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "audit" / "timestamp_adapter_preflight_comparison.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--process-qwen", action="store_true", help="Call qwen_vl_utils.process_vision_info for each adapter message.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = compare_adapters(args.manifest, args.output, limit=args.limit, process_qwen=args.process_qwen)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

