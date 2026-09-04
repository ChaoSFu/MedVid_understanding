from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import RunConfig
from .io_utils import read_jsonl, write_json, write_jsonl
from .selectors.videoitg import selector_output_paths, validate_selection


def dedupe_rows(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    seen_sample_ids: set[str] = set()
    seen_cache_keys: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        cache_key = str(row.get("cache_key", ""))
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        if cache_key and cache_key in seen_cache_keys:
            raise ValueError(f"Duplicate cache_key in {source}: {cache_key}")
        seen_sample_ids.add(sample_id)
        if cache_key:
            seen_cache_keys.add(cache_key)
        out.append(row)
    out.sort(key=lambda row: str(row.get("sample_id", "")))
    return out


def merge_selector_shards(output_root: Path, num_shards: int, allow_missing: bool = False) -> dict[str, Any]:
    cfg = RunConfig(output_root=output_root)
    top_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for shard_index in range(num_shards):
        top_path, scores_path, errors_path = selector_output_paths(cfg, num_shards, shard_index)
        if top_path.exists():
            rows = read_jsonl(top_path)
            for row in rows:
                validate_selection(row)
            top_rows.extend(rows)
        else:
            missing.append(str(top_path))
        if scores_path.exists():
            score_rows.extend(read_jsonl(scores_path))
        elif not allow_missing:
            missing.append(str(scores_path))
        if errors_path.exists():
            error_rows.extend(read_jsonl(errors_path))

    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing selector shard files: {missing[:10]}")

    top_rows = dedupe_rows(top_rows, "top32 shards")
    score_rows = dedupe_rows(score_rows, "score shards") if score_rows else []
    error_rows.sort(key=lambda row: str(row.get("sample_id", "")))

    final_top = cfg.selector_dir / "videoitg_top32.jsonl"
    final_scores = cfg.selector_dir / "videoitg_scores.jsonl"
    final_errors = cfg.selector_dir / "selector_errors.jsonl"
    write_jsonl(final_top, top_rows)
    write_jsonl(final_scores, score_rows)
    write_jsonl(final_errors, error_rows)

    report = {
        "output_root": str(output_root),
        "num_shards": num_shards,
        "top32_rows": len(top_rows),
        "score_rows": len(score_rows),
        "error_rows": len(error_rows),
        "missing": missing,
        "final_top32": str(final_top),
        "final_scores": str(final_scores),
        "final_errors": str(final_errors),
    }
    write_json(cfg.selector_dir / "merge_selector_shards_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    p = argparse.ArgumentParser(description="Merge VideoITG selector shard JSONL outputs.")
    p.add_argument("--output-root", type=Path, default=cfg.output_root)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--allow-missing", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = merge_selector_shards(args.output_root, args.num_shards, allow_missing=args.allow_missing)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
