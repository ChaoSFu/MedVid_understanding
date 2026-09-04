from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import traceback
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from ..config import REGION_CAPTION_QA_TYPES, RunConfig
from ..io_utils import append_jsonl, first_gt_leak, load_completed_keys, read_jsonl, sha256_json, write_json, write_jsonl
from .cafs_adapter import OfficialCAFSAdapter
from .dig_adapter import build_selector_record, global_uniform_positions, refine_and_select
from .query_identifier import OfficialQueryIdentifier, QueryIdentifierUnavailable
from .reward_adapter import OfficialRewardAssigner


ALLOWED_AUDIT_KEYS = {"gt_information_used"}


def assert_selector_no_gt_leak(row: dict[str, Any]) -> None:
    audit_value = row.pop("gt_information_used", None) if "gt_information_used" in row else None
    try:
        leak = first_gt_leak(row)
        if leak is not None:
            raise AssertionError(f"GT-derived field leaked into DIG selector artifact at {leak}")
    finally:
        if audit_value is not None:
            row["gt_information_used"] = audit_value
    if audit_value is not False:
        raise AssertionError("gt_information_used must be exactly false")


def dig_git_commit(dig_repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(dig_repo_dir), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selector_cache_key(row: dict[str, Any], dig_commit: str, selector_cfg: dict[str, Any]) -> str:
    return sha256_json(
        {
            "sample_id": row["sample_id"],
            "question_hash": sha256_json(row.get("question", "")),
            "ordered_frame_identity": sha256_json(
                {
                    "positions": list(range(int(row["n_medvidu_frames"]))),
                    "paths": row.get("video", []),
                    "sampled_video_frames": row.get("sampled_video_frames", []),
                }
            ),
            "dig_commit": dig_commit,
            "query_identifier_model": selector_cfg["query_identifier_model"],
            "cafs_model": selector_cfg["cafs_model"],
            "cafs_sample_per_sec": selector_cfg["cafs_sample_per_sec"],
            "reward_lmm": selector_cfg["reward_lmm"],
            "requested_k": selector_cfg["requested_k"],
            "wlen": selector_cfg["video_refinement_wlen"],
            "selector_version": selector_cfg["selector_version"],
        }
    )


def bypass_rc_record(row: dict[str, Any], dig_commit_value: str, selector_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = {**selector_cfg, "question_hash": sha256_json(row.get("question", ""))}
    record = build_selector_record(
        manifest_row={**row, "selector_reason": "RC_SELECTOR_ADAPTATION_REQUIRED: mandatory region frame/bbox must be preserved"},
        query_type="not_applicable",
        selected_score_order=[],
        selected_chronological=[],
        dig_commit=dig_commit_value,
        selector_config=cfg,
    )
    record["selector_applicable"] = False
    record["selector_reason"] = "RC_SELECTOR_ADAPTATION_REQUIRED"
    record["cache_key"] = selector_cache_key(row, dig_commit_value, selector_cfg)
    return record


async def run_selector_async(args: argparse.Namespace) -> dict[str, Any]:
    cfg = RunConfig(
        data_path=args.data_path,
        output_root=args.output_root,
        dig_repo_dir=args.dig_repo_dir,
        old_data_root=args.old_data_root,
        new_data_root=args.new_data_root,
    )
    cfg.make_dirs()
    selector_cfg = cfg.selector.to_jsonable()
    selector_cfg["requested_k"] = int(args.k)
    selector_cfg["video_refinement_wlen"] = int(args.wlen)
    write_json(cfg.provenance_dir / "selector_config.json", selector_cfg)
    dig_commit_value = dig_git_commit(args.dig_repo_dir)
    (cfg.provenance_dir / "dig_git_commit.txt").write_text(dig_commit_value + "\n", encoding="utf-8")

    rows = read_jsonl(args.manifest)
    if not rows:
        raise RuntimeError(f"No manifest rows in {args.manifest}")

    frame_counts = [int(row["n_medvidu_frames"]) for row in rows]
    if any(n < args.k for n in frame_counts) and not args.allow_short_videos:
        summary = {
            "status": "SHORT_VIDEO_REVIEW_REQUIRED",
            "requested_k": args.k,
            "n_samples_lt_k": sum(n < args.k for n in frame_counts),
            "policy": "effective_k=min(32,N) is implemented, but full selector run stops until reviewed.",
        }
        write_json(cfg.selector_dir / "selector_summary.json", summary)
        raise RuntimeError(json.dumps(summary, ensure_ascii=False))

    query_identifier = OfficialQueryIdentifier(
        dig_repo_dir=args.dig_repo_dir,
        model=args.query_identifier_model,
        base_url=args.query_base_url,
        api_key=args.query_api_key,
        concurrency=args.query_concurrency,
    )
    query_preflight = query_identifier.preflight()
    write_json(cfg.provenance_dir / "selector_model_fingerprints.json", {"query_identifier": query_preflight})
    if query_preflight["status"] != "OK":
        summary = {
            "status": "QUERY_IDENTIFIER_UNAVAILABLE",
            "official_required_model": query_preflight["required_model"],
            "current_configured_model": query_preflight["configured_model"],
            "base_url": query_preflight["base_url"],
            "current_available_models": query_preflight.get("server_models", {}).get("available_models", []),
            "server_status": query_preflight.get("server_models"),
            "possible_alternatives_require_review": [
                "run the official Qwen/Qwen3-Next-80B-A3B-Instruct server",
                "review and document an adapted query-identifier model",
            ],
        }
        write_json(cfg.selector_dir / "selector_summary.json", summary)
        raise QueryIdentifierUnavailable(json.dumps(summary, ensure_ascii=False))

    cafs = OfficialCAFSAdapter(
        dig_repo_dir=args.dig_repo_dir,
        model_name=selector_cfg["cafs_model"],
        samples_per_sec=selector_cfg["cafs_sample_per_sec"],
        infer_batch_size=selector_cfg["cafs_infer_batch_size"],
        device=args.cafs_device,
    )
    rewarder = OfficialRewardAssigner(
        dig_repo_dir=args.dig_repo_dir,
        model=args.reward_model_path,
        display_name=selector_cfg["reward_lmm"],
        base_url=args.reward_base_url,
        api_key=args.reward_api_key,
        concurrency=args.reward_concurrency,
    )

    query_types_path = cfg.selector_dir / "query_types.jsonl"
    rframes_path = cfg.selector_dir / "rframes.jsonl"
    rewards_path = cfg.selector_dir / "rewards.jsonl"
    selector_path = cfg.selector_dir / args.output_name
    errors_path = cfg.selector_dir / "selector_errors.jsonl"
    completed = load_completed_keys(selector_path)
    counts = Counter()
    duplicate_examples: list[dict[str, Any]] = []
    oob_examples: list[dict[str, Any]] = []

    for row in rows:
        cache_key = selector_cache_key(row, dig_commit_value, selector_cfg)
        if cache_key in completed:
            counts["skipped_completed"] += 1
            continue
        try:
            qa_type = str(row.get("qa_type", ""))
            if qa_type in REGION_CAPTION_QA_TYPES:
                record = bypass_rc_record(row, dig_commit_value, selector_cfg)
                record["cache_key"] = cache_key
                assert_selector_no_gt_leak(record)
                append_jsonl(selector_path, record)
                completed.add(cache_key)
                counts["rc_not_applicable"] += 1
                continue

            query_result = (await query_identifier.classify_many([row]))[0]
            query_type = str(query_result["query_type"])
            append_jsonl(query_types_path, {**query_result, "cache_key": cache_key})
            if query_type == "global":
                selected = global_uniform_positions(int(row["n_medvidu_frames"]), k=args.k)
                cfg_for_record = {**selector_cfg, "question_hash": sha256_json(row.get("question", ""))}
                record = build_selector_record(
                    manifest_row=row,
                    query_type="global",
                    selected_score_order=selected,
                    selected_chronological=selected,
                    dig_commit=dig_commit_value,
                    selector_config=cfg_for_record,
                )
                counts["global"] += 1
            else:
                cafs_result = cafs.get_r_frames(row)
                r_positions = [int(x) for x in cafs_result["r_frame_original_positions"]]
                boundaries = [int(x) for x in cafs_result["boundaries_original_positions"]]
                append_jsonl(rframes_path, {**cafs_result, "sample_id": row["sample_id"], "cache_key": cache_key})
                reward_values = await rewarder.score_many(row, r_positions)
                append_jsonl(
                    rewards_path,
                    {
                        "sample_id": row["sample_id"],
                        "reward_lmm": selector_cfg["reward_lmm"],
                        "r_frame_original_positions": r_positions,
                        "reward_values": reward_values,
                        "cache_key": cache_key,
                    },
                )
                intervals, selected_score_order = refine_and_select(
                    reward_values,
                    boundaries,
                    dig_repo_dir=args.dig_repo_dir,
                    k=args.k,
                    wlen=args.wlen,
                )
                selected_chronological = sorted(selected_score_order)
                cfg_for_record = {**selector_cfg, "question_hash": sha256_json(row.get("question", ""))}
                record = build_selector_record(
                    manifest_row=row,
                    query_type="local",
                    selected_score_order=selected_score_order,
                    selected_chronological=selected_chronological,
                    dig_commit=dig_commit_value,
                    selector_config=cfg_for_record,
                    r_frame_positions=r_positions,
                    reward_values=reward_values,
                    reward_boundaries=boundaries,
                    refined_intervals=intervals,
                )
                counts["local"] += 1

            selected_positions = [int(x) for x in record["selected_original_positions_chronological"]]
            if any(pos < 0 or pos >= int(row["n_medvidu_frames"]) for pos in selected_positions):
                oob_examples.append({"sample_id": row["sample_id"], "selected": selected_positions[:40]})
                raise AssertionError("Selected DIG index out of MedVidU logical-position bounds")
            if len(set(selected_positions)) != len(selected_positions):
                duplicate_examples.append({"sample_id": row["sample_id"], "selected": selected_positions})
                if not args.allow_duplicate_selected_indices:
                    raise RuntimeError("DUPLICATE_SELECTED_INDICES_REVIEW_REQUIRED")
            if int(row["n_medvidu_frames"]) >= args.k and len(selected_positions) != args.k:
                raise AssertionError("effective_k is not 32 for N>=32")
            record["cache_key"] = cache_key
            assert_selector_no_gt_leak(record)
            append_jsonl(selector_path, record)
            completed.add(cache_key)
            counts["completed"] += 1
        except Exception as exc:
            append_jsonl(
                errors_path,
                {
                    "sample_id": row.get("sample_id"),
                    "stage": "dig_selector",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            counts["errors"] += 1

    manifest_sha = sha256_file(selector_path) if selector_path.exists() else None
    if manifest_sha:
        (cfg.selector_dir / "dig32_selector_manifest.sha256").write_text(manifest_sha + "  " + selector_path.name + "\n", encoding="utf-8")

    records = read_jsonl(selector_path)
    summary = build_selector_summary(rows, records, counts, args, dig_commit_value, selector_path, manifest_sha, duplicate_examples, oob_examples)
    write_json(cfg.selector_dir / "selector_summary.json", summary)
    (cfg.selector_dir / "selector_summary.md").write_text(render_selector_summary_md(summary), encoding="utf-8")
    return summary


def build_selector_summary(
    manifest_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    counts: Counter,
    args: argparse.Namespace,
    dig_commit_value: str,
    selector_path: Path,
    manifest_sha: str | None,
    duplicate_examples: list[dict[str, Any]],
    oob_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    frame_counts = [int(row["n_medvidu_frames"]) for row in manifest_rows]
    qa_counts = Counter(str(row.get("qa_type", "")) for row in manifest_rows)
    query_counts = Counter(str(row.get("query_type", "")) for row in records if row.get("query_type"))
    return {
        "status": "OK" if counts.get("errors", 0) == 0 else "ERRORS",
        "upstream_dig": {
            "repo_path": str(args.dig_repo_dir),
            "git_commit": dig_commit_value,
            "upstream_modified": bool(subprocess.run(["git", "-C", str(args.dig_repo_dir), "status", "--porcelain"], text=True, stdout=subprocess.PIPE).stdout.strip()),
            "patch": None,
        },
        "fixed_selector": {
            "selector_version": args.selector_version,
            "k": args.k,
            "wlen": args.wlen,
            "query_identifier": args.query_identifier_model,
            "cafs_model": args.cafs_model,
            "reward_lmm": args.reward_lmm,
            "gt_used": False,
            "selector_deterministic": True,
        },
        "medvidu": {
            "total_samples": len(manifest_rows),
            "samples_by_qa_type": dict(sorted(qa_counts.items())),
            "frame_count_min": min(frame_counts) if frame_counts else None,
            "frame_count_median": median(frame_counts) if frame_counts else None,
            "frame_count_max": max(frame_counts) if frame_counts else None,
            "n_samples_lt_32": sum(n < 32 for n in frame_counts),
            "n_samples_eq_32": sum(n == 32 for n in frame_counts),
            "n_samples_gt_32": sum(n > 32 for n in frame_counts),
        },
        "rc_compatibility": "RC_SELECTOR_ADAPTATION_REQUIRED",
        "query_identification": {
            "global": query_counts.get("global", 0),
            "localized": query_counts.get("local", 0),
            "errors": counts.get("errors", 0),
        },
        "cafs": {"completed": counts.get("local", 0), "errors": counts.get("errors", 0)},
        "reward": {"completed": counts.get("local", 0), "errors": counts.get("errors", 0)},
        "refinement": {
            "completed": counts.get("local", 0),
            "errors": counts.get("errors", 0),
            "duplicate_selected_index_cases": len(duplicate_examples),
            "duplicate_examples": duplicate_examples[:5],
            "out_of_bound_cases": len(oob_examples),
            "out_of_bound_examples": oob_examples[:5],
        },
        "frozen_manifest": {"path": str(selector_path), "sha256": manifest_sha},
        "counts": dict(counts),
        "full_run_executed": False,
    }


def render_selector_summary_md(summary: dict[str, Any]) -> str:
    lines = ["# Fixed DIG-32 Selector Summary", ""]
    lines.append(f"- status: {summary['status']}")
    lines.append(f"- DIG commit: {summary['upstream_dig']['git_commit']}")
    lines.append(f"- selector version: {summary['fixed_selector']['selector_version']}")
    lines.append(f"- K: {summary['fixed_selector']['k']}")
    lines.append(f"- wlen: {summary['fixed_selector']['wlen']}")
    lines.append(f"- GT used: {summary['fixed_selector']['gt_used']}")
    lines.append(f"- RC compatibility: {summary['rc_compatibility']}")
    lines.append(f"- frozen manifest: {summary['frozen_manifest']['path']}")
    lines.append(f"- SHA256: {summary['frozen_manifest']['sha256']}")
    lines.append(f"- full run executed: {summary['full_run_executed']}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    parser = argparse.ArgumentParser(description="Run and freeze the official DIG-32 selector on a GT-free MedVidU manifest.")
    parser.add_argument("--manifest", type=Path, default=cfg.selector_dir / "gt_free_manifest.smoke.jsonl")
    parser.add_argument("--data-path", type=Path, default=cfg.data_path)
    parser.add_argument("--output-root", type=Path, default=cfg.output_root)
    parser.add_argument("--dig-repo-dir", type=Path, default=cfg.dig_repo_dir)
    parser.add_argument("--old-data-root", default=cfg.old_data_root)
    parser.add_argument("--new-data-root", default=cfg.new_data_root)
    parser.add_argument("--selector-version", default=cfg.selector.selector_version)
    parser.add_argument("--k", type=int, default=cfg.selector.requested_k)
    parser.add_argument("--wlen", type=int, default=cfg.selector.video_refinement_wlen)
    parser.add_argument("--query-identifier-model", default=cfg.selector.query_identifier_model)
    parser.add_argument("--query-base-url", default=cfg.query_base_url)
    parser.add_argument("--query-api-key", default=cfg.query_api_key)
    parser.add_argument("--query-concurrency", type=int, default=20)
    parser.add_argument("--reward-model-path", default=cfg.selector.reward_lmm_path)
    parser.add_argument("--reward-lmm", default=cfg.selector.reward_lmm)
    parser.add_argument("--reward-base-url", default=cfg.reward_base_url)
    parser.add_argument("--reward-api-key", default=cfg.reward_api_key)
    parser.add_argument("--reward-concurrency", type=int, default=64)
    parser.add_argument("--cafs-model", default=cfg.selector.cafs_model)
    parser.add_argument("--cafs-device", default="cuda")
    parser.add_argument("--allow-short-videos", action="store_true")
    parser.add_argument("--allow-duplicate-selected-indices", action="store_true")
    parser.add_argument("--output-name", default="dig32_selector_manifest.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = asyncio.run(run_selector_async(args))
    except QueryIdentifierUnavailable as exc:
        print(str(exc))
        return 2
    print(summary)
    return 0 if summary.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
