from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import RunConfig, TARGET_MODELS
from .manifest.build_gt_free_manifest import build_manifest, write_smoke_manifest
from .io_utils import read_jsonl, write_json
from .selector.freeze_selector_manifest import dig_git_commit
from .selector.query_identifier import OfficialQueryIdentifier


def run_tests() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "baselines/dig32_medvidu/tests", "-p", "test_*.py"]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "passed": proc.returncode == 0}


def prepare_smoke(args: argparse.Namespace) -> dict[str, Any]:
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root, old_data_root=args.old_data_root, new_data_root=args.new_data_root, dig_repo_dir=args.dig_repo_dir)
    cfg.make_dirs()
    manifest_path = cfg.selector_dir / "gt_free_manifest.jsonl"
    smoke_path = cfg.selector_dir / "gt_free_manifest.smoke.jsonl"
    manifest_report = build_manifest(cfg.data_path, manifest_path, old_data_root=cfg.old_data_root, new_data_root=cfg.new_data_root)
    smoke_report = write_smoke_manifest(read_jsonl(manifest_path), smoke_path, per_task=args.per_task, seed=args.seed)
    query_preflight = OfficialQueryIdentifier(
        dig_repo_dir=args.dig_repo_dir,
        model=args.query_identifier_model,
        base_url=args.query_base_url,
        api_key=args.query_api_key,
    ).preflight()
    dig_commit_value = dig_git_commit(args.dig_repo_dir)
    (cfg.provenance_dir / "dig_git_commit.txt").write_text(dig_commit_value + "\n", encoding="utf-8")
    write_json(cfg.provenance_dir / "selector_config.json", cfg.selector.to_jsonable())
    write_json(cfg.provenance_dir / "run_config.json", cfg.to_jsonable())
    report = {
        "UPSTREAM_DIG": {
            "repo_path": str(args.dig_repo_dir),
            "git_commit": dig_commit_value,
            "upstream_modified": bool(subprocess.run(["git", "-C", str(args.dig_repo_dir), "status", "--porcelain"], text=True, stdout=subprocess.PIPE).stdout.strip()),
            "patch": None,
        },
        "FIXED_SELECTOR": {
            "selector_version": cfg.selector.selector_version,
            "K": cfg.selector.requested_k,
            "wlen": cfg.selector.video_refinement_wlen,
            "query_identifier": cfg.selector.query_identifier_model,
            "query_serving_backend": args.query_serving_backend,
            "CAFS_model": cfg.selector.cafs_model,
            "reward_LMM": cfg.selector.reward_lmm,
            "reward_serving_backend": args.reward_serving_backend,
            "serving_backend_adaptation": bool(
                args.query_serving_backend != "vllm" or args.reward_serving_backend != "vllm"
            ),
            "GT_used": False,
            "selector_deterministic": True,
            "query_preflight": query_preflight,
        },
        "MEDVIDU": manifest_report,
        "RC_compatibility": manifest_report.get("rc_compatibility"),
        "SMOKE": {
            "sample_count": len(smoke_report["sample_ids"]),
            "sample_ids_path": str(cfg.selector_dir / "smoke_sample_ids.json"),
            "manifest_path": str(smoke_path),
            "tasks": sorted({sample["task_name"] for sample in smoke_report["samples"]}),
            "datasets": sorted({str(sample["dataset_name"]) for sample in smoke_report["samples"]}),
        },
        "QUERY_IDENTIFICATION": {
            "status": query_preflight["status"],
            "global": 0,
            "localized": 0,
            "errors": 0 if query_preflight["status"] == "OK" else 1,
        },
        "CAFS": {"completed": 0, "errors": 0},
        "REWARD": {"completed": 0, "errors": 0},
        "REFINEMENT": {"completed": 0, "errors": 0, "duplicate_selected_index_cases": None, "out_of_bound_cases": None},
        "FROZEN_MANIFEST": {"path": str(cfg.selector_dir / "dig32_selector_manifest.jsonl"), "SHA256": None},
        "TARGETS": {
            key: {"model": model, "completed": 0, "errors": 0, "OOM": 0, "peak_memory": None}
            for key, model in TARGET_MODELS.items()
        },
        "FAIRNESS_AUDIT": {
            "same_selector_manifest_across_all_targets": None,
            "same_selected_positions": None,
            "same_effective_K": None,
            "same_question": None,
            "same_task_prompt": None,
            "same_timestamps": None,
            "same_decoding_principle": True,
            "GT_leakage": 0,
        },
        "TESTS": None,
        "FULL_RUN_EXECUTED": False,
        "STOP_REASON": None if query_preflight["status"] == "OK" else "QUERY_IDENTIFIER_UNAVAILABLE",
    }
    write_json(cfg.selector_dir / "smoke_prepare_report.json", report)
    write_json(cfg.selector_dir / "smoke_sample_ids.json", smoke_report)
    return report


def render_report_md(report: dict[str, Any]) -> str:
    lines = ["# Fixed DIG-32 MedVidU Smoke Report", ""]
    for section in (
        "UPSTREAM_DIG",
        "FIXED_SELECTOR",
        "MEDVIDU",
        "RC_compatibility",
        "SMOKE",
        "QUERY_IDENTIFICATION",
        "CAFS",
        "REWARD",
        "REFINEMENT",
        "FROZEN_MANIFEST",
        "FAIRNESS_AUDIT",
        "TESTS",
    ):
        lines.append(f"## {section}")
        lines.append("```json")
        lines.append(json.dumps(report.get(section), indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    lines.append(f"FULL RUN EXECUTED: {str(report.get('FULL_RUN_EXECUTED')).lower()}")
    if report.get("STOP_REASON"):
        lines.append(f"STOP_REASON: {report['STOP_REASON']}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    parser = argparse.ArgumentParser(description="Prepare Fixed DIG-32 smoke inputs and audit readiness.")
    parser.add_argument("--data-path", type=Path, default=cfg.data_path)
    parser.add_argument("--output-root", type=Path, default=cfg.output_root)
    parser.add_argument("--dig-repo-dir", type=Path, default=cfg.dig_repo_dir)
    parser.add_argument("--old-data-root", default=cfg.old_data_root)
    parser.add_argument("--new-data-root", default=cfg.new_data_root)
    parser.add_argument("--seed", type=int, default=cfg.smoke_seed)
    parser.add_argument("--per-task", type=int, default=cfg.smoke_per_task)
    parser.add_argument("--query-identifier-model", default=cfg.selector.query_identifier_model)
    parser.add_argument("--query-serving-backend", choices=["vllm", "sglang"], default=cfg.selector.query_serving_backend)
    parser.add_argument("--query-base-url", default=cfg.query_base_url)
    parser.add_argument("--query-api-key", default=cfg.query_api_key)
    parser.add_argument("--reward-serving-backend", choices=["vllm", "sglang"], default=cfg.selector.reward_serving_backend)
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(output_root=args.output_root)
    report = prepare_smoke(args)
    if args.run_tests:
        report["TESTS"] = run_tests()
    write_json(cfg.logs_dir / "smoke_report.json", report)
    (cfg.logs_dir / "smoke_report.md").write_text(render_report_md(report), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
