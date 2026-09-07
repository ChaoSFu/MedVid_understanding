#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.spatial import (  # noqa: E402
    H4_PROTOCOL_VERSION,
    audit_stg_schema,
    build_spatial_pointer_prompt,
    prompt_sha256,
    write_spatial_threshold_review_md,
    write_stg_schema_md,
    write_temporal_protocol_review_md,
)
from evidence_stability.utils import read_json, write_json  # noqa: E402


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc!r}"


def environment_snapshot() -> str:
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
    ]
    for module_name in ["torch", "transformers", "PIL", "numpy"]:
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}: {getattr(module, '__version__', 'UNKNOWN')}")
        except Exception as exc:
            lines.append(f"{module_name}: UNAVAILABLE ({exc!r})")
    return "\n".join(lines) + "\n"


def write_run_status_md(path: Path, status: dict) -> None:
    lines = [
        "# H4 Spatial v1 Protocol Audit",
        "",
        "## Status",
        f"- protocol version: {status['protocol_version']}",
        f"- preflight executed: {status['preflight_executed']}",
        f"- discovery executed: {status['discovery_executed']}",
        f"- stop reason: {status['stop_reason']}",
        "",
        "## Repo Inspection",
        f"- git commit: {status['repo']['git_commit']}",
        f"- working tree: {status['repo']['git_status_short']!r}",
        "",
        "## STG Schema",
        f"- STG samples: {status['stg_schema']['n_stg']}",
        f"- spatial GT type: {status['stg_schema']['spatial_gt']['type']}",
        f"- temporal GT fields: {status['stg_schema']['temporal_gt']['fields']}",
        f"- official spatial metric: {status['stg_schema']['official_stg_metric']['record_score']}",
        f"- official reported thresholds: {status['stg_schema']['official_stg_metric']['reported_metrics']}",
        f"- canonical positive threshold: {status['stg_schema']['official_stg_metric']['single_canonical_positive_threshold']}",
        "",
        "H4-v1 is stopped before model preflight/discovery because post-hoc TRUE_SPATIAL_SUPPORT labels require a frozen canonical positive spatial threshold.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MedVidU STG schema and H4 spatial-label freeze points.")
    parser.add_argument("--data_json", default="data_json/medvidu_filtered/trainval/stg.json")
    parser.add_argument("--output_dir", default="outputs/stg_pilot/phase_g/h4_spatial_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["audit", "summary", "provenance", "manifest", "predictions", "joined", "candidate", "qa", "visualizations/preflight", "visualizations/discovery"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    samples = read_json(args.data_json)
    if not isinstance(samples, list):
        raise TypeError("STG data_json must contain a list of samples")
    audit = audit_stg_schema(samples)
    exemplar_question = audit["examples"][0]["question"] if audit["examples"] else ""
    pointer_prompt = build_spatial_pointer_prompt(exemplar_question)
    prompt_fingerprint = {
        "spatial_pointer_prompt_version": "spatial_pointer_v1",
        "spatial_pointer_prompt_sha256_for_first_example": prompt_sha256(pointer_prompt),
        "structured_output": False,
        "parser": "strict_json_object_with_bbox_only",
    }
    repo = {
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
    }
    status = {
        "protocol_version": H4_PROTOCOL_VERSION,
        "data_json": args.data_json,
        "output_dir": str(out_dir),
        "stg_schema": audit,
        "prompt_fingerprint": prompt_fingerprint,
        "repo": repo,
        "preflight_executed": False,
        "discovery_executed": False,
        "formal_statistics_run": False,
        "independent_confirmation_run": False,
        "stop_reason": "STOP_NEEDS_SPATIAL_LABEL_THRESHOLD_FREEZE",
        "required_human_decisions": [
            "Freeze H4 temporal eligibility rule: H2 frame-overlap rule vs another explicit STG-compatible rule.",
            "Freeze H4 spatial positive criterion for TRUE_SPATIAL_SUPPORT, because official STG evaluator reports mIoU and iou@0.3/0.5/0.7 without naming one H4 label threshold.",
        ],
    }

    write_json(out_dir / "audit" / "medvidu_stg_schema.json", audit)
    write_stg_schema_md(str(out_dir / "audit" / "medvidu_stg_schema.md"), audit)
    write_stg_schema_md(str(out_dir / "provenance" / "medvidu_stg_schema.md"), audit)
    write_temporal_protocol_review_md(str(out_dir / "audit" / "temporal_eligibility_protocol_review.md"))
    write_spatial_threshold_review_md(str(out_dir / "audit" / "spatial_label_threshold_review.md"), audit)
    write_json(out_dir / "summary" / "h4_protocol_audit_status.json", status)
    write_run_status_md(out_dir / "summary" / "h4_protocol_audit_status.md", status)
    write_json(out_dir / "provenance" / "prompt_fingerprint.json", prompt_fingerprint)
    (out_dir / "provenance" / "environment_snapshot.txt").write_text(environment_snapshot(), encoding="utf-8")
    (out_dir / "provenance" / "git_status.txt").write_text(repo["git_status_short"] + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
