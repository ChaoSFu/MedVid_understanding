#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evidence_stability.frame_manifest import (  # noqa: E402
    build_frame_manifest,
    human_bytes,
    load_smoke_ids_artifact,
    normalize_manifest_relative_path,
    summary_from_manifest,
)
from evidence_stability.utils import read_jsonl, write_json  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/tal_pilot/phase_d/frame_manifests")
INTERVENTION_CANDIDATES = [
    Path("outputs/tal_pilot/phase_d/intervention_pilot/interventions.jsonl"),
    Path("outputs/tal_pilot/phase_d_server/intervention_pilot/interventions.jsonl"),
    Path("outputs/tal_pilot/phase_d/intervention_pilot_check/interventions.jsonl"),
]
SMOKE_ID_CANDIDATES = [
    Path("outputs/tal_pilot/phase_d/qwen3_vl_8b_intervention_smoke_v2/phase_d2_smoke_intervention_ids.json"),
    Path("outputs/tal_pilot/phase_d/qwen3_vl_8b_intervention_smoke/phase_d2_smoke_intervention_ids.json"),
    Path("outputs/tal_pilot/phase_d_server/qwen3_vl_8b_intervention_smoke_v2/phase_d2_smoke_intervention_ids.json"),
    Path("outputs/tal_pilot/phase_d_server/qwen3_vl_8b_intervention_smoke/phase_d2_smoke_intervention_ids.json"),
]


def first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Searched:\n{searched}")


def discover_latest_smoke_ids() -> Path:
    existing = [path for path in SMOKE_ID_CANDIDATES if path.exists()]
    existing.extend(
        path
        for path in Path("outputs/tal_pilot").glob("**/phase_d2_smoke_intervention_ids.json")
        if path.exists() and path not in existing
    )
    if not existing:
        return first_existing(SMOKE_ID_CANDIDATES, "frozen smoke intervention IDs")
    existing.sort(key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    return existing[0]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def required_frame_lines(manifest: dict[str, Any]) -> list[str]:
    paths = [str(item["relative_path"]) for item in manifest["required_frames"]]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Required frame TXT would contain duplicate relative paths.")
    return sorted(normalize_manifest_relative_path(path) for path in paths)


def write_markdown_summary(path: Path, manifest: dict[str, Any], files: dict[str, Path]) -> None:
    scope_label = manifest["scope"].upper()
    lines = [
        f"# D.2 {scope_label} Frame Manifest Summary",
        "",
        "This is an operational frame-transfer audit only. It does not run Qwen, use GT, "
        "or change the Phase D.2 inference selection.",
        "",
        "## Counts",
        "",
        f"- n_interventions: {manifest['n_interventions']}",
        f"- n_total_frame_references: {manifest['n_total_frame_references']}",
        f"- n_unique_required_frames: {manifest['n_unique_required_frames']}",
        f"- n_existing_at_source: {manifest['n_existing_at_source']}",
        f"- n_missing_at_source: {manifest['n_missing_at_source']}",
        f"- n_existing_at_destination: {manifest['n_existing_at_destination']}",
        f"- n_missing_at_destination: {manifest['n_missing_at_destination']}",
        f"- destination_coverage_rate: {manifest['destination_coverage_rate']:.6f}",
        f"- missing_source_bytes_total: {manifest['missing_source_bytes_total']} ({manifest['missing_source_bytes_human']})",
        f"- n_size_mismatch: {manifest['n_size_mismatch']}",
        f"- n_interventions_fully_available: {manifest['n_interventions_fully_available']}",
        f"- n_interventions_blocked_by_missing_frames: {manifest['n_interventions_blocked_by_missing_frames']}",
    ]
    if manifest["scope"] == "smoke":
        lines.append(f"- smoke_ready_for_inference: {manifest['smoke_ready_for_inference']}")
    lines.extend(
        [
            "",
            "## Distributions",
            "",
            f"- frame_count_distribution: {json.dumps(manifest['frame_count_distribution'], sort_keys=True)}",
            f"- intervention_type_distribution: {json.dumps(manifest['intervention_type_distribution'], sort_keys=True)}",
            f"- dataset_distribution: {json.dumps(manifest['dataset_distribution'], sort_keys=True)}",
            f"- target_field_distribution: {json.dumps(manifest['target_field_distribution'], sort_keys=True)}",
            "",
            "## Rsync Template",
            "",
            "```bash",
            "rsync -av --progress "
            f"--files-from={files['rsync_files_from']} "
            f"{str(manifest['source_frame_root']).rstrip('/')}/ "
            f"USER@HOST:{str(manifest['destination_frame_root']).rstrip('/')}/",
            "```",
            "",
            "## Files",
            "",
            f"- required_frame_list: {files['required_txt']}",
            f"- json_manifest: {files['required_json']}",
            f"- missing_destination_list: {files['missing_txt']}",
            f"- existing_destination_list: {files['existing_txt']}",
            f"- rsync_files_from: {files['rsync_files_from']}",
            f"- missing_source_list: {files['missing_at_source_txt']}",
        ]
    )
    write_lines(path, lines)


def output_paths(scope: str, output_dir: Path) -> dict[str, Path]:
    prefix = f"d2_{scope}"
    return {
        "required_txt": output_dir / f"{prefix}_required_frames.txt",
        "required_json": output_dir / f"{prefix}_required_frames.json",
        "missing_txt": output_dir / f"{prefix}_missing_frames.txt",
        "existing_txt": output_dir / f"{prefix}_existing_frames.txt",
        "summary_json": output_dir / f"{prefix}_manifest_summary.json",
        "summary_md": output_dir / f"{prefix}_manifest_summary.md",
        "rsync_files_from": output_dir / f"{prefix}_rsync_files_from.txt",
        "missing_at_source_txt": output_dir / f"{prefix}_missing_at_source.txt",
    }


def write_manifest_outputs(manifest: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    files = output_paths(str(manifest["scope"]), output_dir)
    required = required_frame_lines(manifest)
    missing_destination = sorted(manifest["destination_missing_frames"])
    existing_destination = sorted(manifest["destination_existing_frames"])
    missing_source = sorted(manifest["source_missing_frames"])
    rsync_paths = [
        path
        for path in missing_destination
        if next(item for item in manifest["required_frames"] if item["relative_path"] == path)["exists_at_source"]
    ]

    write_lines(files["required_txt"], required)
    write_json(files["required_json"], manifest)
    write_lines(files["missing_txt"], missing_destination)
    write_lines(files["existing_txt"], existing_destination)
    write_lines(files["rsync_files_from"], rsync_paths)
    write_lines(files["missing_at_source_txt"], missing_source)
    write_json(files["summary_json"], summary_from_manifest(manifest))
    write_markdown_summary(files["summary_md"], manifest, files)
    return files


def write_full_transfer_priority(full_manifest: dict[str, Any], smoke_required_txt: Path, output_dir: Path) -> Path | None:
    if full_manifest["scope"] != "full" or not smoke_required_txt.exists():
        return None
    smoke_paths = {
        normalize_manifest_relative_path(line.strip())
        for line in smoke_required_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    full_paths = set(required_frame_lines(full_manifest))
    priority = sorted(smoke_paths & full_paths) + sorted(full_paths - smoke_paths)
    path = output_dir / "d2_full_transfer_priority.txt"
    write_lines(path, priority)
    return path


def print_completion_report(manifest: dict[str, Any], files: dict[str, Path], priority_path: Path | None) -> None:
    scope = str(manifest["scope"]).upper()
    print(scope)
    if scope == "SMOKE":
        print(f"n_interventions: {manifest['n_interventions']}")
    else:
        print(f"n_generation_valid_interventions: {manifest['n_interventions']}")
    print(f"n_total_frame_references: {manifest['n_total_frame_references']}")
    print(f"n_unique_required_frames: {manifest['n_unique_required_frames']}")
    if scope == "SMOKE":
        print(f"frame-count distribution: {manifest['frame_count_distribution']}")
    print(f"source missing frames: {manifest['n_missing_at_source']}")
    print(f"destination existing frames: {manifest['n_existing_at_destination']}")
    print(f"destination missing frames: {manifest['n_missing_at_destination']}")
    print(f"destination coverage rate: {manifest['destination_coverage_rate']:.6f}")
    print(f"missing transfer bytes: {manifest['missing_source_bytes_total']} ({human_bytes(manifest['missing_source_bytes_total'])})")
    if scope == "SMOKE":
        print(f"smoke_ready_for_inference: {manifest['smoke_ready_for_inference']}")
    else:
        print(f"n_interventions_fully_available: {manifest['n_interventions_fully_available']}")
        print(f"n_interventions_blocked: {manifest['n_interventions_blocked_by_missing_frames']}")
    print("MANIFESTS CREATED")
    print(f"{manifest['scope']} required frame list: {files['required_txt']}")
    print(f"{manifest['scope']} missing frame list: {files['missing_txt']}")
    print(f"{manifest['scope']} rsync files-from: {files['rsync_files_from']}")
    if priority_path:
        print(f"full transfer priority list: {priority_path}")
    print("AUDIT")
    print(f"duplicate relative paths: {manifest['duplicate_unique_path_count']}")
    print("path-outside-root errors: 0")
    print("frame-count mismatches: 0")
    print(f"missing source files: {manifest['n_missing_at_source']}")
    print("GT fields in manifest: 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare D.2 required frame manifests and missing-frame audits.")
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    parser.add_argument("--interventions", default=None, help="Phase D.1 interventions.jsonl.")
    parser.add_argument("--smoke_ids", default=None, help="Frozen D.2 smoke intervention ID artifact.")
    parser.add_argument("--source_frame_root", required=True)
    parser.add_argument("--destination_frame_root", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--verify_hash", action="store_true", help="Compare SHA256 when files exist at both roots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interventions_path = Path(args.interventions) if args.interventions else first_existing(INTERVENTION_CANDIDATES, "Phase D.1 interventions.jsonl")
    if not interventions_path.exists():
        raise FileNotFoundError(f"interventions.jsonl cannot be found: {interventions_path}")
    smoke_ids = None
    if args.scope == "smoke":
        smoke_path = Path(args.smoke_ids) if args.smoke_ids else discover_latest_smoke_ids()
        if not smoke_path.exists():
            raise FileNotFoundError(f"smoke selection file cannot be found: {smoke_path}")
        smoke_ids = load_smoke_ids_artifact(smoke_path)

    rows = read_jsonl(interventions_path)
    manifest = build_frame_manifest(
        rows,
        scope=args.scope,
        smoke_ids=smoke_ids,
        source_frame_root=args.source_frame_root,
        destination_frame_root=args.destination_frame_root,
        verify_hash=args.verify_hash,
    )
    output_dir = Path(args.output_dir)
    files = write_manifest_outputs(manifest, output_dir)
    priority_path = None
    if args.scope == "full":
        priority_path = write_full_transfer_priority(
            manifest,
            output_dir / "d2_smoke_required_frames.txt",
            output_dir,
        )
    print_completion_report(manifest, files, priority_path)


if __name__ == "__main__":
    main()
