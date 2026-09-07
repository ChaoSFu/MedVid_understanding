from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .audit.gt_leakage import audit_manifest_gt_leakage
from .config import DEFAULT_DATA_PATH, DEFAULT_NEW_DATA_ROOT, DEFAULT_OLD_DATA_ROOT, DEFAULT_OUTPUT_ROOT, EVQA_ROOT, RunConfig
from .io_utils import write_json
from .manifest import build_manifests, write_smoke_manifests
from .provenance import ensure_evqa_present, model_fingerprint, record_upstream_provenance, write_environment_snapshot


def build_report(cfg: RunConfig, model_path: Path | None, upstream: dict[str, Any], inventory: dict[str, Any], fingerprint: dict[str, Any]) -> dict[str, Any]:
    stop_reasons = []
    official_success = False
    if model_path is None:
        stop_reasons.append("MODEL_PATH_NOT_PROVIDED")
    elif not model_path.exists():
        stop_reasons.append("MODEL_PATH_NOT_FOUND")
    elif fingerprint.get("config_read_error"):
        stop_reasons.append("MODEL_CONFIG_READ_FAILED")
    if not stop_reasons:
        next_recommendation = "Run run_official_reproduction.py on 2-5 official ST-Evidence samples before any MedVidU smoke run."
    else:
        next_recommendation = "Resolve STOP_REASONS, rerun run_preflight.py, then run official reproduction before MedVidU smoke."
    report = {
        "title": "E-VQA to MedVidU Preflight Report",
        "UPSTREAM": {
            "repo": str(cfg.evqa_root),
            "commit": upstream.get("commit"),
            "initial_status": upstream.get("status", ""),
            "upstream_modified": bool(upstream.get("diff_after")),
        },
        "MODEL": {
            "model_path": str(model_path) if model_path else None,
            "model_class": fingerprint.get("model_config_class"),
            "base_VLM": fingerprint.get("base_model_path") or "Qwen2.5-VL family per upstream PixelQwen2_5_VL code",
            "segmentation_backend": fingerprint.get("sam2_config") or "SAM2 per upstream PixelQwen2_5_VL code",
            "dtype": "bfloat16 default",
            "GPU": "see provenance/environment_snapshot.txt",
            "fingerprint": fingerprint,
        },
        "OFFICIAL_REPRODUCTION": {
            "samples": 0,
            "success": official_success,
            "temporal_output": None,
            "mask_output": None,
            "errors": stop_reasons if stop_reasons else ["NOT_RUN_BY_PREFLIGHT_ENTRYPOINT"],
        },
        "MEDVIDU_INVENTORY": {
            "STG": inventory.get("manifest_rows", {}).get("stg"),
            "RC": inventory.get("manifest_rows", {}).get("rc"),
            "CVS": inventory.get("manifest_rows", {}).get("cvs"),
            "datasets": inventory.get("datasets"),
        },
        "VISUAL_BOUNDARY": {"benchmark_frames_only": True, "source_raw_video_accessed": False, "fake_MP4_used": False},
        "SAMPLING": {
            "official_policy": "fps=1.0, max_frames=128, sample_frames=int(duration*fps), uniform",
            "MedVidU_adaptation": "nearest logical MedVidU benchmark frame to official target clip-local times",
            "additional_sampling": False,
            "timestamp_mapper": "baselines.timelens_medvidu.temporal_mapper.TemporalMapper",
        },
        "STG": {"smoke": cfg.smoke_size, "completed": False, "temporal_parse_valid": None, "mask_valid": None, "bbox_conversion_valid": None, "schema_mismatch": None, "OOM_or_errors": stop_reasons},
        "RC": {
            "native_region_conditioned_input_supported": True,
            "MedVidU_region_schema": "RC_info.start_frame + RC_info.start_frame_bbox -> provided_region frame_path_plus_xyxy_bbox",
            "adapter_status": "implemented, not smoke-run in preflight without model",
            "smoke_run": False,
            "reason_if_STOPPED": stop_reasons,
        },
        "CVS": {"smoke": cfg.smoke_size, "completed": False, "answer_parse_valid": None, "parse_invalid": None, "OOM_or_errors": stop_reasons},
        "GT_ISOLATION": {"STG_GT_visible": False, "CVS_GT_visible": False, "RC_provided_region_visible": True, "RC_target_caption_visible": False},
        "CACHE": {"STG_rerun_new_inference": None, "RC_rerun_new_inference": None, "CVS_rerun_new_inference": None},
        "FULL_RUN_EXECUTED": False,
        "FILES_CREATED": [
            str(cfg.provenance_dir),
            str(cfg.manifest_dir),
            str(cfg.audit_dir),
            str(cfg.prediction_dir),
            str(cfg.evaluation_dir),
        ],
        "NEXT_RECOMMENDATION": next_recommendation,
        "STOP_REASONS": stop_reasons,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E-VQA MedVidU preflight: provenance, inventory, manifests, audits, and STOP report.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evqa-root", type=Path, default=EVQA_ROOT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--old-data-root", default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new-data-root", default=DEFAULT_NEW_DATA_ROOT)
    parser.add_argument("--smoke-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root, evqa_root=args.evqa_root, old_data_root=args.old_data_root, new_data_root=args.new_data_root, smoke_size=args.smoke_size, smoke_seed=args.seed)
    cfg.make_dirs()
    ensure_evqa_present(cfg.evqa_root)
    upstream = record_upstream_provenance(cfg.evqa_root, cfg.provenance_dir)
    write_environment_snapshot(cfg.provenance_dir / "environment_snapshot.txt")
    inventory = build_manifests(cfg)
    inventory["smoke"] = write_smoke_manifests(cfg)
    write_json(cfg.manifest_dir / "medvidu_task_inventory.json", inventory)
    audit_manifest_gt_leakage(cfg.manifest_dir, cfg.audit_dir / "gt_leakage.json")
    fingerprint = model_fingerprint(args.model_path) if args.model_path else {"exists": False, "error": "MODEL_PATH_NOT_PROVIDED", "auto_download_attempted": False}
    write_json(cfg.provenance_dir / "model_fingerprint.json", fingerprint)
    write_json(cfg.provenance_dir / "run_config.json", cfg.to_jsonable())
    report = build_report(cfg, args.model_path, upstream, inventory, fingerprint)
    write_json(cfg.official_reproduction_dir / "predictions.json", {})
    write_json(
        cfg.official_reproduction_dir / "report.json",
        {
            "samples": 0,
            "success": False,
            "temporal_output": None,
            "mask_output": None,
            "errors": report["STOP_REASONS"],
            "stop_reason": "Official reproduction requires a local --model-path and official/sample media; no model download was attempted.",
        },
    )
    write_json(cfg.output_root / "preflight_report.json", report)
    print(report)
    return 2 if report["STOP_REASONS"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
