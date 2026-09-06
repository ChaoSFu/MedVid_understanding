from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import traceback
from typing import Any

from .cache import build_cache_key
from .config import (
    BASELINE_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    FRAME_LOADER_VERSION,
    OFFICIAL_VTUNE_GROUNDING_PROMPT,
    PROMPT_VERSION,
    QUERY_ADAPTER_VERSION,
    SCIENTIFIC_NAME,
    TIMESTAMP_ADAPTER_VERSION,
    GenerationConfig,
    RunConfig,
)
from .io_utils import append_jsonl, assert_no_gt_leak, completed_cache, read_jsonl, sha256_text, write_json
from .manifest import build_manifest, write_smoke_manifest
from .model_adapter import MedVidUTimeChatVTune, static_model_fingerprint
from .provenance import write_environment, write_official_behavior, write_run_config, write_upstream_git_provenance


def prediction_path_for(output_root: Path, stage: str) -> Path:
    pred_dir = output_root / "predictions"
    if stage == "one":
        return pred_dir / "preflight_prediction.jsonl"
    if stage in {"smoke", "cache-rerun"}:
        return pred_dir / "smoke_predictions.jsonl"
    if stage == "full":
        return pred_dir / "full_predictions.jsonl"
    raise ValueError(f"unknown prediction stage: {stage}")


def ensure_manifest(cfg: RunConfig, smoke: bool) -> Path:
    manifest_path = cfg.manifest_dir / "medvidu_tal_timechat_gt_free.jsonl"
    if not manifest_path.exists():
        build_manifest(cfg.data_path, cfg.output_root, old_data_root=cfg.old_data_root, new_data_root=cfg.new_data_root)
    if smoke:
        write_smoke_manifest(cfg.output_root, size=cfg.smoke_size, seed=cfg.smoke_seed)
        return cfg.manifest_dir / "medvidu_tal_timechat_gt_free.smoke.jsonl"
    return manifest_path


def choose_one_preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no TAL rows available")
    return sorted(rows, key=lambda row: sha256_text(row["sample_id"]))[0]


def row_cache_key(row: dict[str, Any], checkpoint_sha256: str, generation: GenerationConfig, model_fp: dict[str, Any]) -> str:
    return build_cache_key(
        row,
        checkpoint_sha256=checkpoint_sha256,
        prompt_sha256=sha256_text(OFFICIAL_VTUNE_GROUNDING_PROMPT),
        generation_config={
            "num_beams": generation.num_beams,
            "do_sample": generation.do_sample,
            "temperature": generation.temperature,
            "max_new_tokens": generation.max_new_tokens,
            "max_length": generation.max_length,
        },
        model_fingerprint=model_fp,
    )


def write_model_load_artifacts(load_result: Any, cfg: RunConfig) -> None:
    write_json(cfg.provenance_dir / "checkpoint_inspection.json", load_result.checkpoint_inspection)
    write_json(cfg.audit_dir / "checkpoint_load_audit.json", load_result.checkpoint_load_audit)
    write_json(cfg.provenance_dir / "model_fingerprint.json", load_result.model_fingerprint)


def run_rows(
    rows: list[dict[str, Any]],
    stage: str,
    cfg: RunConfig,
    timechat_repo: Path,
    vtune_ckpt: Path,
    vit_model: Path,
    q_former_model: Path,
    llama_model: Path,
    gpu_id: int,
) -> dict[str, Any]:
    generation = GenerationConfig()
    model_fp = static_model_fingerprint(vtune_ckpt, vit_model, q_former_model, llama_model)
    checkpoint_sha256 = model_fp["vtune_sha256"]
    pred_path = prediction_path_for(cfg.output_root, stage)
    error_path = cfg.prediction_dir / "errors.jsonl"
    cached = completed_cache(pred_path)
    completed_at_start = len(cached)
    keys = [row_cache_key(row, checkpoint_sha256, generation, model_fp) for row in rows]
    all_cached = all(key in cached for key in keys)
    summary = {
        "stage": stage,
        "n_requested": len(rows),
        "completed_at_start": completed_at_start,
        "new_inference": 0,
        "skipped_cache": 0,
        "errors": 0,
        "prediction_path": str(pred_path),
        "error_path": str(error_path),
    }
    processor_audit_rows: list[dict[str, Any]] = []
    if all_cached:
        summary["skipped_cache"] = len(rows)
        write_json(cfg.audit_dir / ("cache_restart_audit.json" if stage == "cache-rerun" else f"{stage}_cache_audit.json"), summary)
        return summary

    load_result = MedVidUTimeChatVTune.load(timechat_repo, vtune_ckpt, vit_model, q_former_model, llama_model, gpu_id=gpu_id)
    write_model_load_artifacts(load_result, cfg)
    runner = load_result.runner
    model_fp = load_result.model_fingerprint
    checkpoint_sha256 = model_fp["vtune_sha256"]
    cached = completed_cache(pred_path)

    for row in rows:
        prompt = OFFICIAL_VTUNE_GROUNDING_PROMPT.format(event=row["human_question"])
        cache_key = row_cache_key(row, checkpoint_sha256, generation, model_fp)
        if cache_key in cached:
            summary["skipped_cache"] += 1
            continue
        try:
            out = runner.run_one(row, generation)
            result = {
                "sample_id": row["sample_id"],
                "qa_id": row.get("qa_id"),
                "dataset_name": row.get("dataset_name"),
                "data_source": row.get("data_source"),
                "question": row["human_question"],
                "n_input_frames": row["n_input_frames"],
                "n_selected_frames": out["n_selected_frames"],
                "selected_logical_indices": out["selected_logical_indices"],
                "selected_local_timestamps": out["selected_local_timestamps"],
                "selected_rounded_timestamps": out["selected_rounded_timestamps"],
                "model": BASELINE_NAME,
                "checkpoint_path": str(vtune_ckpt),
                "checkpoint_sha256": checkpoint_sha256,
                "raw_answer": out["raw_answer"],
                "official_parsed_span": out["official_parsed_span"],
                "parse_valid": out["parse_valid"],
                "prompt_version": PROMPT_VERSION,
                "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
                "query_adapter_version": QUERY_ADAPTER_VERSION,
                "frame_loader_version": FRAME_LOADER_VERSION,
                "timechat_msg": out["timechat_msg"],
                "timestamp_texts": out["timestamp_texts"],
                "question_sent_to_model": prompt,
                "cache_key": cache_key,
                "inference_error": None,
            }
            processor_audit_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset_name": row.get("dataset_name"),
                    "original_logical_frame_count": row["n_input_frames"],
                    "selected_count": out["n_selected_frames"],
                    "selected_logical_indices": out["selected_logical_indices"],
                    "selected_file_names": [Path(row["ordered_frame_paths"][i]).name for i in out["selected_logical_indices"]],
                    "selected_timestamps": out["selected_local_timestamps"],
                    "input_tensor_shape": out.get("video_tensor_shape"),
                    "video_embedding_shape": out.get("video_embedding_shape"),
                    "timestamp_msg_consistent": True,
                    "duplicates_preserved": len(set(row["ordered_frame_paths"])) < len(row["ordered_frame_paths"]),
                }
            )
            assert_no_gt_leak(result)
            append_jsonl(pred_path, result)
            cached[cache_key] = result
            summary["new_inference"] += 1
        except RuntimeError as exc:
            summary["errors"] += 1
            error_type = "OOM" if "out of memory" in str(exc).lower() else "RuntimeError"
            append_jsonl(error_path, {"sample_id": row["sample_id"], "cache_key": cache_key, "stage": stage, "error_type": error_type, "error": repr(exc), "traceback": traceback.format_exc()})
            if error_type == "OOM":
                raise
        except Exception as exc:
            summary["errors"] += 1
            append_jsonl(error_path, {"sample_id": row["sample_id"], "cache_key": cache_key, "stage": stage, "error_type": type(exc).__name__, "error": repr(exc), "traceback": traceback.format_exc()})
    if processor_audit_rows:
        write_json(cfg.audit_dir / "processor_frame_audit.json", processor_audit_rows)
    write_json(cfg.audit_dir / f"{stage}_run_summary.json", summary)
    return summary


def run_model_load(args: argparse.Namespace, cfg: RunConfig) -> dict[str, Any]:
    load_result = MedVidUTimeChatVTune.load(
        args.timechat_repo,
        args.vtune_ckpt,
        args.vit_model,
        args.q_former_model,
        args.llama_model,
        gpu_id=args.gpu_id,
    )
    write_model_load_artifacts(load_result, cfg)
    gpu_memory: dict[str, Any]
    try:
        import torch

        gpu_memory = {
            "cuda_available": torch.cuda.is_available(),
            "allocated_bytes": torch.cuda.memory_allocated(args.gpu_id) if torch.cuda.is_available() else None,
            "reserved_bytes": torch.cuda.memory_reserved(args.gpu_id) if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        gpu_memory = {"error": repr(exc)}
    summary = {"model_loaded": True, "checkpoint_load_audit": load_result.checkpoint_load_audit, "gpu_memory_after_load": gpu_memory}
    write_json(cfg.audit_dir / "model_load_preflight.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {SCIENTIFIC_NAME} with strict GT-free MedVidU TAL inference.")
    parser.add_argument("--stage", choices=("model-load", "one", "smoke", "cache-rerun", "full"), required=True)
    parser.add_argument("--timechat-repo", type=Path, required=True)
    parser.add_argument("--vtune-ckpt", type=Path, required=True)
    parser.add_argument("--vit-model", type=Path, required=True)
    parser.add_argument("--q-former-model", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--old-data-root", default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new-data-root", default=DEFAULT_NEW_DATA_ROOT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root, old_data_root=args.old_data_root, new_data_root=args.new_data_root)
    cfg.make_dirs()
    git_info = write_upstream_git_provenance(args.timechat_repo, cfg.provenance_dir, suffix="before")
    write_official_behavior(args.timechat_repo, cfg.provenance_dir, git_info)
    write_run_config(cfg.output_root, args.vtune_ckpt)
    write_environment(cfg.output_root)
    if args.stage == "model-load":
        summary = run_model_load(args, cfg)
    else:
        manifest_path = ensure_manifest(cfg, smoke=args.stage in {"smoke", "cache-rerun"})
        rows = read_jsonl(manifest_path)
        if args.stage == "one":
            rows = [choose_one_preflight(rows)]
        if args.limit is not None:
            rows = rows[: args.limit]
        summary = run_rows(rows, args.stage, cfg, args.timechat_repo, args.vtune_ckpt, args.vit_model, args.q_former_model, args.llama_model, args.gpu_id)
        if args.stage == "smoke":
            preds = read_jsonl(prediction_path_for(cfg.output_root, "smoke"))
            summary["parse_valid"] = sum(1 for pred in preds if pred.get("parse_valid"))
            summary["parse_invalid"] = sum(1 for pred in preds if not pred.get("parse_valid"))
            summary["by_dataset"] = dict(sorted(Counter(pred.get("dataset_name") for pred in preds).items()))
            write_json(cfg.audit_dir / "smoke_audit.json", summary)
    write_upstream_git_provenance(args.timechat_repo, cfg.provenance_dir, suffix="after")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
