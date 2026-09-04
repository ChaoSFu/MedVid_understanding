from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .cache import build_cache_key
from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GROUNDER_PROMPT,
    MIN_TOKENS,
    MODEL_NAME,
    PROMPT_VERSION,
    TIMESTAMP_ADAPTER_VERSION,
    TOTAL_TOKENS,
    DecodingConfig,
    PixelConfig,
    RunConfig,
)
from .dataset import MedVidUTALTimeLensDataset, build_official_prompt
from .io_utils import append_jsonl, assert_no_gt_leak, completed_cache, read_jsonl, write_json
from .manifest import build_manifest, write_smoke_manifest
from .parser import parse_timelens_answer
from .provenance import write_environment_and_model, write_official_behavior, write_timelens_git_provenance


def load_model_and_processor(model_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    return model, processor


def run_inference(
    rows: list[dict[str, Any]],
    model_path: Path,
    prediction_path: Path,
    error_path: Path,
    model_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    import torch

    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist; refusing to download checkpoint: {model_path}")

    pixel = PixelConfig()
    decoding = DecodingConfig()
    pixel_payload = pixel.to_jsonable()
    decoding_payload = {
        "do_sample": decoding.do_sample,
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "top_k": decoding.top_k,
        "max_new_tokens": decoding.max_new_tokens,
    }

    cached = completed_cache(prediction_path)
    completed_cache_at_start = len(cached)
    model, processor = load_model_and_processor(model_path)
    dataset = MedVidUTALTimeLensDataset(rows, processor, min_tokens=MIN_TOKENS, total_tokens=TOTAL_TOKENS)
    new_inference_count = 0
    skipped_cached_count = 0
    errors = 0

    for idx in range(len(dataset)):
        row = rows[idx]
        prompt = build_official_prompt(row["human_question"])
        cache_key = build_cache_key(row, model_fingerprint, prompt, pixel_payload, decoding_payload)
        if cache_key in cached:
            skipped_cached_count += 1
            continue
        try:
            data = dataset[idx]
            inputs = data["inputs"].to("cuda", non_blocking=True)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_new_tokens=512,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            raw_answer = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            parse_status, parsed_timestamps = parse_timelens_answer(raw_answer)
            result = {
                "sample_id": row["sample_id"],
                "qa_id": row.get("qa_id"),
                "dataset_name": row.get("dataset_name"),
                "data_source": row.get("data_source"),
                "question": row["human_question"],
                "n_medvidu_frames": row["n_medvidu_frames"],
                "first_local_time": row["first_local_time"],
                "last_local_time": row["last_local_time"],
                "effective_fps": row["effective_fps"],
                "model": MODEL_NAME,
                "model_path": str(model_path),
                "raw_answer": raw_answer,
                "parsed_timestamps": parsed_timestamps,
                "parse_valid": parse_status == "OK",
                "parse_status": parse_status,
                "prompt_version": PROMPT_VERSION,
                "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
                "cache_key": cache_key,
                "inference_error": None,
            }
            assert_no_gt_leak(result)
            append_jsonl(prediction_path, result)
            cached[cache_key] = result
            new_inference_count += 1
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                errors += 1
                append_jsonl(error_path, {"sample_id": row["sample_id"], "cache_key": cache_key, "error": repr(exc), "error_type": "OOM"})
                raise
            errors += 1
            append_jsonl(error_path, {"sample_id": row["sample_id"], "cache_key": cache_key, "error": repr(exc), "error_type": "RuntimeError"})
        except Exception as exc:
            errors += 1
            append_jsonl(error_path, {"sample_id": row["sample_id"], "cache_key": cache_key, "error": repr(exc), "error_type": type(exc).__name__})
    return {
        "completed_cache_at_start": completed_cache_at_start,
        "new_inference_count": new_inference_count,
        "skipped_cached_count": skipped_cached_count,
        "errors": errors,
        "prediction_path": str(prediction_path),
        "error_path": str(error_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen TimeLens-8B on MedVidU TAL frame-list samples.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--old-data-root", default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new-data-root", default=DEFAULT_NEW_DATA_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.exists():
        raise SystemExit(f"STOP: model path does not exist; refusing to download checkpoint: {args.model_path}")
    cfg = RunConfig(data_path=args.data_path, output_root=args.output_root, old_data_root=args.old_data_root, new_data_root=args.new_data_root)
    cfg.make_dirs()
    git_info = write_timelens_git_provenance(cfg.timelens_root, cfg.provenance_dir)
    write_official_behavior(cfg.timelens_root, cfg.provenance_dir, git_info)
    model_fp = write_environment_and_model(args.model_path, cfg.provenance_dir) or {"model_path": str(args.model_path)}
    write_json(cfg.provenance_dir / "run_config.json", cfg.to_jsonable())

    manifest_path = args.manifest or cfg.manifest_dir / "medvidu_tal_timelens_manifest_gt_free.jsonl"
    if not manifest_path.exists():
        build_manifest(cfg.data_path, manifest_path, old_data_root=cfg.old_data_root, new_data_root=cfg.new_data_root)
    rows = read_jsonl(manifest_path)
    if args.smoke:
        smoke_path = cfg.manifest_dir / "medvidu_tal_timelens_manifest_gt_free.smoke.jsonl"
        write_smoke_manifest(rows, smoke_path, size=cfg.smoke_size, seed=cfg.smoke_seed)
        rows = read_jsonl(smoke_path)
        pred_path = cfg.prediction_dir / "timelens8b_tal_smoke_predictions.jsonl"
    else:
        pred_path = cfg.prediction_dir / "timelens8b_tal_predictions.jsonl"
    if args.limit is not None:
        rows = rows[: args.limit]
    incompatible = [row["sample_id"] for row in rows if not row["timestamp_spacing_audit"]["qwen_frame_list_single_fps_compatible"]]
    if incompatible:
        write_json(
            cfg.audit_dir / "timestamp_metadata_audit.json",
            {
                "status": "STOP",
                "reason": "MedVidU frame timestamps cannot be represented faithfully by a single frame-list fps for all selected samples.",
                "incompatible_sample_ids": incompatible,
                "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
            },
        )
        raise SystemExit(f"STOP: {len(incompatible)} selected samples are not single-fps compatible; see audit/timestamp_metadata_audit.json")
    summary = run_inference(rows, args.model_path, pred_path, cfg.prediction_dir / "timelens8b_tal_errors.jsonl", model_fp)
    write_json(cfg.audit_dir / "cache_restart_audit.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
