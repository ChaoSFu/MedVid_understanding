from __future__ import annotations

import argparse
import asyncio
import os
import traceback
from pathlib import Path
from typing import Any

from .config import DecodeConfig, PixelConfig, RunConfig, TARGET_MODELS
from .io_utils import append_jsonl, environment_snapshot, load_completed_keys, read_jsonl, write_json
from .models.base import build_frame_prompt, load_inputs, selected_positions_hash, target_cache_key
from .models.qwen35 import make_qwen35_runner
from .models.qwen38 import Qwen38SGLangRunner
from .models.qwen3vl import make_qwen3vl_runner
from .tasks import get_adapter


def build_runner(args: argparse.Namespace):
    decode = DecodeConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        num_beams=1,
        thinking=args.enable_thinking,
    )
    pixel = PixelConfig(
        total_pixels=args.total_pixels,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        do_resize=args.do_resize,
    )
    model_path = args.model_path or TARGET_MODELS[args.target]
    model_name = args.model_name or model_path
    if args.target in {"qwen3vl_4b", "qwen3vl_8b"}:
        return make_qwen3vl_runner(
            args.target,
            model_path=model_path,
            model_name=model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attention_implementation=args.attn_implementation,
            decode=decode,
            pixel=pixel,
        )
    if args.target == "qwen35_4b":
        return make_qwen35_runner(
            model_path=model_path,
            model_name=model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attention_implementation=args.attn_implementation,
            decode=decode,
            pixel=pixel,
        )
    if args.target == "qwen38_27b":
        return Qwen38SGLangRunner(
            model_path=model_path,
            model_name=model_name,
            base_url=args.base_url,
            api_key=args.api_key,
            media_schema=args.media_schema,
            decode=decode,
            pixel=pixel,
            request_timeout=args.request_timeout,
        )
    raise ValueError(f"Unknown target: {args.target}")


def validate_same_evidence(selection: dict[str, Any]) -> None:
    positions = [int(x) for x in selection.get("selected_original_positions_chronological", [])]
    expected_hash = selection.get("selected_positions_hash")
    if expected_hash and expected_hash != selected_positions_hash(selection):
        raise RuntimeError("SELECTED_POSITIONS_HASH_MISMATCH")
    if selection.get("selector_applicable", False) and len(positions) != int(selection.get("effective_k", len(positions))):
        raise RuntimeError("EFFECTIVE_K_MISMATCH")


async def run_async(args: argparse.Namespace) -> dict[str, Any]:
    cfg = RunConfig(output_root=args.output_root)
    cfg.make_dirs()
    target_dir = cfg.targets_dir / args.target
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_path or (target_dir / "predictions.jsonl")
    errors_path = args.errors_path or (target_dir / "errors.jsonl")

    manifest_by_id, selections, selector_sha = load_inputs(args.manifest, args.selector, args.selector_sha256)
    runner = build_runner(args)
    fingerprint = runner.preflight()
    write_json(target_dir / "model_fingerprint.json", fingerprint)
    (target_dir / "environment.txt").write_text(environment_snapshot(("openai",)), encoding="utf-8")
    if args.preflight_only:
        summary = {"status": "PREFLIGHT_OK", "target": args.target, "model_fingerprint": fingerprint}
        write_json(target_dir / "summary.json", summary)
        return summary

    completed = load_completed_keys(output_path)
    counts = {"completed": 0, "skipped_completed": 0, "errors": 0, "parser_failures": 0, "oom": 0}
    fairness_rows = []
    for selection in selections:
        sample_id = str(selection["sample_id"])
        if not selection.get("selector_applicable", False) and not args.include_rc:
            continue
        try:
            validate_same_evidence(selection)
            manifest_row = manifest_by_id[sample_id]
            adapter = get_adapter(str(manifest_row["qa_type"]))
            prompt, _timestamps = build_frame_prompt(manifest_row, selection)
            decode_config = {
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "repetition_penalty": 1.0,
                "num_beams": 1,
                "thinking": bool(args.enable_thinking),
                "max_new_tokens": adapter.max_new_tokens(),
            }
            pixel_config = {
                "total_pixels": args.total_pixels,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "do_resize": args.do_resize,
            }
            cache_key = target_cache_key(manifest_row, selection, fingerprint, selector_sha, prompt, decode_config, pixel_config)
            if cache_key in completed:
                counts["skipped_completed"] += 1
                continue
            if args.target == "qwen38_27b":
                row = await runner.generate(manifest_row, selection, selector_sha)
            else:
                row = await asyncio.to_thread(runner.generate, manifest_row, selection, selector_sha)
            append_jsonl(output_path, row)
            completed.add(str(row["cache_key"]))
            counts["completed"] += 1
            if row.get("parser_status") != "OK":
                counts["parser_failures"] += 1
            fairness_rows.append(
                {
                    "sample_id": sample_id,
                    "selector_manifest_sha256": row["selector_manifest_sha256"],
                    "selected_positions_hash": row["selected_positions_hash"],
                    "effective_k": len(row.get("selected_original_positions_chronological", [])),
                    "prompt_hash": target_cache_key(manifest_row, selection, {"prompt_only": True}, selector_sha, prompt, {}, {}),
                }
            )
        except Exception as exc:
            text = repr(exc)
            if "out of memory" in text.lower() or "cuda oom" in text.lower():
                counts["oom"] += 1
            append_jsonl(
                errors_path,
                {
                    "sample_id": sample_id,
                    "target": args.target,
                    "stage": "target_inference",
                    "error": text,
                    "traceback": traceback.format_exc(),
                },
            )
            counts["errors"] += 1

    summary = {
        "target": args.target,
        "model_path": args.model_path or TARGET_MODELS[args.target],
        "selector_manifest": str(args.selector),
        "selector_manifest_sha256": selector_sha,
        "output_path": str(output_path),
        "errors_path": str(errors_path),
        "counts": counts,
        "model_fingerprint": fingerprint,
        "fairness_audit": {
            "same_selector_manifest_sha256": len({row["selector_manifest_sha256"] for row in fairness_rows}) <= 1,
            "records": len(fairness_rows),
            "unique_selected_positions_hashes": sorted({row["selected_positions_hash"] for row in fairness_rows}),
        },
    }
    write_json(target_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    parser = argparse.ArgumentParser(description="Run one target Qwen backbone on a frozen DIG-32 selector manifest.")
    parser.add_argument("--target", choices=sorted(TARGET_MODELS), required=True)
    parser.add_argument("--manifest", type=Path, default=cfg.selector_dir / "gt_free_manifest.smoke.jsonl")
    parser.add_argument("--selector", type=Path, default=cfg.selector_dir / "dig32_selector_manifest.jsonl")
    parser.add_argument("--selector-sha256", default=None)
    parser.add_argument("--output-root", type=Path, default=cfg.output_root)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--errors-path", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-name", default=None, help="Human-readable run label; model loading uses --model-path.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default=os.getenv("SGLANG_API_KEY", "EMPTY"))
    parser.add_argument("--media-schema", choices=["image_sequence", "qwen_video"], default="image_sequence")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--total-pixels", type=int, default=32000 * 32 * 32)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--do-resize", action="store_true")
    parser.add_argument("--include-rc", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = asyncio.run(run_async(args))
    print(summary)
    return 0 if summary.get("counts", {}).get("errors", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
