#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pipelined local Qwen3.8-27B + vLLM inference for MedVidU samples.

This script keeps the MedGRPO/MedVidU inference protocol used in this repo:
one-shot system examples, medical video preprocessing, per-sample FPS, RC bbox
rendering, non-thinking chat template, and greedy decoding by default.

Compared with qwen38_27b_vllm_infer.py, it overlaps CPU/video preprocessing
with GPU batched generation by using a ThreadPoolExecutor. vLLM still owns the
GPU scheduling; do not launch multiple copies of this script on the same GPU
unless you intentionally split GPU resources.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from qwen38_27b_vllm_infer import (
    DEFAULT_EXAMPLES_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    TARGET_QA_TYPES,
    create_submission,
    logger,
    prepare_request,
    remap_sample_paths,
    save_json_atomic,
    setup_logging,
    validate_data,
    load_json_if_exists,
    load_oneshot_examples,
)


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "qwen38_27b_vllm_trainval_50_per_qa_type_seed42"
)


PreparedRequest = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_completion_tokens,
    )


def count_by_qa_type(samples: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        qa_type = sample.get("qa_type", "")
        counts[qa_type] = counts.get(qa_type, 0) + 1
    return counts


def select_samples(
    data: Sequence[Dict[str, Any]],
    qa_types: set[str],
    old_root: str,
    new_root: str,
    limit: int | None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    for original_idx, original_sample in enumerate(data):
        qa_type = original_sample.get("qa_type", "")
        if qa_type not in qa_types:
            continue

        sample = remap_sample_paths(
            original_sample,
            old_root=old_root,
            new_root=new_root,
        )
        sample["original_idx"] = original_idx
        selected.append(sample)

        if limit is not None and len(selected) >= limit:
            break

    return selected


def prepare_one(
    sample: Dict[str, Any],
    examples: Dict[str, Dict[str, str]],
    processor: Any,
    debug_rc_dir: str | None,
) -> PreparedRequest:
    return prepare_request(
        sample=sample,
        examples=examples,
        processor=processor,
        debug_rc_dir=debug_rc_dir,
    )


def save_preprocess_failure(
    sample: Dict[str, Any],
    exc: BaseException,
    failures: Dict[str, Any],
    failure_path: str,
) -> None:
    idx = int(sample["original_idx"])
    key = str(idx)
    failures[key] = {
        "stage": "preprocess",
        "original_idx": idx,
        "id": sample.get("id"),
        "qa_type": sample.get("qa_type"),
        "metadata": sample.get("metadata", {}) or {},
        "error": repr(exc),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json_atomic(failures, failure_path)


def save_generation_failure(
    sample: Dict[str, Any],
    exc: BaseException | str,
    failures: Dict[str, Any],
    failure_path: str,
) -> None:
    idx = int(sample["original_idx"])
    key = str(idx)
    failures[key] = {
        "stage": "generation",
        "original_idx": idx,
        "id": sample.get("id"),
        "qa_type": sample.get("qa_type"),
        "metadata": sample.get("metadata", {}) or {},
        "error": repr(exc),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json_atomic(failures, failure_path)


def save_success(
    prepared: PreparedRequest,
    request_output: Any,
    args: argparse.Namespace,
    results: Dict[str, Any],
    failures: Dict[str, Any],
    batch_elapsed: float,
) -> None:
    sample, prep_info, _ = prepared
    idx = int(sample["original_idx"])
    key = str(idx)
    qa_type = sample["qa_type"]
    metadata = sample.get("metadata", {}) or {}

    if not request_output.outputs:
        raise RuntimeError("vLLM returned no output candidate")

    candidate = request_output.outputs[0]
    prompt_tokens = len(getattr(request_output, "prompt_token_ids", []) or [])
    completion_tokens = len(getattr(candidate, "token_ids", []) or [])
    finish_reason = (
        getattr(candidate, "finish_reason", None)
        or getattr(candidate, "stop_reason", None)
        or ""
    )

    results[key] = {
        "id": sample.get("id"),
        "metadata": metadata,
        "qa_type": qa_type,
        "struc_info": sample.get("struc_info"),
        "question": prep_info["question"],
        "answer": candidate.text or "",
        "data_source": sample.get("data_source"),
        "inference_info": {
            "model": "Qwen3.8-27B",
            "model_path": args.model_path,
            "backend": "vLLM offline pipelined",
            "protocol": "MedGRPO official one-shot format protocol",
            "thinking": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "repetition_penalty": args.repetition_penalty,
            "max_completion_tokens": args.max_completion_tokens,
            "min_pixels_per_frame": MIN_PIXELS_PER_FRAME,
            "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
            "num_processed_frames": prep_info["num_processed_frames"],
            "processed_height": prep_info["processed_height"],
            "processed_width": prep_info["processed_width"],
            "fps": metadata.get("fps"),
            "video_kwargs": prep_info["video_kwargs"],
            "preprocess_elapsed_sec": prep_info["preprocess_elapsed_sec"],
            "batch_generation_elapsed_sec": round(batch_elapsed, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": str(finish_reason),
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
        },
    }
    save_json_atomic(results, args.output_path)

    if key in failures:
        failures.pop(key, None)
        save_json_atomic(failures, args.failure_path)

    logger.info(
        "idx=%d id=%s type=%s video=%s status=success frames=%d size=%dx%d "
        "fps=%s prompt_tokens=%d completion_tokens=%d finish=%s",
        idx,
        sample.get("id"),
        qa_type,
        metadata.get("video_id", "unknown"),
        prep_info["num_processed_frames"],
        prep_info["processed_width"],
        prep_info["processed_height"],
        metadata.get("fps"),
        prompt_tokens,
        completion_tokens,
        finish_reason,
    )


def run_generation_batch(
    batch: List[PreparedRequest],
    llm: LLM,
    sampling_params: SamplingParams,
    args: argparse.Namespace,
    results: Dict[str, Any],
    failures: Dict[str, Any],
) -> Tuple[int, int]:
    if not batch:
        return 0, 0

    generation_start = time.time()
    success = 0
    failed = 0

    try:
        outputs = llm.generate(
            [item[2] for item in batch],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        batch_elapsed = time.time() - generation_start

        for prepared, output in zip(batch, outputs):
            try:
                save_success(
                    prepared=prepared,
                    request_output=output,
                    args=args,
                    results=results,
                    failures=failures,
                    batch_elapsed=batch_elapsed,
                )
                success += 1
            except Exception as exc:
                failed += 1
                sample = prepared[0]
                logger.exception(
                    "SAVE/OUTPUT FAILED idx=%s type=%s error=%s",
                    sample.get("original_idx"),
                    sample.get("qa_type"),
                    exc,
                )
                save_generation_failure(
                    sample,
                    exc,
                    failures,
                    args.failure_path,
                )

        return success, failed

    except Exception as batch_exc:
        logger.exception(
            "Batch generation failed for %d samples: %s",
            len(batch),
            batch_exc,
        )

        if len(batch) == 1:
            save_generation_failure(
                batch[0][0],
                batch_exc,
                failures,
                args.failure_path,
            )
            return 0, 1

        logger.info("Retrying failed batch sample-by-sample")
        for prepared in batch:
            one_start = time.time()
            try:
                output = llm.generate(
                    [prepared[2]],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )[0]
                save_success(
                    prepared=prepared,
                    request_output=output,
                    args=args,
                    results=results,
                    failures=failures,
                    batch_elapsed=time.time() - one_start,
                )
                success += 1
            except Exception as exc:
                failed += 1
                sample = prepared[0]
                logger.exception(
                    "GENERATION FAILED idx=%s type=%s error=%s",
                    sample.get("original_idx"),
                    sample.get("qa_type"),
                    exc,
                )
                save_generation_failure(
                    sample,
                    exc,
                    failures,
                    args.failure_path,
                )

        return success, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipelined async-style local Qwen3.8-27B vLLM inference for "
            "MedVidU/MedVidBench QA JSON files."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--examples_path", default=str(DEFAULT_EXAMPLES_PATH))
    parser.add_argument("--old_data_root", default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new_data_root", default=DEFAULT_NEW_DATA_ROOT)

    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_DIR / "results.json"),
    )
    parser.add_argument(
        "--submission_path",
        default=str(DEFAULT_OUTPUT_DIR / "submission.json"),
    )
    parser.add_argument(
        "--failure_path",
        default=str(DEFAULT_OUTPUT_DIR / "failures.json"),
    )
    parser.add_argument(
        "--log_path",
        default=str(DEFAULT_OUTPUT_DIR / "inference.log"),
    )
    parser.add_argument("--debug_rc_dir", default=None)

    parser.add_argument(
        "--qa_types",
        nargs="+",
        default=sorted(TARGET_QA_TYPES),
        choices=sorted(TARGET_QA_TYPES),
        help="qa_type values to run. Defaults to all six supported trainval tasks.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Build real processor inputs for a few samples, then exit before "
            "initializing vLLM. Useful on machines without usable GPUs."
        ),
    )
    parser.add_argument(
        "--dry_run_limit",
        type=int,
        default=1,
        help="Number of selected samples to preprocess when --dry_run is set.",
    )

    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of already-preprocessed samples per vLLM generate call.",
    )
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=4,
        help="CPU worker threads used to load/process video frames in parallel.",
    )
    parser.add_argument(
        "--disable_enforce_eager",
        action="store_true",
        help="Allow vLLM CUDA graph optimizations instead of enforce_eager=True.",
    )

    parser.add_argument("--max_completion_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_path)

    logger.info("=" * 80)
    logger.info("Qwen3.8-27B local vLLM pipelined inference")
    logger.info("model=%s", args.model_path)
    logger.info("data=%s", args.data_path)
    logger.info("examples=%s", args.examples_path)
    logger.info(
        "path mapping: %s -> %s",
        args.old_data_root,
        args.new_data_root,
    )
    logger.info("qa_types=%s", ", ".join(args.qa_types))
    logger.info(
        "TP=%d gpu_mem=%.2f max_model_len=%d batch=%d workers=%d",
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.max_model_len,
        args.batch_size,
        args.preprocess_workers,
    )
    logger.info(
        "sampling: temperature=%.3f top_p=%.3f top_k=%d min_p=%.3f "
        "presence_penalty=%.3f frequency_penalty=%.3f repetition_penalty=%.3f "
        "max_tokens=%d",
        args.temperature,
        args.top_p,
        args.top_k,
        args.min_p,
        args.presence_penalty,
        args.frequency_penalty,
        args.repetition_penalty,
        args.max_completion_tokens,
    )
    logger.info("=" * 80)

    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if not isinstance(all_data, list):
        raise TypeError("Input JSON must be a list of samples.")

    selected = select_samples(
        data=all_data,
        qa_types=set(args.qa_types),
        old_root=args.old_data_root,
        new_root=args.new_data_root,
        limit=args.limit,
    )
    counts = count_by_qa_type(selected)

    logger.info("all=%d selected=%d", len(all_data), len(selected))
    for qa_type in sorted(counts):
        logger.info("  %s: %d", qa_type, counts[qa_type])

    if not selected:
        raise RuntimeError("No selected samples.")

    if args.validate_only:
        bad = validate_data(selected)
        if bad:
            raise SystemExit(2)
        logger.info("All selected paths valid")
        return

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")

    examples = load_oneshot_examples(args.examples_path)

    logger.info("Loading AutoProcessor...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    logger.info("Processor loaded in %.1fs", time.time() - t0)

    if args.dry_run:
        dry_samples = selected[: max(args.dry_run_limit, 0)]
        if not dry_samples:
            raise RuntimeError("No samples selected for dry run.")

        logger.info(
            "Dry run: preprocessing %d sample(s); vLLM will not be initialized",
            len(dry_samples),
        )
        failures: Dict[str, Any] = {}
        success = 0
        for sample in dry_samples:
            try:
                prepared = prepare_one(
                    sample,
                    examples,
                    processor,
                    args.debug_rc_dir,
                )
                prompt = prepared[2].get("prompt", "")
                mm_data = prepared[2].get("multi_modal_data", {})
                logger.info(
                    "DRY RUN OK idx=%s id=%s type=%s frames=%d prompt_chars=%d mm_keys=%s",
                    sample.get("original_idx"),
                    sample.get("id"),
                    sample.get("qa_type"),
                    prepared[1]["num_processed_frames"],
                    len(prompt),
                    sorted(mm_data.keys()),
                )
                success += 1
            except Exception as exc:
                logger.exception(
                    "DRY RUN FAILED idx=%s id=%s type=%s error=%s",
                    sample.get("original_idx"),
                    sample.get("id"),
                    sample.get("qa_type"),
                    exc,
                )
                save_preprocess_failure(
                    sample,
                    exc,
                    failures,
                    args.failure_path,
                )

        logger.info(
            "Dry run complete success=%d failed=%d. No predictions were generated.",
            success,
            len(dry_samples) - success,
        )
        return

    logger.info("Loading Qwen3.8-27B into vLLM...")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1},
        dtype=args.dtype,
        enforce_eager=not args.disable_enforce_eager,
    )
    logger.info("vLLM loaded in %.1fs", time.time() - t0)

    sampling_params = build_sampling_params(args)
    results: Dict[str, Any] = load_json_if_exists(args.output_path, {})
    failures: Dict[str, Any] = load_json_if_exists(args.failure_path, {})
    completed = set(results.keys())
    pending = [
        sample
        for sample in selected
        if str(sample["original_idx"]) not in completed
    ]

    logger.info(
        "already_completed=%d pending=%d",
        len(completed),
        len(pending),
    )

    run_start = time.time()
    success = 0
    failed = 0
    prepared_batch: List[PreparedRequest] = []

    with ThreadPoolExecutor(max_workers=args.preprocess_workers) as executor:
        futures: Dict[Future[PreparedRequest], Dict[str, Any]] = {
            executor.submit(
                prepare_one,
                sample,
                examples,
                processor,
                args.debug_rc_dir,
            ): sample
            for sample in pending
        }

        logger.info("Submitted %d preprocessing jobs", len(futures))

        for done_count, future in enumerate(as_completed(futures), start=1):
            sample = futures[future]
            try:
                prepared = future.result()
                prepared_batch.append(prepared)
                logger.info(
                    "prepared=%d/%d idx=%s type=%s frames=%d",
                    done_count,
                    len(futures),
                    sample.get("original_idx"),
                    sample.get("qa_type"),
                    prepared[1]["num_processed_frames"],
                )
            except Exception as exc:
                failed += 1
                logger.exception(
                    "PREPROCESS FAILED idx=%s type=%s error=%s",
                    sample.get("original_idx"),
                    sample.get("qa_type"),
                    exc,
                )
                save_preprocess_failure(
                    sample,
                    exc,
                    failures,
                    args.failure_path,
                )
                continue

            if len(prepared_batch) >= args.batch_size:
                batch_success, batch_failed = run_generation_batch(
                    prepared_batch,
                    llm,
                    sampling_params,
                    args,
                    results,
                    failures,
                )
                success += batch_success
                failed += batch_failed
                prepared_batch = []

    if prepared_batch:
        batch_success, batch_failed = run_generation_batch(
            prepared_batch,
            llm,
            sampling_params,
            args,
            results,
            failures,
        )
        success += batch_success
        failed += batch_failed

    create_submission(results, args.submission_path)

    logger.info("=" * 80)
    logger.info(
        "Run complete success=%d failed=%d total_completed=%d elapsed=%.1fs",
        success,
        failed,
        len(results),
        time.time() - run_start,
    )
    logger.info("Results: %s", args.output_path)
    logger.info("Failures: %s", args.failure_path)
    logger.info("Submission: %s", args.submission_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
