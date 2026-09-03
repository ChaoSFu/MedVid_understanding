#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Async OpenAI-compatible inference against a local SGLang Qwen3.5 server.

This client intentionally does not import SGLang. Run SGLang in a separate
environment or Docker container, then point this script at its /v1 endpoint.
The script reuses the repo's MedGRPO visual preprocessing and sends concurrent
HTTP requests with resume/failure logging.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI

from qwen38_max_api_infer_v2 import (
    DEFAULT_EXAMPLES_PATH,
    DEFAULT_MAX_BASE64_FRAMES,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    TARGET_QA_TYPES,
    build_system_instruction,
    create_submission,
    find_missing_frame_paths,
    get_question,
    load_json_if_exists,
    load_oneshot_examples,
    logger,
    maybe_save_rc_debug_frame,
    preprocess_video,
    remap_sample_paths,
    save_json_atomic,
    setup_logging,
    validate_selected_data,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "qwen38_27b_sglang_trainval_50_per_qa_type_seed42"


def describe_exception(exc: BaseException) -> str:
    details = [repr(exc)]
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        details.append(f"status_code={status_code}")
    body = getattr(exc, "body", None)
    if body is not None:
        details.append(f"body={body}")
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        details.append(f"response_text={response_text}")
    return " | ".join(details)


def build_messages(
    sample: Dict[str, Any],
    frame_data_urls: List[str],
    examples: Dict[str, Dict[str, str]],
    media_schema: str,
) -> List[Dict[str, Any]]:
    qa_type = sample["qa_type"]
    question = get_question(sample)
    metadata = sample.get("metadata", {}) or {}
    fps = float(metadata["fps"])

    messages: List[Dict[str, Any]] = []
    system_prompt = build_system_instruction(qa_type, examples)
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    if media_schema == "qwen_video":
        user_content: List[Dict[str, Any]] = [
            {
                "type": "video",
                "video": frame_data_urls,
                "fps": fps,
                "min_pixels": MIN_PIXELS_PER_FRAME,
                "max_pixels": MAX_PIXELS_PER_FRAME,
            },
            {"type": "text", "text": question},
        ]
    elif media_schema == "image_sequence":
        intro = (
            f"The following images are ordered video frames sampled at {fps} fps. "
            "Use the frame order and sampling rate as temporal evidence.\n\n"
            f"{question}"
        )
        user_content = [{"type": "text", "text": intro}]
        user_content.extend(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
            for data_url in frame_data_urls
        )
    else:
        raise ValueError(f"Unknown media_schema: {media_schema}")

    messages.append({"role": "user", "content": user_content})
    return messages


async def call_sglang(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[str, float, Dict[str, Any], str]:
    last_error: BaseException | None = None

    for attempt in range(1, args.max_attempts + 1):
        try:
            start = time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_completion_tokens,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                },
            )
            elapsed = time.time() - start

            if not response.choices:
                raise RuntimeError("SGLang returned no choices.")

            message = response.choices[0].message
            answer = message.content or ""
            finish_reason = getattr(response.choices[0], "finish_reason", "") or ""

            usage: Dict[str, Any] = {}
            if getattr(response, "usage", None) is not None:
                usage_obj = response.usage
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }

            return answer, elapsed, usage, finish_reason

        except Exception as exc:
            last_error = exc
            logger.warning(
                "SGLang request failed on attempt %d/%d: %s",
                attempt,
                args.max_attempts,
                describe_exception(exc),
            )
            if attempt < args.max_attempts:
                await asyncio.sleep(args.retry_base_seconds * (2 ** (attempt - 1)))

    raise RuntimeError(
        "SGLang request failed after "
        f"{args.max_attempts} attempts: {describe_exception(last_error)}"
    )


def select_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    import json

    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if not isinstance(all_data, list):
        raise TypeError("Input JSON must be a list of samples.")

    selected_qa_types = None
    if args.qa_types != ["all"]:
        selected_qa_types = set(args.qa_types)

    selected: List[Dict[str, Any]] = []
    for original_idx, original_sample in enumerate(all_data):
        qa_type = original_sample.get("qa_type", "")
        if selected_qa_types is not None and qa_type not in selected_qa_types:
            continue
        sample = remap_sample_paths(
            original_sample,
            old_root=args.old_data_root,
            new_root=args.new_data_root,
        )
        sample["original_idx"] = original_idx
        selected.append(sample)
        if args.limit is not None and len(selected) >= args.limit:
            break

    logger.info("all=%d selected=%d", len(all_data), len(selected))
    counts: Dict[str, int] = {}
    for sample in selected:
        qa_type = sample.get("qa_type", "")
        counts[qa_type] = counts.get(qa_type, 0) + 1
    for qa_type in sorted(counts):
        logger.info("  %s: %d", qa_type, counts[qa_type])

    if not selected:
        raise RuntimeError("No selected samples.")
    return selected


def prepare_sample(
    sample: Dict[str, Any],
    examples: Dict[str, Dict[str, str]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[str], int, float, str]:
    start = time.time()
    missing = find_missing_frame_paths(sample)
    if missing:
        raise FileNotFoundError(f"First missing frame: {missing[0]}")

    frame_data_urls, pil_frames = preprocess_video(
        sample,
        max_base64_frames=args.max_base64_frames,
    )
    maybe_save_rc_debug_frame(sample, pil_frames, args.debug_rc_dir)
    num_frames = len(frame_data_urls)
    question = get_question(sample)
    return sample, frame_data_urls, num_frames, time.time() - start, question


def get_media_schemas_to_try(args: argparse.Namespace) -> List[str]:
    if args.media_schema == "auto":
        return ["qwen_video", "image_sequence"]
    if args.media_schema == "qwen_video" and not args.no_schema_fallback:
        return ["qwen_video", "image_sequence"]
    return [args.media_schema]


async def process_one(
    sample: Dict[str, Any],
    examples: Dict[str, Dict[str, str]],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    results: Dict[str, Any],
    failures: Dict[str, Any],
    io_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    position: int,
    total: int,
) -> Tuple[int, int]:
    async with semaphore:
        idx = int(sample["original_idx"])
        key = str(idx)
        qa_type = sample["qa_type"]
        metadata = sample.get("metadata", {}) or {}
        video_id = metadata.get("video_id", "unknown")
        sample_start = time.time()

        try:
            prepared_sample, frame_data_urls, num_frames, prep_elapsed, question = await asyncio.to_thread(
                prepare_sample,
                sample,
                examples,
                args,
            )
            answer = ""
            api_elapsed = 0.0
            usage: Dict[str, Any] = {}
            finish_reason = ""
            used_media_schema = ""
            schema_errors: List[str] = []

            for media_schema in get_media_schemas_to_try(args):
                messages = build_messages(
                    sample=sample,
                    frame_data_urls=frame_data_urls,
                    examples=examples,
                    media_schema=media_schema,
                )
                try:
                    answer, api_elapsed, usage, finish_reason = await call_sglang(
                        client=client,
                        model=args.model,
                        messages=messages,
                        args=args,
                    )
                    used_media_schema = media_schema
                    break
                except Exception as schema_exc:
                    schema_errors.append(f"{media_schema}: {describe_exception(schema_exc)}")
                    if media_schema == "qwen_video" and "image_sequence" in get_media_schemas_to_try(args):
                        logger.warning(
                            "media_schema=qwen_video failed for idx=%d; trying image_sequence",
                            idx,
                        )
                        continue
                    raise

            if not used_media_schema:
                raise RuntimeError("; ".join(schema_errors))

            total_elapsed = time.time() - sample_start

            results[key] = {
                "id": prepared_sample.get("id"),
                "metadata": metadata,
                "qa_type": qa_type,
                "struc_info": prepared_sample.get("struc_info"),
                "question": question,
                "answer": answer,
                "data_source": prepared_sample.get("data_source"),
                "inference_info": {
                    "model": args.model,
                    "backend": "SGLang OpenAI-compatible API",
                    "media_schema": used_media_schema,
                    "requested_media_schema": args.media_schema,
                    "protocol": "MedGRPO official one-shot format protocol",
                    "thinking": False,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "max_completion_tokens": args.max_completion_tokens,
                    "min_pixels_per_frame": MIN_PIXELS_PER_FRAME,
                    "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
                    "num_processed_frames": num_frames,
                    "fps": metadata.get("fps"),
                    "preprocess_elapsed_sec": round(prep_elapsed, 3),
                    "api_elapsed_sec": round(api_elapsed, 3),
                    "total_elapsed_sec": round(total_elapsed, 3),
                    "finish_reason": finish_reason,
                    "usage": usage,
                },
            }

            if key in failures:
                failures.pop(key, None)

            async with io_lock:
                save_json_atomic(results, args.output_path)
                save_json_atomic(failures, args.failure_path)

            logger.info(
                "[%d/%d] idx=%d type=%s video=%s status=success frames=%d "
                "api_elapsed=%.1fs total_elapsed=%.1fs",
                position,
                total,
                idx,
                qa_type,
                video_id,
                num_frames,
                api_elapsed,
                total_elapsed,
            )
            return 1, 0

        except Exception as exc:
            failures[key] = {
                "stage": "preprocess_or_api",
                "original_idx": idx,
                "id": sample.get("id"),
                "qa_type": qa_type,
                "metadata": metadata,
                "error": describe_exception(exc),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            async with io_lock:
                save_json_atomic(failures, args.failure_path)

            logger.exception(
                "[%d/%d] idx=%d type=%s video=%s status=failed error=%s",
                position,
                total,
                idx,
                qa_type,
                video_id,
                exc,
            )
            return 0, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async MedVidU inference through a local SGLang OpenAI API server."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--examples_path", default=str(DEFAULT_EXAMPLES_PATH))
    parser.add_argument("--old_data_root", default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new_data_root", default=DEFAULT_NEW_DATA_ROOT)
    parser.add_argument("--base_url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api_key", default=os.getenv("SGLANG_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="/mnt/hdd3/huihui/models/Qwen3.8-27B")

    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_DIR / "results.json"))
    parser.add_argument("--submission_path", default=str(DEFAULT_OUTPUT_DIR / "submission.json"))
    parser.add_argument("--failure_path", default=str(DEFAULT_OUTPUT_DIR / "failures.json"))
    parser.add_argument("--log_path", default=str(DEFAULT_OUTPUT_DIR / "inference.log"))
    parser.add_argument("--debug_rc_dir", default=None)

    parser.add_argument(
        "--qa_types",
        nargs="+",
        default=sorted(TARGET_QA_TYPES),
        help=(
            "qa_type values to run. Use '--qa_types all' to run every sample "
            "in the input JSON. Default keeps the original six targeted tasks."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--max_base64_frames", type=int, default=DEFAULT_MAX_BASE64_FRAMES)

    parser.add_argument(
        "--media_schema",
        choices=["auto", "qwen_video", "image_sequence"],
        default="image_sequence",
        help=(
            "qwen_video matches the repo's Qwen API video frame-list schema. "
            "image_sequence uses standard OpenAI image_url parts. "
            "auto tries qwen_video first, then image_sequence."
        ),
    )
    parser.add_argument(
        "--no_schema_fallback",
        action="store_true",
        help="Do not retry qwen_video requests as image_sequence after schema errors.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--retry_base_seconds", type=float, default=2.0)
    parser.add_argument("--request_timeout", type=float, default=900.0)

    parser.add_argument("--max_completion_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    setup_logging(args.log_path)

    logger.info("=" * 80)
    logger.info("Qwen3.5/Qwen3.8 local SGLang async API inference")
    logger.info("base_url=%s", args.base_url)
    logger.info("model=%s", args.model)
    logger.info("data=%s", args.data_path)
    logger.info("media_schema=%s concurrency=%d", args.media_schema, args.concurrency)
    logger.info("=" * 80)

    selected = select_samples(args)
    if args.validate_only:
        bad = validate_selected_data(selected)
        if bad:
            raise SystemExit(2)
        logger.info("All selected paths valid")
        return

    examples = load_oneshot_examples(args.examples_path)
    results: Dict[str, Any] = load_json_if_exists(args.output_path, {})
    failures: Dict[str, Any] = load_json_if_exists(args.failure_path, {})
    completed = set(results.keys())
    pending = [sample for sample in selected if str(sample["original_idx"]) not in completed]
    logger.info("already_completed=%d pending=%d", len(completed), len(pending))

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=0,
    )
    io_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(args.concurrency, 1))
    run_start = time.time()

    tasks = [
        process_one(
            sample=sample,
            examples=examples,
            client=client,
            args=args,
            results=results,
            failures=failures,
            io_lock=io_lock,
            semaphore=semaphore,
            position=position,
            total=len(pending),
        )
        for position, sample in enumerate(pending, start=1)
    ]

    success = 0
    failed = 0
    for task in asyncio.as_completed(tasks):
        ok, bad = await task
        success += ok
        failed += bad

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


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
