from __future__ import annotations

import argparse
import asyncio
import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from ..config import QWEN35_CHECKPOINT, RunConfig
from ..io_utils import append_jsonl, assert_no_gt_leak, environment_snapshot, load_completed_keys, read_jsonl, sha256_json, write_json
from ..tasks import get_adapter


def pil_to_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def describe_exception(exc: BaseException) -> str:
    details = [repr(exc)]
    for attr in ("status_code", "body"):
        value = getattr(exc, attr, None)
        if value is not None:
            details.append(f"{attr}={value}")
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if response_text:
        details.append(f"response_text={response_text}")
    return " | ".join(details)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def extract_message_text(message: Any) -> tuple[str, str]:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content, "content"
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "\n".join(parts), "content_list_text"
    for attr in ("reasoning_content", "reasoning", "reasoning_text"):
        text = getattr(message, attr, None)
        if isinstance(text, str) and text.strip():
            return text, attr
    message_dict = to_jsonable(message)
    if isinstance(message_dict, dict):
        for key in ("content", "reasoning_content", "reasoning", "reasoning_text"):
            text = message_dict.get(key)
            if isinstance(text, str) and text.strip():
                return text, key
    return "", "empty"


def sglang_cache_key(
    manifest_row: dict[str, Any],
    selection_row: dict[str, Any],
    served_model: str,
    prompt: str,
    decode_config: dict[str, Any],
    media_config: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "backend": "sglang_openai_compatible",
            "sample_id": manifest_row["sample_id"],
            "qa_type": manifest_row["qa_type"],
            "selected_ordered_positions": selection_row.get("selected_original_positions_chronological", []),
            "timestamps": selection_row.get("selected_local_times_chronological", []),
            "served_model": served_model,
            "prompt_hash": sha256_json(prompt),
            "decode_config": decode_config,
            "media_config": media_config,
        }
    )


def build_sglang_messages(
    prompt: str,
    frame_data_urls: list[str],
    timestamps: list[float],
    media_schema: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    if media_schema == "image_sequence":
        content: list[dict[str, Any]] = []
        for i, (url, timestamp) in enumerate(zip(frame_data_urls, timestamps), start=1):
            content.append({"type": "text", "text": f"Frame {i}: {timestamp:.3f} seconds"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    if media_schema == "qwen_video":
        video_item: dict[str, Any] = {
            "type": "video",
            "video": frame_data_urls,
        }
        if min_pixels is not None:
            video_item["min_pixels"] = min_pixels
        if max_pixels is not None:
            video_item["max_pixels"] = max_pixels
        timestamp_text = "\n".join(
            f"Frame {i}: {timestamp:.3f} seconds"
            for i, timestamp in enumerate(timestamps, start=1)
        )
        return [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": timestamp_text + "\n\n" + prompt},
                ],
            }
        ]

    raise ValueError(f"Unsupported media_schema: {media_schema}")


def prepare_frame_data_urls(selection_row: dict[str, Any]) -> tuple[list[str], list[list[int]]]:
    from PIL import Image

    urls: list[str] = []
    sizes: list[list[int]] = []
    for path in selection_row.get("selected_frame_paths_chronological") or []:
        with Image.open(path) as image:
            image = image.convert("RGB")
            sizes.append([int(image.size[0]), int(image.size[1])])
            urls.append(pil_to_data_url(image))
    return urls, sizes


async def call_sglang(client: Any, args: argparse.Namespace, messages: list[dict[str, Any]]) -> tuple[str, float, dict[str, Any], str, dict[str, Any]]:
    last_error: BaseException | None = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            start = time.time()
            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_completion_tokens,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                },
            )
            elapsed = time.time() - start
            if not response.choices:
                raise RuntimeError("SGLang returned no choices")
            choice = response.choices[0]
            answer, answer_source = extract_message_text(choice.message)
            finish_reason = getattr(choice, "finish_reason", "") or ""
            usage = {}
            if getattr(response, "usage", None) is not None:
                usage_obj = response.usage
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
            raw_info = {"answer_source": answer_source}
            if not answer and not args.allow_empty_answers:
                raw_info["empty_choice"] = to_jsonable(choice)
                raise RuntimeError(f"SGLang returned an empty answer: finish_reason={finish_reason}")
            return answer.strip(), elapsed, usage, finish_reason, raw_info
        except Exception as exc:
            last_error = exc
            if attempt < args.max_attempts:
                await asyncio.sleep(args.retry_base_seconds * (2 ** (attempt - 1)))
            else:
                break
    raise RuntimeError(f"SGLang request failed after {args.max_attempts} attempts: {describe_exception(last_error)}")


async def check_sglang_server(client: Any, args: argparse.Namespace) -> dict[str, Any]:
    try:
        models = await client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to SGLang OpenAI-compatible server. "
            f"base_url={args.base_url!r} error={describe_exception(exc)}"
        ) from exc

    model_ids = []
    for item in getattr(models, "data", []) or []:
        model_id = getattr(item, "id", None)
        if model_id is not None:
            model_ids.append(str(model_id))
    return {
        "base_url": args.base_url,
        "model_arg": args.model,
        "served_model_ids": model_ids,
    }


async def process_one(
    manifest_row: dict[str, Any],
    selection_row: dict[str, Any],
    client: Any,
    args: argparse.Namespace,
    output_path: Path,
    errors_path: Path,
    completed: set[str],
    io_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    position: int,
    total: int,
) -> tuple[int, int, int]:
    async with semaphore:
        sample_id = str(manifest_row["sample_id"])
        try:
            adapter = get_adapter(str(manifest_row["qa_type"]))
            timestamps = [float(x) for x in selection_row.get("selected_local_times_chronological", [])]
            frame_lines = [
                f"Frame {i + 1}: {timestamp:.3f} seconds"
                for i, timestamp in enumerate(timestamps)
            ]
            prompt = adapter.build_prompt(manifest_row, frame_lines=frame_lines)
            decode_config = {
                "do_sample": args.temperature > 0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "num_beams": 1,
                "thinking": args.enable_thinking,
                "max_new_tokens": args.max_completion_tokens,
            }
            media_config = {
                "media_schema": args.media_schema,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
            cache_key = sglang_cache_key(manifest_row, selection_row, args.model, prompt, decode_config, media_config)
            if cache_key in completed:
                return 0, 0, 1

            prep_start = time.time()
            frame_data_urls, image_sizes = await asyncio.to_thread(prepare_frame_data_urls, selection_row)
            messages = build_sglang_messages(
                prompt=prompt,
                frame_data_urls=frame_data_urls,
                timestamps=timestamps,
                media_schema=args.media_schema,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
            prep_elapsed = time.time() - prep_start
            answer, api_elapsed, usage, finish_reason, raw_info = await call_sglang(client, args, messages)
            parser_status, parsed_prediction = adapter.parse_prediction(answer)
            row = {
                "sample_id": sample_id,
                "original_index": manifest_row["original_index"],
                "id": manifest_row.get("id"),
                "qa_type": manifest_row["qa_type"],
                "dataset_name": manifest_row.get("dataset_name"),
                "data_source": manifest_row.get("data_source"),
                "selector_applicable": selection_row.get("selector_applicable", False),
                "selector_reason": selection_row.get("selector_reason"),
                "selected_original_positions_chronological": selection_row.get("selected_original_positions_chronological", []),
                "selected_source_frame_indices_chronological": selection_row.get("selected_source_frame_indices_chronological", []),
                "selected_local_times_chronological": timestamps,
                "prediction": answer,
                "parser_status": parser_status,
                "parsed_prediction": parsed_prediction,
                "cache_key": cache_key,
                "inference_info": {
                    "model": args.model,
                    "backend": "SGLang OpenAI-compatible API",
                    "base_url": args.base_url,
                    "media_config": media_config,
                    "decode_config": decode_config,
                    "num_selected_frames": len(frame_data_urls),
                    "original_image_sizes": image_sizes,
                    "preprocess_elapsed_sec": round(prep_elapsed, 3),
                    "api_elapsed_sec": round(api_elapsed, 3),
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "raw_response_info": raw_info,
                    "position": position,
                    "total": total,
                },
            }
            assert_no_gt_leak(row)
            async with io_lock:
                if cache_key not in completed:
                    append_jsonl(output_path, row)
                    completed.add(cache_key)
            return 1, 0, 0
        except Exception as exc:
            async with io_lock:
                append_jsonl(
                    errors_path,
                    {
                        "sample_id": sample_id,
                        "stage": "qwen_sglang",
                        "error": describe_exception(exc),
                        "position": position,
                        "total": total,
                    },
                )
            return 0, 1, 0


async def run_async(args: argparse.Namespace) -> dict[str, int]:
    from openai import AsyncOpenAI

    cfg = RunConfig(output_root=args.output_root, qwen_model=args.model)
    cfg.make_dirs()
    output_path = args.output_path or (cfg.prediction_dir / "qwen35_videoitg_predictions.jsonl")
    errors_path = args.errors_path or (cfg.prediction_dir / "qwen35_errors.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    (cfg.provenance_dir / "qwen_sglang_environment.txt").write_text(environment_snapshot(("openai",)), encoding="utf-8")
    write_json(
        cfg.provenance_dir / "qwen_sglang_model_fingerprint.json",
        {
            "backend": "SGLang OpenAI-compatible API",
            "base_url": args.base_url,
            "served_model": args.model,
            "media_schema": args.media_schema,
            "thinking": args.enable_thinking,
        },
    )

    manifest_by_id = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    selections = read_jsonl(args.selector)
    pending_pairs = []
    for selection in selections:
        if not selection.get("selector_applicable", False) and not args.include_rc_direct:
            continue
        manifest_row = manifest_by_id[str(selection["sample_id"])]
        pending_pairs.append((manifest_row, selection))

    completed = load_completed_keys(output_path)
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=0,
    )
    if not args.no_server_check:
        server_report = await check_sglang_server(client, args)
        write_json(cfg.logs_dir / "qwen_sglang_server_check.json", server_report)
    io_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(args.concurrency, 1))
    tasks = [
        process_one(
            manifest_row=manifest_row,
            selection_row=selection,
            client=client,
            args=args,
            output_path=output_path,
            errors_path=errors_path,
            completed=completed,
            io_lock=io_lock,
            semaphore=semaphore,
            position=i,
            total=len(pending_pairs),
        )
        for i, (manifest_row, selection) in enumerate(pending_pairs, start=1)
    ]
    counts = {"completed": 0, "errors": 0, "skipped_completed": 0}
    for task in asyncio.as_completed(tasks):
        ok, bad, skipped = await task
        counts["completed"] += ok
        counts["errors"] += bad
        counts["skipped_completed"] += skipped
    write_json(cfg.logs_dir / "qwen_sglang_run_summary.json", counts)
    return counts


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    p = argparse.ArgumentParser(description="Run async SGLang inference on VideoITG-selected MedVidU frames.")
    p.add_argument("--manifest", type=Path, default=cfg.manifest_dir / "medvidu_videoitg_manifest_gt_free.jsonl")
    p.add_argument("--selector", type=Path, default=cfg.selector_dir / "videoitg_top32.jsonl")
    p.add_argument("--output-root", type=Path, default=cfg.output_root)
    p.add_argument("--output-path", type=Path, default=None)
    p.add_argument("--errors-path", type=Path, default=None)
    p.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    p.add_argument("--api-key", default=os.getenv("SGLANG_API_KEY", "EMPTY"))
    p.add_argument("--model", default=QWEN35_CHECKPOINT)
    p.add_argument("--media-schema", choices=["image_sequence", "qwen_video"], default="image_sequence")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-server-check", action="store_true", help="Skip startup /v1/models connectivity check.")
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--retry-base-seconds", type=float, default=2.0)
    p.add_argument("--allow-empty-answers", action="store_true")
    p.add_argument("--include-rc-direct", action="store_true")
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    counts = asyncio.run(run_async(args))
    print(counts)
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
