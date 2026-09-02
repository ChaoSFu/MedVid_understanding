#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local Qwen3.8-27B + vLLM inference for selected MedVidBench tasks.

Tasks: TAL, STG, Region Caption (_gpt/_gemini), Dense Captioning (_gpt/_gemini).
The script keeps the same one-shot prompt, MedGRPO frame preprocessing,
path remapping, non-thinking mode, raw-answer storage, resume/failure logging,
and configurable output length used by qwen38_max_api_infer_v2.py.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MEDGRPO_INFERENCE_DIR = REPO_ROOT / "MedGRPO-Code-main" / "inference"
DEFAULT_EXAMPLES_PATH = SCRIPT_DIR / "oneshot_examples.json"

sys.path.insert(0, str(SCRIPT_DIR))

if not (SCRIPT_DIR / "vision_process_medical.py").exists():
    sys.path.insert(0, str(MEDGRPO_INFERENCE_DIR))

if not DEFAULT_EXAMPLES_PATH.exists():
    DEFAULT_EXAMPLES_PATH = MEDGRPO_INFERENCE_DIR / "oneshot_examples.json"

from vision_process_medical import process_vision_info_medical  # noqa: E402

TARGET_QA_TYPES = {
    "tal",
    "stg",
    "region_caption_gpt",
    "region_caption_gemini",
    "dense_captioning_gpt",
    "dense_captioning_gemini",
}

DEFAULT_OLD_DATA_ROOT = "/root/data"
DEFAULT_NEW_DATA_ROOT = "/mnt/hdd3/huihui/hh_datas/MedVidBench/testdata"
DEFAULT_MODEL_PATH = "/mnt/hdd3/huihui/models/Qwen3.8-27B"
MIN_PIXELS_PER_FRAME = 8 * 28 * 28
MAX_PIXELS_PER_FRAME = 48 * 28 * 28
DEFAULT_MAX_COMPLETION_TOKENS = 256

logger = logging.getLogger("Qwen38_27B_MedVidBench")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def setup_logging(log_path: str) -> None:
    ensure_parent(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def remap_path(path: str, old_root: str, new_root: str) -> str:
    if not isinstance(path, str) or not path:
        return path
    old_root = old_root.rstrip("/")
    new_root = new_root.rstrip("/")
    if path == old_root:
        return new_root
    prefix = old_root + "/"
    if path.startswith(prefix):
        return os.path.join(new_root, path[len(prefix):])
    return path


def remap_nested(value: Any, old_root: str, new_root: str) -> Any:
    if isinstance(value, str):
        return remap_path(value, old_root, new_root)
    if isinstance(value, list):
        return [remap_nested(v, old_root, new_root) for v in value]
    if isinstance(value, tuple):
        return tuple(remap_nested(v, old_root, new_root) for v in value)
    if isinstance(value, dict):
        return {k: remap_nested(v, old_root, new_root) for k, v in value.items()}
    return value


def remap_sample_paths(sample: Dict[str, Any], old_root: str, new_root: str) -> Dict[str, Any]:
    sample = copy.deepcopy(sample)
    sample["video"] = remap_nested(sample.get("video", []), old_root, new_root)
    if sample.get("RC_info") is not None:
        sample["RC_info"] = remap_nested(sample["RC_info"], old_root, new_root)
    return sample


def find_missing(sample: Dict[str, Any], max_report: int = 10) -> List[str]:
    missing = []
    for p in sample.get("video", []):
        if not isinstance(p, str) or p.startswith(("http://", "https://", "data:")):
            continue
        if not os.path.isfile(p):
            missing.append(p)
            if len(missing) >= max_report:
                break
    return missing


def load_oneshot_examples(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    missing = TARGET_QA_TYPES - set(examples.keys())
    if missing:
        raise KeyError(f"oneshot_examples.json missing: {sorted(missing)}")
    return examples


def build_system_instruction(qa_type: str, examples: Dict[str, Dict[str, str]]) -> Optional[str]:
    ex = examples.get(qa_type)
    if ex is None:
        return None
    return (
        "You are an expert medical video analyst. "
        "Below is an example of the expected question and answer format for this task.\n\n"
        "--- Example ---\n"
        f"Question: {ex['question']}\n\n"
        f"Answer: {ex['answer']}\n"
        "--- End Example ---\n\n"
        "Follow the same answer format exactly. Be concise and precise."
    )


def get_question(sample: Dict[str, Any]) -> str:
    convs = sample["conversations"]
    if not convs:
        raise ValueError("Sample has no conversations")
    return convs[0]["value"].replace("<video>\n", "")


def build_prompt(sample: Dict[str, Any], examples: Dict[str, Dict[str, str]], processor) -> str:
    messages: List[Dict[str, Any]] = []
    sys_prompt = build_system_instruction(sample["qa_type"], examples)
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "video", "video": "placeholder"},
            {"type": "text", "text": get_question(sample)},
        ],
    })
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=False,
    )


def preprocess_video(sample: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    metadata = sample.get("metadata", {}) or {}
    if "fps" not in metadata:
        raise KeyError("metadata['fps'] missing")
    fps = float(metadata["fps"])
    video_content: Dict[str, Any] = {
        "type": "video",
        "video": sample["video"],
        "sample_fps": fps,
        "min_pixels": MIN_PIXELS_PER_FRAME,
        "max_pixels": MAX_PIXELS_PER_FRAME,
    }
    if sample.get("is_RC", False) and sample.get("RC_info") is not None:
        video_content["is_RC"] = True
        video_content["RC_info"] = sample["RC_info"]
    visual_message = {
        "role": "user",
        "content": [video_content, {"type": "text", "text": "placeholder"}],
    }
    _, video_inputs, video_kwargs = process_vision_info_medical(
        [visual_message], return_video_kwargs=True
    )
    if not video_inputs:
        raise RuntimeError("No video frames after MedGRPO preprocessing")
    video_tensor = video_inputs[0]
    if isinstance(video_tensor, tuple):
        video_tensor = video_tensor[0]
    if not isinstance(video_tensor, torch.Tensor) or video_tensor.ndim != 4:
        raise TypeError(f"Unexpected processed video type/shape: {type(video_tensor)}")
    video_kwargs = dict(video_kwargs or {})
    video_kwargs["do_sample_frames"] = False
    video_kwargs["fps"] = [fps]
    return video_tensor, video_kwargs


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu()
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).round().to(torch.uint8)
    arr = frame.permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    return Image.fromarray(arr, mode="RGB")


def maybe_save_rc_debug(sample: Dict[str, Any], video_tensor: torch.Tensor, debug_dir: Optional[str]) -> None:
    if not debug_dir or sample.get("qa_type") not in {"region_caption_gpt", "region_caption_gemini"}:
        return
    os.makedirs(debug_dir, exist_ok=True)
    marker = os.path.join(debug_dir, "_saved_one_rc_example.txt")
    if os.path.exists(marker):
        return
    video_id = str((sample.get("metadata", {}) or {}).get("video_id", "unknown"))
    for i, frame in enumerate(video_tensor):
        tensor_frame_to_pil(frame).save(os.path.join(debug_dir, f"rc_{video_id}_{i:04d}.png"))
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"Saved {len(video_tensor)} frames for video_id={video_id}\n")
    logger.info("Saved Region Caption debug frames: %s", debug_dir)


def load_json_if_exists(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(data: Any, path: str) -> None:
    ensure_parent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def create_submission(results: Dict[str, Any], output_path: str) -> None:
    submission = []
    for _, rec in sorted(results.items(), key=lambda x: int(x[0])):
        submission_id = rec.get("id")
        if not submission_id:
            m = rec.get("metadata", {}) or {}
            video_id = m.get("video_id", "")
            start_frame = m.get("input_video_start_frame", "") or m.get("start_frame", "")
            end_frame = m.get("input_video_end_frame", "") or m.get("end_frame", "")
            fps = m.get("fps", "")
            submission_id = f"{video_id}&&{start_frame}&&{end_frame}&&{fps}"
        submission.append({
            "id": submission_id,
            "qa_type": rec.get("qa_type", ""),
            "prediction": rec.get("answer", ""),
        })
    save_json_atomic(submission, output_path)


def validate_data(selected: List[Dict[str, Any]]) -> int:
    bad = 0
    frames = 0
    for s in selected:
        frames += len(s.get("video", []))
        missing = find_missing(s)
        if missing:
            bad += 1
            m = s.get("metadata", {}) or {}
            logger.error(
                "Missing frame idx=%s type=%s video=%s first=%s",
                s.get("original_idx"), s.get("qa_type"), m.get("video_id"), missing[0]
            )
    logger.info("Validation: samples=%d frames=%d bad_samples=%d", len(selected), frames, bad)
    return bad


def build_sampling_params(max_completion_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_completion_tokens,
    )


def prepare_request(
    sample: Dict[str, Any],
    examples: Dict[str, Dict[str, str]],
    processor,
    debug_rc_dir: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.time()
    missing = find_missing(sample)
    if missing:
        raise FileNotFoundError(f"First missing frame: {missing[0]}")
    prompt = build_prompt(sample, examples, processor)
    video_tensor, video_kwargs = preprocess_video(sample)
    maybe_save_rc_debug(sample, video_tensor, debug_rc_dir)
    req = {
        "prompt": prompt,
        "multi_modal_data": {"video": video_tensor},
        "mm_processor_kwargs": video_kwargs,
    }
    info = {
        "question": get_question(sample),
        "num_processed_frames": int(video_tensor.shape[0]),
        "processed_height": int(video_tensor.shape[-2]),
        "processed_width": int(video_tensor.shape[-1]),
        "video_kwargs": video_kwargs,
        "preprocess_elapsed_sec": round(time.time() - t0, 3),
    }
    return req, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--data_path", required=True)
    p.add_argument("--examples_path", default=str(DEFAULT_EXAMPLES_PATH))
    p.add_argument("--old_data_root", default=DEFAULT_OLD_DATA_ROOT)
    p.add_argument("--new_data_root", default=DEFAULT_NEW_DATA_ROOT)
    p.add_argument("--output_path", default=str(REPO_ROOT / "results/qwen38_27b_vllm_4tasks/results.json"))
    p.add_argument("--submission_path", default=str(REPO_ROOT / "results/qwen38_27b_vllm_4tasks/submission.json"))
    p.add_argument("--failure_path", default=str(REPO_ROOT / "results/qwen38_27b_vllm_4tasks/failures.json"))
    p.add_argument("--log_path", default=str(REPO_ROOT / "results/qwen38_27b_vllm_4tasks/inference.log"))
    p.add_argument("--debug_rc_dir", default=None)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_completion_tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--validate_only", action="store_true")
    args = p.parse_args()

    setup_logging(args.log_path)
    logger.info("=" * 80)
    logger.info("Qwen3.8-27B local vLLM MedVidBench")
    logger.info("model=%s", args.model_path)
    logger.info("data=%s", args.data_path)
    logger.info("path mapping: %s -> %s", args.old_data_root, args.new_data_root)
    logger.info(
        "TP=%d gpu_mem=%.2f max_model_len=%d batch=%d max_output=%d",
        args.tensor_parallel_size, args.gpu_memory_utilization,
        args.max_model_len, args.batch_size, args.max_completion_tokens
    )
    logger.info("=" * 80)

    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    if not isinstance(all_data, list):
        raise TypeError("Input JSON must be a list")

    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for idx, original in enumerate(all_data):
        qa_type = original.get("qa_type", "")
        if qa_type not in TARGET_QA_TYPES:
            continue
        s = remap_sample_paths(original, args.old_data_root, args.new_data_root)
        s["original_idx"] = idx
        selected.append(s)
        counts[qa_type] = counts.get(qa_type, 0) + 1
    if args.limit is not None:
        selected = selected[:args.limit]

    logger.info("all=%d selected=%d", len(all_data), len(selected))
    for k in sorted(counts):
        logger.info("  %s: %d", k, counts[k])
    if not selected:
        raise RuntimeError("No selected task samples")

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
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    logger.info("Processor loaded in %.1fs", time.time() - t0)

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
        enforce_eager=True,
    )
    logger.info("vLLM loaded in %.1fs", time.time() - t0)
    sampling_params = build_sampling_params(args.max_completion_tokens)

    results: Dict[str, Any] = load_json_if_exists(args.output_path, {})
    failures: Dict[str, Any] = load_json_if_exists(args.failure_path, {})
    completed = set(results.keys())
    pending = [s for s in selected if str(s["original_idx"]) not in completed]
    logger.info("Already completed=%d pending=%d", len(completed), len(pending))

    run_start = time.time()
    success = 0
    failed = 0

    for batch_start in range(0, len(pending), args.batch_size):
        batch_samples = pending[batch_start:batch_start + args.batch_size]
        prepared = []

        for sample in batch_samples:
            idx = int(sample["original_idx"])
            key = str(idx)
            qa_type = sample["qa_type"]
            meta = sample.get("metadata", {}) or {}
            try:
                req, prep_info = prepare_request(sample, examples, processor, args.debug_rc_dir)
                prepared.append((sample, prep_info, req))
            except Exception as exc:
                failed += 1
                logger.exception("PREPROCESS FAILED idx=%d type=%s error=%s", idx, qa_type, exc)
                failures[key] = {
                    "stage": "preprocess", "original_idx": idx, "qa_type": qa_type,
                    "metadata": meta, "error": repr(exc),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_json_atomic(failures, args.failure_path)

        if not prepared:
            continue

        infer_start = time.time()
        try:
            outputs = llm.generate(
                [x[2] for x in prepared],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            pairs = list(zip(prepared, outputs))
        except Exception as batch_exc:
            logger.exception("Batch generation failed: %s", batch_exc)
            pairs = []
            if len(prepared) > 1:
                logger.info("Retrying batch sample-by-sample")
                for item in prepared:
                    try:
                        out = llm.generate([item[2]], sampling_params=sampling_params, use_tqdm=False)[0]
                        pairs.append((item, out))
                    except Exception as exc:
                        sample = item[0]
                        idx = int(sample["original_idx"])
                        key = str(idx)
                        failed += 1
                        failures[key] = {
                            "stage": "generation", "original_idx": idx,
                            "qa_type": sample["qa_type"], "metadata": sample.get("metadata", {}),
                            "error": repr(exc), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        save_json_atomic(failures, args.failure_path)
            else:
                sample = prepared[0][0]
                idx = int(sample["original_idx"])
                key = str(idx)
                failed += 1
                failures[key] = {
                    "stage": "generation", "original_idx": idx,
                    "qa_type": sample["qa_type"], "metadata": sample.get("metadata", {}),
                    "error": repr(batch_exc), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_json_atomic(failures, args.failure_path)
                continue

        batch_elapsed = time.time() - infer_start

        for (sample, prep_info, _), request_output in pairs:
            idx = int(sample["original_idx"])
            key = str(idx)
            qa_type = sample["qa_type"]
            meta = sample.get("metadata", {}) or {}
            if not request_output.outputs:
                failed += 1
                failures[key] = {
                    "stage": "empty_output", "original_idx": idx, "qa_type": qa_type,
                    "metadata": meta, "error": "No vLLM output candidate",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_json_atomic(failures, args.failure_path)
                continue

            cand = request_output.outputs[0]
            prompt_tokens = len(getattr(request_output, "prompt_token_ids", []) or [])
            completion_tokens = len(getattr(cand, "token_ids", []) or [])
            finish_reason = getattr(cand, "finish_reason", None) or getattr(cand, "stop_reason", None) or ""

            results[key] = {
                "id": sample.get("id"),
                "metadata": meta,
                "qa_type": qa_type,
                "struc_info": sample.get("struc_info"),
                "question": prep_info["question"],
                "answer": cand.text or "",
                "data_source": sample.get("data_source"),
                "inference_info": {
                    "model": "Qwen3.8-27B",
                    "model_path": args.model_path,
                    "backend": "vLLM offline",
                    "protocol": "MedGRPO one-shot + medical frame preprocessing",
                    "thinking": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "presence_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "max_completion_tokens": args.max_completion_tokens,
                    "min_pixels_per_frame": MIN_PIXELS_PER_FRAME,
                    "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
                    "num_processed_frames": prep_info["num_processed_frames"],
                    "processed_height": prep_info["processed_height"],
                    "processed_width": prep_info["processed_width"],
                    "fps": meta.get("fps"),
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
            completed.add(key)
            if key in failures:
                failures.pop(key, None)
                save_json_atomic(failures, args.failure_path)
            success += 1
            logger.info(
                "idx=%d type=%s video=%s status=success frames=%d size=%dx%d fps=%s "
                "prompt_tokens=%d completion_tokens=%d finish=%s",
                idx, qa_type, meta.get("video_id", "unknown"),
                prep_info["num_processed_frames"], prep_info["processed_width"],
                prep_info["processed_height"], meta.get("fps"), prompt_tokens,
                completion_tokens, finish_reason,
            )

    create_submission(results, args.submission_path)
    logger.info("=" * 80)
    logger.info("Run complete success=%d failed=%d total_completed=%d elapsed=%.1fs", success, failed, len(results), time.time() - run_start)
    logger.info("Results: %s", args.output_path)
    logger.info("Failures: %s", args.failure_path)
    logger.info("Submission: %s", args.submission_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
