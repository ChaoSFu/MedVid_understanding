#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3.8-Max API inference for selected MedVidBench tasks.

Selected task families:
    1) TAL
    2) STG
    3) Region Caption
       - region_caption_gpt
       - region_caption_gemini
    4) Dense Captioning
       - dense_captioning_gpt
       - dense_captioning_gemini

Fair-comparison protocol:
    - Use the official MedVidBench sampled frame list.
    - Preserve each sample's metadata["fps"].
    - Reuse MedGRPO's official vision_process_medical.py preprocessing.
    - Reuse MedGRPO's official oneshot_examples.json.
    - Preserve Region Caption bounding-box rendering.
    - Use deterministic/non-thinking Qwen3.8-Max decoding.
    - Save raw model predictions without answer-side correction.

This script additionally:
    - Remaps the authors' original /root/data/... frame paths to the local
      MedVidBench test-data directory.
    - Supports resume after interruption.
    - Retries transient API failures.
    - Saves results after every successful sample.
    - Writes a leaderboard-format submission JSON.
    - Provides --validate_only to check local frame paths before spending API cost.

Expected location:
    MedGRPO-Code/inference/qwen38_max_api_infer.py
"""

import argparse
import base64
import copy
import json
import logging
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from openai import OpenAI


# =============================================================================
# Repository paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the official MedGRPO preprocessing implementation.
from vision_process_medical import process_vision_info_medical  # noqa: E402


# =============================================================================
# Task selection
# =============================================================================

TARGET_QA_TYPES = {
    "tal",
    "stg",
    "region_caption_gpt",
    "region_caption_gemini",
    "dense_captioning_gpt",
    "dense_captioning_gemini",
}


# =============================================================================
# Default local path mapping
# =============================================================================
#
# JSON examples from the authors:
#   /root/data/AVOS/frames_15fps/VsKw5d-4rq8/13561.jpg
#
# Your local location:
#   /mnt/hdd3/huihui/hh_datas/MedVidBench/testdata/
#       AVOS/frames_15fps/VsKw5d-4rq8/13561.jpg
#
# These defaults can also be overridden from the command line.
# =============================================================================

DEFAULT_OLD_DATA_ROOT = "/root/data"
DEFAULT_NEW_DATA_ROOT = "/mnt/hdd3/huihui/hh_datas/MedVidBench/testdata"


# =============================================================================
# MedGRPO visual/generation settings
# =============================================================================

MIN_PIXELS_PER_FRAME = 8 * 28 * 28      # 6,272
MAX_PIXELS_PER_FRAME = 48 * 28 * 28     # 37,632
MAX_COMPLETION_TOKENS = 256

# Qwen3.8-Max OpenAI-compatible API accepts Base64 image inputs.
# Keep this explicit so a very long sample fails loudly rather than being
# silently downsampled and made incomparable.
DEFAULT_MAX_BASE64_FRAMES = 250


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("Qwen38MedVidBench")


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def setup_logging(log_path: str) -> None:
    ensure_parent(log_path)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# =============================================================================
# Path remapping
# =============================================================================

def remap_path(path: str, old_root: str, new_root: str) -> str:
    """
    Map an author-machine absolute path to the local dataset path.

    Example:
        /root/data/AVOS/.../13561.jpg

    becomes:
        /mnt/hdd3/huihui/hh_datas/MedVidBench/testdata/AVOS/.../13561.jpg
    """
    if not isinstance(path, str) or not path:
        return path

    old_root = old_root.rstrip("/")
    new_root = new_root.rstrip("/")

    if path == old_root:
        return new_root

    prefix = old_root + "/"

    if path.startswith(prefix):
        relative_path = path[len(prefix):]
        return os.path.join(new_root, relative_path)

    return path


def remap_nested_paths(
    value: Any,
    old_root: str,
    new_root: str,
) -> Any:
    """
    Recursively remap strings under structures such as RC_info.
    Only strings beginning with old_root are changed.
    """
    if isinstance(value, str):
        return remap_path(value, old_root, new_root)

    if isinstance(value, list):
        return [
            remap_nested_paths(v, old_root, new_root)
            for v in value
        ]

    if isinstance(value, tuple):
        return tuple(
            remap_nested_paths(v, old_root, new_root)
            for v in value
        )

    if isinstance(value, dict):
        return {
            k: remap_nested_paths(v, old_root, new_root)
            for k, v in value.items()
        }

    return value


def remap_sample_paths(
    sample: Dict[str, Any],
    old_root: str,
    new_root: str,
) -> Dict[str, Any]:
    """
    Remap all frame paths needed by the selected sample.

    Important for Region Caption:
    RC_info.start_frame (and any other path stored in RC_info) must be remapped
    together with sample["video"]. Otherwise the official code cannot match the
    RC frame path and the green bounding box will not be drawn.
    """
    sample = copy.deepcopy(sample)

    if "video" in sample:
        sample["video"] = remap_nested_paths(
            sample["video"],
            old_root,
            new_root,
        )

    if sample.get("RC_info") is not None:
        sample["RC_info"] = remap_nested_paths(
            sample["RC_info"],
            old_root,
            new_root,
        )

    return sample


def find_missing_frame_paths(
    sample: Dict[str, Any],
    max_report: int = 10,
) -> List[str]:
    """
    Return up to max_report missing/unreadable frame paths for this sample.
    """
    missing: List[str] = []

    for frame_path in sample.get("video", []):
        if not isinstance(frame_path, str):
            continue

        # The official test JSON is expected to contain local filesystem paths.
        # If a URL is ever present, do not test it with os.path.isfile.
        if frame_path.startswith(("http://", "https://", "data:")):
            continue

        if not os.path.isfile(frame_path):
            missing.append(frame_path)

            if len(missing) >= max_report:
                break

    return missing


# =============================================================================
# One-shot prompt
# =============================================================================

def load_oneshot_examples(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    required = TARGET_QA_TYPES - set(examples.keys())

    if required:
        raise KeyError(
            "oneshot_examples.json is missing required qa_type(s): "
            + ", ".join(sorted(required))
        )

    return examples


def build_system_instruction(
    qa_type: str,
    examples: Dict[str, Dict[str, str]],
) -> Optional[str]:
    """
    Keep the same one-shot system-instruction pattern used by
    MedGRPO's vllm_infer_oneshot.py.
    """
    example = examples.get(qa_type)

    if example is None:
        return None

    return (
        "You are an expert medical video analyst. "
        "Below is an example of the expected question and answer format for this task.\n\n"
        "--- Example ---\n"
        f"Question: {example['question']}\n\n"
        f"Answer: {example['answer']}\n"
        "--- End Example ---\n\n"
        "Follow the same answer format exactly. Be concise and precise."
    )


def get_question(sample: Dict[str, Any]) -> str:
    """
    Same question extraction convention used in the official inference code.
    """
    conversations = sample["conversations"]

    if not conversations:
        raise ValueError("Sample has no conversations.")

    question = conversations[0]["value"]

    # Match the official code.
    question = question.replace("<video>\n", "")

    return question


# =============================================================================
# Official MedGRPO visual preprocessing
# =============================================================================

def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """
    Convert [C,H,W] tensor returned by vision_process_medical.py to RGB PIL.
    """
    frame = frame.detach().cpu()

    if frame.ndim != 3:
        raise ValueError(
            f"Expected frame tensor [C,H,W], got shape {tuple(frame.shape)}"
        )

    if frame.shape[0] not in (1, 3, 4):
        raise ValueError(
            f"Unexpected channel count in frame: {tuple(frame.shape)}"
        )

    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).round().to(torch.uint8)

    array = frame.permute(1, 2, 0).numpy()

    if array.shape[2] == 1:
        array = array[:, :, 0]
        return Image.fromarray(array, mode="L").convert("RGB")

    if array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")

    return Image.fromarray(array, mode="RGB")


def pil_to_base64_png(image: Image.Image) -> str:
    """
    Use lossless PNG so that no extra JPEG compression is introduced after the
    official preprocessing step.
    """
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="PNG",
        optimize=False,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def preprocess_video(
    sample: Dict[str, Any],
    max_base64_frames: int,
) -> Tuple[List[str], List[Image.Image]]:
    """
    Run the official MedGRPO visual preprocessing:
      - use sample["video"] directly
      - preserve metadata fps
      - apply min/max pixel settings
      - draw Region Caption green bbox through RC_info

    Returns:
      1) Base64 PNG data URLs for Qwen API
      2) processed PIL frames (useful for optional debugging)
    """
    metadata = sample.get("metadata", {}) or {}

    if "fps" not in metadata:
        raise KeyError("Sample metadata does not contain 'fps'.")

    video_content: Dict[str, Any] = {
        "type": "video",
        "video": sample["video"],
        "sample_fps": float(metadata["fps"]),
        "min_pixels": MIN_PIXELS_PER_FRAME,
        "max_pixels": MAX_PIXELS_PER_FRAME,
    }

    # Critical for Region Caption.
    if sample.get("is_RC", False) and sample.get("RC_info") is not None:
        video_content["is_RC"] = True
        video_content["RC_info"] = sample["RC_info"]

    message = {
        "role": "user",
        "content": [
            video_content,
            {
                "type": "text",
                "text": "placeholder",
            },
        ],
    }

    _, video_inputs, _ = process_vision_info_medical(
        [message],
        return_video_kwargs=True,
    )

    if not video_inputs:
        raise RuntimeError(
            "Official visual preprocessing returned no video frames."
        )

    video_tensor = video_inputs[0]

    # If future versions return (video, metadata), retain compatibility.
    if isinstance(video_tensor, tuple):
        video_tensor = video_tensor[0]

    if not isinstance(video_tensor, torch.Tensor):
        raise TypeError(
            "Expected process_vision_info_medical() to return a torch.Tensor "
            f"for the video, got {type(video_tensor)}"
        )

    if video_tensor.ndim != 4:
        raise ValueError(
            "Expected preprocessed video tensor [T,C,H,W], "
            f"got shape {tuple(video_tensor.shape)}"
        )

    num_frames = int(video_tensor.shape[0])

    if num_frames > max_base64_frames:
        raise RuntimeError(
            f"Sample has {num_frames} processed frames, exceeding the configured "
            f"Base64 image-list limit ({max_base64_frames}). "
            "The script intentionally does NOT downsample silently because that "
            "would change the benchmark input. Use URL-based frame hosting or "
            "another supported input route for this sample."
        )

    pil_frames = [
        tensor_frame_to_pil(frame)
        for frame in video_tensor
    ]

    frame_data_urls = [
        pil_to_base64_png(image)
        for image in pil_frames
    ]

    return frame_data_urls, pil_frames


# =============================================================================
# Qwen API message construction
# =============================================================================

def build_api_messages(
    sample: Dict[str, Any],
    frame_data_urls: List[str],
    examples: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    qa_type = sample["qa_type"]
    question = get_question(sample)

    metadata = sample.get("metadata", {}) or {}
    fps = float(metadata["fps"])

    system_prompt = build_system_instruction(
        qa_type,
        examples,
    )

    messages: List[Dict[str, Any]] = []

    if system_prompt is not None:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    # Qwen OpenAI-compatible image-list video input.
                    "type": "video",
                    "video": frame_data_urls,

                    # Critical for TAL/STG and timestamp-sensitive captioning.
                    "fps": fps,

                    # Keep spatial evidence in the same approximate envelope
                    # used by the official MedGRPO inference code.
                    "min_pixels": MIN_PIXELS_PER_FRAME,
                    "max_pixels": MAX_PIXELS_PER_FRAME,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    )

    return messages


# =============================================================================
# Qwen3.8-Max API
# =============================================================================

def call_qwen(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_attempts: int,
    max_completion_tokens: int,
) -> Tuple[str, float, Dict[str, Any], str]:
    """
    Call Qwen3.8-Max with a deterministic, non-thinking configuration.

    The MedGRPO reference code uses temperature=0 and max_tokens=256.
    Qwen3.8 non-thinking mode has model-specific defaults such as a nonzero
    presence penalty, so those are explicitly neutralized here.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=messages,

                # Deterministic / greedy-style decoding.
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,

                # Thinking is disabled, so this caps the visible answer.
                max_completion_tokens=max_completion_tokens,

                extra_body={
                    "enable_thinking": False,
                    "top_k": 1,
                    "repetition_penalty": 1.0,
                    "vl_high_resolution_images": False,
                },

                stream=False,
            )

            elapsed = time.time() - start_time

            if not response.choices:
                raise RuntimeError("API returned no choices.")

            message = response.choices[0].message
            answer = message.content or ""

            usage: Dict[str, Any] = {}

            if getattr(response, "usage", None) is not None:
                usage_obj = response.usage

                usage = {
                    "prompt_tokens": getattr(
                        usage_obj,
                        "prompt_tokens",
                        None,
                    ),
                    "completion_tokens": getattr(
                        usage_obj,
                        "completion_tokens",
                        None,
                    ),
                    "total_tokens": getattr(
                        usage_obj,
                        "total_tokens",
                        None,
                    ),
                }

            finish_reason = getattr(
                response.choices[0],
                "finish_reason",
                "",
            ) or ""

            return answer, elapsed, usage, finish_reason

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Qwen request failed on attempt %d/%d: %s",
                attempt,
                max_attempts,
                str(exc),
            )

            if attempt < max_attempts:
                sleep_seconds = 5 * (2 ** (attempt - 1))
                logger.info(
                    "Retrying in %d seconds...",
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Qwen API failed after {max_attempts} attempts: {last_error}"
    )


# =============================================================================
# Persistence
# =============================================================================

def load_json_if_exists(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(data: Any, path: str) -> None:
    ensure_parent(path)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(tmp_path, path)


# =============================================================================
# Leaderboard submission
# =============================================================================

def create_submission(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    submission: List[Dict[str, Any]] = []

    for _, record in sorted(
        results.items(),
        key=lambda item: int(item[0]),
    ):
        metadata = record.get("metadata", {}) or {}

        video_id = metadata.get("video_id", "")

        start_frame = (
            metadata.get("input_video_start_frame", "")
            or metadata.get("start_frame", "")
        )

        end_frame = (
            metadata.get("input_video_end_frame", "")
            or metadata.get("end_frame", "")
        )

        fps = metadata.get("fps", "")

        submission_id = (
            f"{video_id}&&{start_frame}&&{end_frame}&&{fps}"
        )

        submission.append(
            {
                "id": submission_id,
                "qa_type": record.get("qa_type", ""),
                "prediction": record.get("answer", ""),
            }
        )

    save_json_atomic(
        submission,
        output_path,
    )


# =============================================================================
# Optional Region Caption visual sanity check
# =============================================================================

def maybe_save_rc_debug_frame(
    sample: Dict[str, Any],
    pil_frames: List[Image.Image],
    debug_dir: Optional[str],
) -> None:
    """
    Save one processed frame for the first Region Caption sample encountered.

    This is only a visual sanity check. It does not affect inference.
    """
    if not debug_dir:
        return

    if sample.get("qa_type") not in {
        "region_caption_gpt",
        "region_caption_gemini",
    }:
        return

    os.makedirs(debug_dir, exist_ok=True)

    marker_path = os.path.join(
        debug_dir,
        "_saved_one_rc_example.txt",
    )

    if os.path.exists(marker_path):
        return

    metadata = sample.get("metadata", {}) or {}
    video_id = str(metadata.get("video_id", "unknown"))

    # Save all processed frames for this single sample because RC_info may point
    # to a specific frame and the official preprocessing may pad the sequence.
    for idx, image in enumerate(pil_frames):
        out_path = os.path.join(
            debug_dir,
            f"rc_{video_id}_{idx:04d}.png",
        )
        image.save(out_path)

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(
            f"Saved {len(pil_frames)} processed frames for "
            f"video_id={video_id}\n"
        )

    logger.info(
        "Saved Region Caption debug frames to %s",
        debug_dir,
    )


# =============================================================================
# Validation
# =============================================================================

def validate_selected_data(
    selected_data: List[Dict[str, Any]],
) -> int:
    """
    Validate remapped local frame paths without invoking the API.
    Returns number of samples containing missing frame paths.
    """
    bad_samples = 0
    total_frames = 0

    for sample in selected_data:
        total_frames += len(sample.get("video", []))

        missing = find_missing_frame_paths(sample)

        if missing:
            bad_samples += 1

            metadata = sample.get("metadata", {}) or {}
            logger.error(
                "Missing frames: idx=%s type=%s video_id=%s first_missing=%s",
                sample.get("original_idx"),
                sample.get("qa_type"),
                metadata.get("video_id"),
                missing[0],
            )

    logger.info(
        "Validation summary: samples=%d frames=%d bad_samples=%d",
        len(selected_data),
        total_frames,
        bad_samples,
    )

    return bad_samples


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3.8-Max API inference for selected MedVidBench tasks "
            "using the official MedGRPO one-shot protocol."
        )
    )

    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to cleaned_test_data_11_04.json",
    )

    parser.add_argument(
        "--examples_path",
        default=str(SCRIPT_DIR / "oneshot_examples.json"),
        help="Official MedGRPO one-shot examples JSON.",
    )

    parser.add_argument(
        "--old_data_root",
        default=DEFAULT_OLD_DATA_ROOT,
        help="Original path prefix stored in the benchmark JSON.",
    )

    parser.add_argument(
        "--new_data_root",
        default=DEFAULT_NEW_DATA_ROOT,
        help="Local MedVidBench test-data root.",
    )

    parser.add_argument(
        "--model",
        default="qwen3.8-max",
        help="Qwen model ID.",
    )

    parser.add_argument(
        "--base_url",
        default=os.getenv("DASHSCOPE_BASE_URL"),
        help=(
            "OpenAI-compatible Model Studio base URL. "
            "Defaults to DASHSCOPE_BASE_URL."
        ),
    )

    parser.add_argument(
        "--output_path",
        default=str(
            REPO_ROOT
            / "results"
            / "qwen38_max_4tasks"
            / "results.json"
        ),
    )

    parser.add_argument(
        "--submission_path",
        default=str(
            REPO_ROOT
            / "results"
            / "qwen38_max_4tasks"
            / "submission.json"
        ),
    )

    parser.add_argument(
        "--failure_path",
        default=str(
            REPO_ROOT
            / "results"
            / "qwen38_max_4tasks"
            / "failures.json"
        ),
    )

    parser.add_argument(
        "--log_path",
        default=str(
            REPO_ROOT
            / "results"
            / "qwen38_max_4tasks"
            / "inference.log"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected samples for testing.",
    )

    parser.add_argument(
        "--max_attempts",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=MAX_COMPLETION_TOKENS,
        help=(
            "Maximum number of generated answer tokens. "
            "Default: 256, matching the MedGRPO reference inference setting."
        ),
    )

    parser.add_argument(
        "--max_base64_frames",
        type=int,
        default=DEFAULT_MAX_BASE64_FRAMES,
        help=(
            "Fail loudly if a processed sample exceeds this many Base64 frames. "
            "No silent temporal downsampling is performed."
        ),
    )

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help=(
            "Only check remapped local frame paths; do not call Qwen API."
        ),
    )

    parser.add_argument(
        "--debug_rc_dir",
        default=None,
        help=(
            "Optional directory for saving processed frames from the first "
            "Region Caption sample to visually confirm the green bbox."
        ),
    )

    args = parser.parse_args()

    setup_logging(args.log_path)

    logger.info("=" * 80)
    logger.info("Qwen3.8-Max MedVidBench inference")
    logger.info("Model: %s", args.model)
    logger.info("Data: %s", args.data_path)
    logger.info(
        "Path mapping: %s -> %s",
        args.old_data_root,
        args.new_data_root,
    )
    logger.info(
        "Target qa_types: %s",
        ", ".join(sorted(TARGET_QA_TYPES)),
    )
    logger.info(
        "Max completion tokens: %d",
        args.max_completion_tokens,
    )
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Load and filter benchmark data
    # -------------------------------------------------------------------------

    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if not isinstance(all_data, list):
        raise TypeError(
            "Expected MedVidBench test JSON to be a list of samples."
        )

    selected_data: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {}

    for original_idx, original_sample in enumerate(all_data):
        qa_type = original_sample.get("qa_type", "")

        if qa_type not in TARGET_QA_TYPES:
            continue

        sample = remap_sample_paths(
            original_sample,
            old_root=args.old_data_root,
            new_root=args.new_data_root,
        )

        sample["original_idx"] = original_idx

        selected_data.append(sample)

        type_counts[qa_type] = type_counts.get(qa_type, 0) + 1

    if args.limit is not None:
        selected_data = selected_data[:args.limit]

    logger.info(
        "All benchmark samples: %d",
        len(all_data),
    )
    logger.info(
        "Selected samples for this run: %d",
        len(selected_data),
    )

    for qa_type in sorted(type_counts):
        logger.info(
            "  %s: %d",
            qa_type,
            type_counts[qa_type],
        )

    if not selected_data:
        raise RuntimeError(
            "No selected TAL/STG/Region Caption/Dense Captioning samples found."
        )

    # -------------------------------------------------------------------------
    # Validate remapped paths
    # -------------------------------------------------------------------------

    if args.validate_only:
        bad_samples = validate_selected_data(selected_data)

        if bad_samples:
            raise SystemExit(2)

        logger.info("All selected local frame paths are valid.")
        return

    # -------------------------------------------------------------------------
    # API configuration
    # -------------------------------------------------------------------------

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. Example:\n"
            'export DASHSCOPE_API_KEY="sk-..."'
        )

    if not args.base_url:
        raise RuntimeError(
            "DASHSCOPE_BASE_URL is not set and --base_url was not provided."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,

        # Multimodal requests may be slow.
        timeout=900.0,

        # Retry explicitly in call_qwen().
        max_retries=0,
    )

    examples = load_oneshot_examples(
        args.examples_path
    )

    # -------------------------------------------------------------------------
    # Resume state
    # -------------------------------------------------------------------------

    results: Dict[str, Any] = load_json_if_exists(
        args.output_path,
        {},
    )

    failures: Dict[str, Any] = load_json_if_exists(
        args.failure_path,
        {},
    )

    completed_keys = set(results.keys())

    logger.info(
        "Already completed in results.json: %d",
        len(completed_keys),
    )

    run_start_time = time.time()
    num_success_this_run = 0
    num_failed_this_run = 0

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    for position, sample in enumerate(
        selected_data,
        start=1,
    ):
        original_idx = int(sample["original_idx"])
        result_key = str(original_idx)

        if result_key in completed_keys:
            logger.info(
                "[%d/%d] idx=%d already completed; skipping",
                position,
                len(selected_data),
                original_idx,
            )
            continue

        qa_type = sample["qa_type"]
        metadata = sample.get("metadata", {}) or {}
        video_id = metadata.get("video_id", "unknown")

        try:
            sample_start_time = time.time()

            # Fail early with a clear path message instead of a PIL PermissionError.
            missing = find_missing_frame_paths(sample)

            if missing:
                raise FileNotFoundError(
                    "Local frame path is missing or inaccessible after remapping. "
                    f"First missing path: {missing[0]}"
                )

            question = get_question(sample)

            # Official MedGRPO visual preprocessing.
            frame_data_urls, pil_frames = preprocess_video(
                sample,
                max_base64_frames=args.max_base64_frames,
            )

            maybe_save_rc_debug_frame(
                sample,
                pil_frames,
                args.debug_rc_dir,
            )

            # Do not keep decoded PIL frames after request construction.
            num_frames = len(frame_data_urls)

            messages = build_api_messages(
                sample,
                frame_data_urls,
                examples,
            )

            answer, api_elapsed, usage, finish_reason = call_qwen(
                client=client,
                model=args.model,
                messages=messages,
                max_attempts=args.max_attempts,
                max_completion_tokens=args.max_completion_tokens,
            )

            total_elapsed = time.time() - sample_start_time

            result = {
                "metadata": metadata,
                "qa_type": qa_type,
                "struc_info": sample.get("struc_info"),
                "question": question,

                # Keep the raw API answer unchanged.
                "answer": answer,

                "data_source": sample.get("data_source"),
                "inference_info": {
                    "model": args.model,
                    "protocol": "MedGRPO official one-shot format protocol",
                    "thinking": False,
                    "temperature": 0.0,
                    "presence_penalty": 0.0,
                    "top_k": 1,
                    "repetition_penalty": 1.0,
                    "max_completion_tokens": args.max_completion_tokens,
                    "min_pixels_per_frame": MIN_PIXELS_PER_FRAME,
                    "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
                    "num_processed_frames": num_frames,
                    "fps": metadata.get("fps"),
                    "api_elapsed_sec": round(api_elapsed, 3),
                    "total_elapsed_sec": round(total_elapsed, 3),
                    "finish_reason": finish_reason,
                    "usage": usage,
                },
            }

            results[result_key] = result
            save_json_atomic(
                results,
                args.output_path,
            )

            completed_keys.add(result_key)

            # If this sample failed in a previous run, clear the stale failure.
            if result_key in failures:
                failures.pop(result_key, None)
                save_json_atomic(
                    failures,
                    args.failure_path,
                )

            num_success_this_run += 1

            logger.info(
                "[%d/%d] idx=%d type=%s video=%s status=success "
                "frames=%d fps=%s api_elapsed=%.1fs total_elapsed=%.1fs",
                position,
                len(selected_data),
                original_idx,
                qa_type,
                video_id,
                num_frames,
                metadata.get("fps"),
                api_elapsed,
                total_elapsed,
            )

        except Exception as exc:
            num_failed_this_run += 1

            logger.exception(
                "[%d/%d] idx=%d type=%s video=%s status=FAILED error=%s",
                position,
                len(selected_data),
                original_idx,
                qa_type,
                video_id,
                str(exc),
            )

            failures[result_key] = {
                "original_idx": original_idx,
                "qa_type": qa_type,
                "metadata": metadata,
                "error": repr(exc),
                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }

            save_json_atomic(
                failures,
                args.failure_path,
            )

            # Do not mark it as completed.
            # Re-running the same command will retry it.
            continue

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------

    create_submission(
        results,
        args.submission_path,
    )

    total_run_elapsed = time.time() - run_start_time

    logger.info("=" * 80)
    logger.info("Run complete")
    logger.info(
        "Success this run: %d",
        num_success_this_run,
    )
    logger.info(
        "Failed this run: %d",
        num_failed_this_run,
    )
    logger.info(
        "Total completed in results.json: %d",
        len(results),
    )
    logger.info(
        "Run elapsed: %.1fs",
        total_run_elapsed,
    )
    logger.info(
        "Results: %s",
        args.output_path,
    )
    logger.info(
        "Failures: %s",
        args.failure_path,
    )
    logger.info(
        "Submission: %s",
        args.submission_path,
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
