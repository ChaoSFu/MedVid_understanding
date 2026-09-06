from __future__ import annotations

from typing import Any

from .config import GROUNDER_PROMPT, QWEN3_DOWNSAMPLE_RATE, QWEN3_IMAGE_PATCH_SIZE, TIMESTAMP_ADAPTER_VERSION


class MedVidUTALTimeLensDataset:
    """Inference-only dataset. Rows must already be GT-free manifest rows."""

    forbidden_fields = {"conversations", "struc_info", "answer", "gt", "ground_truth", "gt_spans"}

    def __init__(
        self,
        rows: list[dict[str, Any]],
        processor: Any,
        min_tokens: int,
        total_tokens: int,
        timestamp_adapter: str = TIMESTAMP_ADAPTER_VERSION,
    ):
        self.rows = rows
        self.processor = processor
        self.min_tokens = min_tokens
        self.total_tokens = total_tokens
        self.timestamp_adapter = timestamp_adapter

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        leaked = self.forbidden_fields.intersection(row)
        if leaked:
            raise AssertionError(f"inference dataset received GT-capable fields: {sorted(leaked)}")
        messages = build_messages_for_adapter(row, self.min_tokens, self.total_tokens, self.timestamp_adapter)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs, video_metadatas = process_qwen_messages(messages)
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "images": images,
            "padding": True,
            "return_tensors": "pt",
            **video_kwargs,
        }
        if videos:
            processor_kwargs["videos"] = videos
            processor_kwargs["video_metadata"] = video_metadatas
        inputs = self.processor(**processor_kwargs)
        return {"inputs": inputs, "row": row, "messages": messages}


def build_official_prompt(question: str) -> str:
    return GROUNDER_PROMPT.format(question)


def build_qwen3_frame_list_messages(row: dict[str, Any], min_tokens: int, total_tokens: int) -> list[dict[str, Any]]:
    effective_fps = row.get("effective_fps")
    if not effective_fps or effective_fps <= 0:
        raise ValueError(f"sample {row.get('sample_id')} has invalid effective_fps={effective_fps}")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": list(row["video"]),
                    "min_pixels": min_tokens * QWEN3_DOWNSAMPLE_RATE * QWEN3_DOWNSAMPLE_RATE,
                    "total_pixels": total_tokens * QWEN3_DOWNSAMPLE_RATE * QWEN3_DOWNSAMPLE_RATE,
                    "fps": float(effective_fps),
                },
                {"type": "text", "text": build_official_prompt(row["human_question"])},
            ],
        }
    ]


def build_textual_timestamp_image_sequence_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    observations = list(row.get("frame_observations") or [])
    video = list(row.get("video") or [])
    if len(observations) != len(video):
        raise ValueError("frame_observations must align one-to-one with video")
    for obs, frame_path in zip(observations, video):
        frame_no = int(obs["frame_position"]) + 1
        local_time = float(obs["local_time"])
        content.append({"type": "text", "text": f"Frame {frame_no} timestamp: {local_time:.6f} seconds"})
        content.append({"type": "image", "image": frame_path})
    content.append({"type": "text", "text": build_official_prompt(row["human_question"])})
    return [{"role": "user", "content": content}]


def build_messages_for_adapter(row: dict[str, Any], min_tokens: int, total_tokens: int, timestamp_adapter: str) -> list[dict[str, Any]]:
    if timestamp_adapter == TIMESTAMP_ADAPTER_VERSION or timestamp_adapter == "strict_single_fps_subset":
        audit = row.get("timestamp_spacing_audit") or {}
        if not audit.get("qwen_frame_list_single_fps_compatible"):
            raise ValueError(f"sample {row.get('sample_id')} is not single-fps compatible")
        return build_qwen3_frame_list_messages(row, min_tokens, total_tokens)
    if timestamp_adapter == "textual_timestamp_image_sequence":
        return build_textual_timestamp_image_sequence_messages(row)
    raise ValueError(f"unknown timestamp adapter: {timestamp_adapter}")


def process_qwen_messages(messages: list[dict[str, Any]]) -> tuple[Any, list[Any], dict[str, Any], list[Any]]:
    from qwen_vl_utils import process_vision_info

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=QWEN3_IMAGE_PATCH_SIZE,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_tensors, video_metadatas = zip(*videos) if videos else ([], [])
    return images, list(video_tensors), video_kwargs, list(video_metadatas)


def process_qwen3_frame_list_messages(messages: list[dict[str, Any]]) -> tuple[Any, list[Any], dict[str, Any], list[Any]]:
    return process_qwen_messages(messages)
