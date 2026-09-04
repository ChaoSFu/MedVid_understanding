from __future__ import annotations

from typing import Any

from .config import GROUNDER_PROMPT, QWEN3_DOWNSAMPLE_RATE, QWEN3_IMAGE_PATCH_SIZE


class MedVidUTALTimeLensDataset:
    """Inference-only dataset. Rows must already be GT-free manifest rows."""

    forbidden_fields = {"conversations", "struc_info", "answer", "gt", "ground_truth", "gt_spans"}

    def __init__(self, rows: list[dict[str, Any]], processor: Any, min_tokens: int, total_tokens: int):
        self.rows = rows
        self.processor = processor
        self.min_tokens = min_tokens
        self.total_tokens = total_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        leaked = self.forbidden_fields.intersection(row)
        if leaked:
            raise AssertionError(f"inference dataset received GT-capable fields: {sorted(leaked)}")
        messages = build_qwen3_frame_list_messages(row, self.min_tokens, self.total_tokens)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs, video_metadatas = process_qwen3_frame_list_messages(messages)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
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


def process_qwen3_frame_list_messages(messages: list[dict[str, Any]]) -> tuple[Any, list[Any], dict[str, Any], list[Any]]:
    from qwen_vl_utils import process_vision_info

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=QWEN3_IMAGE_PATCH_SIZE,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_tensors, video_metadatas = zip(*videos) if videos else ([], [])
    return images, list(video_tensors), video_kwargs, list(video_metadatas)

