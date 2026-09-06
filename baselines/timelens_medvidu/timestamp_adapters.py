from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    GROUNDER_PROMPT,
    MIN_TOKENS,
    QWEN3_DOWNSAMPLE_RATE,
    TOTAL_TOKENS,
)
from .dataset import build_official_prompt, build_qwen3_frame_list_messages, textual_timestamp_image_max_pixels
from .dataset import build_textual_timestamp_image_sequence_messages as _build_textual_timestamp_image_sequence_messages
from .io_utils import assert_no_gt_leak, sha256_json


STRICT_SINGLE_FPS_ADAPTER = "strict_single_fps_subset"
TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER = "textual_timestamp_image_sequence"


@dataclass(frozen=True)
class AdapterAudit:
    adapter: str
    sample_id: str
    dataset_name: str | None
    status: str
    applicable: bool
    reason: str | None
    n_medvidu_frames: int
    clip_duration: float | None
    prompt_sha256: str | None
    official_prompt_exact_match: bool
    official_prompt_included_verbatim: bool
    uses_sample_video_only: bool
    preserves_frame_order: bool
    preserves_duplicate_logical_frames: bool
    additional_temporal_resampling: bool
    qwen_content_types: list[str]
    per_image_max_pixels: int | None = None
    total_visual_pixels_budget: int | None = None
    qwen_processing_status: str | None = None
    qwen_processing_error: str | None = None
    qwen_video_tensor_count: int | None = None
    qwen_image_count: int | None = None
    qwen_video_metadata_repr: list[str] | None = None
    qwen_video_kwargs_repr: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "status": self.status,
            "applicable": self.applicable,
            "reason": self.reason,
            "n_medvidu_frames": self.n_medvidu_frames,
            "clip_duration": self.clip_duration,
            "prompt_sha256": self.prompt_sha256,
            "official_prompt_exact_match": self.official_prompt_exact_match,
            "official_prompt_included_verbatim": self.official_prompt_included_verbatim,
            "uses_sample_video_only": self.uses_sample_video_only,
            "preserves_frame_order": self.preserves_frame_order,
            "preserves_duplicate_logical_frames": self.preserves_duplicate_logical_frames,
            "additional_temporal_resampling": self.additional_temporal_resampling,
            "qwen_content_types": self.qwen_content_types,
            "per_image_max_pixels": self.per_image_max_pixels,
            "total_visual_pixels_budget": self.total_visual_pixels_budget,
            "qwen_processing_status": self.qwen_processing_status,
            "qwen_processing_error": self.qwen_processing_error,
            "qwen_video_tensor_count": self.qwen_video_tensor_count,
            "qwen_image_count": self.qwen_image_count,
            "qwen_video_metadata_repr": self.qwen_video_metadata_repr,
            "qwen_video_kwargs_repr": self.qwen_video_kwargs_repr,
        }


def duplicate_logical_frames_present(row: dict[str, Any]) -> bool:
    frames = list(row.get("sampled_video_frames") or [])
    paths = list(row.get("video") or [])
    return len(frames) != len(set(frames)) or len(paths) != len(set(paths))


def build_strict_single_fps_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    audit = row.get("timestamp_spacing_audit") or {}
    if not audit.get("qwen_frame_list_single_fps_compatible"):
        raise ValueError(f"sample {row.get('sample_id')} is not single-fps compatible")
    return build_qwen3_frame_list_messages(row, MIN_TOKENS, TOTAL_TOKENS)


def build_textual_timestamp_image_sequence_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = _build_textual_timestamp_image_sequence_messages(row, TOTAL_TOKENS)
    assert_no_gt_leak(
        {
            "adapter": TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER,
            "messages": messages,
            "gt_information_available_to_model": False,
        }
    )
    return messages


def content_types(messages: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("type")) for message in messages for item in message.get("content", [])]


def process_messages_for_audit(messages: list[dict[str, Any]]) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_tensors: list[Any] = []
    video_metadatas: list[Any] = []
    if videos:
        for item in videos:
            if isinstance(item, tuple) and len(item) == 2:
                video_tensors.append(item[0])
                video_metadatas.append(item[1])
            else:
                video_tensors.append(item)
    return {
        "qwen_processing_status": "OK",
        "qwen_image_count": len(images or []),
        "qwen_video_tensor_count": len(video_tensors),
        "qwen_video_metadata_repr": [repr(x) for x in video_metadatas],
        "qwen_video_kwargs_repr": {k: repr(v) for k, v in video_kwargs.items()},
    }


def audit_adapter_for_row(row: dict[str, Any], adapter: str, process_qwen: bool) -> AdapterAudit:
    assert_no_gt_leak(row)
    official_prompt = build_official_prompt(row["human_question"])
    duplicate_present = duplicate_logical_frames_present(row)
    try:
        per_image_max_pixels = None
        total_visual_pixels_budget = None
        if adapter == STRICT_SINGLE_FPS_ADAPTER:
            messages = build_strict_single_fps_messages(row)
            applicable = True
            status = "OK"
            reason = None
            official_exact = True
            official_included = True
        elif adapter == TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER:
            messages = build_textual_timestamp_image_sequence_messages(row)
            applicable = True
            status = "OK"
            reason = "Uses explicit GT-free local_time text before each image; not official TimeLens-8B Qwen3 video timestamp encoding."
            official_exact = False
            official_included = True
            per_image_max_pixels = textual_timestamp_image_max_pixels(len(row.get("video") or []), TOTAL_TOKENS)
            total_visual_pixels_budget = TOTAL_TOKENS * QWEN3_DOWNSAMPLE_RATE * QWEN3_DOWNSAMPLE_RATE
        else:
            raise ValueError(f"unknown adapter: {adapter}")
    except Exception as exc:
        return AdapterAudit(
            adapter=adapter,
            sample_id=str(row.get("sample_id")),
            dataset_name=row.get("dataset_name"),
            status="SKIP",
            applicable=False,
            reason=repr(exc),
            n_medvidu_frames=int(row.get("n_medvidu_frames") or 0),
            clip_duration=row.get("clip_duration"),
            prompt_sha256=None,
            official_prompt_exact_match=False,
            official_prompt_included_verbatim=False,
            uses_sample_video_only=True,
            preserves_frame_order=True,
            preserves_duplicate_logical_frames=duplicate_present,
            additional_temporal_resampling=False,
            qwen_content_types=[],
            per_image_max_pixels=None,
            total_visual_pixels_budget=None,
        )

    qwen_payload: dict[str, Any] = {}
    if process_qwen:
        try:
            qwen_payload = process_messages_for_audit(messages)
        except Exception as exc:
            qwen_payload = {
                "qwen_processing_status": "ERROR",
                "qwen_processing_error": repr(exc),
            }
    prompt_material = {
        "adapter": adapter,
        "messages": messages,
        "official_prompt": official_prompt,
    }
    return AdapterAudit(
        adapter=adapter,
        sample_id=str(row.get("sample_id")),
        dataset_name=row.get("dataset_name"),
        status=status,
        applicable=applicable,
        reason=reason,
        n_medvidu_frames=int(row.get("n_medvidu_frames") or 0),
        clip_duration=row.get("clip_duration"),
        prompt_sha256=sha256_json(prompt_material),
        official_prompt_exact_match=official_exact,
        official_prompt_included_verbatim=official_included,
        uses_sample_video_only=True,
        preserves_frame_order=True,
        preserves_duplicate_logical_frames=duplicate_present,
        additional_temporal_resampling=False,
        qwen_content_types=content_types(messages),
        per_image_max_pixels=per_image_max_pixels,
        total_visual_pixels_budget=total_visual_pixels_budget,
        qwen_processing_status=qwen_payload.get("qwen_processing_status"),
        qwen_processing_error=qwen_payload.get("qwen_processing_error"),
        qwen_video_tensor_count=qwen_payload.get("qwen_video_tensor_count"),
        qwen_image_count=qwen_payload.get("qwen_image_count"),
        qwen_video_metadata_repr=qwen_payload.get("qwen_video_metadata_repr"),
        qwen_video_kwargs_repr=qwen_payload.get("qwen_video_kwargs_repr"),
    )


def adapter_scientific_profile(adapter: str) -> dict[str, Any]:
    if adapter == STRICT_SINGLE_FPS_ADAPTER:
        return {
            "adapter": adapter,
            "description": "Official TimeLens-8B/Qwen3 frame-list video input with fps=effective_fps, restricted to samples whose GT-free local timestamps are exactly single-fps representable.",
            "best_use": "diagnostic official-behavior subset",
            "full_medvidu_coverage": False,
            "time_fidelity": "exact only on compatible samples",
            "official_timelens_behavior_fidelity": "highest",
            "recommended_primary_external_baseline": False,
        }
    if adapter == TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER:
        return {
            "adapter": adapter,
            "description": "GT-free image sequence with explicit per-frame local_time text labels followed by the unchanged official TimeLens grounding prompt.",
            "best_use": "MedVidU-faithful adapted TimeLens variant",
            "full_medvidu_coverage": True,
            "time_fidelity": "preserves original per-frame local_time including non-uniform gaps and duplicates",
            "official_timelens_behavior_fidelity": "lower; does not use official TimeLens-8B Qwen3 video timestamp encoding",
            "recommended_primary_external_baseline": True,
        }
    raise ValueError(f"unknown adapter: {adapter}")
