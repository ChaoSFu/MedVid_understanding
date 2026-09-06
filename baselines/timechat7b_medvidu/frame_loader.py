from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import IMAGE_SIZE, MAX_FRAMES


@dataclass(frozen=True)
class SelectionAudit:
    n_input_frames: int
    n_selected_frames: int
    selected_logical_indices: list[int]
    unique_selected_indices: int
    first_selected_index: int | None
    last_selected_index: int | None
    policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input_frames": self.n_input_frames,
            "n_selected_frames": self.n_selected_frames,
            "selected_logical_indices": self.selected_logical_indices,
            "unique_selected_indices": self.unique_selected_indices,
            "first_selected_index": self.first_selected_index,
            "last_selected_index": self.last_selected_index,
            "policy": self.policy,
        }


def official_uniform_indices(n_frames: int, max_frames: int = MAX_FRAMES) -> list[int]:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if n_frames <= max_frames:
        return list(range(n_frames))
    import numpy as np

    indices = np.arange(0, n_frames, n_frames / max_frames).astype(int).tolist()
    if len(indices) != max_frames:
        raise AssertionError(f"official-style uniform sampling produced {len(indices)} indices, expected {max_frames}")
    if len(set(indices)) != len(indices):
        raise AssertionError("official-style uniform sampling unexpectedly produced duplicate logical indices")
    return [int(i) for i in indices]


def selection_audit(n_frames: int, indices: list[int]) -> SelectionAudit:
    return SelectionAudit(
        n_input_frames=n_frames,
        n_selected_frames=len(indices),
        selected_logical_indices=list(indices),
        unique_selected_indices=len(set(indices)),
        first_selected_index=indices[0] if indices else None,
        last_selected_index=indices[-1] if indices else None,
        policy="all_frames_if_N_le_96_else_np_arange_uniform_official_style",
    )


def round_timechat_timestamp(seconds: float) -> str:
    return str(round(float(seconds), 1))


def timestamp_texts(local_timestamps: list[float]) -> list[str]:
    return [f"This frame is sampled at {round_timechat_timestamp(t)} second." for t in local_timestamps]


def build_timechat_msg(rounded_timestamps: list[str]) -> str:
    return f"The video contains {len(rounded_timestamps)} frames sampled at {', '.join(rounded_timestamps)} seconds. "


def build_timechat_visual_message(msg: str) -> str:
    return f" <Video><ImageHere></Video> {msg}"


def assert_timestamp_msg_consistency(msg: str, qformer_timestamp_texts: list[str]) -> None:
    rounded_from_qformer = [text.removeprefix("This frame is sampled at ").removesuffix(" second.") for text in qformer_timestamp_texts]
    expected = build_timechat_msg(rounded_from_qformer)
    if msg != expected:
        raise AssertionError(f"TimeChat msg/timestamp mismatch: {msg!r} != {expected!r}")


def select_frame_inputs(frame_paths: list[str], local_timestamps: list[float], max_frames: int = MAX_FRAMES) -> dict[str, Any]:
    if len(frame_paths) != len(local_timestamps):
        raise ValueError("frame_paths and local_timestamps must align one-to-one")
    indices = official_uniform_indices(len(frame_paths), max_frames=max_frames)
    selected_paths = [frame_paths[i] for i in indices]
    selected_timestamps = [float(local_timestamps[i]) for i in indices]
    rounded = [round_timechat_timestamp(t) for t in selected_timestamps]
    texts = timestamp_texts(selected_timestamps)
    msg = build_timechat_msg(rounded)
    assert_timestamp_msg_consistency(msg, texts)
    return {
        "selected_frame_paths": selected_paths,
        "selected_local_timestamps": selected_timestamps,
        "selected_rounded_timestamps": rounded,
        "timestamp_texts": texts,
        "msg": msg,
        "selection_audit": selection_audit(len(frame_paths), indices).to_dict(),
    }


def load_frames_as_timechat_tensor(frame_paths: list[str], image_size: int = IMAGE_SIZE):
    """Load MedVidU frame-list evidence as official TimeChat-style uint8 C x T x H x W tensor."""
    if not frame_paths:
        raise ValueError("frame_paths is empty")
    import numpy as np
    from PIL import Image
    import torch

    arrays = []
    for frame_path in frame_paths:
        path = Path(frame_path)
        if not path.exists():
            raise FileNotFoundError(f"MedVidU frame does not exist: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
            arrays.append(np.asarray(rgb, dtype=np.uint8))
    video = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()
    if tuple(tensor.shape) != (3, len(frame_paths), image_size, image_size):
        raise AssertionError(f"unexpected TimeChat frame tensor shape: {tuple(tensor.shape)}")
    if tensor.dtype != torch.uint8:
        raise AssertionError(f"unexpected TimeChat frame tensor dtype: {tensor.dtype}")
    return tensor


def build_video_features_from_frames(model: Any, vis_processor: Any, frame_paths: list[str], local_timestamps: list[float], device: str | int):
    selected = select_frame_inputs(frame_paths, local_timestamps)
    video = load_frames_as_timechat_tensor(selected["selected_frame_paths"])
    video = vis_processor.transform(video)
    video = video.unsqueeze(0).to(device)
    timestamps = model.tokenizer(
        selected["timestamp_texts"],
        return_tensors="pt",
        padding="longest",
        max_length=32,
        truncation=True,
    )
    if getattr(model, "qformer_text_input", False):
        image_emb, _ = model.encode_videoQformer_visual(video, timestamp=timestamps)
    else:
        image_emb, _ = model.encode_videoQformer_visual(video)
    selected["video_tensor_shape"] = list(video.shape)
    selected["video_embedding_shape"] = list(image_emb.shape) if hasattr(image_emb, "shape") else None
    return [image_emb], selected["msg"], selected
