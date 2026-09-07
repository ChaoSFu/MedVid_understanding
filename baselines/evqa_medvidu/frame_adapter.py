from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import FRAME_ADAPTER_VERSION, OFFICIAL_FPS, OFFICIAL_MAX_FRAMES, SAMPLING_VERSION
from .io_utils import sha256_json


@dataclass(frozen=True)
class SamplingResult:
    selected_logical_indices: list[int]
    selected_timestamps: list[float]
    selected_frame_paths: list[str]
    target_timestamps: list[float]
    policy: str = SAMPLING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_logical_indices": self.selected_logical_indices,
            "selected_timestamps": self.selected_timestamps,
            "selected_frame_paths": self.selected_frame_paths,
            "target_timestamps": self.target_timestamps,
            "n_selected": len(self.selected_logical_indices),
            "policy": self.policy,
            "frame_adapter_version": FRAME_ADAPTER_VERSION,
        }


def official_sample_frame_count(
    clip_duration: float,
    total_frames: int,
    fps: float = OFFICIAL_FPS,
    max_frames: int | None = OFFICIAL_MAX_FRAMES,
) -> int:
    if max_frames is None:
        return int(total_frames)
    sample_frames = int(float(clip_duration) * float(fps))
    return max(1, min(sample_frames, min(int(max_frames), int(total_frames))))


def nearest_indices_for_target_times(local_timestamps: list[float], target_timestamps: list[float]) -> list[int]:
    if not local_timestamps:
        raise ValueError("local_timestamps is empty")
    out: list[int] = []
    for target in target_timestamps:
        best = min(range(len(local_timestamps)), key=lambda i: (abs(float(local_timestamps[i]) - target), i))
        out.append(best)
    return out


def select_model_frames(
    frame_paths: list[str],
    local_timestamps: list[float],
    clip_duration: float,
    fps: float = OFFICIAL_FPS,
    max_frames: int | None = None,
) -> SamplingResult:
    if len(frame_paths) != len(local_timestamps):
        raise ValueError("frame_paths and local_timestamps must have the same length")
    sample_frames = official_sample_frame_count(clip_duration, len(frame_paths), fps=fps, max_frames=max_frames)
    if sample_frames >= len(frame_paths):
        indices = list(range(len(frame_paths)))
        targets = [float(local_timestamps[i]) for i in indices]
    else:
        if sample_frames == 1:
            targets = [0.0]
        else:
            targets = np.arange(0, clip_duration, clip_duration / sample_frames)[:sample_frames].astype(float).tolist()
        indices = nearest_indices_for_target_times(local_timestamps, targets)
    return SamplingResult(
        selected_logical_indices=[int(i) for i in indices],
        selected_timestamps=[float(local_timestamps[i]) for i in indices],
        selected_frame_paths=[frame_paths[i] for i in indices],
        target_timestamps=[float(t) for t in targets],
    )


def logical_frame_identities(frame_paths: list[str], source_frame_indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "logical_position": i,
            "source_frame_index": int(source_frame_indices[i]),
            "frame_path": frame_paths[i],
            "frame_identity_hash": sha256_json(
                {"logical_position": i, "source_frame_index": int(source_frame_indices[i]), "frame_path": frame_paths[i]}
            ),
        }
        for i in range(len(frame_paths))
    ]


def path_exists_audit(frame_paths: list[str]) -> dict[str, Any]:
    missing = [path for path in frame_paths if not Path(path).exists()]
    return {"n_paths": len(frame_paths), "n_missing": len(missing), "missing_preview": missing[:10]}
