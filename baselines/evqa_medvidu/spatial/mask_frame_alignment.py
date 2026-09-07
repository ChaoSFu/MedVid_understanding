from __future__ import annotations

from typing import Any


def build_mask_frame_alignment(row: dict[str, Any], n_mask_frames: int) -> list[dict[str, Any]]:
    frame_paths = row["ordered_frame_paths"]
    timestamps = row["local_timestamps"]
    source_indices = row["source_frame_indices"]
    if n_mask_frames != len(frame_paths):
        raise ValueError(f"mask frames ({n_mask_frames}) do not match benchmark frames ({len(frame_paths)})")
    return [
        {
            "mask_frame_index": i,
            "logical_medvidu_frame_index": i,
            "source_frame_index": int(source_indices[i]),
            "clip_local_timestamp": float(timestamps[i]),
            "frame_path": frame_paths[i],
        }
        for i in range(n_mask_frames)
    ]

