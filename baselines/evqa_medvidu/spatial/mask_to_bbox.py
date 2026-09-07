from __future__ import annotations

from typing import Any

import numpy as np

from baselines.evqa_medvidu.config import MASK_TO_BBOX_VERSION


def tight_bbox(mask: Any) -> list[float] | None:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {arr.shape}")
    ys, xs = np.nonzero(arr > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def masklet_to_bboxes(masklet: Any) -> list[list[float] | None]:
    arr = np.asarray(masklet)
    if arr.ndim != 3:
        raise ValueError(f"masklet must be [T,H,W], got shape {arr.shape}")
    return [tight_bbox(arr[i]) for i in range(arr.shape[0])]


def mask_to_bbox_audit(sample_id: str, bboxes: list[list[float] | None]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "mask_to_bbox_version": MASK_TO_BBOX_VERSION,
        "n_frames": len(bboxes),
        "n_empty": sum(1 for box in bboxes if box is None),
        "n_non_empty": sum(1 for box in bboxes if box is not None),
    }

